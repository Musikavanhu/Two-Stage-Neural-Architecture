import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from core import GenesisDirectiveModel
import numpy as np
import time
import gc
import os

def prepare_data(num_samples=500):
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    print(f"Loading {num_samples} samples from WikiText-2 (seed 42)...")
    # Load more data to accommodate larger sample sizes
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20%]")
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 30)
    
    pairs = []
    for item in dataset:
        text = item['text'].strip()
        words = text.split()[:150]
        if len(words) < 10: continue
        mid = len(words) // 2
        prompt = " ".join(words[:mid])
        target = " ".join(words)
        pairs.append((prompt, target))
        if len(pairs) >= num_samples:
            break
    return pairs

def run_exp7_extended_training(planner_name, decoder_name, pairs, device, steps=1500, margin=4.0, prefix_len=64, dry_run=False):
    print("\n" + "="*70)
    print("EXPERIMENT 7: Phased Dual Projection Training")
    print(f"Goal: Close zero-gap (Phase 1) then impose structure sensitivity (Phase 2)")
    print("="*70)
    print(f"Config: arch=dual, total_steps={steps}, margin={margin}, prefix_len={prefix_len}")
    
    model = GenesisDirectiveModel(planner_name, decoder_name, device=device, proj_type='dual')
    model.to(device)

    proj_params = list(model.projection.parameters())
    print(f"Decoder frozen. Training {sum(p.numel() for p in proj_params):,} projection params.")
    
    optimizer = optim.AdamW(proj_params, lr=1e-4)
    
    model.eval()
    model.projection.train()
    import torch.nn.functional as F_

    ACCUMULATION_STEPS = 8

    train_pairs = pairs[:-50]
    val_pairs = pairs[-50:]

    # Phase split: 1200 alignment (drive to <0.03), 1500 contrastive
    PHASE1_STEPS = 1200 if not dry_run else 3
    PHASE2_STEPS = 1500 if not dry_run else 2
    TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS

    def get_target_embeds(target):
        """Extract decoder's own token embeddings as alignment target (no grad)."""
        dec_inputs = model.decoder.tokenizer(
            target, return_tensors="pt", padding=True, truncation=True, max_length=prefix_len
        )
        dec_input_ids = dec_inputs['input_ids'].to(device)
        target_embeds = model.decoder.model.get_input_embeddings()(dec_input_ids)
        t_len = target_embeds.shape[1]
        if t_len < prefix_len:
            pad = torch.zeros(1, prefix_len - t_len, target_embeds.shape[2],
                              device=device, dtype=target_embeds.dtype)
            target_embeds = torch.cat([target_embeds, pad], dim=1)
        else:
            target_embeds = target_embeds[:, :prefix_len, :]
        return target_embeds.float()

    def get_projected_z(prompt):
        """Run planner → interpolate → align → normalize → project (grad flows through projection)."""
        with torch.no_grad():
            z_raw = model.planner(prompt)
            z_raw = F_.interpolate(
                z_raw.transpose(1, 2), size=prefix_len, mode='linear', align_corners=False
            ).transpose(1, 2)
        z_aln = model._align_z(z_raw, prefix_len)
        z_normed = model.z_norm(z_aln)
        return model.projection(z_normed.to(torch.float32))

    def run_eval(label, max_samples=None):
        """Run ablation evaluation and print results. Returns (r, z, s, S)."""
        model.eval()
        losses = {"real": [], "zero": [], "shuffle": []}
        with torch.no_grad():
            for i, (prompt, target) in enumerate(val_pairs):
                if dry_run and i >= 2: break
                if max_samples and i >= max_samples: break
                losses["real"].append(model(prompt, target, prefix_len=prefix_len)["local_fluency_loss"].item())
                losses["zero"].append(model(prompt, target, ablation_type="zero", prefix_len=prefix_len)["local_fluency_loss"].item())
                losses["shuffle"].append(model(prompt, target, ablation_type="shuffle", prefix_len=prefix_len)["local_fluency_loss"].item())
        r = np.mean(losses['real'])
        z = np.mean(losses['zero'])
        s = np.mean(losses['shuffle'])
        dz = z - r
        ds = s - r
        S = ds - dz
        print(f"\n  [{label}] Real: {r:.4f} | Zero: {z:.4f} (Δ_z={dz:+.4f}) | Shuf: {s:.4f} (Δ_s={ds:+.4f}) | S={S:+.4f}")
        model.projection.train()
        return r, z, s, S

    # ================================================================
    # PHASE 1: Alignment — teach projection to speak the decoder's language
    # Extended to 1200 steps to drive alignment loss below 0.03
    # ================================================================
    print(f"\n--- PHASE 1: Embedding Alignment ({PHASE1_STEPS} steps, lr=1e-4) ---")
    print(f"  Objective: MSE(projected_z, decoder_target_embeds)")
    print(f"  Target: align loss < 0.03")

    step = 0
    accum_count = 0
    nan_count = 0
    start_time = time.time()
    optimizer.zero_grad()

    while step < PHASE1_STEPS:
        for prompt, target in train_pairs:
            if step >= PHASE1_STEPS:
                break
            with torch.no_grad():
                target_embeds = get_target_embeds(target)
            pz = get_projected_z(prompt)
            L_align = F_.mse_loss(pz, target_embeds.detach())

            if torch.isnan(L_align) or torch.isinf(L_align):
                nan_count += 1
                accum_count += 1
                if accum_count % ACCUMULATION_STEPS == 0:
                    optimizer.zero_grad()
                    step += ACCUMULATION_STEPS
                continue

            (L_align / ACCUMULATION_STEPS).backward()
            accum_count += 1

            if accum_count % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(proj_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += ACCUMULATION_STEPS
                if step % 100 == 0 or step == PHASE1_STEPS or (dry_run and step > 0):
                    elapsed = time.time() - start_time
                    print(f"  [P1] Step {step:4d}/{PHASE1_STEPS} | Align: {L_align.item():.4f} | {elapsed:.1f}s")

    # Flush
    if accum_count % ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(proj_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    print(f"  Phase 1 complete. Final align loss: {L_align.item():.4f}")

    # --- MID-PHASE CHECKPOINT EVALUATION ---
    print("\n--- Mid-Phase Evaluation (after Phase 1, before Phase 2) ---")
    run_eval("POST-P1", max_samples=20)

    # ================================================================
    # PHASE 2: Orthogonal Structural Head Training
    #
    # KEY INSIGHT from Phase 1 eval: alignment works (Δ_z = +1.61).
    # But when Phase 2 trains ALL projection weights, contrastive loss
    # escapes the decoder manifold entirely. When training just a gate,
    # it lacks capacity (S ~ -0.10).
    #
    # FIX: Train the dual-head architecture.
    # We freeze the semantic head (which now speaks the decoder's language)
    # and only train the structural head and the mixing scalar alpha.
    # To prevent the structural head from overwriting semantic information,
    # we add an orthogonality penalty: cos_sim(Z_sem, Z_struct)^2 -> 0.
    # ================================================================

    # Freeze semantic head
    for param in model.projection.semantic_head.parameters():
        param.requires_grad = False

    # Train structural head and alpha
    struct_params = list(model.projection.structural_head.parameters()) + [model.projection.alpha]
    struct_param_count = sum(p.numel() for p in struct_params)
    total_proj = sum(p.numel() for p in model.projection.parameters())
    frozen_count = total_proj - struct_param_count

    LR_PHASE2 = 3e-5
    MARGIN_START = 2.0
    MARGIN_END = margin  # 4.0
    ORTHO_WEIGHT = 0.5
    optimizer_p2 = optim.AdamW(struct_params, lr=LR_PHASE2)

    print(f"\n--- PHASE 2: Orthogonal Structural Head ({PHASE2_STEPS} steps, lr={LR_PHASE2}) ---")
    print(f"  Frozen: semantic_head ({frozen_count:,} params)")
    print(f"  Training: structural_head + alpha ({struct_param_count:,} params)")
    print(f"  Objective: relu(m(t) - ||Z - Z_shuffled||) + λ * cos_sim(Z_sem, Z_struct)²")
    print(f"  Margin schedule: {MARGIN_START} → {MARGIN_END}")

    step2 = 0
    accum_count = 0
    optimizer_p2.zero_grad()

    while step2 < PHASE2_STEPS:
        for prompt, target in train_pairs:
            if step2 >= PHASE2_STEPS:
                break

            # Margin schedule: linear ramp
            progress = step2 / max(PHASE2_STEPS, 1)
            current_margin = MARGIN_START + (MARGIN_END - MARGIN_START) * progress

            with torch.no_grad():
                z_raw = model.planner(prompt)
                z_raw = F_.interpolate(
                    z_raw.transpose(1, 2), size=prefix_len, mode='linear', align_corners=False
                ).transpose(1, 2)
            z_aln = model._align_z(z_raw, prefix_len)
            z_normed = model.z_norm(z_aln)
            
            z_fused, z_sem, z_struct = model.projection(z_normed.to(torch.float32), return_components=True)
            pz = z_fused

            # Contrastive: push ordered away from shuffled
            idx = torch.randperm(pz.shape[1], device=pz.device)
            shuffled = pz[:, idx, :].detach()
            dist_shuf = F_.mse_loss(pz, shuffled)
            L_contrastive = F_.relu(current_margin - dist_shuf)

            # Orthogonality constraint: cosine similarity squared
            cos_sim = F_.cosine_similarity(z_sem.flatten(1), z_struct.flatten(1), dim=-1).mean()
            L_ortho = cos_sim ** 2

            total = L_contrastive + ORTHO_WEIGHT * L_ortho

            if torch.isnan(total) or torch.isinf(total):
                nan_count += 1
                accum_count += 1
                if accum_count % ACCUMULATION_STEPS == 0:
                    optimizer_p2.zero_grad()
                    step2 += ACCUMULATION_STEPS
                continue

            (total / ACCUMULATION_STEPS).backward()
            accum_count += 1

            if accum_count % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(struct_params, max_norm=1.0)
                optimizer_p2.step()
                optimizer_p2.zero_grad()
                step2 += ACCUMULATION_STEPS
                if step2 % 100 == 0 or step2 == PHASE2_STEPS or (dry_run and step2 > 0):
                    elapsed = time.time() - start_time
                    print(f"  [P2] Step {step2:4d}/{PHASE2_STEPS} | m={current_margin:.2f} | Contrast: {L_contrastive.item():.4f} | dist: {dist_shuf.item():.4f} | Ortho: {L_ortho.item():.4f} | α: {model.projection.alpha.item():.4f} | {elapsed:.1f}s")

    # Flush
    if accum_count % ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
        optimizer_p2.step()
        optimizer_p2.zero_grad()

    print(f"\nBoth phases complete ({PHASE1_STEPS + PHASE2_STEPS} total steps, {nan_count} NaN).")

    # --- FINAL EVALUATION ---
    print("\n--- Final Evaluation ---")
    r, z, s, S_metric = run_eval("FINAL")
    
    zeroed_gap = z - r
    shuffled_gap = s - r

    print("\n" + "-"*50)
    print("EXPERIMENT 7 FINAL RESULTS")
    print("-"*50)
    print(f"Real Latent Plan Fluency Loss:    {r:.4f}")
    print(f"Zeroed Latent Plan Fluency Loss:  {z:.4f} (Δ_z = {zeroed_gap:+.4f})")
    print(f"Shuffled Latent Plan Fluency Loss:{s:.4f} (Δ_s = {shuffled_gap:+.4f})")
    print(f"Wrong-Structure Penalty (S):      {S_metric:+.4f} nats")
    
    print("\n--- Verdict ---")
    if zeroed_gap > 0 and S_metric > 0:
        print("-> SUCCESS: ZERO-GAP CLOSED!")
        print("   The real plan is now strictly better than silence (Δ_z > 0), and")
        print("   wrong structure is strictly worse than silence (S > 0).")
        print("   The projection has fully aligned with the decoder's expectations.")
    elif S_metric > 0:
        print("-> PARTIAL: S-metric confirmed (Structure sensitive), but Zero-Gap still negative.")
        print("   Try training for more steps (e.g. 3000) or lowering the learning rate.")
    else:
        print("-> FAILED: Model lost structure sensitivity.")

    if not dry_run:
        save_path = "exp7_gated_projection.pt"
        torch.save(model.projection.state_dict(), save_path)
        print(f"\nSaved trained projection to {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a quick 5-step validation to check for errors")
    parser.add_argument("--steps", type=int, default=1500, help="Number of training steps")
    parser.add_argument("--samples", type=int, default=500, help="Number of data pairs to load")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    planner_name = "/Users/tinomusikavanhu/.lmstudio/models/MCES10/gpt2-large-hf"
    decoder_name = "gpt2-xl"
    
    pairs = prepare_data(num_samples=args.samples)
    
    run_exp7_extended_training(
        planner_name, 
        decoder_name, 
        pairs, 
        device, 
        steps=args.steps, 
        margin=4.0, 
        prefix_len=64,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
