import torch
import torch.nn as nn
import torch.nn.functional as F
from models.diffusion_planner import DiffusionPlanner
from models.autoregressive_decoder import AutoregressiveDecoder
from models.projection import LatentProjection, DualProjection

class GenesisDirectiveModel(nn.Module):
    def __init__(self, planner_path, decoder_path, device="mps", proj_type="standard"):
        super().__init__()
        self.device = device

        print("Loading Diffusion Planner (LLaDA)...")
        self.planner = DiffusionPlanner(planner_path, device=device)

        print("Loading Autoregressive Decoder (GPT-2)...")
        self.decoder = AutoregressiveDecoder(decoder_path, device=device)

        in_features  = self.planner.model.config.hidden_size
        out_features = self.decoder.model.config.n_embd

        self.z_norm   = nn.LayerNorm(in_features).to(device)
        print(f"Initializing Projection Network ({in_features} -> {out_features}) [Type: {proj_type}]...")
        if proj_type == "dual":
            self.projection = DualProjection(in_features, out_features).to(device)
        else:
            self.projection = LatentProjection(in_features, out_features, proj_type=proj_type).to(device)

    def _align_z(self, z, target_len):
        b, seq, h = z.shape
        if seq > target_len:
            return z[:, :target_len, :]
        elif seq < target_len:
            pad = torch.zeros((b, target_len - seq, h), device=self.device, dtype=z.dtype)
            return torch.cat([z, pad], dim=1)
        return z

    def _project_fp32(self, z_raw, target_len):
        """Planner output → aligned → normed → projected, all in fp32."""
        z = self._align_z(z_raw, target_len)
        z = self.z_norm(z)
        out = self.projection(z.to(torch.float32))   # fp32 matmul (no fp16 anywhere)
        return out   # fp32, no cast

    def forward(
        self,
        prompt,
        target_text,
        wrong_prompt=None,
        ablation_type=None,
        prefix_len=None,
        output_attentions=False,
        contrastive_margin=1.0,
        contrastive_lambda=0.05,
        shuffled_z_contrastive=False,
        # When True: fluency is a NO-GRAD reference; contrastive trains projection
        # using embedding-space L2 distance (pure fp32, no fp16 decoder backward)
        contrastive_grad_only=False,
    ):
        # --- PHASE 1: PLANNER (always no_grad) ---
        with torch.no_grad():
            z_star_embeds, z_star_mask = self.planner.get_target_latents(target_text)
            z_predicted = self.planner(prompt)

        z_star_len = z_star_embeds.shape[1]
        if prefix_len is not None and prefix_len != z_star_len:
            z_predicted = F.interpolate(
                z_predicted.transpose(1, 2), size=prefix_len, mode='linear', align_corners=False
            ).transpose(1, 2)
            z_star_len = prefix_len

        # Coherence loss (diagnostic only) — only valid when z has same length as z_star
        with torch.no_grad():
            if prefix_len is None or prefix_len == z_star_embeds.shape[1]:
                z_aln_diag = self._align_z(z_predicted, z_star_len)
                mask_exp = z_star_mask.unsqueeze(-1).expand_as(z_star_embeds)
                coherence_loss = F.mse_loss(z_aln_diag * mask_exp, z_star_embeds * mask_exp)
            else:
                coherence_loss = torch.tensor(0.0, device=self.device)


        # --- PHASE 2: PROJECT (fp32) ---
        projected_z_fp32 = self._project_fp32(z_predicted, z_star_len)   # fp32, has gradient
        decoder_dtype = next(self.decoder.model.parameters()).dtype
        # Cast to fp16 for decoder, clamped to safe range
        projected_z = projected_z_fp32.clamp(-5.0, 5.0).to(decoder_dtype)

        if ablation_type == "zero":
            projected_z = torch.zeros_like(projected_z)
        elif ablation_type == "shuffle":
            idx = torch.randperm(projected_z.shape[1], device=projected_z.device)
            projected_z = projected_z[:, idx, :]

        # --- PHASE 3: DECODER ---
        if contrastive_grad_only:
            # Detach: no fp16 gradient flows back to the fp32 projection through the decoder
            with torch.no_grad():
                fluency_loss, logits, attentions = self.decoder(
                    projected_z.detach(), target_text, output_attentions=output_attentions
                )
        else:
            fluency_loss, logits, attentions = self.decoder(
                projected_z, target_text, output_attentions=output_attentions
            )

        # --- PHASE 4: CONTRASTIVE COUPLING ---
        contrastive_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        if shuffled_z_contrastive:
            with torch.no_grad():
                idx = torch.randperm(projected_z.shape[1], device=projected_z.device)
                shuffled_z = projected_z[:, idx, :]
                shuffled_fluency, _, _ = self.decoder(shuffled_z, target_text)
            r = fluency_loss.float().clamp(-50, 50)
            n = shuffled_fluency.float().clamp(-50, 50)
            contrastive_loss = F.relu(contrastive_margin - (n - r))

        elif wrong_prompt is not None:
            with torch.no_grad():
                wrong_z_raw = self.planner(wrong_prompt)

            if contrastive_grad_only:
                # Pure fp32 embedding-space contrastive loss.
                # No decoder involved: push projected_z and wrong_projected_z apart in L2 space.
                # Gradient is entirely in fp32, never touches fp16 LoRA.
                wrong_projected_fp32 = self._project_fp32(wrong_z_raw, z_star_len)  # fp32, grad
                # L2 distance between real and wrong projections (mean over seq × hidden)
                dist = F.mse_loss(projected_z_fp32, wrong_projected_fp32.detach())
                # Hinge: penalise if the projections are NOT separated by margin
                contrastive_loss = F.relu(contrastive_margin - dist)
                # Also: we still need fluency reference for total_loss scale
                # Use detached fluency_loss (no gradient)
            else:
                with torch.no_grad():
                    wrong_p_fp32 = self._project_fp32(wrong_z_raw, z_star_len)
                    wrong_p = wrong_p_fp32.clamp(-5.0, 5.0).to(decoder_dtype)
                    wrong_fluency, _, _ = self.decoder(wrong_p, target_text)
                r = fluency_loss.float().clamp(-50, 50)
                n = wrong_fluency.float().clamp(-50, 50)
                contrastive_loss = F.relu(contrastive_margin - (n - r))

        # --- TOTAL LOSS ---
        if contrastive_grad_only:
            # Only contrastive_loss has gradient (embedding-space, pure fp32)
            # fluency_loss is detached reference for logging
            total_loss = contrastive_lambda * contrastive_loss
        else:
            total_loss = fluency_loss.float() + contrastive_lambda * contrastive_loss

        return {
            "global_coherence_loss": coherence_loss,
            "local_fluency_loss":    fluency_loss,
            "contrastive_loss":      contrastive_loss,
            "total_loss":            total_loss,
            "projected_z":           projected_z,
            "attentions":            attentions,
        }
