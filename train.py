import torch
import torch.optim as optim
from core import GenesisDirectiveModel

def overfit_test(model, prompt, target, epochs=100, lr=1e-3, device="mps"):
    """
    Overfits the model on a single prompt-target pair to prove that the projection network 
    and any adapter layers successfully learn to pipe the latent plan 'z' into the autoregressive decoder.
    """
    print(f"Starting Overfit Test on device: {device}")
    
    # In the MVP, we only train the projection network. 
    # Planners and Decoders are frozen to isolate the Handoff Mechanism's effectiveness.
    trainable_params = list(model.projection.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr)
    
    # Store history
    history = {"total_loss": [], "coherence_loss": [], "fluency_loss": []}
    
    model.train()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(prompt, target)
        
        # Backward Pass
        loss = outputs["total_loss"]
        loss.backward()
        optimizer.step()
        
        # Logging
        c_loss = outputs["global_coherence_loss"].item()
        f_loss = outputs["local_fluency_loss"].item()
        t_loss = loss.item()
        
        history["coherence_loss"].append(c_loss)
        history["fluency_loss"].append(f_loss)
        history["total_loss"].append(t_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Total Loss: {t_loss:.4f} "
                  f"| Coherence: {c_loss:.4f} | Fluency: {f_loss:.4f}")
            
    print("Overfit test complete!")
    return history

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Using tiny models for the integration test pipeline since 
    # the 16+ GB native PyTorch models exceed the Apple Silicon unified memory
    # when loaded alongside all training gradients.
    planner_name = "sshleifer/tiny-gpt2"
    decoder_name = "sshleifer/tiny-gpt2"
    
    print("Initializing Genesis Directive Core Pipeline...")
    model = GenesisDirectiveModel(planner_name, decoder_name, device=device)
    model.to(torch.float32) # MPS safety
    model.to(device)
    
    prompt = "Create a brief story about a clever fox."
    target = "The clever brown fox quickly devised a plan to jump over the sleeping dog."
    
    print("\n--- Running 100 Epoch Overfit Test ---")
    print(f"Prompt: {prompt}")
    print(f"Target: {target}")
    
    history = overfit_test(model, prompt, target, epochs=100, lr=5e-3, device=device)

    print("\nVerifying learning...")
    loss_reduction = history["total_loss"][0] - history["total_loss"][-1]
    if loss_reduction > 0:
        print(f"SUCCESS: Model learned! Loss reduced by {loss_reduction:.4f}")
    else:
        print("WARNING: Model failed to learn. Check gradient flow.")
        
if __name__ == "__main__":
    main()
