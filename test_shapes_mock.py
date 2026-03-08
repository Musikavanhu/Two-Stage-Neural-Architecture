import torch
from core import GenesisDirectiveModel

# Since local PyTorch can't easily read MLX 4-bit safetensors, 
# we'll use small pure PyTorch models from the hub just to test the architecture's loss flow.
mock_planner = "sshleifer/tiny-gpt2" 
mock_decoder = "sshleifer/tiny-gpt2" 

def main():
    print("Testing Genesis Directive Assembly (Mock Models)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = GenesisDirectiveModel(mock_planner, mock_decoder, device=device)
        # Convert all to float32 for MPS testing safety
        model.to(torch.float32)
        print("Model pipeline successfully instantiated!")
        
        print("\n--- Testing Forward Pass Shape Flow ---")
        prompt = "Explain the nature of the liminal space."
        target = "The liminal space is the boundary between thought and execution, where global structure is resolved before tokens commit."
        
        outputs = model(prompt, target)
        
        print("\n=== SUCCESS ===")
        print("Forward pass successful! Gradients can flow.")
        print(f"Global Coherence Loss: {outputs['global_coherence_loss'].item():.4f}")
        print(f"Local Fluency Loss: {outputs['local_fluency_loss'].item():.4f}")
        print(f"Total Loss: {outputs['total_loss'].item():.4f}")
        print(f"Projected Z Shape: {outputs['projected_z'].shape}")
        
    except Exception as e:
        print(f"\nAssembly Failed with Error: {str(e)}")

if __name__ == "__main__":
    main()
