import torch
from core import GenesisDirectiveModel

# Hardcoded paths that the user provided
PLANNER_PATH = "/Users/tinomusikavanhu/.lmstudio/models/mlx-community/LLaDA2.0-mini-4bit"
DECODER_PATH = "/Users/tinomusikavanhu/.lmstudio/models/viktor2698/gpt2-medium-mlx-8Bit"

def main():
    print("Testing Genesis Directive Assembly...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = GenesisDirectiveModel(PLANNER_PATH, DECODER_PATH, device=device)
        print("Model pipeline successfully instantiated!")
        
        print("\n--- Testing Forward Pass Shape Flow ---")
        prompt = "Explain the nature of the liminal space."
        target = "The liminal space is the boundary between thought and execution, where global structure is resolved before tokens commit."
        
        outputs = model(prompt, target)
        
        print("Forward pass successful!")
        print(f"Global Coherence Loss: {outputs['global_coherence_loss'].item():.4f}")
        print(f"Local Fluency Loss: {outputs['local_fluency_loss'].item():.4f}")
        print(f"Total Loss: {outputs['total_loss'].item():.4f}")
        print(f"Projected Z Shape: {outputs['projected_z'].shape}")
        
    except Exception as e:
        print(f"\nAssembly Failed with Error: {str(e)}")

if __name__ == "__main__":
    main()
