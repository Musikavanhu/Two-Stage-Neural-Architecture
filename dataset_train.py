import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from core import GenesisDirectiveModel
import matplotlib.pyplot as plt

def prepare_dataset(dataset_name="wikitext", config_name="wikitext-2-raw-v1", split="train[:1%]"):
    print(f"Loading {dataset_name} ({config_name} - {split})...")
    dataset = load_dataset(dataset_name, config_name, split=split)
    # Filter empty lines
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 30)
    
    pairs = []
    for item in dataset:
        text = item['text'].strip()
        # Split into prompt and target
        words = text.split()
        if len(words) < 10:
            continue
        mid = len(words) // 2
        prompt = " ".join(words[:mid])
        target = text
        pairs.append((prompt, target))
    return pairs

def train_loop(model, dataloader, epochs=5, lr=1e-3, device="mps"):
    print(f"Starting Dataset Training on device: {device}")
    
    trainable_params = list(model.projection.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr)
    
    history = {"total_loss": [], "coherence_loss": [], "fluency_loss": []}
    
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_t_loss = 0
        epoch_c_loss = 0
        epoch_f_loss = 0
        
        for batch_idx, (prompts, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # For simplicity in MVP, we process batch sizes of 1 inside the loop, 
            # modifying core.py for batched inputs natively requires padding logic.
            batch_loss = 0
            for prompt, target in zip(prompts, targets):
                outputs = model(prompt, target)
                loss = outputs["total_loss"]
                loss.backward()
                
                batch_loss += loss.item()
                epoch_c_loss += outputs["global_coherence_loss"].item()
                epoch_f_loss += outputs["local_fluency_loss"].item()
            
            optimizer.step()
            epoch_t_loss += batch_loss
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {batch_loss/len(prompts):.4f}")
        
        num_samples = len(dataloader.dataset)
        avg_t_loss = epoch_t_loss / num_samples
        avg_c_loss = epoch_c_loss / num_samples
        avg_f_loss = epoch_f_loss / num_samples
        
        history["total_loss"].append(avg_t_loss)
        history["coherence_loss"].append(avg_c_loss)
        history["fluency_loss"].append(avg_f_loss)
        
        print(f"--- Epoch {epoch} Summary ---")
        print(f"Total Loss: {avg_t_loss:.4f} | Coherence: {avg_c_loss:.4f} | Fluency: {avg_f_loss:.4f}")
            
    print("Training complete!")
    return history

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    planner_name = "sshleifer/tiny-gpt2"
    decoder_name = "sshleifer/tiny-gpt2"
    
    print("Initializing Genesis Directive Core Pipeline...")
    model = GenesisDirectiveModel(planner_name, decoder_name, device=device)
    model.to(torch.float32)
    model.to(device)
    
    pairs = prepare_dataset(split="train[:2%]") # Grab a small slice of WikiText
    print(f"Created {len(pairs)} prompt-target pairs.")
    
    # Batch size 4 means 4 forward passes accumulated before stepping optimizer
    dataloader = DataLoader(pairs, batch_size=4, shuffle=True)
    
    history = train_loop(model, dataloader, epochs=5, lr=2e-3, device=device)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 6), history["total_loss"], marker='o', label="Total Loss")
    plt.plot(range(1, 6), history["coherence_loss"], marker='s', label="Coherence Loss")
    plt.plot(range(1, 6), history["fluency_loss"], marker='^', label="Fluency Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Genesis Directive: Training Over WikiText-2 (Tiny Models)")
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("Saved loss_curve.png to disk!")

if __name__ == "__main__":
    main()
