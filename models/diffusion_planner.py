import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class DiffusionPlanner(nn.Module):
    def __init__(self, model_path, device="mps"):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # NOTE: use `dtype` not `torch_dtype` to suppress the deprecation warning
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16,
        ).to(self.device)

        # Freeze the planner — it acts as a fixed encoder
        for param in self.model.parameters():
            param.requires_grad = False

    def get_target_latents(self, text):
        """
        Single forward pass → last-layer hidden states as z*.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            return_attention_mask=True,
            truncation=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_size)

        return hidden_states, inputs['attention_mask']

    def denoise(self, prompt, gen_length=64, steps=32, temperature=0.0):
        """
        Latent plan extraction: ONE forward pass, return last-layer hidden states.

        Why not generate():
          - generate(max_new_tokens=64) = 64 sequential steps through 774M params
            ≈ 60-120 seconds per sample on MPS  →  unusable
          - Single forward pass ≈ 0.1-0.3 seconds per sample (200-400x faster)

        The hidden states are a contextualised encoding of the prompt across all
        36 transformer layers — richer than raw token embeddings and sufficient
        to serve as Z in the Genesis Directive architecture.

        gen_length: controls output Z sequence length (pad/trim to match).
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=128
        )
        input_ids = inputs.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # (1, prompt_len, 1280)

            prompt_len = hidden_states.shape[1]
            if prompt_len >= gen_length:
                z = hidden_states[:, :gen_length, :]
            else:
                # Repeat the final token's hidden state to fill to gen_length
                pad = hidden_states[:, -1:, :].expand(1, gen_length - prompt_len, -1)
                z = torch.cat([hidden_states, pad], dim=1)

        return z  # (1, gen_length, 1280)

    def forward(self, prompt, gen_length=64, steps=32):
        return self.denoise(prompt, gen_length=gen_length, steps=steps)
