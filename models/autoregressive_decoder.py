import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class AutoregressiveDecoder(nn.Module):
    def __init__(self, model_path, device="mps"):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Add a pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        ).to(self.device)
        


        # Apply LoRA for memory-safe scaling
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj", "c_fc"] # GPT-2 specific Linear layers
        )
        self.model = get_peft_model(self.model, lora_config)
        # Strategy: base model weights stay fp16 (memory efficient),
        # but LoRA adapter weights (lora_A, lora_B) are kept in fp32 for stable AdamW gradient computation.
        # PEFT internally casts the lora output to the base dtype before addition, so forward works fine.
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(torch.float16)
        self.model.print_trainable_parameters()
            
    def forward(self, projected_z, target_text, output_attentions=False):
        """
        Training forward pass. 
        Prepends the `projected_z` continuous vectors to the target text embeddings 
        to compute Local Fluency Loss (Causal LM loss).
        """
        # 1. Get input IDs and raw embeddings for target text
        inputs = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        text_embeds = self.model.get_input_embeddings()(inputs['input_ids'])
        
        # 2. Soft Prefixing: Concat [Projected_Z, Text_Embeddings] along sequence dimension
        # projected_z: (batch, planner_seq_len, gpt2_dim)
        # text_embeds: (batch, target_seq_len, gpt2_dim)
        # Clamp projected_z to fp16-safe range BEFORE casting.
        # PEFT Conv1D base layer has fp16 weights — must cast to fp16.
        # After optimizer steps, projection outputs can drift to ±8+ causing fp16 LayerNorm NaN.
        # ±5.0 is ~5σ from normal init, preserves all meaningful signal.
        projected_z = projected_z.clamp(-5.0, 5.0).to(text_embeds.dtype)
        combined_embeds = torch.cat([projected_z, text_embeds], dim=1)


        # 3. Create extended attention mask (assuming projected_z is fully attended to)
        z_seq_len = projected_z.shape[1]
        batch_size = projected_z.shape[0]
        z_mask = torch.ones((batch_size, z_seq_len), dtype=inputs['attention_mask'].dtype, device=self.device)
        combined_mask = torch.cat([z_mask, inputs['attention_mask']], dim=1)
        
        # 4. Create dummy labels. We don't want to calculate loss over the projected_z prefix tokens,
        # so we set their labels to -100 (CrossEntropy ignores -100).
        z_labels = torch.full((batch_size, z_seq_len), -100, dtype=torch.long, device=self.device)
        text_labels = inputs['input_ids'].clone()
        # Set padding tokens in text labels to -100 too
        text_labels[text_labels == self.tokenizer.pad_token_id] = -100
        combined_labels = torch.cat([z_labels, text_labels], dim=1)
        
        # 5. Forward pass — get logits WITHOUT asking the model to compute loss internally.
        # Computing loss inside the fp16 model causes NaN gradients on MPS.
        # Instead we compute CE loss manually in fp32.
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        # 6. Compute CE loss in fp32 (logits are fp16 from model, cast to fp32 for stability)
        logits_fp32 = outputs.logits.float()  # cast to fp32
        # Shift: labels are from position 1 onward, logits from position 0
        shift_logits = logits_fp32[:, :-1, :].contiguous()
        shift_labels = combined_labels[:, 1:].contiguous()
        import torch.nn.functional as F
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
        
        return loss, outputs.logits, (outputs.attentions if output_attentions else None)


    def generate(self, projected_z, max_new_tokens=50):
        """
        Inference pass: generate tokens autoregressively conditioned solely on the 
        projected latent plan prefix.
        """
        z_seq_len = projected_z.shape[1]
        batch_size = projected_z.shape[0]
        z_mask = torch.ones((batch_size, z_seq_len), dtype=torch.long, device=self.device)
        
        # Generate kwargs bypassing input_ids initially
        output_ids = self.model.generate(
            inputs_embeds=projected_z,
            attention_mask=z_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        
        # Decode the generated output
        generated_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return generated_text
