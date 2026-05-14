import torch
import torch.nn as nn


class LatentProjection(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, **kwargs):
        """
        Bridges the continuous latent space of the diffusion planner (gpt2-large)
        to the embedding space of the autoregressive decoder (gpt2-xl).
        """
        super().__init__()
        if hidden_features is None:
            hidden_features = (in_features + out_features) // 2

        self.proj_type = kwargs.get("proj_type", "standard")

        if self.proj_type == "standard":
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.LayerNorm(hidden_features),
                nn.Linear(hidden_features, out_features)
            )
        elif self.proj_type == "wide":
            hidden_features = hidden_features * 4
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.LayerNorm(hidden_features),
                nn.Linear(hidden_features, out_features)
            )
        elif self.proj_type == "residual":
            self.up  = nn.Linear(in_features, out_features)
            self.mlp = nn.Sequential(
                nn.LayerNorm(out_features),
                nn.Linear(out_features, hidden_features * 2),
                nn.GELU(),
                nn.Linear(hidden_features * 2, out_features)
            )
        elif self.proj_type == "gated":
            # SwiGLU-style: gate(x) * up(x) -> norm -> out
            # LayerNorm after gate*up is CRITICAL — without it the product
            # explodes to inf in fp32 and overflows to NaN when cast to fp16.
            self.w1   = nn.Linear(in_features, hidden_features)
            self.w2   = nn.Linear(in_features, hidden_features)
            self.norm = nn.LayerNorm(hidden_features)   # the missing stabiliser
            self.w3   = nn.Linear(hidden_features, out_features)
            self.act  = nn.SiLU()

    def forward(self, z):
        """
        Args:
            z: (batch, seq_len, in_features) in fp32
        Returns:
            (batch, seq_len, out_features) in fp32
        """
        if self.proj_type in ["standard", "wide"]:
            return self.net(z)
        elif self.proj_type == "residual":
            z_proj = self.up(z)
            return z_proj + self.mlp(z_proj)
        elif self.proj_type == "gated":
            gate   = self.act(self.w1(z))
            up     = self.w2(z)
            hidden = self.norm(gate * up)   # LayerNorm prevents explosion
            return self.w3(hidden)
        return self.net(z)


class DualProjection(nn.Module):
    """
    Orthogonal dual-head projection: Z = Z_semantic + α · Z_structural
    
    Experimental motivation (Experiment 7):
      - Full alignment training achieves Δ_z = +1.61 (semantic utility)
      - Full contrastive training achieves S = +1.37 (structure sensitivity)  
      - But they COMPETE in a shared manifold: training one destroys the other
    
    Solution: separate heads with orthogonality constraint.
      - semantic_head: aligned to decoder token embeddings (trained Phase 1)
      - structural_head: encodes order-sensitive perturbations (trained Phase 2)
      - orthogonality loss: cos_sim(Z_sem, Z_struct)² → 0
      - alpha: learned scalar controlling structural contribution magnitude
    """
    def __init__(self, in_features, out_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = (in_features + out_features) // 2

        # Semantic head: full-capacity MLP aligned to decoder manifold
        self.semantic_head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, out_features)
        )

        # Structural head: smaller MLP for order-sensitive perturbations
        # Half hidden size — it only needs to encode structural deltas,
        # not the full semantic content
        struct_hidden = hidden_features // 2
        self.structural_head = nn.Sequential(
            nn.Linear(in_features, struct_hidden),
            nn.GELU(),
            nn.LayerNorm(struct_hidden),
            nn.Linear(struct_hidden, out_features)
        )

        # Learned mixing scalar (initialized small so structural starts near-zero)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, return_components=False):
        """
        Args:
            z: (batch, seq_len, in_features) in fp32
            return_components: if True, also return Z_sem and Z_struct separately
        Returns:
            Z_fused: (batch, seq_len, out_features) = Z_sem + α * Z_struct
            (optionally) Z_sem, Z_struct
        """
        z_sem = self.semantic_head(z)
        z_struct = self.structural_head(z)
        z_fused = z_sem + self.alpha * z_struct

        if return_components:
            return z_fused, z_sem, z_struct
        return z_fused
