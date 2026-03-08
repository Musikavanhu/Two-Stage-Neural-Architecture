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
