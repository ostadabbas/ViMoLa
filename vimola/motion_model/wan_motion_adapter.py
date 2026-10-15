"""
WanMotionAdapter: injects spatial motion tokens into a Wan DiT block via
parallel cross-attention.

Design principles:
    - Zero-gated init: at initialisation the adapter contributes exactly zero
      to the visual stream, so the base Wan model generates normally before
      any motion training signal arrives.
    - Drop-in: wraps the forward call of any nn.Module that returns visual
      tokens (B, L, D).  The motion cross-attention is added in parallel —
      visual self-attention is unchanged.
    - Standalone: no dependency on the Wan codebase; can be unit-tested
      with synthetic tensors.

Typical integration into Wan's transformer block:
    # Inside WanTransformerBlock.forward():
    hidden = self_attn(hidden) + ffn(hidden)
    if self.motion_adapter is not None:
        hidden = self.motion_adapter(hidden, motion_tokens)

Args (WanMotionAdapter.__init__):
    wan_hidden_dim:  hidden dim of the Wan DiT model (5B → 1536)
    motion_dim:      latent_dim of the VQ-VAE embeddings (default 128)
    num_heads:       attention heads for cross-attention (default 8)
    dropout:         attention dropout (default 0.0)
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WanMotionAdapter(nn.Module):
    """
    Parallel cross-attention adapter that injects motion tokens into a Wan
    DiT hidden state.

    Forward:
        visual_out = visual_tokens + gate * Normalize(CrossAttn(Q=visual, KV=motion))

    The gate is a learnable scalar initialised to 0 via tanh parametrisation:
        gate = tanh(gate_param)   (gate_param init = 0  →  gate = 0)

    Args:
        wan_hidden_dim: Wan DiT hidden dimension (e.g. 1536 for 5B)
        motion_dim:     VQ-VAE latent_dim (default 128)
        num_heads:      number of attention heads
        dropout:        attention dropout rate
    """

    def __init__(
        self,
        wan_hidden_dim: int = 1536,
        motion_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        rank: int = 0,
    ):
        super().__init__()
        assert wan_hidden_dim % num_heads == 0, (
            f"wan_hidden_dim ({wan_hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.wan_hidden_dim = wan_hidden_dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        self.head_dim = wan_hidden_dim // num_heads
        # rank=0 → full-width projections (published 1.14B config).
        # rank>0 → bottleneck inner dim for the Q/K/V/out maps (X3 low-rank).
        self.rank = rank
        inner = rank if rank and rank > 0 else wan_hidden_dim

        # Project motion tokens from VQ latent space → Wan hidden space
        self.motion_proj = nn.Linear(motion_dim, wan_hidden_dim, bias=False)

        # Layer norms
        self.norm_visual = nn.LayerNorm(wan_hidden_dim)
        self.norm_motion = nn.LayerNorm(wan_hidden_dim)

        # Q / K / V projections. Low-rank: D → r → D (two linears, no bias).
        if rank and rank > 0:
            self.q_proj = nn.Sequential(
                nn.Linear(wan_hidden_dim, inner, bias=False),
                nn.Linear(inner, wan_hidden_dim, bias=False),
            )
            self.k_proj = nn.Sequential(
                nn.Linear(wan_hidden_dim, inner, bias=False),
                nn.Linear(inner, wan_hidden_dim, bias=False),
            )
            self.v_proj = nn.Sequential(
                nn.Linear(wan_hidden_dim, inner, bias=False),
                nn.Linear(inner, wan_hidden_dim, bias=False),
            )
            self.out_proj = nn.Sequential(
                nn.Linear(wan_hidden_dim, inner, bias=False),
                nn.Linear(inner, wan_hidden_dim, bias=False),
            )
        else:
            self.q_proj = nn.Linear(wan_hidden_dim, wan_hidden_dim, bias=False)
            self.k_proj = nn.Linear(wan_hidden_dim, wan_hidden_dim, bias=False)
            self.v_proj = nn.Linear(wan_hidden_dim, wan_hidden_dim, bias=False)
            self.out_proj = nn.Linear(wan_hidden_dim, wan_hidden_dim, bias=False)

        self.attn_dropout = dropout

        # Zero-gate: gate = tanh(gate_param), init gate_param=0 → gate=0
        self.gate_param = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for module in [self.motion_proj, self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if hasattr(layer, "weight"):
                        nn.init.xavier_uniform_(layer.weight)
            elif hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        motion_tokens: torch.Tensor,
        motion_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, L, D_wan)  — visual sequence from Wan DiT
            motion_tokens: (B, P, D_motion) — spatial motion prior tokens
                           from SpatialMotionVQVAE.encode()
            motion_mask:   (B, P) bool, True = valid token (optional)
        Returns:
            (B, L, D_wan) updated visual tokens
        """
        B, L, D = visual_tokens.shape
        _, P, _ = motion_tokens.shape

        # Project + norm
        m = self.norm_motion(self.motion_proj(motion_tokens))  # (B, P, D_wan)
        v = self.norm_visual(visual_tokens)                    # (B, L, D_wan)

        # Q from visual, K/V from motion
        q = self.q_proj(v)   # (B, L, D_wan)
        k = self.k_proj(m)   # (B, P, D_wan)
        v_ = self.v_proj(m)  # (B, P, D_wan)

        # Reshape to multi-head: (B, heads, seq, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            b, s, _ = t.shape
            return t.reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)   # (B, H, L, hd)
        k = split_heads(k)   # (B, H, P, hd)
        v_ = split_heads(v_) # (B, H, P, hd)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, H, L, P)

        if motion_mask is not None:
            # motion_mask: (B, P) bool, True=valid → False=mask out
            inv_mask = ~motion_mask                              # (B, P)
            inv_mask = inv_mask[:, None, None, :]               # (B, 1, 1, P)
            attn = attn.masked_fill(inv_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        if self.training and self.attn_dropout > 0:
            attn = F.dropout(attn, p=self.attn_dropout)

        out = attn @ v_  # (B, H, L, hd)
        out = out.transpose(1, 2).reshape(B, L, D)  # (B, L, D_wan)
        out = self.out_proj(out)

        gate = torch.tanh(self.gate_param)
        # Normalize cross-attn output to hidden-state scale before gating.
        # Prevents cross-attn weights from growing unboundedly to compensate
        # for small gate values (root cause of hidden-state norm explosion).
        hidden_norm = visual_tokens.norm(dim=-1, keepdim=True).detach()
        out_norm    = out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out         = out * (hidden_norm / out_norm)
        return visual_tokens + gate * out

    def gate_value(self) -> float:
        """Return current gate magnitude (for logging)."""
        return torch.tanh(self.gate_param).item()


class MotionAdaptedBlock(nn.Module):
    """
    Convenience wrapper: takes an existing Wan transformer block and augments
    it with a WanMotionAdapter.  The block's original forward() is called
    first, then motion cross-attention is applied to its output.

    Usage:
        block = MotionAdaptedBlock(wan_block, wan_hidden_dim=1536, motion_dim=128)
        visual = block(visual, motion_tokens=motion_tokens)

    Note: only the adapter parameters are updated during motion training;
    ``wan_block`` can be frozen.
    """

    def __init__(
        self,
        wan_block: nn.Module,
        wan_hidden_dim: int = 1536,
        motion_dim: int = 128,
        num_heads: int = 8,
    ):
        super().__init__()
        self.wan_block = wan_block
        self.motion_adapter = WanMotionAdapter(wan_hidden_dim, motion_dim, num_heads)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        motion_tokens: Optional[torch.Tensor] = None,
        **wan_block_kwargs,
    ) -> torch.Tensor:
        visual_tokens = self.wan_block(visual_tokens, **wan_block_kwargs)
        if motion_tokens is not None:
            visual_tokens = self.motion_adapter(visual_tokens, motion_tokens)
        return visual_tokens


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _smoke():
    B, L, D_wan = 2, 512, 1536
    P, D_motion = 64, 128

    adapter = WanMotionAdapter(wan_hidden_dim=D_wan, motion_dim=D_motion, num_heads=8)

    visual = torch.randn(B, L, D_wan)
    motion = torch.randn(B, P, D_motion)

    # At init, gate=0 → output should equal input
    out = adapter(visual, motion)
    assert out.shape == (B, L, D_wan), out.shape
    assert torch.allclose(out, visual, atol=1e-6), "gate=0 init check failed"

    print("WanMotionAdapter smoke test passed")
    print(f"  gate (init): {adapter.gate_value():.6f}  (expected ~0.0)")
    print(f"  output shape: {out.shape}")

    # Verify gradients flow through gate
    loss = out.sum()
    loss.backward()
    assert adapter.gate_param.grad is not None
    print("  gradient through gate_param: OK")


if __name__ == "__main__":
    _smoke()
