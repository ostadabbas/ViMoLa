"""
TextToMotionTransformer: predicts spatial motion tokens from text + initial positions.

This is the "sampling from the prior" step in the ViMoLa pipeline:
    text prompt + initial point positions → codebook indices (B, P)

Architecture (parallel, non-autoregressive):
    1. Patch initial positions: (B, N, 2) → spatial patch centroids (B, P, 2)
       → linear embed + learned patch-position encoding → patch queries (B, P, D)
    2. Encode text: (B, L, D_text) → (B, L, D_model) via linear projection
    3. Alternating cross-attn (Q=patches, KV=text) + self-attn blocks
    4. Output head: (B, P, D) → (B, P, codebook_size) logits
       → argmax at inference, softmax-cross-entropy loss during training

Training target: VQ indices produced by SpatialMotionVQVAE.encode(real_velocities)
Loss: cross-entropy on (B*P, codebook_size) logits vs (B*P,) target indices

Args:
    codebook_size:  number of VQ codes K (must match SpatialMotionVQVAE)
    num_patches:    P spatial patches (must match SpatialMotionVQVAE.num_patches)
    patch_size:     spatial patch side (needed to compute patch centroids)
    grid_size:      point grid side (needed to compute patch centroids)
    text_dim:       input text embedding dimension (e.g. 4096 for T5-XXL)
    model_dim:      internal transformer hidden dimension
    num_heads:      attention heads
    num_layers:     number of cross-attn + self-attn stacked blocks
    dropout:        dropout rate
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal encoding for (x, y) coordinates in [0, 1]²."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D sinusoidal encoding"
        self.dim = dim

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: (..., 2) normalised coordinates in [0, 1]²
        Returns:
            encoding: (..., dim)
        """
        half = self.dim // 2
        freq = torch.arange(half // 2, device=xy.device, dtype=xy.dtype)
        freq = 1.0 / (10000 ** (freq / (half // 2)))

        x = xy[..., 0:1] * freq  # (..., half//2)
        y = xy[..., 1:2] * freq

        enc = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)  # (..., dim)
        return enc


class MotionPatchEmbedding(nn.Module):
    """
    Embeds the initial positions of each spatial patch into D_model.

    For each patch: compute the centroid of its point initial positions,
    then embed via sinusoidal encoding + learned linear projection.
    """

    def __init__(
        self,
        grid_size: int = 32,
        patch_size: int = 4,
        model_dim: int = 512,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        pps = grid_size // patch_size
        self.num_patches = pps * pps
        self.patch_pts = patch_size * patch_size

        self.pos_enc = SinusoidalPositionEncoding(model_dim)
        # Additional learned offset per patch index
        self.patch_embed = nn.Embedding(self.num_patches, model_dim)

    def forward(self, initial_position: torch.Tensor) -> torch.Tensor:
        """
        Args:
            initial_position: (B, N, 2) normalised to [0, 1]²
                              N = grid_size²
        Returns:
            patch_queries: (B, P, model_dim)
        """
        B, N, _ = initial_position.shape
        gs, ps = self.grid_size, self.patch_size
        pps = gs // ps
        P = self.num_patches

        # Reshape to grid → group into patches → compute centroid
        # (B, N, 2) → (B, gs, gs, 2)
        pos = initial_position.reshape(B, gs, gs, 2)
        # (B, pps, ps, pps, ps, 2) → permute → (B, pps, pps, ps, ps, 2)
        pos = pos.reshape(B, pps, ps, pps, ps, 2)
        pos = pos.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, pps, pps, ps, ps, 2)
        pos = pos.reshape(B, P, self.patch_pts, 2)
        centroids = pos.mean(dim=2)  # (B, P, 2)

        # Sinusoidal embedding from centroids
        emb = self.pos_enc(centroids)  # (B, P, model_dim)

        # Add learned patch-index embedding
        patch_idx = torch.arange(P, device=initial_position.device)
        emb = emb + self.patch_embed(patch_idx).unsqueeze(0)  # (B, P, D)
        return emb


class CrossAttentionBlock(nn.Module):
    """One cross-attention layer: Q from patches, KV from text."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(model_dim)
        self.norm_kv = nn.LayerNorm(model_dim)
        self.norm_out = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries:      (B, P, D)
            context:      (B, L, D)
            context_mask: (B, L) bool, True = valid token
        Returns:
            (B, P, D)
        """
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask  # MHA expects True = ignore

        q = self.norm_q(queries)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_out(queries))
        return queries


class SelfAttentionBlock(nn.Module):
    """One self-attention layer over patch tokens."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TextToMotionTransformer(nn.Module):
    """
    Predicts spatial motion prior tokens from text and initial point positions.

    At inference: argmax over logits → (B, P) indices → VQ-VAE decode → motion field
    At training:  cross-entropy loss against real VQ indices from SpatialMotionVQVAE

    The model is deliberately small (default ~30M params) since the motion
    space is strongly constrained by the codebook.

    Args:
        codebook_size:  K codes in the VQ-VAE (default 1024)
        num_patches:    P spatial patches (default 64)
        patch_size:     VQ-VAE patch_size (default 4, needed for centroid calc)
        grid_size:      VQ-VAE grid_size  (default 32)
        text_dim:       dimension of input text embeddings (default 4096 for T5-XXL)
        model_dim:      internal hidden dimension (default 512)
        num_heads:      attention heads (default 8)
        num_layers:     stacked cross+self attention blocks (default 4)
        dropout:        dropout rate
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        num_patches: int = 64,
        patch_size: int = 4,
        grid_size: int = 32,
        text_dim: int = 4096,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        head_type: str = "discrete",
        continuous_dim: int = 128,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_patches = num_patches
        self.model_dim = model_dim
        self.head_type = head_type
        self.continuous_dim = continuous_dim

        # Initial position → patch queries
        self.patch_embed = MotionPatchEmbedding(grid_size, patch_size, model_dim)

        # Text → model_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # Alternating cross-attn + self-attn blocks
        self.cross_attn_blocks = nn.ModuleList(
            [CrossAttentionBlock(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.self_attn_blocks = nn.ModuleList(
            [SelfAttentionBlock(model_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.norm_out = nn.LayerNorm(model_dim)
        self.logit_head = nn.Linear(model_dim, codebook_size)
        # X4: only allocated for the continuous-head retrain so published
        # discrete checkpoints still load with strict=True.
        if head_type == "continuous":
            self.continuous_head = nn.Linear(model_dim, continuous_dim)
        else:
            self.continuous_head = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.logit_head.weight, std=0.02)
        nn.init.zeros_(self.logit_head.bias)
        if self.continuous_head is not None:
            nn.init.normal_(self.continuous_head.weight, std=0.02)
            nn.init.zeros_(self.continuous_head.bias)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        initial_position: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_embeddings:  (B, L, text_dim)  — e.g. T5 encoder output
            initial_position: (B, N, 2)          — normalised to [0, 1]²
            text_mask:        (B, L) bool, True = valid token
        Returns:
            logits: (B, P, codebook_size)
        """
        # Patch queries from spatial initial positions
        queries = self.patch_embed(initial_position)  # (B, P, D)

        # Project text
        context = self.text_proj(text_embeddings)     # (B, L, D)

        # Alternating cross-attn (text grounding) + self-attn (inter-patch)
        for cross_blk, self_blk in zip(self.cross_attn_blocks, self.self_attn_blocks):
            queries = cross_blk(queries, context, text_mask)
            queries = self_blk(queries)

        queries = self.norm_out(queries)              # (B, P, D)
        if self.head_type == "continuous":
            return self.continuous_head(queries)      # (B, P, continuous_dim)
        return self.logit_head(queries)               # (B, P, codebook_size)

    def predict_indices(
        self,
        text_embeddings: torch.Tensor,
        initial_position: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample or argmax motion token indices.

        Args:
            text_embeddings:  (B, L, text_dim)
            initial_position: (B, N, 2)
            text_mask:        (B, L) bool optional
            temperature:      >1 → more diversity; =0 → greedy argmax
        Returns:
            indices: (B, P) int64
        """
        logits = self.forward(text_embeddings, initial_position, text_mask)  # (B, P, K)
        if temperature == 0.0 or not self.training:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)  # (B, P, K)
        B, P, K = probs.shape
        return torch.multinomial(probs.reshape(B * P, K), num_samples=1).reshape(B, P)

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss over all patches.

        Args:
            logits:         (B, P, codebook_size)
            target_indices: (B, P) int64 — from SpatialMotionVQVAE.encode()
        Returns:
            loss: scalar
        """
        B, P, K = logits.shape
        return F.cross_entropy(logits.reshape(B * P, K), target_indices.reshape(B * P))

    @staticmethod
    def compute_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE over continuous motion embeddings. pred/target: (B, P, D)."""
        return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _smoke():
    B, L, D_text = 2, 77, 4096
    N = 1024  # 32×32 grid
    codebook_size = 1024
    num_patches = 64

    model = TextToMotionTransformer(
        codebook_size=codebook_size,
        num_patches=num_patches,
        text_dim=D_text,
        model_dim=256,  # smaller for test
        num_layers=2,
    )

    text_emb = torch.randn(B, L, D_text)
    init_pos = torch.rand(B, N, 2)  # [0,1]² normalised
    text_mask = torch.ones(B, L, dtype=torch.bool)

    logits = model(text_emb, init_pos, text_mask)
    assert logits.shape == (B, num_patches, codebook_size), logits.shape

    # Test prediction
    indices = model.predict_indices(text_emb, init_pos, temperature=0.0)
    assert indices.shape == (B, num_patches), indices.shape
    assert indices.dtype == torch.int64

    # Test loss
    fake_targets = torch.randint(0, codebook_size, (B, num_patches))
    loss = TextToMotionTransformer.compute_loss(logits, fake_targets)
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("TextToMotionTransformer smoke test passed")
    print(f"  params: {n_params:.1f}M")
    print(f"  logits: {logits.shape}, loss: {loss.item():.4f}")
    print(f"  indices: {indices.shape}, dtype: {indices.dtype}")


if __name__ == "__main__":
    _smoke()
