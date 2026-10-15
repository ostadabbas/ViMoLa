"""
Spatial Motion VQ-VAE (v2): tokenizes the full point-track field.

The 32×32=1024-point grid is divided into P spatial patches (default: 4×4
patch size → 8×8=64 patches). Each patch is independently encoded and
VQ-quantized using a **shared** encoder/codebook, so the codebook becomes a
spatial motion-primitive prior.

Terminology:
    gs  = grid_size       (default 32, so N = gs² = 1024 points)
    ps  = patch_size      (default 4, so patch_pts = ps² = 16 points/patch)
    pps = gs // ps        (patches-per-side, default 8)
    P   = pps²            (total spatial patches, default 64)
    T   = seq_len         (velocity timesteps, default 48 = Wan 49 frames − 1)

Usage:
    model = SpatialMotionVQVAE()
    indices, embeddings = model.encode(velocities)   # (B, P), (B, P, D)
    recon_vel, recon_pos, vq_loss, indices, perplexity = model(velocities, init_pos)
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchVelocityEncoder(nn.Module):
    """Encodes one patch's velocity sequence (B, T, patch_pts, 2) → (B, latent_dim).

    Shared across all P patches; called with batch dimension = B × P.
    """

    def __init__(
        self,
        seq_len: int = 48,
        patch_pts: int = 16,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_pts = patch_pts

        in_ch = patch_pts * 2  # flatten (patch_pts, 2) into channel dim

        layers: list = []
        ch = in_ch
        for i in range(num_layers):
            out_ch = hidden_dim
            layers += [nn.Conv1d(ch, out_ch, kernel_size=3, padding=1), nn.GELU()]
            ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, patch_pts, 2)
        Returns:
            latent: (B, latent_dim)
        """
        B, T, P, _ = x.shape
        x = x.reshape(B, T, P * 2).permute(0, 2, 1)  # (B, in_ch, T)
        x = self.conv(x)                               # (B, hidden_dim, T)
        x = x.mean(dim=-1)                             # global avg pool → (B, hidden_dim)
        return self.to_latent(x)                       # (B, latent_dim)


class PatchVelocityDecoder(nn.Module):
    """Decodes a latent code → velocity patch (B, T, patch_pts, 2)."""

    def __init__(
        self,
        seq_len: int = 48,
        patch_pts: int = 16,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_pts = patch_pts
        self.hidden_dim = hidden_dim

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * seq_len),
            nn.GELU(),
        )

        out_ch = patch_pts * 2
        layers: list = []
        ch = hidden_dim
        for i in range(num_layers):
            c_out = out_ch if i == num_layers - 1 else hidden_dim
            layers.append(nn.Conv1d(ch, c_out, kernel_size=3, padding=1))
            if i < num_layers - 1:
                layers.append(nn.GELU())
            ch = c_out
        self.conv = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim)
        Returns:
            velocities: (B, T, patch_pts, 2)
        """
        B = latent.size(0)
        x = self.from_latent(latent)                            # (B, hidden_dim * T)
        x = x.reshape(B, self.hidden_dim, self.seq_len)         # (B, hidden_dim, T)
        x = self.conv(x)                                         # (B, patch_pts*2, T)
        x = x.permute(0, 2, 1)                                  # (B, T, patch_pts*2)
        return x.reshape(B, self.seq_len, self.patch_pts, 2)    # (B, T, patch_pts, 2)


class VectorQuantizer(nn.Module):
    """VQ layer with EMA codebook updates and dead-code reset. Works on arbitrary leading dims."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        reset_threshold: float = 1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.reset_threshold = reset_threshold

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embeddings", embed)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (..., embedding_dim) — any number of leading dims
        Returns:
            quantized_st: same shape as z, straight-through gradient
            vq_loss:      scalar commitment loss
            indices:      (...,) int64 nearest-code indices
        """
        leading = z.shape[:-1]
        flat = z.reshape(-1, self.embedding_dim)           # (M, D)

        dists = (
            flat.pow(2).sum(1, keepdim=True)
            + self.embeddings.pow(2).sum(1)
            - 2 * flat @ self.embeddings.t()
        )  # (M, K)

        indices_flat = dists.argmin(1)                     # (M,)
        quantized_flat = self.embeddings[indices_flat]     # (M, D)

        # EMA codebook update (training only)
        if self.training:
            one_hot = F.one_hot(indices_flat, self.num_embeddings).float()  # (M, K)
            new_cluster_size = one_hot.sum(0)
            new_embed_sum = one_hot.t() @ flat.detach()                     # (K, D)

            self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(new_embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            smoothed = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.embeddings.copy_(self.embed_avg / smoothed.unsqueeze(1))

            # Dead-code reset: reinitialise unused entries with random encoder outputs
            dead = self.cluster_size < self.reset_threshold
            if dead.any():
                n_dead = int(dead.sum().item())
                rand_idx = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
                self.embeddings[dead] = flat[rand_idx].detach()
                self.embed_avg[dead] = flat[rand_idx].detach()
                self.cluster_size[dead] = self.reset_threshold

        quantized_st = z + (quantized_flat.reshape(z.shape) - z).detach()
        vq_loss = self.commitment_cost * F.mse_loss(quantized_flat.detach(), flat)

        return quantized_st, vq_loss, indices_flat.reshape(leading)


class SpatialMotionVQVAE(nn.Module):
    """
    Spatial Motion VQ-VAE: the codebook is the motion prior.

    The 32×32 point-track grid is spatially divided into P=64 patches (4×4
    each). Every patch is encoded independently via a shared encoder, then
    VQ-quantized against a shared codebook.  This yields P discrete motion
    tokens per clip — a spatial representation of *what motion primitive is
    happening in each region*.

    These tokens are the inputs to the cross-attention layers in Wan DiT and
    the targets for the text-to-motion model.

    Args:
        grid_size:      side length of the point grid (default 32 → 1024 pts)
        patch_size:     side length of each spatial patch (default 4 → 16 pts)
        seq_len:        number of velocity timesteps (frames − 1, default 48)
        hidden_dim:     conv hidden channels
        latent_dim:     codebook embedding dimension
        codebook_size:  number of discrete codes K
        num_layers:     conv layers in encoder / decoder
        commitment_cost: VQ commitment loss weight β
    """

    def __init__(
        self,
        grid_size: int = 32,
        patch_size: int = 4,
        seq_len: int = 48,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        codebook_size: int = 1024,
        num_layers: int = 3,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        assert grid_size % patch_size == 0, "grid_size must be divisible by patch_size"

        self.grid_size = grid_size
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        pps = grid_size // patch_size
        self.num_patches = pps * pps          # P
        self.patch_pts = patch_size * patch_size  # points per patch

        self.encoder = PatchVelocityEncoder(seq_len, self.patch_pts, hidden_dim, latent_dim, num_layers)
        self.vq = VectorQuantizer(codebook_size, latent_dim, commitment_cost)
        self.decoder = PatchVelocityDecoder(seq_len, self.patch_pts, hidden_dim, latent_dim, num_layers)

    # ------------------------------------------------------------------
    # Patchify helpers
    # ------------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, N, 2) → (B*P, T, patch_pts, 2) in raster-order patches."""
        B, T, N, C = x.shape
        gs, ps = self.grid_size, self.patch_size
        pps = gs // ps

        x = x.reshape(B, T, gs, gs, C)
        x = x.reshape(B, T, pps, ps, pps, ps, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # (B, T, pps, pps, ps, ps, C)
        x = x.reshape(B, T, self.num_patches, self.patch_pts, C)
        x = x.permute(0, 2, 1, 3, 4).contiguous()         # (B, P, T, patch_pts, C)
        return x.reshape(B * self.num_patches, T, self.patch_pts, C)

    def _unpatchify(self, x: torch.Tensor, B: int) -> torch.Tensor:
        """(B*P, T, patch_pts, 2) → (B, T, N, 2)."""
        gs, ps = self.grid_size, self.patch_size
        pps = gs // ps
        T = x.size(1)
        C = x.size(3)

        x = x.reshape(B, self.num_patches, T, self.patch_pts, C)
        x = x.permute(0, 2, 1, 3, 4).contiguous()          # (B, T, P, patch_pts, C)
        x = x.reshape(B, T, pps, pps, ps, ps, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()   # (B, T, pps, ps, pps, ps, C)
        return x.reshape(B, T, gs * gs, C)                  # (B, T, N, C)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, velocities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode velocity field to codebook indices and quantized embeddings.

        Args:
            velocities: (B, T, N, 2)  — N must equal grid_size²
        Returns:
            indices:    (B, P) int64 codebook indices
            embeddings: (B, P, latent_dim) quantized embeddings (straight-through)
        """
        B = velocities.size(0)
        patches = self._patchify(velocities)                         # (B*P, T, pts, 2)
        latents = self.encoder(patches)                              # (B*P, D)
        latents_2d = latents.reshape(B, self.num_patches, self.latent_dim)  # (B, P, D)
        quantized, _, indices = self.vq(latents_2d)                  # (B,P,D), _, (B,P)
        return indices, quantized

    def encode_continuous(self, velocities: torch.Tensor) -> torch.Tensor:
        """Pre-VQ latents (B, P, D) — target for the X4 regression head."""
        B = velocities.size(0)
        patches = self._patchify(velocities)
        latents = self.encoder(patches)
        return latents.reshape(B, self.num_patches, self.latent_dim)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook indices to reconstructed velocity field.

        Args:
            indices: (B, P) int64
        Returns:
            velocities: (B, T, N, 2)
        """
        B = indices.size(0)
        quantized = self.vq.embeddings[indices]                # (B, P, D)
        flat = quantized.reshape(B * self.num_patches, self.latent_dim)
        patches = self.decoder(flat)                           # (B*P, T, pts, 2)
        return self._unpatchify(patches, B)                    # (B, T, N, 2)

    def forward(
        self,
        velocities: torch.Tensor,
        initial_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VQ-VAE forward pass.

        Args:
            velocities:       (B, T, N, 2) input velocity field
            initial_position: (B, N, 2) optional; zeros if None
        Returns:
            recon_velocities: (B, T, N, 2)
            recon_positions:  (B, T, N, 2)
            vq_loss:          scalar
            indices:          (B, P) int64
            perplexity:       scalar (codebook usage metric)
        """
        B, T, N, _ = velocities.shape

        patches = self._patchify(velocities)                         # (B*P, T, pts, 2)
        latents = self.encoder(patches)                              # (B*P, D)
        latents_2d = latents.reshape(B, self.num_patches, self.latent_dim)
        quantized_2d, vq_loss, indices = self.vq(latents_2d)        # (B,P,D), scalar, (B,P)

        flat_q = quantized_2d.reshape(B * self.num_patches, self.latent_dim)
        recon_patches = self.decoder(flat_q)                         # (B*P, T, pts, 2)
        recon_velocities = self._unpatchify(recon_patches, B)        # (B, T, N, 2)

        if initial_position is None:
            initial_position = torch.zeros(B, N, 2, device=velocities.device, dtype=velocities.dtype)

        recon_positions = initial_position.unsqueeze(1) + torch.cumsum(recon_velocities, dim=1)

        # Codebook perplexity
        flat_idx = indices.reshape(-1)
        avg_probs = F.one_hot(flat_idx, self.codebook_size).float().mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return recon_velocities, recon_positions, vq_loss, indices, perplexity


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _smoke():
    B, T, N = 2, 48, 1024
    model = SpatialMotionVQVAE(grid_size=32, patch_size=4, seq_len=T, codebook_size=128)
    vel = torch.randn(B, T, N, 2)
    init = torch.randn(B, N, 2)

    recon_vel, recon_pos, vq_loss, indices, perp = model(vel, init)
    assert recon_vel.shape == (B, T, N, 2), recon_vel.shape
    assert recon_pos.shape == (B, T, N, 2), recon_pos.shape
    assert indices.shape == (B, model.num_patches), indices.shape

    # Test encode/decode path
    idx2, emb = model.encode(vel)
    recon2 = model.decode(idx2)
    assert recon2.shape == (B, T, N, 2), recon2.shape

    print(f"SpatialMotionVQVAE smoke test passed")
    print(f"  patches={model.num_patches}, patch_pts={model.patch_pts}")
    print(f"  recon_vel: {recon_vel.shape}, vq_loss: {vq_loss.item():.4f}")
    print(f"  indices: {indices.shape}, perplexity: {perp.item():.2f}")


if __name__ == "__main__":
    _smoke()
