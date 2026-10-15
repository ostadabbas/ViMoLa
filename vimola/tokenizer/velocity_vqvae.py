"""
Motion VQ-VAE: Encodes velocity sequences per trajectory.

Architecture:
- Encoder: Conv1D over time → MLP → VQ codebook
- Decoder: Code → upsample → Conv1D → velocity predictions
- Reconstruction: velocities + initial position → cumsum → positions
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ layer with EMA codebook updates and dead-code reset to prevent collapse."""

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
            z: (B, D) continuous latent
        Returns:
            quantized_st: (B, D) with straight-through gradient
            vq_loss:       commitment loss scalar
            indices:       (B,) codebook indices
        """
        flat_z = z.reshape(-1, self.embedding_dim)

        # Distances to codebook entries
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=1)
            - 2 * flat_z @ self.embeddings.t()
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings[indices]

        # EMA codebook update (training only)
        if self.training:
            one_hot = F.one_hot(indices, self.num_embeddings).float()  # (B, K)
            new_cluster_size = one_hot.sum(0)
            new_embed_sum = one_hot.t() @ flat_z.detach()             # (K, D)

            self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(new_embed_sum, alpha=1 - self.decay)

            # Laplace smoothing
            n = self.cluster_size.sum()
            smoothed = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.embeddings.copy_(self.embed_avg / smoothed.unsqueeze(1))

            # Dead code reset: reinitialise with random encoder outputs
            dead = self.cluster_size < self.reset_threshold
            if dead.any():
                n_dead = dead.sum().item()
                rand_idx = torch.randint(0, flat_z.size(0), (int(n_dead),), device=flat_z.device)
                self.embeddings[dead] = flat_z[rand_idx].detach()
                self.embed_avg[dead] = flat_z[rand_idx].detach()
                self.cluster_size[dead] = self.reset_threshold

        # Straight-through estimator (commitment loss only; codebook updated via EMA)
        quantized_st = flat_z + (quantized - flat_z).detach()
        vq_loss = self.commitment_cost * F.mse_loss(quantized.detach(), flat_z)

        return quantized_st, vq_loss, indices


class VelocityEncoder(nn.Module):
    """
    Encodes velocity sequence (B, T, 2) to a single latent per trajectory.
    Per-trajectory tokenization for v1.
    """
    
    def __init__(
        self,
        seq_len: int = 30,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Conv1D over time: (B, 2, T) -> (B, hidden_dim, T)
        conv_layers = []
        in_ch = 2
        for i in range(num_layers):
            out_ch = hidden_dim if i == num_layers - 1 else hidden_dim // 2
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.conv_encoder = nn.Sequential(*conv_layers)
        
        # Global pooling + MLP to latent
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            velocities: (B, T, 2) velocity sequence
        Returns:
            latent: (B, latent_dim) continuous latent before VQ
        """
        # (B, T, 2) -> (B, 2, T)
        x = velocities.transpose(1, 2)
        x = self.conv_encoder(x)  # (B, hidden_dim, T)
        x = x.reshape(x.size(0), -1)  # Flatten
        latent = self.to_latent(x)  # (B, latent_dim)
        return latent


class VelocityDecoder(nn.Module):
    """
    Decodes latent code to velocity sequence (B, T, 2).
    """
    
    def __init__(
        self,
        seq_len: int = 30,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.seq_len = seq_len
        
        # MLP to reshape
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * seq_len),
            nn.ReLU(),
        )
        
        # Conv1D decoder
        deconv_layers = []
        in_ch = hidden_dim
        for i in range(num_layers):
            out_ch = 2 if i == num_layers - 1 else hidden_dim
            deconv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
            ])
            in_ch = out_ch
        self.conv_decoder = nn.Sequential(*deconv_layers)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) quantized code
        Returns:
            velocities: (B, T, 2) predicted velocity sequence
        """
        x = self.from_latent(latent)  # (B, hidden_dim * T)
        x = x.reshape(x.size(0), -1, self.seq_len)  # (B, hidden_dim, T)
        x = self.conv_decoder(x)  # (B, 2, T)
        velocities = x.transpose(1, 2)  # (B, T, 2)
        return velocities


class VelocityVQVAE(nn.Module):
    """
    Complete VQ-VAE for velocity sequences.
    
    Forward:
        velocities -> encoder -> VQ -> decoder -> reconstructed velocities
    
    Reconstruction:
        cumsum(reconstructed velocities, dim=1) + initial_position -> positions
    """
    
    def __init__(
        self,
        seq_len: int = 30,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        codebook_size: int = 512,
        num_layers: int = 3,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        
        self.encoder = VelocityEncoder(seq_len, hidden_dim, latent_dim, num_layers)
        self.vq = VectorQuantizer(codebook_size, latent_dim, commitment_cost)
        self.decoder = VelocityDecoder(seq_len, latent_dim, hidden_dim, num_layers)
    
    def encode(self, velocities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to codebook indices.
        
        Args:
            velocities: (B, T, 2)
        Returns:
            indices: (B,) codebook indices
            quantized: (B, latent_dim) quantized latents
        """
        latent = self.encoder(velocities)
        quantized, _, indices = self.vq(latent)
        return indices, quantized
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from codebook indices.
        
        Args:
            indices: (B,) codebook indices
        Returns:
            velocities: (B, T, 2) reconstructed velocities
        """
        quantized = self.vq.embeddings(indices)
        velocities = self.decoder(quantized)
        return velocities
    
    def forward(
        self,
        velocities: torch.Tensor,
        initial_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with full reconstruction.

        Supports two input modes:
            Single-point:  velocities (B, T, 2),       initial_position (B, 2)
            Multi-point:   velocities (B, T, N, 2),     initial_position (B, N, 2)
                           N points are flattened into the batch dimension.

        Returns:
            recon_velocities: same shape as input velocities
            recon_positions:  same spatial shape, time dim reconstructed
            vq_loss:          scalar VQ commitment loss
            indices:          (B,) or (B*N,) codebook indices
            perplexity:       scalar codebook usage metric
        """
        multi_point = velocities.dim() == 4  # (B, T, N, 2)

        if multi_point:
            B, T, N, C = velocities.shape
            # Flatten points into batch: (B*N, T, 2)
            velocities_in = velocities.permute(0, 2, 1, 3).reshape(B * N, T, C)
            if initial_position is not None:
                init_in = initial_position.reshape(B * N, C)  # (B*N, 2)
            else:
                init_in = torch.zeros(B * N, C, device=velocities.device)
        else:
            velocities_in = velocities
            init_in = initial_position if initial_position is not None else \
                torch.zeros(velocities.size(0), 2, device=velocities.device)

        # Encode -> VQ -> Decode
        latent = self.encoder(velocities_in)
        quantized, vq_loss, indices = self.vq(latent)
        recon_velocities_flat = self.decoder(quantized)

        # Reconstruct positions via cumsum
        position_deltas = torch.cumsum(recon_velocities_flat, dim=1)
        recon_positions_flat = init_in.unsqueeze(1) + position_deltas

        # Codebook usage
        encodings = F.one_hot(indices, self.vq.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if multi_point:
            # Unflatten back: (B*N, T, 2) -> (B, T, N, 2)
            recon_velocities = recon_velocities_flat.reshape(B, N, T, C).permute(0, 2, 1, 3)
            recon_positions = recon_positions_flat.reshape(B, N, T, C).permute(0, 2, 1, 3)
        else:
            recon_velocities = recon_velocities_flat
            recon_positions = recon_positions_flat

        return recon_velocities, recon_positions, vq_loss, indices, perplexity


def test_velocity_vqvae():
    """Quick sanity test."""
    model = VelocityVQVAE(seq_len=30, codebook_size=128)
    
    # Synthetic data
    B, T = 4, 30
    velocities = torch.randn(B, T, 2)
    initial_pos = torch.randn(B, 2)
    
    # Forward
    recon_vel, recon_pos, vq_loss, indices, perplexity = model(velocities, initial_pos)
    
    print(f"Input velocities: {velocities.shape}")
    print(f"Recon velocities: {recon_vel.shape}")
    print(f"Recon positions: {recon_pos.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Codebook indices: {indices.shape}, unique: {indices.unique().numel()}")
    print(f"Perplexity: {perplexity.item():.2f}")
    print("✓ Test passed")


if __name__ == "__main__":
    test_velocity_vqvae()
