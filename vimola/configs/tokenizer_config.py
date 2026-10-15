"""
Dataclass config for velocity tokenizer.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class VelocityTokenizerConfig:
    """Configuration for velocity VQ-VAE tokenizer."""
    
    # Model architecture
    seq_len: int = 30  # Number of timesteps per trajectory
    hidden_dim: int = 256  # Hidden dimension for conv layers
    latent_dim: int = 128  # Dimension of codebook embeddings
    codebook_size: int = 512  # Number of discrete codes
    num_layers: int = 3  # Number of conv layers in encoder/decoder
    commitment_cost: float = 0.25  # VQ commitment loss weight
    
    # Data
    trajectory_type: str = "mixed"  # "random", "smooth", "bouncing", "mixed"
    
    # Training
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_epochs: int = 50
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "vimola-tokenizer"
    
    def to_dict(self):
        """Convert to dictionary for logging/saving."""
        return {
            k: v for k, v in self.__dict__.items()
        }
