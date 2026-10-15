"""
Training script for velocity VQ-VAE tokenizer.
Trains on synthetic trajectories first.
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# Add vimola to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.spatial_vqvae import SpatialMotionVQVAE
from data.synthetic_trajectories import generate_synthetic_dataset
from data.trajectory_dataset import TrajectoryDataset, collate_fn as real_collate_fn


def compute_reconstruction_loss(pred_vel, true_vel, pred_pos, true_pos):
    """Combined velocity + position reconstruction loss."""
    vel_loss = nn.functional.mse_loss(pred_vel, true_vel)
    pos_loss = nn.functional.mse_loss(pred_pos, true_pos)
    return vel_loss + pos_loss


def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_vq_loss = 0.0
    total_recon_loss = 0.0
    total_perplexity = 0.0
    
    for batch_idx, (velocities, positions, initial_pos) in enumerate(dataloader):
        velocities = velocities.to(device)
        positions = positions.to(device)
        initial_pos = initial_pos.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        recon_vel, recon_pos, vq_loss, indices, perplexity = model(
            velocities, initial_pos
        )
        
        # Reconstruction loss
        recon_loss = compute_reconstruction_loss(
            recon_vel, velocities, recon_pos, positions
        )
        
        # Total loss
        loss = recon_loss + vq_loss
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_vq_loss += vq_loss.item()
        total_recon_loss += recon_loss.item()
        total_perplexity += perplexity.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}, "
                  f"recon={recon_loss.item():.4f}, vq={vq_loss.item():.4f}, "
                  f"perplexity={perplexity.item():.2f}")
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        'vq_loss': total_vq_loss / n,
        'perplexity': total_perplexity / n,
    }


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_perplexity = 0.0
    
    with torch.no_grad():
        for velocities, positions, initial_pos in dataloader:
            velocities = velocities.to(device)
            positions = positions.to(device)
            initial_pos = initial_pos.to(device)
            
            recon_vel, recon_pos, vq_loss, indices, perplexity = model(
                velocities, initial_pos
            )
            
            recon_loss = compute_reconstruction_loss(
                recon_vel, velocities, recon_pos, positions
            )
            loss = recon_loss + vq_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'recon_loss': total_recon_loss / n,
        'vq_loss': total_vq_loss / n,
        'perplexity': total_perplexity / n,
    }


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # WandB
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"tokenizer_{datetime.now().strftime('%m%d_%H%M')}",
            config=vars(args),
        )
    
    # Model
    print(f"Creating model: seq_len={args.seq_len}, codebook={args.codebook_size}, patches=64")
    model = SpatialMotionVQVAE(
        grid_size=32,
        patch_size=4,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_layers=args.num_layers,
        commitment_cost=args.commitment_cost,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Datasets
    if args.data_dir:
        print(f"Loading real trajectory dataset from: {args.data_dir}")
        full_dataset = TrajectoryDataset(
            trajectory_dir=args.data_dir,
            grid_size=32,
            seq_len=args.seq_len,
            max_samples=args.max_samples,
        )
        val_size = max(1, int(len(full_dataset) * 0.05))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"Real data: {train_size} train, {val_size} val samples")

        def _collate(batch):
            # TrajectoryDataset returns (vel, init_pos, text)
            # vel: (B, T, N, 2), init_pos: (B, N, 2)
            vels, inits, _ = real_collate_fn(batch)
            # positions: cumsum of vels prepended with init_pos
            # (B, T, N, 2) -> cumsum -> (B, T, N, 2), then add init_pos (B, 1, N, 2)
            positions = inits.unsqueeze(1) + torch.cumsum(vels, dim=1)
            return vels, positions, inits

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, collate_fn=_collate,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=_collate,
        )
    else:
        print(f"Generating synthetic dataset: {args.num_train_samples} train, {args.num_val_samples} val")
        train_dataset = generate_synthetic_dataset(
            num_samples=args.num_train_samples,
            num_frames=args.seq_len,
            trajectory_type=args.trajectory_type,
            seed=args.seed,
        )
        val_dataset = generate_synthetic_dataset(
            num_samples=args.num_val_samples,
            num_frames=args.seq_len,
            trajectory_type=args.trajectory_type,
            seed=args.seed + 10000,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        print(f"Train: loss={train_metrics['loss']:.4f}, "
              f"recon={train_metrics['recon_loss']:.4f}, "
              f"vq={train_metrics['vq_loss']:.4f}, "
              f"perplexity={train_metrics['perplexity']:.2f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        print(f"Val:   loss={val_metrics['loss']:.4f}, "
              f"recon={val_metrics['recon_loss']:.4f}, "
              f"vq={val_metrics['vq_loss']:.4f}, "
              f"perplexity={val_metrics['perplexity']:.2f}")
        
        # Log
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/recon': train_metrics['recon_loss'],
                'train/vq': train_metrics['vq_loss'],
                'train/perplexity': train_metrics['perplexity'],
                'val/loss': val_metrics['loss'],
                'val/recon': val_metrics['recon_loss'],
                'val/vq': val_metrics['vq_loss'],
                'val/perplexity': val_metrics['perplexity'],
            })
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'args': vars(args),
            }, checkpoint_path)
            print(f"Saved best model (val_loss={best_val_loss:.4f})")
        
        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'args': vars(args),
            }, checkpoint_path)
    
    # Save final
    final_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    
    # Data
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to extracted .pt trajectory files (real data). If not set, uses synthetic data.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset size (useful for smoke tests).")
    parser.add_argument("--num_train_samples", type=int, default=10000)
    parser.add_argument("--num_val_samples", type=int, default=1000)
    parser.add_argument("--trajectory_type", type=str, default="mixed",
                        choices=["random", "smooth", "bouncing", "mixed"])
    parser.add_argument("--seed", type=int, default=42)
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="/projects/aclab/bishoy/outputs/vimola/tokenizer")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="vimola-tokenizer")
    
    args = parser.parse_args()
    main(args)
