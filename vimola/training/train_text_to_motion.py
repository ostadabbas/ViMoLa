"""
Train TextToMotionTransformer on real (text, trajectory) pairs.

Pipeline:
  1. Load frozen SpatialMotionVQVAE tokenizer from checkpoint.
  2. For each batch: encode velocities → target indices (B, P=64).
  3. Encode text captions with frozen CLIP → text_embeddings (B, 77, 512).
  4. Forward T2M → logits (B, P, K).
  5. Cross-entropy loss → backward → optimise T2M only.

Usage:
    python vimola/training/train_text_to_motion.py \
        --tokenizer_checkpoint /path/to/tokenizer/best_model.pt \
        --data_dir /projects/aclab/bishoy/data/vimola/trajectories \
        --output_dir /projects/aclab/bishoy/outputs/vimola/t2m \
        --num_epochs 100 --batch_size 32
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.spatial_vqvae import SpatialMotionVQVAE
from motion_model.text_to_motion import TextToMotionTransformer
from data.trajectory_dataset import TrajectoryDataset, collate_fn


# ---------------------------------------------------------------------------
# Text encoder helpers (CLIP for smoke tests, T5 cache for production)
# ---------------------------------------------------------------------------


class CachedT5Embeddings:
    """
    Loads pre-computed T5 embeddings from cache_text_embeddings.py output.
    Falls back to zero tensor for video_ids not found in cache.
    """

    def __init__(self, cache_path: str):
        print(f"Loading T5 embedding cache from: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False,
                          mmap=True)          # memory-mapped: stays on disk, no RAM copy
        self.cache    = data["embeddings"]   # {caption_text: (L, D_text)}
        self.text_dim = data["text_dim"]
        self.seq_len  = data["max_length"]
        print(f"  Cached {len(self.cache):,} embeddings  "
              f"text_dim={self.text_dim}  seq_len={self.seq_len}")

    def encode(self, texts_or_ids: list, device: torch.device) -> torch.Tensor:
        """Return (B, L, D_text) float32."""
        out = []
        for key in texts_or_ids:
            emb = self.cache.get(key)
            if emb is None:
                emb = torch.zeros(self.seq_len, self.text_dim)
            out.append(emb)
        return torch.stack(out, dim=0).to(device=device, dtype=torch.float32)  # (B, L, D)


# ---------------------------------------------------------------------------
# CLIP text encoder helper

class CLIPTextEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 text encoder returning per-token features (B, 77, 512)."""

    TEXT_DIM = 512
    SEQ_LEN  = 77

    def __init__(self, device: torch.device):
        super().__init__()
        try:
            import clip as openai_clip
            self._clip_mod = "openai"
            model, _ = openai_clip.load("ViT-B/32", device=device)
            self.model = model
            self._tokenize = openai_clip.tokenize
        except ImportError:
            try:
                import open_clip
                self._clip_mod = "open_clip"
                model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
                model = model.to(device)
                self.model = model
                self._tokenize = open_clip.get_tokenizer("ViT-B-32")
            except ImportError:
                raise ImportError("Install clip: pip install git+https://github.com/openai/CLIP.git")

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list, device: torch.device) -> torch.Tensor:
        """
        Returns:
            (B, 77, 512) per-token CLIP features
        """
        tokens = self._tokenize(texts, truncate=True).to(device)   # (B, 77)

        # Use the public encode_text API for both backends to avoid internal dtype issues.
        # Expand the single pooled vector to a sequence of length SEQ_LEN so T2M
        # receives (B, L, D_text) regardless of encoder.
        if self._clip_mod == "openai":
            x = self.model.encode_text(tokens).float()               # (B, 512)
        else:
            x = self.model.encode_text(tokens, normalize=False).float()  # (B, 512)

        x = x.unsqueeze(1).expand(-1, self.SEQ_LEN, -1).contiguous()  # (B, 77, 512)
        return x


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(t2m, tokenizer, text_encoder, loader, optimizer, device, epoch):
    t2m.train()
    total_loss = total_acc = 0.0

    for i, (vels, inits, texts) in enumerate(loader):
        vels  = vels.to(device)
        inits = inits.to(device)

        with torch.no_grad():
            text_emb = text_encoder.encode(texts, device)

        if t2m.head_type == "continuous":
            with torch.no_grad():
                target = tokenizer.encode_continuous(vels)
            pred = t2m(text_emb, inits)
            loss = TextToMotionTransformer.compute_regression_loss(pred, target)
            acc = 0.0
        else:
            with torch.no_grad():
                target_indices, _ = tokenizer.encode(vels)
            logits = t2m(text_emb, inits)
            loss   = TextToMotionTransformer.compute_loss(logits, target_indices)
            with torch.no_grad():
                preds = logits.argmax(-1)
                acc   = (preds == target_indices).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(t2m.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc  += acc

        if i % 20 == 0:
            print(f"  Batch {i:4d}: loss={loss.item():.4f}  acc={acc:.3f}")

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def validate(t2m, tokenizer, text_encoder, loader, device):
    t2m.eval()
    total_loss = total_acc = 0.0

    for vels, inits, texts in loader:
        vels  = vels.to(device)
        inits = inits.to(device)

        text_emb = text_encoder.encode(texts, device)
        if t2m.head_type == "continuous":
            target = tokenizer.encode_continuous(vels)
            pred = t2m(text_emb, inits)
            loss = TextToMotionTransformer.compute_regression_loss(pred, target)
            acc = 0.0
        else:
            target_indices, _ = tokenizer.encode(vels)
            logits = t2m(text_emb, inits)
            loss   = TextToMotionTransformer.compute_loss(logits, target_indices)
            preds  = logits.argmax(-1)
            acc    = (preds == target_indices).float().mean().item()

        total_loss += loss.item()
        total_acc  += acc

    n = len(loader)
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_dir", required=True)
    p.add_argument("--text_embeddings_cache", default=None,
                   help="Path to pre-computed T5 cache (.pt). If set, uses T5 "
                        "instead of CLIP. Recommended for production training.")
    p.add_argument("--tokenizer_checkpoint", required=True,
                   help="Path to trained SpatialMotionVQVAE checkpoint (.pt)")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--val_fraction", type=float, default=0.05)
    # Tokenizer (must match checkpoint)
    p.add_argument("--seq_len",       type=int, default=48)
    p.add_argument("--codebook_size", type=int, default=512)
    p.add_argument("--latent_dim",    type=int, default=128)
    # T2M model
    p.add_argument("--model_dim",  type=int, default=512)
    p.add_argument("--num_heads",  type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--head_type",  default="discrete", choices=["discrete", "continuous"],
                   help="discrete=VQ classification (paper); continuous=X4 regression")
    # Training
    p.add_argument("--num_epochs",  type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--save_every",  type=int,   default=10)
    p.add_argument("--output_dir",  default="/tmp/t2m_training")
    # Logging
    p.add_argument("--use_wandb",   action="store_true", default=True)
    p.add_argument("--no-use_wandb", dest="use_wandb", action="store_false")
    p.add_argument("--wandb_project", default="vimola-t2m")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        wandb.init(project=args.wandb_project,
                   name=f"t2m_{datetime.now().strftime('%m%d_%H%M')}",
                   config=vars(args))

    # ------------------------------------------------------------------
    # Frozen tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from: {args.tokenizer_checkpoint}")
    ckpt = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    tokenizer = SpatialMotionVQVAE(
        grid_size=32, patch_size=4, seq_len=args.seq_len,
        hidden_dim=256, latent_dim=args.latent_dim, codebook_size=args.codebook_size,
    ).to(device)
    tokenizer.load_state_dict(ckpt["model_state_dict"])
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    print(f"  Tokenizer loaded, codebook={args.codebook_size}")

    # ------------------------------------------------------------------
    # Text encoder: T5 cache (production) or CLIP (smoke test fallback)
    # ------------------------------------------------------------------
    if args.text_embeddings_cache:
        text_encoder = CachedT5Embeddings(args.text_embeddings_cache)
        text_dim = text_encoder.text_dim
    else:
        print("No T5 cache provided — using CLIP (smoke test only)")
        text_encoder = CLIPTextEncoder(device)
        text_dim = CLIPTextEncoder.TEXT_DIM
    print(f"  text_dim={text_dim}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset from: {args.data_dir}")
    full_ds = TrajectoryDataset(args.data_dir, grid_size=32, seq_len=args.seq_len,
                                max_samples=args.max_samples)

    val_n   = max(1, int(len(full_ds) * args.val_fraction))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    print(f"  Train: {train_n}, Val: {val_n}")

    # ------------------------------------------------------------------
    # T2M model
    # ------------------------------------------------------------------
    t2m = TextToMotionTransformer(
        codebook_size=args.codebook_size,
        num_patches=64,
        patch_size=4,
        grid_size=32,
        text_dim=text_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head_type=args.head_type,
        continuous_dim=args.latent_dim,
    ).to(device)
    n_params = sum(p.numel() for p in t2m.parameters()) / 1e6
    print(f"  T2M: {n_params:.1f}M params")

    optimizer = torch.optim.AdamW(t2m.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        tr_loss, tr_acc = train_one_epoch(
            t2m, tokenizer, text_encoder, train_loader, optimizer, device, epoch)
        val_loss, val_acc = validate(
            t2m, tokenizer, text_encoder, val_loader, device)

        scheduler.step()

        print(f"Train: loss={tr_loss:.4f}  acc={tr_acc:.3f}")
        print(f"Val:   loss={val_loss:.4f}  acc={val_acc:.3f}")

        if args.use_wandb:
            wandb.log({"train/loss": tr_loss, "train/acc": tr_acc,
                       "val/loss": val_loss, "val/acc": val_acc,
                       "lr": scheduler.get_last_lr()[0]}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": t2m.state_dict(),
                        "val_loss": val_loss, "val_acc": val_acc, "args": vars(args)},
                       run_dir / "best_model.pt")
            print(f"  Saved best model (val_loss={val_loss:.4f}  val_acc={val_acc:.3f})")

        if epoch % args.save_every == 0:
            torch.save({"epoch": epoch, "model_state_dict": t2m.state_dict(),
                        "val_loss": val_loss, "args": vars(args)},
                       run_dir / f"checkpoint_epoch{epoch:04d}.pt")

    torch.save({"epoch": args.num_epochs, "model_state_dict": t2m.state_dict(),
                "args": vars(args)},
               run_dir / "final_model.pt")
    print(f"\nTraining complete. Final model saved to {run_dir}/final_model.pt")


if __name__ == "__main__":
    main()
