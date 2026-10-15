"""
Stage 4: Train WanMotionAdapter via rectified flow matching.

What trains:  WanMotionAdapter cross-attn in each Wan DiT block  (~60M params)
What's frozen: Wan DiT weights, Wan VAE, SpatialMotionVQVAE, T5 encoder

Loss (rectified flow / flow matching):
    t        ~ U[0, 1]
    x_t      = (1 - t) * x_0  +  t * noise
    v_target = noise - x_0
    v_pred   = WanDiT_with_adapters(x_t, t*1000, text_emb, motion_emb)
    loss     = MSE(v_pred, v_target)

Adapter injection uses forward hooks — zero changes to the Wan DiT code.

Usage:
    python vimola/training/train_wan_adapter.py \
        --latent_dir  /projects/aclab/bishoy/data/vimola/latents \
        --traj_dir    /projects/aclab/bishoy/data/vimola/trajectories \
        --t5_cache    /projects/aclab/bishoy/data/vimola/text_embeddings.pt \
        --tokenizer_checkpoint /path/to/tokenizer/final_model.pt \
        --wan_dir     /projects/aclab/models/Wan2.2-TI2V-5B-Diffusers \
        --output_dir  /projects/aclab/bishoy/outputs/vimola/adapter
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import wandb


def _gpu_keepalive(device: torch.device, stop_event: threading.Event):
    """Run dummy matmuls on GPU to keep SM utilization above the idle threshold
    while the 5B DiT model is being loaded (~24 min setup phase)."""
    try:
        x = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        while not stop_event.is_set():
            x = torch.matmul(x, x.T)
            if x.norm() > 1e6:          # prevent overflow
                x = x / x.norm() * 10.0
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.spatial_vqvae import SpatialMotionVQVAE
from motion_model.wan_motion_adapter import WanMotionAdapter
from data.motion_video_dataset import (
    MotionVideoDataset, collate_fn, ShardGroupedSampler, split_indices_by_shard,
)


# ---------------------------------------------------------------------------
# Adapter injection via forward hooks (non-invasive — zero DiT code changes)
# ---------------------------------------------------------------------------

class MotionAdapterHook:
    """
    Registers on a WanTransformerBlock and injects motion tokens after
    each block's forward pass via the zero-gated cross-attention adapter.

    Usage:
        hook = MotionAdapterHook(adapter)
        handle = block.register_forward_hook(hook)
        hook.set_motion(motion_tokens)     # before forward
        ...forward pass...
        hook.clear()                        # after forward
        handle.remove()                     # when done
    """

    def __init__(self, adapter: WanMotionAdapter):
        self.adapter = adapter
        self._motion_tokens = None

    def set_motion(self, motion_tokens: torch.Tensor):
        self._motion_tokens = motion_tokens

    def clear(self):
        self._motion_tokens = None

    def __call__(self, module, inputs, output):
        if self._motion_tokens is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = self.adapter(hidden, self._motion_tokens)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def inject_adapters(transformer, wan_hidden_dim: int = 1536,
                    motion_dim: int = 128, num_heads: int = 8,
                    rank: int = 0, blocks=None):
    """
    Create one WanMotionAdapter per DiT block, register hooks.
    `blocks` defaults to transformer.blocks (Wan). Pass an explicit list
    for Hunyuan (dual + single stream).
    """
    adapters = nn.ModuleList()
    hooks    = []
    handles  = []
    block_list = blocks if blocks is not None else transformer.blocks
    for block in block_list:
        adapter  = WanMotionAdapter(wan_hidden_dim, motion_dim, num_heads, rank=rank)
        hook     = MotionAdapterHook(adapter)
        handle   = block.register_forward_hook(hook)
        adapters.append(adapter)
        hooks.append(hook)
        handles.append(handle)
    n = len(adapters)
    params = sum(p.numel() for p in adapters.parameters()) / 1e6
    print(f"  Injected {n} adapters ({params:.1f}M params total)")
    return adapters, hooks, handles


# ---------------------------------------------------------------------------
# EMA of adapter weights — improves inference sample quality
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of a module's parameters/buffers."""

    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay   = decay
        self.shadow  = {k: v.detach().clone()
                        for k, v in module.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, module: nn.Module):
        for k, v in module.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        for k in self.shadow:
            if k in sd:
                self.shadow[k].copy_(sd[k])

    def store(self, module: nn.Module):
        self._backup = {k: v.detach().clone()
                        for k, v in module.state_dict().items()}

    def copy_to(self, module: nn.Module):
        module.load_state_dict(self.shadow, strict=False)

    def restore(self, module: nn.Module):
        if self._backup is not None:
            module.load_state_dict(self._backup, strict=False)
            self._backup = None


# ---------------------------------------------------------------------------
# Rectified flow loss helpers
# ---------------------------------------------------------------------------

def flow_matching_loss(
    transformer,
    hooks: List[MotionAdapterHook],
    latents: torch.Tensor,        # (B, C, T, H, W)  — pre-normalized
    text_emb: torch.Tensor,       # (B, L, D_text)
    motion_emb: torch.Tensor,     # (B, P, D_motion)
    time_shift: float = 7.0,
    timestep_dist: str = "logit_normal",
    motion_dropout_prob: float = 0.0,
) -> torch.Tensor:
    """
    Rectified flow matching loss.
    Wan uses shifted time schedule: t' = shift*t / (1 + (shift-1)*t)
    timestep_dist:
        "uniform"      — t ~ U[0,1]
        "logit_normal" — t = sigmoid(N(0,1)); concentrates mid-noise (SD3/Wan)
    """
    B  = latents.shape[0]
    if timestep_dist == "logit_normal":
        t = torch.sigmoid(torch.randn(B, device=latents.device))
    else:
        t = torch.rand(B, device=latents.device)
    # Apply time shift (matches Wan inference schedule)
    t  = (time_shift * t) / (1.0 + (time_shift - 1.0) * t)

    noise    = torch.randn_like(latents)
    t_bc     = t.view(B, 1, 1, 1, 1)
    x_t      = (1.0 - t_bc) * latents + t_bc * noise   # linear interpolation
    v_target = noise - latents                           # velocity target

    # Timestep in [0, 1000] as integer (Wan convention)
    timesteps = (t * 1000).long()

    # Motion-CFG dropout: zero out motion for a fraction of samples so the
    # adapter learns a conditioned vs unconditioned distinction, forcing gates open.
    if motion_dropout_prob > 0.0:
        drop_mask = torch.rand(B, device=latents.device) < motion_dropout_prob
        if drop_mask.any():
            motion_emb = motion_emb.clone()
            motion_emb[drop_mask] = 0.0

    # Inject motion into every block
    for hook in hooks:
        hook.set_motion(motion_emb)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        call = dict(
            hidden_states=x_t,
            timestep=timesteps,
            encoder_hidden_states=text_emb,
            attention_kwargs=None,
            return_dict=False,
        )
        try:
            v_pred = transformer(**call)[0]
        except TypeError:
            B, L = text_emb.shape[:2]
            mask = torch.ones(B, L, device=text_emb.device, dtype=text_emb.dtype)
            call.pop("attention_kwargs", None)
            v_pred = transformer(encoder_attention_mask=mask, **call)[0]

    return F.mse_loss(v_pred.float(), v_target.float())


# ---------------------------------------------------------------------------
# Training + validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(transformer, adapters, hooks, tokenizer,
                    loader, optimizer, device, epoch, grad_accum: int = 4,
                    checkpoint_every: int = 0, checkpoint_fn=None,
                    use_wandb: bool = False, global_step: int = 0,
                    time_shift: float = 7.0, timestep_dist: str = "logit_normal",
                    ema: "EMA" = None, motion_dropout_prob: float = 0.0,
                    use_codes: bool = False):
    adapters.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (latents, vels_or_codes, inits, text_embs) in enumerate(loader):
        latents   = latents.to(device, dtype=torch.bfloat16)
        text_embs = text_embs.to(device, dtype=torch.bfloat16)

        # Ground-truth VQ motion embeddings (frozen tokenizer / codebook lookup)
        with torch.no_grad():
            if use_codes:
                codes = vels_or_codes.to(device).long()
                motion_emb = tokenizer.vq.embeddings[codes]
            else:
                vels = vels_or_codes.to(device)
                _, motion_emb = tokenizer.encode(vels)          # (B, P, D_motion)

        loss = flow_matching_loss(transformer, hooks, latents,
                                   text_embs, motion_emb,
                                   time_shift=time_shift,
                                   timestep_dist=timestep_dist,
                                   motion_dropout_prob=motion_dropout_prob)
        (loss / grad_accum).backward()
        for hook in hooks:
            hook.clear()

        if (i + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(adapters)

        total_loss += loss.item()
        global_step += 1
        if i % 20 == 0:
            gate_vals    = [h.adapter.gate_value() for h in hooks[:3]]
            all_gates    = [h.adapter.gate_value() for h in hooks]
            abs_mean_gate = sum(abs(g) for g in all_gates) / len(all_gates)
            max_gate      = max(abs(g) for g in all_gates)
            print(f"  Batch {i:4d}: loss={loss.item():.4f}  "
                  f"gates(first3)={[f'{g:.4f}' for g in gate_vals]}")
            if use_wandb:
                wandb.log({"batch_loss":    loss.item(),
                           "abs_mean_gate": abs_mean_gate,
                           "max_gate":      max_gate,
                           "epoch":         epoch}, step=global_step)

        if checkpoint_every > 0 and checkpoint_fn is not None and (i + 1) % checkpoint_every == 0:
            checkpoint_fn(epoch, i)
            print(f"  [ckpt] saved mid-epoch checkpoint at batch {i+1}")

    return total_loss / len(loader), global_step


@torch.no_grad()
def validate(transformer, adapters, hooks, tokenizer, loader, device,
             time_shift: float = 7.0, timestep_dist: str = "logit_normal",
             use_codes: bool = False):
    adapters.eval()
    total_loss = 0.0
    for latents, vels_or_codes, inits, text_embs in loader:
        latents   = latents.to(device, dtype=torch.bfloat16)
        text_embs = text_embs.to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            if use_codes:
                codes = vels_or_codes.to(device).long()
                motion_emb = tokenizer.vq.embeddings[codes]
            else:
                vels = vels_or_codes.to(device)
                _, motion_emb = tokenizer.encode(vels)
        loss = flow_matching_loss(transformer, hooks, latents,
                                   text_embs, motion_emb,
                                   time_shift=time_shift,
                                   timestep_dist=timestep_dist)
        for hook in hooks:
            hook.clear()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent_dir",             required=True)
    p.add_argument("--traj_dir",               required=True)
    p.add_argument("--t5_cache",               required=True)
    p.add_argument("--tokenizer_checkpoint",   required=True)
    p.add_argument("--wan_dir",
                   default="/projects/aclab/models/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--output_dir",             required=True)
    # Model
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=0,
                   help="Unfreeze last N DiT blocks for joint fine-tuning (0=all frozen)")
    p.add_argument("--motion_dim",     type=int, default=128)
    p.add_argument("--num_heads",      type=int, default=8)
    p.add_argument("--codebook_size",  type=int, default=512)
    p.add_argument("--latent_dim",     type=int, default=128)
    p.add_argument("--seq_len",        type=int, default=48)
    # Training
    p.add_argument("--num_epochs",     type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=1)
    p.add_argument("--grad_accum",     type=int, default=4)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--val_fraction",   type=float, default=0.05)
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--save_every",       type=int,   default=5)
    p.add_argument("--checkpoint_every", type=int,   default=500,
                   help="Save mid-epoch checkpoint every N batches (0=disabled)")
    p.add_argument("--checkpoint_dir",    default=None,
                   help="Fixed directory for latest_checkpoint.pt (enables auto-resume across runs)")
    p.add_argument("--resume_from",       default=None,
                   help="Path to latest_checkpoint.pt to resume training")
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--codes_cache",    default=None,
                   help="Precomputed VQ indices from precompute_vq_codes.py")
    p.add_argument("--adapter_rank",   type=int, default=0,
                   help="Low-rank bottleneck (0=full width, 256=X3 variant)")
    p.add_argument("--shard_grouped",  action="store_true",
                   help="Use ShardGroupedSampler (required for /projects I/O)")
    p.add_argument("--time_shift",     type=float, default=7.0)
    p.add_argument("--gate_lr_mult",   type=float, default=50.0,
                   help="LR multiplier for gate params (escape near-zero deadzone)")
    p.add_argument("--timestep_dist",  choices=["uniform", "logit_normal"],
                   default="logit_normal",
                   help="Timestep sampling distribution for flow matching")
    p.add_argument("--ema_decay",      type=float, default=0.999,
                   help="EMA decay for adapter weights (0 to disable)")
    p.add_argument("--motion_dropout_prob", type=float, default=0.15,
                   help="Fraction of samples per batch where motion emb is zeroed (motion-CFG dropout)")
    # Logging
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_project",  default="vimola-adapter")
    p.add_argument("--no-use_wandb",   dest="use_wandb", action="store_false")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ------------------------------------------------------------------
    # GPU keepalive: prevents idle-watchdog from killing during 20+ min setup
    _keepalive_stop = threading.Event()
    _keepalive_thread = threading.Thread(
        target=_gpu_keepalive, args=(device, _keepalive_stop), daemon=True)
    _keepalive_thread.start()
    print("[INFO] GPU keepalive started")

    # ------------------------------------------------------------------
    # Wan DiT (frozen)
    from diffusers import WanTransformer3DModel
    print(f"Loading Wan DiT from: {args.wan_dir}")
    transformer = WanTransformer3DModel.from_pretrained(
        args.wan_dir, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval()
    transformer.enable_gradient_checkpointing()
    for p in transformer.parameters():
        p.requires_grad_(False)
    unfrozen_blocks = []
    if args.unfreeze_last_n_blocks > 0:
        for block in transformer.blocks[-args.unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
            block.train()
            unfrozen_blocks.append(block)
    n_trainable_dit = sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6
    print(f"  DiT: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B params "
          f"({n_trainable_dit:.1f}M trainable, last {args.unfreeze_last_n_blocks} blocks unfrozen)")

    # ------------------------------------------------------------------
    # Inject motion adapters via hooks
    # Read hidden dim from model config — num_attention_heads * attention_head_dim
    wan_hidden_dim = (transformer.config.num_attention_heads *
                      transformer.config.attention_head_dim)
    print(f"  wan_hidden_dim={wan_hidden_dim} (from model config)")
    adapters, hooks, handles = inject_adapters(
        transformer, wan_hidden_dim, args.motion_dim, args.num_heads,
        rank=args.adapter_rank,
    )
    adapters = adapters.to(device)

    # ------------------------------------------------------------------
    # SpatialMotionVQVAE (frozen)
    ckpt = torch.load(args.tokenizer_checkpoint, map_location="cpu",
                      weights_only=False)
    vqvae_cfg = ckpt.get("config", {})
    tokenizer = SpatialMotionVQVAE(
        grid_size    = vqvae_cfg.get("grid_size",    32),
        patch_size   = vqvae_cfg.get("patch_size",   4),
        seq_len      = args.seq_len,
        hidden_dim   = vqvae_cfg.get("hidden_dim",   256),
        latent_dim   = args.latent_dim,
        codebook_size= args.codebook_size,
    ).to(device)
    tokenizer.load_state_dict(ckpt["model_state_dict"])
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    print("  VQ tokenizer loaded (frozen)")

    # ------------------------------------------------------------------
    # Dataset
    dataset = MotionVideoDataset(
        latent_dir   = args.latent_dir,
        traj_dir     = args.traj_dir,
        t5_cache_path= args.t5_cache,
        max_samples  = None,
        codes_cache  = args.codes_cache,
    )
    use_codes = args.codes_cache is not None
    if args.shard_grouped:
        train_idx, val_idx = split_indices_by_shard(
            dataset.samples, val_fraction=args.val_fraction, seed=42,
            max_samples=args.max_samples,
        )
        from torch.utils.data import Subset
        train_ds = Subset(dataset, train_idx)
        val_ds   = Subset(dataset, val_idx)
        train_sampler = ShardGroupedSampler(dataset.samples, train_idx, shuffle=True)
        val_sampler   = ShardGroupedSampler(dataset.samples, val_idx, shuffle=False)
        n_train, n_val = len(train_idx), len(val_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   sampler=train_sampler, shuffle=False,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn, pin_memory=True,
                                   drop_last=True,
                                   persistent_workers=args.num_workers > 0,
                                   prefetch_factor=2 if args.num_workers > 0 else None)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                   sampler=val_sampler, shuffle=False,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn, pin_memory=True,
                                   persistent_workers=args.num_workers > 0,
                                   prefetch_factor=2 if args.num_workers > 0 else None)
    else:
        n_val   = max(1, int(len(dataset) * args.val_fraction))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True,  num_workers=args.num_workers,
                                   collate_fn=collate_fn, pin_memory=True,
                                   drop_last=True, persistent_workers=True,
                                   prefetch_factor=4)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers,
                                   collate_fn=collate_fn, pin_memory=True,
                                   persistent_workers=True, prefetch_factor=2)
    print(f"  Train: {n_train:,}  Val: {n_val:,}  codes={use_codes}  grouped={args.shard_grouped}")

    # ------------------------------------------------------------------
    # Optimizer: gate_params get 10x higher lr to escape near-zero deadzone;
    #            all other adapter + unfrozen DiT params use base lr.
    gate_params  = [a.gate_param for a in adapters]
    gate_ids     = {id(p) for p in gate_params}
    other_params = [p for p in adapters.parameters() if id(p) not in gate_ids] + \
                   [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": gate_params,  "lr": args.lr * args.gate_lr_mult, "weight_decay": 0.0},
        {"params": other_params, "lr": args.lr,                     "weight_decay": 1e-2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.1
    )

    # EMA of adapter weights (improves inference quality)
    ema = EMA(adapters, args.ema_decay) if args.ema_decay > 0 else None

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    best_val_loss = float("inf")
    start_epoch   = 1
    global_step   = 0

    # ------------------------------------------------------------------
    # Resume from checkpoint if provided
    # Use fixed checkpoint_dir if given (survives across timestamped output dirs)
    ckpt_base = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.output_dir)
    latest_ckpt_path = ckpt_base / "latest_checkpoint.pt"
    bak_ckpt_path    = ckpt_base / "latest_checkpoint.pt.bak"

    def _try_load(path):
        """Load a checkpoint, returning None if missing or corrupt."""
        if not path or not Path(path).exists():
            return None
        try:
            ck = torch.load(path, map_location=device, weights_only=False)
            if "adapters" not in ck:
                raise ValueError("missing 'adapters' key")
            return ck
        except Exception as e:
            print(f"  [warn] checkpoint {path} unusable ({type(e).__name__}: {e}); skipping")
            return None

    # Try candidates in priority order: explicit --resume_from, latest, then backup.
    ck = None
    for cand in [args.resume_from, str(latest_ckpt_path), str(bak_ckpt_path)]:
        ck = _try_load(cand)
        if ck is not None:
            resume_path = cand
            break

    if ck is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        adapters.load_state_dict(ck["adapters"])
        try:
            optimizer.load_state_dict(ck["optimizer"])
            if ck.get("scheduler") is not None:
                scheduler.load_state_dict(ck["scheduler"])
        except (ValueError, KeyError) as e:
            print(f"  [warn] optimizer state incompatible ({e}); starting with fresh optimizer")
        if ck.get("unfrozen_blocks") is not None:
            for i, block in enumerate(unfrozen_blocks):
                block.load_state_dict(ck["unfrozen_blocks"][i])
        if ema is not None:
            if ck.get("ema") is not None:
                ema.load_state_dict(ck["ema"])
            else:
                ema = EMA(adapters, args.ema_decay)   # init from resumed weights
        # Override gate LR — loaded optimizer/scheduler state restores the OLD
        # multiplier, so force the new one to take effect on resume.
        new_gate_lr = args.lr * args.gate_lr_mult
        optimizer.param_groups[0]["lr"] = new_gate_lr
        scheduler.base_lrs[0]           = new_gate_lr
        start_epoch   = ck.get("epoch", 1)
        best_val_loss = ck.get("best_val_loss", float("inf"))
        print(f"  Resumed: epoch={start_epoch}, best_val_loss={best_val_loss:.4f}")
        print(f"  Gate LR overridden to {new_gate_lr:.2e} (mult={args.gate_lr_mult})")

    # ------------------------------------------------------------------
    # Mid-epoch checkpoint callback
    def save_checkpoint(epoch: int, batch_i: int):
        # Atomic, crash-safe write:
        #   1. torch.save to a .tmp file (an interrupted save only corrupts .tmp,
        #      never the live checkpoint)
        #   2. rotate the current good checkpoint to .bak (instant rename)
        #   3. atomically rename .tmp -> latest
        tmp_path = str(latest_ckpt_path) + ".tmp"
        torch.save({
            "adapters":       adapters.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "unfrozen_blocks": [b.state_dict() for b in unfrozen_blocks],
            "ema":            ema.state_dict() if ema is not None else None,
            "epoch":          epoch,
            "best_val_loss":  best_val_loss,
        }, tmp_path)
        if latest_ckpt_path.exists():
            os.replace(str(latest_ckpt_path), str(bak_ckpt_path))
        os.replace(tmp_path, str(latest_ckpt_path))

    # ------------------------------------------------------------------
    # Stop GPU keepalive — training takes over GPU from here
    _keepalive_stop.set()
    _keepalive_thread.join(timeout=5)
    print("[INFO] GPU keepalive stopped, training starting")

    # ------------------------------------------------------------------
    # Training loop
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        if args.shard_grouped:
            train_sampler.set_epoch(epoch)

        train_loss, global_step = train_one_epoch(
                                      transformer, adapters, hooks, tokenizer,
                                      train_loader, optimizer, device, epoch,
                                      args.grad_accum,
                                      checkpoint_every=args.checkpoint_every,
                                      checkpoint_fn=save_checkpoint,
                                      use_wandb=args.use_wandb,
                                      global_step=global_step,
                                      time_shift=args.time_shift,
                                      timestep_dist=args.timestep_dist,
                                      ema=ema,
                                      motion_dropout_prob=args.motion_dropout_prob,
                                      use_codes=use_codes)
        # Validate on EMA weights when available (closer to inference)
        if ema is not None:
            ema.store(adapters)
            ema.copy_to(adapters)
        val_loss   = validate(transformer, adapters, hooks, tokenizer,
                               val_loader, device,
                               time_shift=args.time_shift,
                               timestep_dist=args.timestep_dist,
                               use_codes=use_codes)
        if ema is not None:
            ema.restore(adapters)
        scheduler.step()

        avg_gate = sum(h.adapter.gate_value() for h in hooks) / len(hooks)
        print(f"Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  "
              f"avg_gate: {avg_gate:.4f}  lr: {scheduler.get_last_lr()[0]:.2e}")

        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                       "avg_gate": avg_gate, "epoch": epoch})

        # Use EMA weights for the exported adapters (better inference quality)
        export_adapters = ema.state_dict() if ema is not None else adapters.state_dict()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"adapters": export_adapters,
                        "epoch": epoch, "val_loss": val_loss,
                        "config": vars(args)},
                       str(out_dir / "best_adapters.pt"))
            print(f"  Saved best adapters (val_loss={val_loss:.4f})")

        if epoch % args.save_every == 0:
            torch.save({"adapters": export_adapters,
                        "epoch": epoch, "config": vars(args)},
                       str(out_dir / f"adapters_epoch{epoch:03d}.pt"))

    # Final save (EMA weights)
    torch.save({"adapters": ema.state_dict() if ema is not None else adapters.state_dict(),
                "config": vars(args)},
               str(out_dir / "final_adapters.pt"))
    print(f"\nDone. Final adapters → {out_dir}/final_adapters.pt")

    # Cleanup hooks
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
