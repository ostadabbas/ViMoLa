#!/usr/bin/env python3
"""
ViMoLa inference: text + image → motion-grounded video via Wan2.2 TI2V.

Full pipeline:
  1. T5-encode text prompt (via WanImageToVideoPipeline encoder)
  2. T2M: text embedding + uniform spatial grid → motion token indices (B, P=64)
  3. VQVAE decode: indices → motion embeddings (B, P, 128)
  4. WanPipeline with motion adapter hooks active → video frames
  5. Save MP4

Usage (single prompt):
  python -m vimola.inference.generate \\
    --wan_dir ./models/Wan2.2-TI2V-5B-Diffusers \\
    --adapter_ckpt ./checkpoints/adapters.pt \\
    --tokenizer_ckpt ./checkpoints/tokenizer.pt \\
    --t2m_ckpt ./checkpoints/t2m.pt \\
    --image ./assets/sisyphus_input.png \\
    --prompt "pushing a boulder up a steep hill" \\
    --output_dir ./outputs/demo \\
    --adapter_strength 0.1

Batch usage (prompts file, one per line):
  python -m vimola.inference.generate ... --prompts_file prompts.txt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_uniform_grid(grid_size: int = 32, batch_size: int = 1,
                      device: str = "cpu") -> torch.Tensor:
    """Uniform normalized grid positions (B, N=grid_size², 2) in [0,1]²."""
    xs = torch.linspace(0.0, 1.0, grid_size, device=device)
    ys = torch.linspace(0.0, 1.0, grid_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (N, 2)
    return grid.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def load_tokenizer(ckpt_path: str, device: torch.device,
                   latent_dim: int = 128, codebook_size: int = 512,
                   seq_len: int = 48):
    from tokenizer.spatial_vqvae import SpatialMotionVQVAE
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = SpatialMotionVQVAE(
        grid_size     = cfg.get("grid_size",    32),
        patch_size    = cfg.get("patch_size",   4),
        seq_len       = seq_len,
        hidden_dim    = cfg.get("hidden_dim",   256),
        latent_dim    = latent_dim,
        codebook_size = codebook_size,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  VQ tokenizer loaded from {Path(ckpt_path).name}")
    return model


def load_t2m(ckpt_path: str, device: torch.device):
    from motion_model.text_to_motion import TextToMotionTransformer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    model = TextToMotionTransformer(
        codebook_size = args.get("codebook_size", 512),
        num_patches   = 64,
        patch_size    = 4,
        grid_size     = 32,
        text_dim      = 4096,
        model_dim     = args.get("model_dim", 512),
        num_heads     = args.get("num_heads", 8),
        num_layers    = args.get("num_layers", 4),
        dropout       = 0.0,
        head_type     = args.get("head_type", "discrete"),
        continuous_dim= args.get("latent_dim", args.get("continuous_dim", 128)),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  T2M loaded from {Path(ckpt_path).name}  "
          f"head={model.head_type} (epoch={ck.get('epoch', '?')})")
    return model


def scale_gate_params(adapters, strength: float):
    """
    Scale effective gate values by `strength` at inference time.
    Operates on gate_param so: tanh(new_param) = strength * tanh(old_param).
    This is the only correct scale point — the output normalization in
    WanMotionAdapter.forward() makes scaling motion_emb ineffective.
    """
    with torch.no_grad():
        for adapter in adapters:
            g = adapter.gate_param.data
            new_g = torch.atanh(
                (strength * torch.tanh(g)).clamp(-0.9999, 0.9999)
            )
            adapter.gate_param.data = new_g


def load_adapters(pipe, adapter_ckpt: str, motion_dim: int = 128,
                  num_heads: int = 8, device: torch.device = None,
                  adapter_strength: float = 1.0, adapter_rank: int = 0,
                  blocks=None):
    from training.train_wan_adapter import inject_adapters
    transformer = pipe.transformer
    wan_hidden_dim = (transformer.config.num_attention_heads *
                      transformer.config.attention_head_dim)
    adapters, hooks, handles = inject_adapters(
        transformer, wan_hidden_dim, motion_dim, num_heads,
        rank=adapter_rank, blocks=blocks,
    )
    adapters = adapters.to(device, dtype=torch.bfloat16)
    ck = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
    adapters.load_state_dict(ck["adapters"])
    adapters.eval()
    for p in adapters.parameters():
        p.requires_grad_(False)
    epoch     = ck.get("epoch", "?")
    val_loss  = ck.get("val_loss", float("nan"))
    print(f"  Adapters loaded (epoch={epoch}, val_loss={val_loss:.4f})")
    avg_gate = sum(abs(h.adapter.gate_value()) for h in hooks) / len(hooks)
    max_gate = max(abs(h.adapter.gate_value()) for h in hooks)
    print(f"  avg_abs_gate={avg_gate:.3f}  max_gate={max_gate:.3f}")
    if adapter_strength != 1.0:
        scale_gate_params(adapters, adapter_strength)
        scaled_avg = sum(abs(h.adapter.gate_value()) for h in hooks) / len(hooks)
        print(f"  → after strength={adapter_strength}: avg_abs_gate={scaled_avg:.3f}")
    return adapters, hooks, handles


# ---------------------------------------------------------------------------
# Text encoding — extracts T5 embeddings via the pipeline's encoder
# ---------------------------------------------------------------------------

def encode_prompt(pipe, prompt: str, device: torch.device,
                  max_sequence_length: int = 226) -> torch.Tensor:
    """
    Run the pipeline's T5 encoder and return (1, L, 4096) float32 tensor.
    Works for WanImageToVideoPipeline (diffusers >= 0.32).
    """
    tokenizer  = pipe.tokenizer
    text_enc   = pipe.text_encoder.to(device)

    tokens = tokenizer(
        [prompt],
        padding        = "max_length",
        max_length     = max_sequence_length,
        truncation     = True,
        return_tensors = "pt",
    ).to(device)

    with torch.no_grad():
        out = text_enc(**tokens)
    emb = out.last_hidden_state.float()      # (1, L, 4096)
    return emb


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def resolve_motion_indices(
    motion_source: str,
    t2m,
    tokenizer_vq,
    text_emb: torch.Tensor,
    init_pos: torch.Tensor,
    device: torch.device,
    temperature: float = 0.0,
    gt_velocities: torch.Tensor = None,
    shuffle_seed: int = 0,
) -> torch.Tensor:
    """Return (1, P) codebook indices for the requested conditioning source."""
    if motion_source == "none":
        return None
    if motion_source == "t2m":
        return t2m.predict_indices(text_emb.to(device), init_pos, temperature=temperature)
    if motion_source in ("gt", "shuffled"):
        if gt_velocities is None:
            raise ValueError(f"--motion_source {motion_source} requires GT velocities")
        with torch.no_grad():
            indices, _ = tokenizer_vq.encode(gt_velocities.to(device))
        if motion_source == "shuffled":
            g = torch.Generator(device=indices.device).manual_seed(shuffle_seed)
            perm = torch.randperm(indices.shape[1], generator=g, device=indices.device)
            indices = indices[:, perm]
        return indices
    raise ValueError(f"unknown motion_source={motion_source}")


@torch.no_grad()
def generate(
    pipe,
    hooks,
    tokenizer_vq,
    t2m,
    prompt: str,
    image: Image.Image,
    device: torch.device,
    num_frames: int = 81,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 42,
    temperature: float = 0.0,
    adapter_strength: float = 0.1,
    motion_stop_frac: float = 1.0,
    motion_source: str = "t2m",
    gt_velocities: torch.Tensor = None,
    low_vram: bool = False,
) -> list:
    """
    Returns list of PIL Images (video frames).
    motion_stop_frac: fraction of denoising steps to keep adapter active.
      Default 1.0 = all steps. Lowering may reduce late-video corruption but
      can also break temporal coherence — use with care.
    """
    # 1. T5-encode prompt
    text_emb = encode_prompt(pipe, prompt, device)          # (1, L, 4096)

    # 2. Motion embeddings: discrete VQ indices or continuous T2M latents
    init_pos  = make_uniform_grid(grid_size=32, batch_size=1, device=device)
    if motion_source == "none":
        for hook in hooks:
            hook.clear()
    elif getattr(t2m, "head_type", "discrete") == "continuous" and motion_source == "t2m":
        motion_emb = t2m(text_emb.to(device), init_pos)
        motion_emb = motion_emb.to(device, dtype=torch.bfloat16)
        for hook in hooks:
            hook.set_motion(motion_emb)
    else:
        indices = resolve_motion_indices(
            motion_source, t2m, tokenizer_vq, text_emb, init_pos, device,
            temperature=temperature, gt_velocities=gt_velocities, shuffle_seed=seed,
        )
        if indices is not None:
            motion_emb = tokenizer_vq.vq.embeddings[indices]
            motion_emb = motion_emb.to(device, dtype=torch.bfloat16)
            for hook in hooks:
                hook.set_motion(motion_emb)
        else:
            for hook in hooks:
                hook.clear()

    # 5. Run Wan TI2V pipeline
    generator = torch.Generator(device=device).manual_seed(seed)
    pipe_kwargs = dict(
        image               = image,
        prompt              = prompt,
        height              = height,
        width               = width,
        num_frames          = num_frames,
        num_inference_steps = num_inference_steps,
        guidance_scale      = guidance_scale,
        generator           = generator,
    )
    if motion_stop_frac < 1.0:
        stop_step = max(1, int(motion_stop_frac * num_inference_steps))
        stopped   = [False]
        def _stop_cb(pipe, step_idx, timestep, callback_kwargs):
            if not stopped[0] and step_idx >= stop_step:
                for h in hooks:
                    h.clear()
                stopped[0] = True
            return callback_kwargs
        pipe_kwargs["callback_on_step_end"] = _stop_cb
    if low_vram:
        pipe_kwargs["output_type"] = "latent"
    output = pipe(**pipe_kwargs)
    if low_vram:
        latents = output.frames
        if isinstance(latents, (list, tuple)):
            latents = latents[0]
        pipe.transformer.to("cpu")
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        vae = pipe.vae
        latents = latents.to(vae.dtype)
        z = vae.config.z_dim
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="pil")
        frames = video[0] if isinstance(video[0], list) else video
        pipe.transformer.to(device)
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.to(device)
    else:
        frames = output.frames[0]    # list of PIL Images

    # 6. Clear motion
    for hook in hooks:
        hook.clear()

    return frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wan_dir",        required=True)
    p.add_argument("--adapter_ckpt",   required=True)
    p.add_argument("--tokenizer_ckpt", required=True)
    p.add_argument("--t2m_ckpt",       required=True)
    p.add_argument("--image",          default=None,
                   help="Path to first frame image (JPG/PNG). Required for TI2V.")
    p.add_argument("--prompt",         default=None)
    p.add_argument("--prompts_file",   default=None,
                   help="Text file with one prompt per line")
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--num_frames",     type=int,   default=81)
    p.add_argument("--height",         type=int,   default=480)
    p.add_argument("--width",          type=int,   default=832)
    p.add_argument("--num_steps",      type=int,   default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--temperature",    type=float, default=0.0,
                   help="T2M sampling temperature (0=greedy argmax)")
    p.add_argument("--adapter_strength", type=float, default=0.1,
                   help="Scale motion contribution: 1.0=full, 0.1=10%% (default), 0.0=off")
    p.add_argument("--motion_stop_frac", type=float, default=1.0,
                   help="Clear motion hooks after this fraction of denoising steps (default 1.0=all steps)")
    p.add_argument("--motion_dim",     type=int,   default=128)
    p.add_argument("--num_heads",      type=int,   default=8)
    p.add_argument("--codebook_size",  type=int,   default=512)
    p.add_argument("--latent_dim",     type=int,   default=128)
    p.add_argument("--seq_len",        type=int,   default=48)
    p.add_argument("--no_motion",      action="store_true",
                   help="Disable motion adapters (baseline Wan generation)")
    p.add_argument("--motion_source",  default="t2m",
                   choices=["t2m", "gt", "shuffled", "none"],
                   help="X1 ladder: t2m=predicted, gt=tokenizer codes, "
                        "shuffled=GT codes spatially permuted, none=no adapter")
    p.add_argument("--gt_traj",        default=None,
                   help="Path to a trajectory .pt (velocities + initial_position) "
                        "for --motion_source gt|shuffled")
    p.add_argument("--eval_set",       default=None,
                   help="eval_set.json — used with --eval_video_id to look up GT traj")
    p.add_argument("--eval_video_id",  default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts += [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError("Provide --prompt or --prompts_file")

    # Load reference image
    if args.image is None:
        print("[WARN] No --image provided; using a blank white 832×480 frame.")
        image = Image.fromarray(np.ones((args.height, args.width, 3),
                                       dtype=np.uint8) * 200)
    else:
        image = Image.open(args.image).convert("RGB")
        image = image.resize((args.width, args.height))

    # -----------------------------------------------------------------------
    print("Loading Wan TI2V pipeline (WanImageToVideoPipeline) …")
    from diffusers import WanImageToVideoPipeline
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.wan_dir, torch_dtype=torch.bfloat16
    )
    if hasattr(pipe, "config") and not getattr(pipe.config, "expand_timesteps", False):
        pipe.register_to_config(expand_timesteps=True)
        print("  forced expand_timesteps=True")
    print(f"  loaded {pipe.__class__.__name__} expand_timesteps={getattr(pipe.config, 'expand_timesteps', None)}")
    pipe.to(device)

    # -----------------------------------------------------------------------
    print("Loading motion adapters …")
    adapters, hooks, handles = load_adapters(
        pipe, args.adapter_ckpt,
        motion_dim=args.motion_dim, num_heads=args.num_heads, device=device,
        adapter_strength=args.adapter_strength,
    )

    # -----------------------------------------------------------------------
    print("Loading VQ tokenizer …")
    tokenizer_vq = load_tokenizer(
        args.tokenizer_ckpt, device,
        latent_dim=args.latent_dim, codebook_size=args.codebook_size,
        seq_len=args.seq_len,
    )

    # -----------------------------------------------------------------------
    print("Loading T2M …")
    t2m = load_t2m(args.t2m_ckpt, device)

    # -----------------------------------------------------------------------
    if args.no_motion:
        args.motion_source = "none"
        print("[INFO] --no_motion: disabling adapter hooks (baseline)")
        for hook in hooks:
            hook.clear()

    gt_vels = None
    if args.motion_source in ("gt", "shuffled"):
        if args.gt_traj:
            td = torch.load(args.gt_traj, map_location="cpu", weights_only=False)
            gt_vels = td["velocities"].float().unsqueeze(0)
        elif args.eval_set and args.eval_video_id:
            import json
            with open(args.eval_set) as f:
                recs = {r["video_id"]: r for r in json.load(f)}
            rec = recs[args.eval_video_id]
            shard = torch.load(rec["gt_traj_shard"], map_location="cpu", weights_only=False)
            gt_vels = shard[rec["gt_traj_key"]]["velocities"].float().unsqueeze(0)
        else:
            raise ValueError("gt/shuffled requires --gt_traj or --eval_set + --eval_video_id")

    print(f"\nGenerating {len(prompts)} video(s)…")
    from diffusers.utils import export_to_video

    for idx, prompt in enumerate(prompts):
        slug = prompt[:60].replace(" ", "_").replace("/", "-")
        out_path = str(out_dir / f"{idx:03d}_{slug}.mp4")
        print(f"\n[{idx+1}/{len(prompts)}] {prompt[:80]}")

        if args.no_motion:
            # Baseline: plain Wan generation, no adapters
            generator = torch.Generator(device=device).manual_seed(args.seed)
            output = pipe(
                image               = image,
                prompt              = prompt,
                height              = args.height,
                width               = args.width,
                num_frames          = args.num_frames,
                num_inference_steps = args.num_steps,
                guidance_scale      = args.guidance_scale,
                generator           = generator,
            )
            frames = output.frames[0]
        else:
            frames = generate(
                pipe, hooks, tokenizer_vq, t2m, prompt, image, device,
                num_frames          = args.num_frames,
                height              = args.height,
                width               = args.width,
                num_inference_steps = args.num_steps,
                guidance_scale      = args.guidance_scale,
                seed                = args.seed,
                temperature         = args.temperature,
                adapter_strength    = args.adapter_strength,
                motion_stop_frac    = args.motion_stop_frac,
                motion_source       = args.motion_source,
                gt_velocities       = gt_vels,
            )

        export_to_video(frames, out_path, fps=16)
        print(f"  Saved → {out_path}")

    # Cleanup hooks
    for handle in handles:
        handle.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
