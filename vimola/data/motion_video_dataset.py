"""
MotionVideoDataset — Stage 4 adapter training dataset.

Each sample provides:
  latent     : (C=48, T_lat, H_lat, W_lat)  float16  — pre-encoded Wan 2.2 VAE latent
  velocities : (T=48, N=1024, 2)            float32  — raw trajectory for VQ encoding
                   OR codes (P=64,) int64 when codes_cache is provided
  init_pos   : (N=1024, 2)                  float32  — initial grid positions
  text_key   : str                          — caption used to look up T5 cache

Pass --codes_cache (precompute_vq_codes.py) to skip the 740MB traj-shard
reads. Pair with ShardGroupedSampler so each 1GB latent shard is loaded once.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler


class MotionVideoDataset(Dataset):
    """
    Loads (latent, trajectory, text) triplets for WanMotionAdapter training.

    Expects:
      - latent_dir/  containing shard_CCCC_NNNNNN.pt shard files
      - traj_dir/    containing {video_id}.pt trajectory files OR
                     traj_shards/ sub-directory with sharded trajectories
      - latent_manifest_*.json  with records:
          {video_id, shard_path, shard_key, trajectory_path, text, status="success"}

    Args:
        latent_dir:     directory containing pre-encoded latents
        traj_dir:       directory containing trajectory .pt files
        t5_cache_path:  path to T5 embedding cache (.pt, loaded mmap)
        max_samples:    cap dataset size (for smoke tests)
    """

    def __init__(
        self,
        latent_dir: str,
        traj_dir: str,
        t5_cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        codes_cache: Optional[str] = None,
        require_t5: bool = True,
    ):
        self.latent_dir = Path(latent_dir)
        self.traj_dir   = Path(traj_dir)
        self.use_codes  = codes_cache is not None
        self._codes_index: dict = {}
        self._codes_tensor = None
        if codes_cache is not None:
            print(f"Loading VQ codes cache from: {codes_cache}", flush=True)
            cc = torch.load(codes_cache, map_location="cpu", weights_only=False)
            self._codes_tensor = cc["codes"]          # (N, P) int16
            self._codes_index  = cc["index"]          # {video_id: row}
            print(f"  Codes cache: {len(self._codes_index):,} video_ids, "
                  f"shape={tuple(self._codes_tensor.shape)}", flush=True)

        # ---------------------------------------------------------------
        # Load T5 cache (memory-mapped — no RAM copy). Optional for I/O benches.
        self.t5_cache = None
        self.t5_dim = 4096
        self.t5_seqlen = 512
        if require_t5 and t5_cache_path:
            print(f"Loading T5 cache from: {t5_cache_path}", flush=True)
            t5_data = torch.load(t5_cache_path, map_location="cpu",
                                 weights_only=False, mmap=True)
            self.t5_cache  = t5_data["embeddings"]
            self.t5_dim    = t5_data["text_dim"]
            self.t5_seqlen = t5_data["max_length"]
            print(f"  T5 cache: {len(self.t5_cache):,} entries, dim={self.t5_dim}",
                  flush=True)
        else:
            print("  Skipping T5 cache (dummy text embeddings)", flush=True)

        # ---------------------------------------------------------------
        # Build sample index from latent manifests
        manifests = sorted(self.latent_dir.glob("latent_manifest_*.json"))
        if not manifests:
            raise FileNotFoundError(
                f"No latent_manifest_*.json found in {latent_dir}. "
                "Run encode_wan_latents.py first."
            )

        self.samples = []
        for mf in manifests:
            with open(mf) as f:
                for rec in json.load(f):
                    if rec.get("status") != "success":
                        continue
                    text = rec.get("text", "")
                    if self.t5_cache is not None and text not in self.t5_cache:
                        continue
                    shard_path = rec.get("shard_path", "")
                    shard_key  = rec.get("shard_key",  rec.get("video_id", ""))
                    traj_path  = Path(rec["trajectory_path"])
                    if not shard_path:
                        continue
                    # Remap absolute cluster path to local latent_dir if needed
                    shard_p = Path(shard_path)
                    if not shard_p.exists():
                        shard_p = self.latent_dir / shard_p.name
                    if not shard_p.exists():
                        continue
                    shard_path = str(shard_p)
                    # Remap trajectory path if it doesn't exist (sharded lookup handles it)
                    if not traj_path.exists():
                        traj_path = self.traj_dir / traj_path.name
                    if not traj_path.exists() and rec.get("video_id") not in getattr(self, "_traj_shard_index_preview", {}):
                        pass  # will be resolved via traj_shard_index at __getitem__ time
                    self.samples.append({
                        "shard_path": shard_path,
                        "shard_key":  shard_key,
                        "traj_path":  str(traj_path),
                        "video_id":   rec.get("video_id", Path(traj_path).stem),
                        "text":       text,
                    })
                    if max_samples and len(self.samples) >= max_samples:
                        break
            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"  Dataset: {len(self.samples):,} valid (latent, traj, text) samples")

        # Latent shard cache: keep last N open shards in memory
        self._shard_cache: dict = {}
        self._shard_cache_max = 4

        # Trajectory shard cache: used when traj_shards/ directory exists
        # Skipped entirely when a codes cache is provided (X3 I/O fix).
        traj_shard_manifest = self.traj_dir / "traj_shards" / "traj_shard_manifest.json"
        if self.use_codes:
            self._traj_shard_index = {}
            print("  Skipping traj shards (using precomputed VQ codes)")
        elif traj_shard_manifest.exists():
            print(f"  Using sharded trajectories: {traj_shard_manifest}")
            with open(traj_shard_manifest) as f:
                traj_index = json.load(f)
            traj_shard_dir = self.traj_dir / "traj_shards"
            self._traj_shard_index: dict = {}
            for rec in traj_index:
                sp = Path(rec["shard_path"])
                if not sp.exists():
                    sp = traj_shard_dir / sp.name
                self._traj_shard_index[rec["video_id"]] = (str(sp), rec["shard_key"])
        else:
            self._traj_shard_index = {}
        self._traj_shard_cache: dict = {}
        self._traj_shard_cache_max = 4

    def __len__(self):
        return len(self.samples)

    def _load_shard(self, shard_path: str) -> dict:
        if shard_path not in self._shard_cache:
            if len(self._shard_cache) >= self._shard_cache_max:
                self._shard_cache.pop(next(iter(self._shard_cache)))
            self._shard_cache[shard_path] = torch.load(
                shard_path, map_location="cpu", weights_only=True)
        return self._shard_cache[shard_path]

    def _load_traj_shard(self, shard_path: str) -> dict:
        if shard_path not in self._traj_shard_cache:
            if len(self._traj_shard_cache) >= self._traj_shard_cache_max:
                self._traj_shard_cache.pop(next(iter(self._traj_shard_cache)))
            sp = Path(shard_path)
            if not sp.exists():
                sp = self.traj_dir / "traj_shards" / sp.name
            self._traj_shard_cache[shard_path] = torch.load(
                str(sp), map_location="cpu", weights_only=False)
        return self._traj_shard_cache[shard_path]

    def __getitem__(self, idx):
        rec = self.samples[idx]

        shard  = self._load_shard(rec["shard_path"])
        latent = shard[rec["shard_key"]]                             # float16

        video_id = rec.get("video_id") or Path(rec["traj_path"]).stem
        dummy_init = torch.zeros(1024, 2)

        if self.use_codes:
            row = self._codes_index.get(video_id)
            if row is None:
                codes = torch.zeros(64, dtype=torch.long)
            else:
                codes = self._codes_tensor[row].long()               # (P,)
            text_emb = self._text_emb(rec)
            return latent, codes, dummy_init, text_emb

        if video_id in self._traj_shard_index:
            shard_path, shard_key = self._traj_shard_index[video_id]
            traj_shard = self._load_traj_shard(shard_path)
            traj_data  = traj_shard[shard_key]
        else:
            traj_data = torch.load(rec["traj_path"], map_location="cpu",
                                   weights_only=False)
        velocities = traj_data["velocities"].float()                 # (T, N, 2)
        init_pos   = traj_data["initial_position"].float()           # (N, 2)
        text_emb = self._text_emb(rec)
        return latent, velocities, init_pos, text_emb

    def _text_emb(self, rec):
        if self.t5_cache is None:
            return torch.zeros(self.t5_seqlen, self.t5_dim)
        return self.t5_cache[rec["text"]].float()


def collate_fn(batch):
    latents, vels, inits, text_embs = zip(*batch)
    return (
        torch.stack(latents,    dim=0),
        torch.stack(vels,       dim=0),
        torch.stack(inits,      dim=0),
        torch.stack(text_embs,  dim=0),
    )


class ShardGroupedSampler(Sampler[int]):
    """Yields Subset positions grouped by latent shard (each 1GB shard read once).

    ``indices`` are indices into the parent ``MotionVideoDataset.samples`` list
    (e.g. the train split).  When paired with ``torch.utils.data.Subset``, the
    sampler must yield positions ``0 .. len(indices)-1``, not the parent indices.
    """

    def __init__(self, samples: List[dict], indices: List[int],
                 shuffle: bool = True, seed: int = 0):
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        groups: dict = defaultdict(list)
        for pos, ds_idx in enumerate(indices):
            groups[samples[ds_idx]["shard_path"]].append(pos)
        self.groups: List[List[int]] = list(groups.values())
        self.num_samples = len(indices)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.groups), generator=g).tolist() if self.shuffle \
            else list(range(len(self.groups)))
        for gi in order:
            members = list(self.groups[gi])
            if self.shuffle and len(members) > 1:
                perm = torch.randperm(len(members), generator=g).tolist()
                members = [members[j] for j in perm]
            yield from members


def split_indices_by_shard(samples: List[dict], val_fraction: float = 0.05,
                           seed: int = 42, max_samples: Optional[int] = None):
    """Split train/val by shard so grouping is preserved in both splits."""
    by_shard = defaultdict(list)
    for i, rec in enumerate(samples):
        by_shard[rec["shard_path"]].append(i)
    shards = sorted(by_shard.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(shards), generator=g).tolist()
    shards = [shards[i] for i in perm]
    n_val = max(1, int(len(shards) * val_fraction))
    val_shards = set(shards[:n_val])
    train_idx, val_idx = [], []
    for sh, idxs in by_shard.items():
        (val_idx if sh in val_shards else train_idx).extend(idxs)
    if max_samples is not None:
        train_idx = train_idx[:max_samples]
    return train_idx, val_idx
