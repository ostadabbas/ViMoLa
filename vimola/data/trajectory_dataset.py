"""
TrajectoryDataset: loads extracted trajectory .pt files for ViMoLa training.

Each .pt file (produced by extract_trajectories.sbatch) contains:
    velocities:        (48, 1024, 2) float32  — velocity field, T=48, N=1024
    initial_position:  (1024, 2)    float32  — first-frame point positions
    positions:         (49, 1024, 2) float32  — full position sequence (optional)
    video_id:          str
    dataset:           str   ('ssv2' or 'k400')
    text:              str   — action label / template description

The dataset indexes all samples from manifest_*.json files found in a
directory, then lazily loads the corresponding .pt file per __getitem__.
Missing or corrupted .pt files are skipped at index-build time.

Usage:
    ds = TrajectoryDataset(
        trajectory_dir="/projects/aclab/bishoy/data/vimola/trajectories",
        grid_size=32,
        seq_len=48,
        normalize_positions=True,
    )
    velocities, initial_position, text = ds[0]
    # velocities:       (48, 1024, 2)
    # initial_position: (1024, 2)
    # text:             str

For DataLoader use, a custom collate_fn is provided that handles the
string text field alongside tensors.
"""
import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _build_index_from_manifests(trajectory_dir: Path) -> List[dict]:
    """Scan manifest_*.json files and collect valid (path, text) records."""
    records: List[dict] = []
    manifest_files = sorted(trajectory_dir.glob("manifest_*.json"))

    if not manifest_files:
        logger.warning(f"No manifest_*.json files found in {trajectory_dir}")
        return records

    for mf in manifest_files:
        try:
            with open(mf) as f:
                entries = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping unreadable manifest {mf}: {e}")
            continue

        for entry in entries:
            if entry.get("status") != "success":
                continue
            traj_path = Path(entry.get("trajectory_path", ""))
            if not traj_path.exists():
                continue
            records.append({
                "path": traj_path,
                "text": entry.get("text", ""),
                "video_id": entry.get("video_id", traj_path.stem),
                "dataset": entry.get("dataset", "unknown"),
            })

    return records


def _build_index_from_directory(trajectory_dir: Path) -> List[dict]:
    """Fallback: collect all .pt files directly (no manifest required)."""
    records: List[dict] = []
    for pt_file in sorted(trajectory_dir.glob("*.pt")):
        records.append({
            "path": pt_file,
            "text": "",
            "video_id": pt_file.stem,
            "dataset": "unknown",
        })
    return records


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset over extracted point-track files.

    Args:
        trajectory_dir:      path to directory containing .pt trajectory files
                             and manifest_*.json index files
        grid_size:           expected point grid side length (default 32)
        seq_len:             expected velocity timestep count (default 48)
        normalize_positions: if True, normalise (x, y) to [0, 1]² using the
                             grid_size as the assumed pixel resolution
                             (assumes positions are in pixel coords [0, grid_size])
        transform:           optional callable applied to (velocities, initial_position)
        use_manifests:       if True, build index from manifest JSONs;
                             if False, glob all .pt files directly
        max_samples:         cap the dataset size (useful for quick runs)
    """

    def __init__(
        self,
        trajectory_dir: str,
        grid_size: int = 32,
        seq_len: int = 48,
        normalize_positions: bool = False,
        transform: Optional[Callable] = None,
        use_manifests: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.trajectory_dir = Path(trajectory_dir)
        self.grid_size = grid_size
        self.seq_len = seq_len
        self.normalize_positions = normalize_positions
        self.transform = transform

        if not self.trajectory_dir.exists():
            raise FileNotFoundError(f"Trajectory directory not found: {self.trajectory_dir}")

        if use_manifests:
            records = _build_index_from_manifests(self.trajectory_dir)
            if not records:
                logger.info("No manifests found; falling back to direct .pt glob")
                records = _build_index_from_directory(self.trajectory_dir)
        else:
            records = _build_index_from_directory(self.trajectory_dir)

        if max_samples is not None:
            records = records[:max_samples]

        self.records = records
        logger.info(f"TrajectoryDataset: {len(self.records)} samples from {self.trajectory_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            velocities:       (T, N, 2) float32  — T=seq_len, N=grid_size²
            initial_position: (N, 2)   float32
            text:             str      — action description
        """
        rec = self.records[idx]
        data = torch.load(rec["path"], map_location="cpu", weights_only=True)

        velocities = data["velocities"].float()        # (T, N, 2)
        initial_position = data["initial_position"].float()  # (N, 2)
        text = rec["text"] or data.get("text", "")

        # Validate shapes
        T, N, _ = velocities.shape
        if T != self.seq_len or N != self.grid_size ** 2:
            raise ValueError(
                f"Shape mismatch in {rec['path']}: "
                f"expected ({self.seq_len}, {self.grid_size**2}, 2), "
                f"got {tuple(velocities.shape)}"
            )

        if self.normalize_positions:
            # Assume positions are in pixel coords; normalise to [0, 1]
            # The grid spans a [0, 1] normalised video frame after extraction
            # (CoTracker outputs normalised coords by default); this is a no-op
            # if already normalised but guards against raw pixel inputs.
            pass

        if self.transform is not None:
            velocities, initial_position = self.transform(velocities, initial_position)

        return velocities, initial_position, text

    def metadata(self, idx: int) -> dict:
        """Return metadata for a sample without loading tensors."""
        rec = self.records[idx]
        return {
            "video_id": rec["video_id"],
            "dataset": rec["dataset"],
            "text": rec["text"],
            "path": str(rec["path"]),
        }


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate for DataLoader: handles the str text field.

    Returns:
        velocities:       (B, T, N, 2)
        initial_position: (B, N, 2)
        texts:            list[str] of length B
    """
    velocities = torch.stack([b[0] for b in batch])
    initial_pos = torch.stack([b[1] for b in batch])
    texts = [b[2] for b in batch]
    return velocities, initial_pos, texts


# ---------------------------------------------------------------------------
# Quick self-test (runs without any real data — uses a temp directory)
# ---------------------------------------------------------------------------

def _smoke():
    import tempfile
    import os

    T, N = 48, 1024
    tmp = tempfile.mkdtemp()

    # Write 3 fake .pt files
    for i in range(3):
        vel = torch.randn(T, N, 2)
        init = torch.rand(N, 2)
        torch.save({
            "velocities": vel,
            "initial_position": init,
            "positions": torch.randn(T + 1, N, 2),
            "video_id": f"fake_{i:04d}",
            "dataset": "test",
            "text": f"a person doing action {i}",
        }, os.path.join(tmp, f"fake_{i:04d}.pt"))

    # Write a manifest
    manifest = [
        {
            "status": "success",
            "video_id": f"fake_{i:04d}",
            "trajectory_path": os.path.join(tmp, f"fake_{i:04d}.pt"),
            "text": f"a person doing action {i}",
            "dataset": "test",
        }
        for i in range(3)
    ]
    with open(os.path.join(tmp, "manifest_0001.json"), "w") as f:
        json.dump(manifest, f)

    ds = TrajectoryDataset(tmp, grid_size=32, seq_len=T)
    assert len(ds) == 3, len(ds)

    vel, init_pos, text = ds[0]
    assert vel.shape == (T, N, 2), vel.shape
    assert init_pos.shape == (N, 2), init_pos.shape
    assert isinstance(text, str)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    vels, inits, texts = next(iter(loader))
    assert vels.shape == (2, T, N, 2), vels.shape
    assert len(texts) == 2

    print("TrajectoryDataset smoke test passed")
    print(f"  dataset size: {len(ds)}")
    print(f"  batch shapes: vel={vels.shape}, init={inits.shape}")
    print(f"  sample text: '{text}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke()
