"""Clean dataset loaders for RadarHD (dataset_5) and ColoRadar 3D heatmap projection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
import torch
from torch.utils.data import Dataset

from .geometry import MAX_RANGE_M, POLAR_AZ_BINS, POLAR_R_BINS


def _sample_id_from_radar_path(path: Path) -> str:
    stem = path.stem  # e.g. R_117_0
    return stem[2:] if stem.startswith("R_") else stem


class RadarHDDataset(Dataset):
    """Standalone PyTorch dataset for RadarHD dataset_5/{train,test}/{radar,lidar}."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        exclude_ids: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(data_root) / split
        self.radar_dir = self.root / "radar"
        self.lidar_dir = self.root / "lidar"
        if not self.radar_dir.is_dir() or not self.lidar_dir.is_dir():
            raise FileNotFoundError(f"Expected radar/ and lidar/ under {self.root}")

        excluded = set(str(x) for x in exclude_ids) if exclude_ids else set()
        radar_files = sorted(self.radar_dir.glob("R_*.png"))
        pairs: List[Tuple[str, Path, Path]] = []
        for r_path in radar_files:
            sid = _sample_id_from_radar_path(r_path)
            if sid in excluded:
                continue
            l_path = self.lidar_dir / f"L_{sid}.png"
            if l_path.is_file():
                pairs.append((sid, r_path, l_path))
        if max_samples is not None:
            pairs = pairs[: int(max_samples)]
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sid, r_path, l_path = self.pairs[idx]
        with Image.open(r_path) as r_img:
            raw_pol = np.asarray(r_img.convert("L"), dtype=np.float32) / 255.0
        with Image.open(l_path) as l_img:
            lidar_u8 = np.asarray(l_img.convert("L"), dtype=np.uint8)
        lidar_bin = (lidar_u8 > 0).astype(np.float32)

        return {
            "sample_id": sid,
            "traj_id": sid.split("_")[0],
            "raw_polar": torch.from_numpy(raw_pol).unsqueeze(0),     # (1, 256, 64)
            "lidar_polar": torch.from_numpy(lidar_bin).unsqueeze(0), # (1, 256, 512)
            "lidar_u8": lidar_u8,
        }


def project_coloradar_heatmap(
    heatmap_4d: np.ndarray,
    ranges: np.ndarray,
    az_bins: np.ndarray,
    el_bins: np.ndarray,
    min_range_m: float = 0.253,
    max_az_deg: float = 70.0,
) -> Tuple[np.ndarray, float]:
    """Extract a 2D 256x64 horizontal slice (mid-2 elevation bins) from a ColoRadar 3D heatmap.

    Args:
        heatmap_4d: (N_el, N_az, N_r, 2) float32 single-chip ColoRadar heatmap (channel 0 = intensity).
        ranges: (N_r,) range bin centers in meters.
        az_bins: (N_az,) azimuth bin angles in radians.
        el_bins: (N_el,) elevation bin angles in radians.
    Returns:
        (img_256x64_float01, max_valid_range_m)
    """
    r_grid = (np.arange(POLAR_R_BINS, dtype=np.float64) + 0.5) * (MAX_RANGE_M / POLAR_R_BINS)
    a_grid = (np.arange(POLAR_AZ_BINS, dtype=np.float64) + 0.5) * (math.pi / POLAR_AZ_BINS) - 0.5 * math.pi
    rr, aa = np.meshgrid(r_grid, a_grid, indexing="ij")

    j_sorted = np.argsort(np.abs(el_bins))
    mid_indices = [int(j_sorted[0]), int(j_sorted[1])]
    max_r = float(ranges[-1] * math.cos(float(el_bins[j_sorted[0]])))

    acc = np.zeros_like(rr, dtype=np.float64)
    for j in mid_indices:
        interp = RegularGridInterpolator((ranges, az_bins), heatmap_4d[j, :, :, 0].T, bounds_error=False, fill_value=0.0)
        vals = interp(np.c_[(rr / math.cos(float(el_bins[j]))).ravel(), aa.ravel()]).reshape(rr.shape)
        acc += np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    img = (acc / len(mid_indices)).astype(np.float32)

    valid = (rr >= min_range_m) & (rr <= max_r) & (np.abs(aa) <= math.radians(max_az_deg))
    img[~valid] = 0.0
    mx = float(img.max())
    if mx > 0.0:
        img[img < 0.05 * mx] = 0.0
        ok = img > 0.0
        if ok.any():
            db = 10.0 * np.log10(img[ok])
            lo, hi = float(db.min()), float(db.max())
            img[ok] = (db - lo) / (hi - lo) if hi > lo else 1.0
            img[~ok] = 0.0
    return np.clip(img, 0.0, 1.0).astype(np.float32), max_r
