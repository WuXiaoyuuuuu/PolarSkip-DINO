"""Zero-training continuous-probability dual-tier budgeted readout."""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import numpy as np

from .geometry import GRID_W, build_cart_grid, indices_to_metric_points

_DEFAULT_ROI_256_NP = build_cart_grid(256, 512)[1].squeeze().cpu().numpy().astype(bool)


def _deterministic_seed(sample_id: str) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def dual_tier_readout(
    prob_map: np.ndarray,
    sample_id: str = "0",
    tau_hi: float = 0.006,
    tau_lo: float = 0.002,
    budget_frac: float = 0.35,
    weak_frac: float = 0.15,
    max_points: int = 1750,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract a 2D metric point cloud from continuous Cartesian probabilities.

    Args:
        prob_map: (256, 512) float probability map in [0, 1].
        sample_id: string key used for deterministic uniform sampling in the weak tier.
        tau_hi: primary-tier probability threshold (0.006 for RadarHD, 0.30 for ColoRadar LOBO).
        tau_lo: weak-support lower bound (0.002 for RadarHD, 0.10 for ColoRadar LOBO).
        budget_frac: frame-adaptive budget factor d (0.35 default, 0.25 compact mode).
        weak_frac: fraction alpha of budget reserved for uniform weak-tier sampling (default: 0.15).
        max_points: hard frame budget cap K_max (default: 1750).
        roi_mask: optional (256, 512) boolean valid ROI mask.
    Returns:
        (N, 3) float32 metric points [x, y, 0].
    """
    mask = _DEFAULT_ROI_256_NP if roi_mask is None else (roi_mask > 0.5)
    p_roi = np.where(mask, prob_map, -1.0).reshape(-1)
    main_idx = np.flatnonzero(p_roi >= tau_hi)
    k0 = int(main_idx.size)
    if k0 == 0:
        return np.empty((0, 3), dtype=np.float32)

    budget_k = min(max(1, math.floor(budget_frac * k0)), max_points)
    weak_idx = np.flatnonzero((p_roi >= tau_lo) & (p_roi < tau_hi))
    n_weak = min(math.floor(weak_frac * budget_k), int(weak_idx.size)) if tau_lo < tau_hi else 0
    keep_main = budget_k - n_weak

    if keep_main >= k0:
        sel_main = main_idx
    else:
        scores = p_roi[main_idx]
        sel_main = main_idx[np.argpartition(scores, -keep_main)[-keep_main:]]

    parts = [sel_main]
    if n_weak > 0:
        rng = np.random.Generator(np.random.PCG64(_deterministic_seed(sample_id)))
        sel_weak = np.sort(rng.choice(weak_idx, size=n_weak, replace=False))
        parts.append(sel_weak)

    sel = np.concatenate(parts)
    return indices_to_metric_points(sel // GRID_W, sel % GRID_W)


def single_threshold_readout(
    prob_map: np.ndarray,
    thresh: float = 0.30,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract all ROI pixels with prob_map >= thresh."""
    mask = _DEFAULT_ROI_256_NP if roi_mask is None else (roi_mask > 0.5)
    sel = np.flatnonzero(np.where(mask, prob_map, -1.0).reshape(-1) >= thresh)
    return indices_to_metric_points(sel // GRID_W, sel % GRID_W)


def subsample_same_k(points: np.ndarray, target_k: int, sample_id: str = "0") -> np.ndarray:
    """Deterministic per-frame random subsampling to target_k points for Same-K comparison."""
    if len(points) <= target_k:
        return points
    if target_k <= 0:
        return np.empty((0, 3), dtype=np.float32)
    rng = np.random.Generator(np.random.PCG64(_deterministic_seed(f"samek_{sample_id}")))
    idx = np.sort(rng.choice(len(points), size=target_k, replace=False))
    return points[idx]
