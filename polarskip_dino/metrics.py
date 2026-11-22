"""Point-cloud evaluation metrics: Chamfer Distance, multi-tolerance F1, and far-field recall."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree


def compute_pointcloud_metrics(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    thresholds: Sequence[float] = (0.20, 0.10, 0.05),
    far_range_m: float = 6.0,
) -> Dict[str, Optional[float]]:
    """Compute bidirectional CD, multi-tolerance Precision/Recall/F1, and far-field recall."""
    n_pred = int(len(pred_pts))
    n_gt = int(len(gt_pts))
    res: Dict[str, Optional[float]] = {
        "pts": float(n_pred),
        "cd": None,
        "p2g": None,
        "g2p": None,
        "far_recall": None,
    }
    for thr in thresholds:
        cm = int(round(thr * 100))
        res[f"f1_{cm}cm"] = 0.0
        res[f"p_{cm}cm"] = 0.0
        res[f"r_{cm}cm"] = 0.0

    if n_gt == 0:
        return res
    gt_xy = gt_pts[:, :2].astype(np.float64)
    gt_r = np.hypot(gt_xy[:, 0], gt_xy[:, 1])
    far_mask = gt_r >= far_range_m
    has_far = bool(far_mask.any())

    if n_pred == 0:
        if has_far:
            res["far_recall"] = 0.0
        return res

    pred_xy = pred_pts[:, :2].astype(np.float64)
    d_p2g = cKDTree(gt_xy).query(pred_xy, k=1)[0]
    d_g2p = cKDTree(pred_xy).query(gt_xy, k=1)[0]

    p2g = float(np.mean(d_p2g))
    g2p = float(np.mean(d_g2p))
    res["p2g"] = p2g
    res["g2p"] = g2p
    res["cd"] = 0.5 * (p2g + g2p)

    for thr in thresholds:
        cm = int(round(thr * 100))
        p = float(np.mean(d_p2g <= thr))
        r = float(np.mean(d_g2p <= thr))
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0
        res[f"p_{cm}cm"] = p
        res[f"r_{cm}cm"] = r
        res[f"f1_{cm}cm"] = f1

    if has_far:
        res["far_recall"] = float(np.mean(d_g2p[far_mask] <= 0.20))
    return res
