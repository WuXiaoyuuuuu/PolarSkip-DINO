"""Batch-invariant ROI weighted BCE + L1-SoftDice training objective."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .geometry import RadarGeometry


def compute_cartesian_target(lidar_polar_batch: torch.Tensor, geom: RadarGeometry, thresh: float = 0.25) -> torch.Tensor:
    """Project paired polar LiDAR binary maps (B, 1, 256, 512) to Cartesian occupancy labels."""
    b = lidar_polar_batch.shape[0]
    grid = geom.cart_grid_256.expand(b, -1, -1, -1)
    l_bil = F.grid_sample(lidar_polar_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return ((l_bil >= thresh) & geom.m_roi_256).float()


def compute_roi_wbce_dice_loss(
    logits: torch.Tensor,
    y_bin: torch.Tensor,
    roi_mask: torch.Tensor,
    pos_weight: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute batch-size-invariant ROI WBCE + L1-SoftDice loss.

    Args:
        logits: (B, 1, 256, 512) raw decoder logits.
        y_bin: (B, 1, 256, 512) binary Cartesian target.
        roi_mask: (1, 1, 256, 512) or broadcastable boolean/float valid ROI mask.
        pos_weight: positive pixel weight for WBCE (default: 3.0).
    Returns:
        (total_loss, wbce_loss_detached, dice_loss_detached)
    """
    m = roi_mask.to(dtype=logits.dtype).expand_as(logits)
    y = y_bin * m
    denom_roi = m.sum().clamp(min=1.0)

    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    weight = torch.where(y == 1.0, float(pos_weight), 1.0)
    wbce_roi = (bce * weight * m).sum() / denom_roi

    prob = torch.sigmoid(logits) * m
    inter = 2.0 * (prob * y).sum()
    dice_loss = 1.0 - (inter + 1e-6) / (prob.sum() + y.sum() + 1e-6)

    total_loss = 0.5 * wbce_roi + 0.5 * dice_loss
    return total_loss, wbce_roi.detach(), dice_loss.detach()
