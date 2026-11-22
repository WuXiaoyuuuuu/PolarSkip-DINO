"""Exact voxel-center polar <-> Cartesian geometry and DINOv3 input preparation."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

MAX_RANGE_M = 10.8
GRID_H = 256
GRID_W = 512
POLAR_R_BINS = 256
POLAR_AZ_BINS = 64

X_CENTERS_NP = (np.arange(GRID_H, dtype=np.float32) + 0.5) * (MAX_RANGE_M / GRID_H)
Y_CENTERS_NP = (np.arange(GRID_W, dtype=np.float32) + 0.5 - GRID_W / 2.0) * (2.0 * MAX_RANGE_M / GRID_W)
DX_M = float(X_CENTERS_NP[1] - X_CENTERS_NP[0])
DY_M = float(Y_CENTERS_NP[1] - Y_CENTERS_NP[0])


def build_cart_grid(height: int, width: int, max_range_m: float = MAX_RANGE_M) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalized Cartesian sampling grids aligned to exact voxel centers.

    Returns:
        grid: (1, H, W, 2) normalized grid for F.grid_sample from polar (r, theta).
        m_roi: (1, 1, H, W) boolean semicircular field-of-view mask (r <= max_range_m).
        coord: (3, H, W) normalized [r / max_range_m, sin(theta), cos(theta)].
    """
    x = (torch.arange(height, dtype=torch.float32) + 0.5) * (max_range_m / height)
    y = (torch.arange(width, dtype=torch.float32) + 0.5 - width / 2.0) * (2.0 * max_range_m / width)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    r = torch.hypot(gx, gy)
    theta = torch.atan2(gy, gx)

    norm_x = theta / (math.pi / 2.0)
    norm_y = 2.0 * (r / max_range_m) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
    m_roi = (r <= max_range_m).view(1, 1, height, width)
    coord = torch.stack([r / max_range_m, torch.sin(theta), torch.cos(theta)], dim=0)
    return grid, m_roi, coord


def build_dino_cart_grid(height: int = 64, width: int = 128, max_range_m: float = MAX_RANGE_M) -> torch.Tensor:
    """Build Cartesian sampling grid from 17x33 DINOv3 tokens accounting for 272x528 padding."""
    x = (torch.arange(height, dtype=torch.float32) + 0.5) * (max_range_m / height)
    y = (torch.arange(width, dtype=torch.float32) + 0.5 - width / 2.0) * (2.0 * max_range_m / width)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    r = torch.hypot(gx, gy)
    theta = torch.atan2(gy, gx)

    y_px = (r / max_range_m) * 255.0 + 0.5
    x_px = ((theta + math.pi / 2.0) / math.pi) * 511.0 + 0.5
    norm_gx = 2.0 * x_px / 528.0 - 1.0
    norm_gy = 2.0 * y_px / 272.0 - 1.0
    return torch.stack([norm_gx, norm_gy], dim=-1).unsqueeze(0)


def build_polar_geo_channels(r_bins: int = POLAR_R_BINS, az_bins: int = POLAR_AZ_BINS, max_range_m: float = MAX_RANGE_M) -> torch.Tensor:
    """Precompute [r / max_range_m, cos(theta), sin(theta)] on the native 256x64 polar grid."""
    r_pol = (torch.arange(r_bins, dtype=torch.float32) + 0.5) * (max_range_m / r_bins)
    th_pol = (torch.arange(az_bins, dtype=torch.float32) + 0.5 - az_bins / 2.0) * (math.pi / az_bins)
    gr, gth = torch.meshgrid(r_pol, th_pol, indexing="ij")
    return torch.stack([gr / max_range_m, torch.cos(gth), torch.sin(gth)], dim=0)


class RadarGeometry:
    """Device-resident physical coordinate grids and ROI masks for PolarSkip-DINO."""

    def __init__(self, device: torch.device | str = "cpu", max_range_m: float = MAX_RANGE_M):
        self.device = torch.device(device)
        self.max_range_m = float(max_range_m)

        g256, m256, c256 = build_cart_grid(256, 512, max_range_m)
        g128, m128, _ = build_cart_grid(128, 256, max_range_m)
        g64, m64, _ = build_cart_grid(64, 128, max_range_m)
        gdino = build_dino_cart_grid(64, 128, max_range_m)
        pgeo = build_polar_geo_channels(POLAR_R_BINS, POLAR_AZ_BINS, max_range_m)

        self.cart_grid_256 = g256.to(self.device)
        self.cart_grid_128 = g128.to(self.device)
        self.cart_grid_64 = g64.to(self.device)
        self.dino_cart_grid_64 = gdino.to(self.device)
        self.m_roi_256 = m256.to(self.device)
        self.m_roi_128 = m128.to(self.device)
        self.m_roi_64 = m64.to(self.device)
        self.coord_256 = c256.to(self.device)
        self.polar_geo_channels = pgeo.to(self.device)

        self.m_roi_256_np = m256.squeeze().cpu().numpy().astype(bool)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.black_pad = torch.tensor(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], device=self.device
        ).view(1, 3, 1, 1)


def make_dino_input(raw_polar_64: torch.Tensor, geom: RadarGeometry) -> torch.Tensor:
    """Convert raw polar radar heatmap (B, 1, 256, 64) in [0, 1] to padded DINOv3 input (B, 3, 272, 528)."""
    batch_size = raw_polar_64.shape[0]
    t_512 = F.interpolate(raw_polar_64, size=(256, 512), mode="bilinear", align_corners=False)
    t_pad = F.pad(t_512, (0, 6, 0, 10), mode="constant", value=0.0)
    t_norm = (t_pad.repeat(1, 3, 1, 1) - geom.imagenet_mean) / geom.imagenet_std
    padded = geom.black_pad.expand(batch_size, 3, 272, 528).clone()
    padded[:, :, :266, :518] = t_norm
    return padded


def indices_to_metric_points(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Convert Cartesian grid indices (row, col) to (N, 3) metric coordinates [x, y, 0] at voxel centers."""
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.zeros((len(rows), 3), dtype=np.float32)
    pts[:, 0] = X_CENTERS_NP[rows]
    pts[:, 1] = Y_CENTERS_NP[cols]
    return pts


def polar_lidar_to_points(lidar_polar_u8: np.ndarray, max_range_m: float = MAX_RANGE_M) -> np.ndarray:
    """Convert a RadarHD 256x512 polar LiDAR occupancy PNG to (N, 3) metric 2D points [x, y, 0]."""
    rr, aa = np.nonzero(lidar_polar_u8 > 0)
    if len(rr) == 0:
        return np.empty((0, 3), dtype=np.float32)
    r_m = rr.astype(np.float64) * (max_range_m / float(lidar_polar_u8.shape[0] - 1))
    theta = -0.5 * math.pi + aa.astype(np.float64) * (math.pi / float(lidar_polar_u8.shape[1] - 1))
    pts = np.zeros((len(rr), 3), dtype=np.float32)
    pts[:, 0] = (r_m * np.cos(theta)).astype(np.float32)
    pts[:, 1] = (r_m * np.sin(theta)).astype(np.float32)
    return pts
