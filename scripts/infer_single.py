#!/usr/bin/env python3
"""Single-frame inference demo: reconstruct a 2D metric point cloud from one 256x64 radar PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polarskip_dino import (
    PolarSkipDINO,
    RadarGeometry,
    compute_pointcloud_metrics,
    dual_tier_readout,
    polar_lidar_to_points,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Single-frame PolarSkip-DINO inference")
    parser.add_argument("--radar-png", type=str, required=True, help="256x64 polar radar intensity PNG")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained PolarSkip-DINO decoder .pth")
    parser.add_argument("--dino-weights", type=str, required=True, help="Frozen DINOv3-S/16 .pth")
    parser.add_argument("--dinov3-repo", type=str, default=None)
    parser.add_argument("--lidar-png", type=str, default=None, help="Optional paired 256x512 LiDAR GT PNG")
    parser.add_argument("--variant", choices=["full", "lite"], default="full")
    parser.add_argument("--tau-hi", type=float, default=0.006)
    parser.add_argument("--tau-lo", type=float, default=0.002)
    parser.add_argument("--budget-frac", type=float, default=0.35)
    parser.add_argument("--out-prefix", type=str, default="outputs/demo_frame")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = RadarGeometry(device=device)
    model = PolarSkipDINO(args.dino_weights, dinov3_repo=args.dinov3_repo, use_polar_skip=(args.variant == "full")).to(device).eval()
    model.load_decoder_checkpoint(args.checkpoint)

    radar_path = Path(args.radar_png)
    with Image.open(radar_path) as img:
        raw_np = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    raw_t = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(raw_t, geom)).squeeze().cpu().numpy()

    pts = dual_tier_readout(
        prob,
        sample_id=radar_path.stem,
        tau_hi=args.tau_hi,
        tau_lo=args.tau_lo,
        budget_frac=args.budget_frac,
        roi_mask=geom.m_roi_256_np,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out_prefix}_points.npy", pts)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    title = f"PolarSkip-DINO ({args.variant}, d={args.budget_frac}) | pts={len(pts)}"
    if args.lidar_png:
        with Image.open(args.lidar_png) as gimg:
            gt_pts = polar_lidar_to_points(np.asarray(gimg.convert("L"), dtype=np.uint8))
        ax.scatter(gt_pts[:, 1], gt_pts[:, 0], s=1.5, c="#2ca02c", alpha=0.55, label="LiDAR GT", edgecolors="none")
        m = compute_pointcloud_metrics(pts, gt_pts)
        title += f" | CD={m['cd']:.3f}m | F1@20={m['f1_20cm']*100:.1f}%"

    ax.scatter(pts[:, 1], pts[:, 0], s=2.2, c="#1f77b4", alpha=0.85, label="PolarSkip-DINO", edgecolors="none")
    ax.set_xlim(10.8, -10.8)
    ax.set_ylim(0.0, 10.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral Y (m)")
    ax.set_ylabel("Forward X (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", markerscale=3)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_bev.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved point cloud -> {out_prefix}_points.npy ({len(pts)} pts) & plot -> {out_prefix}_bev.png")


if __name__ == "__main__":
    main()
