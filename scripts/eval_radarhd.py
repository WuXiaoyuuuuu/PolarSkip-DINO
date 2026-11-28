#!/usr/bin/env python3
"""Evaluate a PolarSkip-DINO checkpoint on the RadarHD test set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

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
from polarskip_dino.dataset import RadarHDDataset


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PolarSkip-DINO on RadarHD")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/radarhd.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dino-weights", type=str, default=None)
    parser.add_argument("--dinov3-repo", type=str, default=None)
    parser.add_argument("--variant", choices=["full", "lite"], default="full")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--exclude-ids-json",
        type=str,
        default=str(REPO_ROOT / "configs/dev500_sids.json"),
        help="JSON list of held-out dev sample IDs to exclude (default: paper's 500 dev IDs -> 18,075 test frames)",
    )
    parser.add_argument("--include-all-test", action="store_true", help="Evaluate all 18,575 test frames without exclusion")
    parser.add_argument("--budget-frac", type=float, default=None, help="0.35 (default) or 0.25 (compact)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional subset size for quick check")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = args.data_root or cfg["data"]["radarhd_root"]
    dino_weights = args.dino_weights or cfg["model"]["dino_weights"]
    dinov3_repo = args.dinov3_repo or cfg["model"].get("dinov3_repo")
    rcfg = cfg["readout"]
    budget_frac = args.budget_frac if args.budget_frac is not None else float(rcfg["budget_frac"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (args.precision == "fp16") and (device.type == "cuda")
    dtype = torch.float16 if use_fp16 else torch.float32

    geom = RadarGeometry(device=device)
    geom_run = geom
    if use_fp16:
        geom_run = RadarGeometry(device=device)
        for attr in (
            "dino_cart_grid_64", "cart_grid_64", "cart_grid_128", "cart_grid_256",
            "m_roi_64", "m_roi_128", "m_roi_256", "polar_geo_channels",
            "imagenet_mean", "imagenet_std", "black_pad",
        ):
            v = getattr(geom_run, attr)
            if torch.is_tensor(v):
                setattr(geom_run, attr, v.to(dtype=dtype))

    model = PolarSkipDINO(dino_weights, dinov3_repo=dinov3_repo, use_polar_skip=(args.variant == "full")).to(device).eval()
    model.load_decoder_checkpoint(args.checkpoint)
    if use_fp16:
        model = model.to(dtype=dtype).eval()

    exclude_ids = None
    if not args.include_all_test and args.exclude_ids_json and Path(args.exclude_ids_json).is_file():
        exclude_ids = json.loads(Path(args.exclude_ids_json).read_text(encoding="utf-8"))
    dataset = RadarHDDataset(data_root=data_root, split="test", exclude_ids=exclude_ids, max_samples=args.max_samples)
    print(f"[Eval] Loaded {len(dataset)} RadarHD test frames | checkpoint={Path(args.checkpoint).name} | d={budget_frac} | precision={args.precision}")

    t0 = time.monotonic()
    all_metrics = []
    per_traj = defaultdict(list)

    for idx in range(len(dataset)):
        item = dataset[idx]
        sid = item["sample_id"]
        raw = item["raw_polar"].unsqueeze(0).to(device=device, dtype=dtype)
        prob = (torch.sigmoid(model(raw, geom_run).float()) * geom.m_roi_256.float()).squeeze().cpu().numpy()
        pred_pts = dual_tier_readout(
            prob,
            sample_id=sid,
            tau_hi=float(rcfg["tau_hi"]),
            tau_lo=float(rcfg["tau_lo"]),
            budget_frac=budget_frac,
            weak_frac=float(rcfg["weak_frac"]),
            max_points=int(rcfg["max_points"]),
            roi_mask=geom.m_roi_256_np,
        )
        gt_pts = polar_lidar_to_points(item["lidar_u8"])
        m = compute_pointcloud_metrics(pred_pts, gt_pts, thresholds=(0.20, 0.10, 0.05), far_range_m=6.0)
        if len(gt_pts) > 0:
            m["empty_pred"] = int(len(pred_pts) == 0)
            all_metrics.append(m)
            per_traj[item["traj_id"]].append(m)
        if (idx + 1) % 2000 == 0 or (idx + 1) == len(dataset):
            print(f"  Evaluated {idx + 1}/{len(dataset)} frames ({time.monotonic() - t0:.1f}s)")

    def _mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "checkpoint": str(args.checkpoint),
        "variant": args.variant,
        "budget_frac": budget_frac,
        "num_frames": len(all_metrics),
        "num_empty_pred_frames": int(sum(r.get("empty_pred", 0) for r in all_metrics)),
        "num_cd_valid_frames": int(sum(1 for r in all_metrics if r.get("cd") is not None)),
        "pts": _mean(all_metrics, "pts"),
        "cd_m": _mean(all_metrics, "cd"),
        "pred_to_gt_m": _mean(all_metrics, "p2g"),
        "gt_to_pred_m": _mean(all_metrics, "g2p"),
        "f1_20cm": _mean(all_metrics, "f1_20cm"),
        "f1_10cm": _mean(all_metrics, "f1_10cm"),
        "f1_5cm": _mean(all_metrics, "f1_5cm"),
        "recall_ge6m_present": _mean(all_metrics, "far_recall"),
        "num_trajectories": len(per_traj),
    }

    print("\n=== RadarHD Evaluation Summary ===")
    print(
        f"Frames={summary['num_frames']} | Pts={summary['pts']:.1f} | CD={summary['cd_m']:.4f} m "
        f"(P->G={summary['pred_to_gt_m']:.4f}, G->P={summary['gt_to_pred_m']:.4f}) | "
        f"F1@20={summary['f1_20cm']*100:.2f}% | F1@10={summary['f1_10cm']*100:.2f}% | "
        f"F1@5={summary['f1_5cm']*100:.2f}% | >=6m_R={summary['recall_ge6m_present']*100:.2f}%"
    )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary -> {args.output_json}")


if __name__ == "__main__":
    main()
