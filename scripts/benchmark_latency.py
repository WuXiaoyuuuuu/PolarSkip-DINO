#!/usr/bin/env python3
"""Strict Batch=1 synchronized end-to-end latency profiler for PolarSkip-DINO."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polarskip_dino import PolarSkipDINO, RadarGeometry, dual_tier_readout, make_dino_input


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark strict Batch=1 latency of PolarSkip-DINO")
    parser.add_argument("--dino-weights", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dinov3-repo", type=str, default=None)
    parser.add_argument("--variant", choices=["full", "lite"], default="full")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--radar-dir", type=str, default=None, help="Optional directory of real R_*.png test frames")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

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

    model = PolarSkipDINO(args.dino_weights, dinov3_repo=args.dinov3_repo, use_polar_skip=(args.variant == "full")).to(device).eval()
    if args.checkpoint:
        model.load_decoder_checkpoint(args.checkpoint)
    if use_fp16:
        model = model.to(dtype=dtype).eval()

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_iters = args.warmup + args.runs
    if args.radar_dir and Path(args.radar_dir).is_dir():
        pngs = sorted(Path(args.radar_dir).glob("R_*.png"))[:total_iters]
        frames = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in pngs]
        sids = [p.stem[2:] for p in pngs]
    else:
        rng = np.random.default_rng(45)
        frames = [rng.random((256, 64), dtype=np.float32)]
        sids = ["synthetic"]

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for i in range(args.warmup):
        raw_np = frames[i % len(frames)]
        t_raw = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
        p_map = (torch.sigmoid(model(t_raw, geom_run).float()) * geom.m_roi_256.float()).squeeze().cpu().numpy()
        _ = dual_tier_readout(p_map, sample_id="warmup", roi_mask=geom.m_roi_256_np)
    _sync()

    e2e_ms, gpu_ms, readout_ms = [], [], []
    for i in range(args.runs):
        idx = (args.warmup + i) % len(frames)
        raw_np = frames[idx]
        sid = sids[idx] if len(sids) > 1 else f"run_{i}"
        _sync()
        t0 = time.perf_counter()
        t_raw = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
        d_in = make_dino_input(t_raw, geom_run)
        logits = model(t_raw, geom_run, dino_input=d_in)
        prob_np = (torch.sigmoid(logits.float()) * geom.m_roi_256.float()).squeeze().cpu().numpy()
        _sync()
        t1 = time.perf_counter()
        _ = dual_tier_readout(prob_np, sample_id=sid, roi_mask=geom.m_roi_256_np)
        t2 = time.perf_counter()

        gpu_ms.append((t1 - t0) * 1000.0)
        readout_ms.append((t2 - t1) * 1000.0)
        e2e_ms.append((t2 - t0) * 1000.0)

    mean_e2e = float(np.mean(e2e_ms))
    p95_e2e = float(np.percentile(e2e_ms, 95))
    print(f"=== Strict Batch=1 Benchmark ({args.variant}, {args.precision}, {device}) ===")
    print(f"Params: {total_p/1e6:.2f}M total ({train_p/1e6:.2f}M trainable)")
    print(f"GPU+Transfer: {np.mean(gpu_ms):.2f} ms | CPU Readout: {np.mean(readout_ms):.2f} ms")
    print(f"End-to-End  : {mean_e2e:.2f} ms ({1000.0/mean_e2e:.1f} Hz) | P95: {p95_e2e:.2f} ms")


if __name__ == "__main__":
    main()
