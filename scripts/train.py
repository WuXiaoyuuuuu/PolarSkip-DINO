#!/usr/bin/env python3
"""Train PolarSkip-DINO (Full or Lite) on the RadarHD dataset."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polarskip_dino import (
    PolarSkipDINO,
    RadarGeometry,
    compute_cartesian_target,
    compute_roi_wbce_dice_loss,
)
from polarskip_dino.dataset import RadarHDDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PolarSkip-DINO on RadarHD")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/radarhd.yaml"))
    parser.add_argument("--data-root", type=str, default=None, help="Override RadarHD dataset_5 path")
    parser.add_argument("--dino-weights", type=str, default=None, help="Path to DINOv3-S/16 .pth weights")
    parser.add_argument("--dinov3-repo", type=str, default=None, help="Optional local facebookresearch/dinov3 path")
    parser.add_argument("--variant", choices=["full", "lite"], default=None, help="full (with PolarSkip) or lite")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = args.data_root or cfg["data"]["radarhd_root"]
    dino_weights = args.dino_weights or cfg["model"]["dino_weights"]
    dinov3_repo = args.dinov3_repo or cfg["model"].get("dinov3_repo")
    variant = args.variant or cfg["model"].get("variant", "full")
    total_steps = args.steps or int(cfg["train"]["steps"])
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    pos_weight = float(cfg["train"].get("pos_weight", 3.0))
    seed = args.seed if args.seed is not None else int(cfg["train"]["seed"])
    out_dir = Path(args.output_dir or cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = RadarGeometry(device=device)

    dataset = RadarHDDataset(data_root=data_root, split="train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    use_polar_skip = variant == "full"
    model = PolarSkipDINO(dino_weights, dinov3_repo=dinov3_repo, use_polar_skip=use_polar_skip).to(device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[PolarSkip-DINO:{variant}] Total={n_total/1e6:.2f}M | Trainable={n_trainable/1e6:.2f}M | Device={device}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    log_csv = out_dir / f"train_log_{variant}_seed{seed}.csv"
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "loss", "wbce", "dice", "lr", "elapsed_min"])

    t0 = time.monotonic()
    data_iter = iter(loader)
    for step in range(1, total_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        raw_polar = batch["raw_polar"].to(device, non_blocking=True)
        lidar_polar = batch["lidar_polar"].to(device, non_blocking=True)
        y_bin = compute_cartesian_target(lidar_polar, geom)

        optimizer.zero_grad(set_to_none=True)
        logits = model(raw_polar, geom)
        loss, l_wbce, l_dice = compute_roi_wbce_dice_loss(logits, y_bin, geom.m_roi_256, pos_weight=pos_weight)
        loss.backward()
        if step == 1:
            assert all(p.grad is None for p in model.dino.parameters()), "Frozen DINOv3 received gradients!"
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 200 == 0 or step == total_steps:
            elapsed_min = (time.monotonic() - t0) / 60.0
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:4d}/{total_steps} | loss={loss.item():.4f} "
                f"(wbce={l_wbce.item():.4f}, dice={l_dice.item():.4f}) | lr={cur_lr:.2e} | {elapsed_min:.2f} min"
            )
            with log_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step, f"{loss.item():.5f}", f"{l_wbce.item():.5f}", f"{l_dice.item():.5f}", f"{cur_lr:.2e}", f"{elapsed_min:.2f}"])

    ckpt_path = out_dir / f"polarskip_dino_{variant}_seed{seed}_step{total_steps}.pth"
    torch.save(
        {
            "arm": "B1" if use_polar_skip else "B0",
            "variant": variant,
            "seed": seed,
            "step": total_steps,
            "model_state": model.trainable_state_dict(),
            "train_time_s": time.monotonic() - t0,
        },
        ckpt_path,
    )
    print(f"Saved lightweight decoder checkpoint ({ckpt_path.stat().st_size / 1e6:.2f} MB) -> {ckpt_path}")


if __name__ == "__main__":
    main()
