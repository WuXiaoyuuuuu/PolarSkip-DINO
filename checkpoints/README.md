# Pretrained Checkpoints

Because PolarSkip-DINO keeps DINOv3-S/16 completely frozen, each released checkpoint stores **only the trainable decoder + PolarSkip weights (~9.7 MB for Full, ~8.9 MB for Lite)**.

## Required Backbone

Download the official **DINOv3 ViT-S/16** pretrained checkpoint (`dinov3_vits16_pretrain_lvd1689m-08c60483.pth`) from Meta's [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) repository under its license.

## Released PolarSkip-DINO Weights (Upload to GitHub Releases / Hugging Face)

| File | Variant | Dataset | Size | Paper Result |
|---|---|---|---:|---|
| `polarskip_dino_radarhd_seed45.pth` | Full (`2.50M` trainable) | RadarHD | 9.7 MB | Default released checkpoint (3-seed mean: `60.52% F1@20`, `0.3000 m CD`) |
| `polarskip_dino_radarhd_seed43.pth` | Full (`2.50M` trainable) | RadarHD | 9.7 MB | Multi-seed evaluation checkpoint |
| `polarskip_dino_radarhd_seed44.pth` | Full (`2.50M` trainable) | RadarHD | 9.7 MB | Multi-seed evaluation checkpoint |
| `polarskip_dino_lite_seed45.pth` | Lite w/o PolarSkip (`2.31M`) | RadarHD | 8.9 MB | `23.3 Hz` (`42.95 ms`) fast ablation |
| `polarskip_dino_coloradar_fold1_aspen.pth` | Full (LOBO, 4 min) | ColoRadar | 9.7 MB | Trained on `ec_hallways + arpg_lab`, tested on `aspen` |
| `polarskip_dino_coloradar_fold2_arpg_lab.pth` | Full (LOBO, 4 min) | ColoRadar | 9.7 MB | Trained on `ec_hallways + aspen`, tested on `arpg_lab` |
| `polarskip_dino_coloradar_fold3_ec_hallways.pth` | Full (LOBO, 4 min) | ColoRadar | 9.7 MB | Trained on `arpg_lab + aspen`, tested on `ec_hallways` |
