"""PolarSkip-DINO architecture: Frozen DINOv3-S/16 + PolarSkip + Isotropic Cartesian U-Net."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import RadarGeometry, make_dino_input


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + res)


class DilatedResBlock(nn.Module):
    """Residual block dilated along azimuth (dilation=(1, 2)) to bridge far-field beam gaps."""

    def __init__(self, channels: int, dilation=(1, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + res)


class PolarSkipEncoder(nn.Module):
    """Lightweight polar-domain encoder (+0.19M params) preserving full 64-bin angular resolution."""

    def __init__(self):
        super().__init__()
        # Input channels: [log1p(255*I)/log(256), I, r/10.8, cos(theta), sin(theta)]
        self.enc0 = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            DilatedResBlock(32, dilation=(1, 2)),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            DilatedResBlock(48, dilation=(1, 2)),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 64, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            DilatedResBlock(64, dilation=(1, 2)),
        )

    def forward(self, polar_5ch: torch.Tensor, geom: RadarGeometry) -> Dict[str, torch.Tensor]:
        p0 = self.enc0(polar_5ch)  # (B, 32, 256, 64)
        p1 = self.enc1(p0)         # (B, 48, 128, 64)
        p2 = self.enc2(p1)         # (B, 64, 64, 64)

        b = polar_5ch.shape[0]
        g256 = geom.cart_grid_256.expand(b, -1, -1, -1)
        g128 = geom.cart_grid_128.expand(b, -1, -1, -1)
        g64 = geom.cart_grid_64.expand(b, -1, -1, -1)

        s256 = F.grid_sample(p0, g256, mode="bilinear", padding_mode="zeros", align_corners=True) * geom.m_roi_256
        s128 = F.grid_sample(p1, g128, mode="bilinear", padding_mode="zeros", align_corners=True) * geom.m_roi_128
        s64 = F.grid_sample(p2, g64, mode="bilinear", padding_mode="zeros", align_corners=True) * geom.m_roi_64
        return {"s256": s256, "s128": s128, "s64": s64}


class CartesianUNetDecoder(nn.Module):
    """Isotropic metric Cartesian multi-scale U-Net decoder (~2.31M trainable params)."""

    def __init__(self, in_channels: int = 384, has_polar_skips: bool = True):
        super().__init__()
        self.has_polar_skips = has_polar_skips

        self.mix = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        c64, c32, c16, c128, c256 = 64, 96, 128, 48, 24

        self.enc64 = nn.Sequential(ResBlock(c64), ResBlock(c64))
        self.down32 = nn.Sequential(nn.Conv2d(c64, c32, 3, stride=2, padding=1), nn.GroupNorm(8, c32), nn.GELU())
        self.enc32 = nn.Sequential(ResBlock(c32), ResBlock(c32))
        self.down16 = nn.Sequential(nn.Conv2d(c32, c16, 3, stride=2, padding=1), nn.GroupNorm(8, c16), nn.GELU())
        self.bottleneck = nn.Sequential(ResBlock(c16), ResBlock(c16))

        self.up32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c16, c32, 3, padding=1),
            nn.GroupNorm(8, c32),
            nn.GELU(),
        )
        self.dec32_fuse = nn.Sequential(nn.Conv2d(c32 * 2, c32, 1), nn.GroupNorm(8, c32), nn.GELU())
        self.dec32 = nn.Sequential(ResBlock(c32), ResBlock(c32))

        self.up64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c32, c64, 3, padding=1),
            nn.GroupNorm(8, c64),
            nn.GELU(),
        )
        fuse64_in = c64 * 2 + (64 if has_polar_skips else 0)
        self.dec64_fuse = nn.Sequential(nn.Conv2d(fuse64_in, c64, 1), nn.GroupNorm(8, c64), nn.GELU())
        self.dec64 = nn.Sequential(ResBlock(c64), ResBlock(c64))

        self.up128 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c64, c128, 3, padding=1),
            nn.GroupNorm(8, c128),
            nn.GELU(),
        )
        fuse128_in = c128 + (48 if has_polar_skips else 0)
        self.dec128_fuse = (
            nn.Sequential(nn.Conv2d(fuse128_in, c128, 1), nn.GroupNorm(8, c128), nn.GELU())
            if has_polar_skips
            else nn.Identity()
        )
        self.dec128 = nn.Sequential(ResBlock(c128), ResBlock(c128))

        self.up256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c128, c256, 3, padding=1),
            nn.GroupNorm(4, c256),
            nn.GELU(),
        )
        fuse256_in = c256 + (32 if has_polar_skips else 0)
        self.dec256_fuse = (
            nn.Sequential(nn.Conv2d(fuse256_in, c256, 1), nn.GroupNorm(4, c256), nn.GELU())
            if has_polar_skips
            else nn.Identity()
        )
        self.dec256 = nn.Sequential(ResBlock(c256), ResBlock(c256))
        self.head = nn.Conv2d(c256, 1, 1)

    def forward(self, x: torch.Tensor, polar_skips: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x64 = self.mix(x)
        feat64 = self.enc64(x64)
        feat32 = self.enc32(self.down32(feat64))
        feat16 = self.bottleneck(self.down16(feat32))

        up32 = self.up32(feat16)
        dec32 = self.dec32(self.dec32_fuse(torch.cat([up32, feat32], dim=1)))

        up64 = self.up64(dec32)
        if self.has_polar_skips and polar_skips is not None and "s64" in polar_skips:
            dec64 = self.dec64_fuse(torch.cat([up64, feat64, polar_skips["s64"]], dim=1))
        else:
            dec64 = self.dec64_fuse(torch.cat([up64, feat64], dim=1))
        dec64 = self.dec64(dec64)

        up128 = self.up128(dec64)
        if self.has_polar_skips and polar_skips is not None and "s128" in polar_skips:
            dec128 = self.dec128_fuse(torch.cat([up128, polar_skips["s128"]], dim=1))
        else:
            dec128 = up128
        dec128 = self.dec128(dec128)

        up256 = self.up256(dec128)
        if self.has_polar_skips and polar_skips is not None and "s256" in polar_skips:
            dec256 = self.dec256_fuse(torch.cat([up256, polar_skips["s256"]], dim=1))
        else:
            dec256 = up256
        dec256 = self.dec256(dec256)
        return self.head(dec256)


def _load_dinov3_vits16(dino_weights_path: str | Path, dinov3_repo: Optional[str | Path] = None) -> nn.Module:
    """Load DINOv3-S/16 backbone from a local facebookresearch/dinov3 clone or torch.hub."""
    if dinov3_repo is not None:
        repo_path = str(Path(dinov3_repo).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    try:
        from dinov3.hub.backbones import dinov3_vits16  # type: ignore

        backbone = dinov3_vits16(pretrained=False)
    except ImportError:
        backbone = torch.hub.load("facebookresearch/dinov3", "dinov3_vits16", pretrained=False)

    state = torch.load(str(dino_weights_path), map_location="cpu", weights_only=True)
    incompat = backbone.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            f"DINOv3 state_dict mismatch: missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}"
        )
    return backbone


class PolarSkipDINO(nn.Module):
    """Complete PolarSkip-DINO model (Full: 24.10M total / 2.50M trainable; Lite: 23.92M / 2.31M)."""

    def __init__(
        self,
        dino_weights_path: str | Path,
        dinov3_repo: Optional[str | Path] = None,
        use_polar_skip: bool = True,
    ):
        super().__init__()
        self.has_polar = bool(use_polar_skip)
        self.dino = _load_dinov3_vits16(dino_weights_path, dinov3_repo)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        self.polar_stem = PolarSkipEncoder() if self.has_polar else None
        self.decoder = CartesianUNetDecoder(in_channels=384, has_polar_skips=self.has_polar)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()  # Backbone stays strictly frozen in eval mode
        return self

    def forward(self, raw_polar_img: torch.Tensor, geom: RadarGeometry, dino_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run single-frame forward pass.

        Args:
            raw_polar_img: (B, 1, 256, 64) raw polar radar heatmap normalized to [0, 1].
            geom: RadarGeometry residing on the same device.
            dino_input: optional precomputed (B, 3, 272, 528) tensor.
        Returns:
            logits: (B, 1, 256, 512) Cartesian BEV occupancy logits.
        """
        d_in = make_dino_input(raw_polar_img, geom) if dino_input is None else dino_input
        with torch.no_grad():
            feat_17x33 = self.dino.get_intermediate_layers(
                d_in,
                n=1,
                reshape=True,
                return_class_token=False,
                return_extra_tokens=False,
            )[0]
            b = feat_17x33.shape[0]
            feat_cart_64 = F.grid_sample(
                feat_17x33,
                geom.dino_cart_grid_64.expand(b, -1, -1, -1),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

        if self.has_polar and self.polar_stem is not None:
            b = raw_polar_img.shape[0]
            ch0 = torch.log1p(raw_polar_img * 255.0) / math.log1p(255.0)
            coords = geom.polar_geo_channels.unsqueeze(0).expand(b, -1, -1, -1)
            polar_5ch = torch.cat([ch0, raw_polar_img, coords], dim=1)
            skips = self.polar_stem(polar_5ch, geom)
            return self.decoder(feat_cart_64, polar_skips=skips)

        return self.decoder(feat_cart_64, polar_skips=None)

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return lightweight (~9.7 MB) trainable weights excluding the frozen DINOv3 backbone."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("dino.")}

    def load_decoder_checkpoint(self, ckpt_path: str | Path) -> dict:
        """Load PolarSkip + Cartesian U-Net weights with strict trainable-key verification."""
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
        filtered = {k: v for k, v in state.items() if not k.startswith("dino.")}
        expected = {k for k in self.state_dict().keys() if not k.startswith("dino.")}
        missing = sorted(expected - set(filtered.keys()))
        unexpected = sorted(set(filtered.keys()) - expected)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.load_state_dict(filtered, strict=False)
        return ck if isinstance(ck, dict) else {}
