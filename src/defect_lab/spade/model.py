from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, norm_channels: int, seg_channels: int = 1, hidden_channels: int = 64) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(seg_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        seg = F.interpolate(seg, size=x.shape[2:], mode="nearest")
        actv = self.shared(seg)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        normalized = self.norm(x)
        return normalized * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, seg_channels: int = 1) -> None:
        super().__init__()
        self.learned_shortcut = in_channels != out_channels
        middle_channels = min(in_channels, out_channels)

        self.spade_1 = SPADE(in_channels, seg_channels=seg_channels)
        self.conv_1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.spade_2 = SPADE(middle_channels, seg_channels=seg_channels)
        self.conv_2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.spade_shortcut = SPADE(in_channels, seg_channels=seg_channels)
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_shortcut(F.leaky_relu(self.spade_shortcut(x, seg), 0.2))
        return x

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x, seg)
        x = self.conv_1(F.leaky_relu(self.spade_1(x, seg), 0.2))
        x = self.conv_2(F.leaky_relu(self.spade_2(x, seg), 0.2))
        return x + residual


class SpadeGenerator(nn.Module):
    def __init__(self, seg_channels: int = 1, base_channels: int = 64, latent_size: int = 4) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.base_channels = base_channels
        self.fc = nn.Conv2d(seg_channels, base_channels * 8, kernel_size=3, padding=1)
        self.head = SPADEResBlock(base_channels * 8, base_channels * 8, seg_channels=seg_channels)
        self.up1 = SPADEResBlock(base_channels * 8, base_channels * 4, seg_channels=seg_channels)
        self.up2 = SPADEResBlock(base_channels * 4, base_channels * 2, seg_channels=seg_channels)
        self.up3 = SPADEResBlock(base_channels * 2, base_channels, seg_channels=seg_channels)
        self.up4 = SPADEResBlock(base_channels, base_channels // 2, seg_channels=seg_channels)
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, seg: torch.Tensor) -> torch.Tensor:
        seg_small = F.interpolate(seg, size=(self.latent_size, self.latent_size), mode="nearest")
        x = self.fc(seg_small)
        x = self.head(x, seg)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x, seg)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x, seg)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x, seg)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x, seg)
        if x.shape[-1] != seg.shape[-1] or x.shape[-2] != seg.shape[-2]:
            x = F.interpolate(x, size=seg.shape[2:], mode="bilinear", align_corners=False)
        return self.to_rgb(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers: list[nn.Module] = []
        current = in_channels
        for idx, out in enumerate(channels):
            stride = 1 if idx == len(channels) - 1 else 2
            layers.append(nn.Conv2d(current, out, kernel_size=4, stride=stride, padding=1))
            if idx > 0:
                layers.append(nn.BatchNorm2d(out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current = out
        layers.append(nn.Conv2d(current, 1, kernel_size=4, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        seg = F.interpolate(seg, size=image.shape[2:], mode="nearest")
        return self.model(torch.cat([image, seg], dim=1))
