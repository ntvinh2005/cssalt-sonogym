"""Small image-only segmentation model for simulator weak interface labels."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class SmallUNet(nn.Module):
    """A bounded two-level U-Net consuming only one clean ultrasound channel."""

    def __init__(self, base_channels: int = 16, class_count: int = 4) -> None:
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be a positive multiple of four")
        self.encoder_0 = ConvBlock(1, base_channels)
        self.encoder_1 = ConvBlock(base_channels, 2 * base_channels)
        self.bottleneck = ConvBlock(2 * base_channels, 4 * base_channels)
        self.decoder_1 = ConvBlock(6 * base_channels, 2 * base_channels)
        self.decoder_0 = ConvBlock(3 * base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, class_count, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("model input must be B x 1 x H x W clean ultrasound")
        height, width = image.shape[-2:]
        pad_height = (-height) % 4
        pad_width = (-width) % 4
        value = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
        first = self.encoder_0(value)
        second = self.encoder_1(F.max_pool2d(first, 2))
        center = self.bottleneck(F.max_pool2d(second, 2))
        up_second = F.interpolate(
            center, size=second.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_second = self.decoder_1(torch.cat((up_second, second), dim=1))
        up_first = F.interpolate(
            decoded_second, size=first.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_first = self.decoder_0(torch.cat((up_first, first), dim=1))
        return self.output(decoded_first)[..., :height, :width]


def normalized_ultrasound(image: torch.Tensor) -> torch.Tensor:
    """Normalize each image by robust actor-visible intensity quantiles."""

    if image.ndim != 3:
        raise ValueError("image batch must be B x H x W")
    flattened = image.flatten(1)
    low = torch.quantile(flattened, 0.05, dim=1)[:, None, None]
    high = torch.quantile(flattened, 0.95, dim=1)[:, None, None]
    return torch.clamp((image - low) / torch.clamp(high - low, min=1.0e-6), 0.0, 1.0)


def interface_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Weighted cross entropy plus foreground Dice on non-ignored pixels."""

    cross_entropy = F.cross_entropy(
        logits, target, weight=class_weights, ignore_index=ignore_index
    )
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    valid_channels = valid[:, None]
    probabilities = torch.softmax(logits, dim=1) * valid_channels
    truth = one_hot.to(probabilities.dtype) * valid_channels
    numerator = 2.0 * torch.sum(probabilities[:, 1:] * truth[:, 1:], dim=(0, 2, 3))
    denominator = torch.sum(
        probabilities[:, 1:] + truth[:, 1:], dim=(0, 2, 3)
    )
    dice = (numerator + 1.0) / (denominator + 1.0)
    return cross_entropy + torch.mean(1.0 - dice)


def negative_presence_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_ids: tuple[int, ...] = (1, 2),
    top_k_pixels: int = 20,
) -> torch.Tensor:
    """Penalize high class probabilities in frames with no visible class interface."""

    if logits.ndim != 4 or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("logits and target shapes are incompatible")
    if top_k_pixels <= 0:
        raise ValueError("top_k_pixels must be positive")
    probabilities = torch.softmax(logits, dim=1)
    losses = []
    for class_id in class_ids:
        if class_id <= 0 or class_id >= logits.shape[1]:
            raise ValueError(f"invalid foreground class: {class_id}")
        negative_frames = ~torch.any(target == class_id, dim=(1, 2))
        if not torch.any(negative_frames):
            continue
        values = probabilities[negative_frames, class_id].flatten(1)
        count = min(top_k_pixels, values.shape[1])
        strongest = torch.topk(values, k=count, dim=1).values
        losses.append(
            -torch.log1p(-torch.clamp(strongest, max=1.0 - 1.0e-6)).mean()
        )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
