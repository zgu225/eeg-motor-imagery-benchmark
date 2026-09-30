"""PyTorch reproduction of EEGNet from Lawhern et al. (2018).

Reference:
Lawhern V J et al. EEGNet: a compact convolutional neural network for
EEG-based brain-computer interfaces. Journal of Neural Engineering, 2018.

The reference implementation uses Keras/TensorFlow. This module reproduces the
recommended EEGNet-8,2 architecture in PyTorch.

Input shape:
    (batch, channels, samples)

Architecture:
    1. Temporal Conv2D learns frequency-selective filters.
    2. Depthwise spatial Conv2D learns D spatial filters per temporal filter.
    3. Separable Conv2D integrates channel-wise temporal features.
    4. A dense classifier returns one logit per class.
"""

from __future__ import annotations

import torch
from torch import nn


class MaxNormConstraint:
    """Apply a Keras-style max-norm constraint to a PyTorch parameter.

    The reference implementation constrains the depthwise convolution and dense
    layer. PyTorch has no equivalent layer constraint, so norms are projected
    after each optimizer update.
    """

    def __init__(self, max_value: float, dim: int | tuple[int, ...]) -> None:
        self.max_value = max_value
        self.dim = dim

    def __call__(self, weight: torch.Tensor) -> None:
        """Project weight norms in place so they do not exceed ``max_value``."""

        with torch.no_grad():
            norm = weight.norm(p=2, dim=self.dim, keepdim=True).clamp(min=1e-8)
            desired = torch.clamp(norm, max=self.max_value)
            weight.mul_(desired / norm)


class LawhernEEGNet(nn.Module):
    """Reproduce the updated EEGNet architecture described by Lawhern et al.

    Default parameters correspond to EEGNet-8,2: F1=8, D=2, and F2=16.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        dropout_rate: float = 0.5,
        kernel_length: int = 64,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int | None = None,
        norm_rate: float = 0.25,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.f1 = f1
        self.depth_multiplier = depth_multiplier
        self.f2 = f2 or f1 * depth_multiplier

        # Block 1: temporal filtering while preserving the sample dimension.
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=f1,
            kernel_size=(1, kernel_length),
            padding="same",
            bias=False,
        )
        self.batch_norm_1 = nn.BatchNorm2d(f1)

        # Block 1: each temporal filter learns D independent spatial filters.
        self.depthwise_spatial_conv = nn.Conv2d(
            in_channels=f1,
            out_channels=f1 * depth_multiplier,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.batch_norm_2 = nn.BatchNorm2d(f1 * depth_multiplier)
        self.activation = nn.ELU()
        self.pool_1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Block 2: depthwise temporal filtering followed by pointwise mixing.
        self.separable_depthwise_conv = nn.Conv2d(
            in_channels=f1 * depth_multiplier,
            out_channels=f1 * depth_multiplier,
            kernel_size=(1, 16),
            padding="same",
            groups=f1 * depth_multiplier,
            bias=False,
        )
        self.separable_pointwise_conv = nn.Conv2d(
            in_channels=f1 * depth_multiplier,
            out_channels=self.f2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.batch_norm_3 = nn.BatchNorm2d(self.f2)
        self.pool_2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout_2 = nn.Dropout(dropout_rate)

        # Infer the flattened dimension to support different epoch lengths.
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_samples)
            flattened_features = self._forward_features(dummy).flatten(start_dim=1).shape[1]

        self.classifier = nn.Linear(flattened_features, n_classes)
        self.train()

        # Constraints are applied explicitly after each optimizer update.
        self.depthwise_max_norm = MaxNormConstraint(max_value=1.0, dim=(1, 2, 3))
        self.classifier_max_norm = MaxNormConstraint(max_value=norm_rate, dim=1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract convolutional EEGNet features."""

        # Add the singleton image channel required by Conv2D.
        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.batch_norm_1(x)

        x = self.depthwise_spatial_conv(x)
        x = self.batch_norm_2(x)
        x = self.activation(x)
        x = self.pool_1(x)
        x = self.dropout_1(x)

        x = self.separable_depthwise_conv(x)
        x = self.separable_pointwise_conv(x)
        x = self.batch_norm_3(x)
        x = self.activation(x)
        x = self.pool_2(x)
        x = self.dropout_2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""

        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

    def apply_max_norm(self) -> None:
        """Apply the max-norm constraints from the reference implementation."""

        self.depthwise_max_norm(self.depthwise_spatial_conv.weight)
        self.classifier_max_norm(self.classifier.weight)


def build_lawhern_eegnet(
    n_channels: int,
    n_samples: int,
    n_classes: int,
    dropout_rate: float = 0.5,
    kernel_length: int = 64,
    f1: int = 8,
    depth_multiplier: int = 2,
    f2: int | None = None,
    norm_rate: float = 0.25,
) -> LawhernEEGNet:
    """Create an EEGNet-8,2 model with configurable architecture parameters."""

    return LawhernEEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        dropout_rate=dropout_rate,
        kernel_length=kernel_length,
        f1=f1,
        depth_multiplier=depth_multiplier,
        f2=f2,
        norm_rate=norm_rate,
    )
