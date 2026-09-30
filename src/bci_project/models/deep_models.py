"""Compact convolutional architectures for EEG decoding."""

from __future__ import annotations

import torch
from torch import nn

from bci_project.models.lawhern_eegnet import build_lawhern_eegnet


class EEGNet(nn.Module):
    """Compact EEGNet-inspired convolutional network used as an ablation."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Temporal convolution learns local time-frequency patterns.
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            # Depthwise spatial convolution learns channel combinations.
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            # Separable convolution integrates features with few parameters.
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Infer the classifier input size to support arbitrary epoch lengths.
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flattened = self.features(dummy).flatten(start_dim=1).shape[1]

        self.classifier = nn.Linear(flattened, n_classes)
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input shaped ``(batch, channels, samples)``."""

        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x.flatten(start_dim=1))


class ShallowConvNet(nn.Module):
    """ShallowConvNet-style model with an FBCSP-like power representation."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.temporal = nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False)
        self.spatial = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False)
        self.batch_norm = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)

        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flattened = self._forward_features(dummy).flatten(start_dim=1).shape[1]

        self.classifier = nn.Linear(flattened, n_classes)
        self.train()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal/spatial filtering followed by log-power extraction."""

        x = self.temporal(x)
        x = self.spatial(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of EEG epochs."""

        x = x.unsqueeze(1)
        x = self._forward_features(x)
        return self.classifier(x.flatten(start_dim=1))


def build_deep_model(name: str, n_channels: int, n_times: int, n_classes: int, dropout: float) -> nn.Module:
    """Instantiate a neural network from its experiment identifier."""

    if name == "lawhern_eegnet":
        return build_lawhern_eegnet(
            n_channels=n_channels,
            n_samples=n_times,
            n_classes=n_classes,
            dropout_rate=dropout,
        )
    if name == "eegnet":
        return EEGNet(n_channels, n_times, n_classes, dropout)
    if name == "shallow_convnet":
        return ShallowConvNet(n_channels, n_times, n_classes, dropout)
    raise ValueError(f"Unsupported deep model: {name}")
