"""Deterministic random-seed configuration."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN kernels trade some speed for repeatability.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
