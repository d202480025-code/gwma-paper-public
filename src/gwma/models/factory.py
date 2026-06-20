from __future__ import annotations

from typing import Any

from torch import nn

from gwma.models.bilstm import BiLSTMDenoiser
from gwma.models.gwma import GWMA
from gwma.models.unet1d import UNet1D


def build_model(config: dict[str, Any]) -> nn.Module:
    config = dict(config)
    name = config.pop("name").lower()
    if name == "gwma":
        return GWMA(**config)
    if name == "unet1d":
        return UNet1D(**config)
    if name == "bilstm":
        return BiLSTMDenoiser(**config)
    raise ValueError(f"Unknown model name: {name}")
