from __future__ import annotations

import torch
from torch import nn


class BiLSTMDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        recurrent_dim = hidden_dim * (2 if bidirectional else 1)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.norm = nn.LayerNorm(recurrent_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(recurrent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, None]:
        del mask_ratio
        if x.ndim == 3:
            x = x.transpose(1, 2)
        elif x.ndim == 2:
            x = x.unsqueeze(-1)
        hidden, _ = self.lstm(self.input_projection(x))
        output = self.output_projection(self.norm(hidden)).transpose(1, 2)
        return output, None
