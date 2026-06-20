from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from gwma.models.framing import frame_signal, number_of_frames, overlap_add
from gwma.models.layers import TransformerBlock


class GWMA(nn.Module):
    """Masked waveform autoencoder for one-dimensional strain reconstruction."""

    def __init__(
        self,
        signal_length: int = 4096,
        frame_length: int = 64,
        hop_length: int = 32,
        embed_dim: int = 768,
        depth: int = 24,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        mask_ratio: float = 0.75,
        local_conv_kernel: int = 3,
        residual_conv_kernel: int = 7,
        learned_position_embedding: bool = False,
        tied_decoder: bool = True,
        embedding_type: str = "frame_linear",
        decoder_type: str = "overlap_add",
        mask_strategy: str = "mask_token",
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")
        if embedding_type not in {"frame_linear", "conv"}:
            raise ValueError("embedding_type must be 'frame_linear' or 'conv'")
        if decoder_type not in {"overlap_add", "conv_transpose"}:
            raise ValueError("decoder_type must be 'overlap_add' or 'conv_transpose'")
        if mask_strategy not in {"mask_token", "zero"}:
            raise ValueError("mask_strategy must be 'mask_token' or 'zero'")

        self.signal_length = signal_length
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.num_frames = number_of_frames(signal_length, frame_length, hop_length)
        self.default_mask_ratio = mask_ratio
        self.tied_decoder = tied_decoder
        self.embedding_type = embedding_type
        self.decoder_type = decoder_type
        self.mask_strategy = mask_strategy
        self.use_checkpointing = use_checkpointing

        self.token_projection = (
            nn.Linear(frame_length, embed_dim, bias=False)
            if embedding_type == "frame_linear" or decoder_type == "overlap_add"
            else None
        )
        self.patch_embed = (
            nn.Conv1d(1, embed_dim, kernel_size=frame_length, stride=hop_length)
            if embedding_type == "conv"
            else None
        )

        # Treat frames as channels so convolution mixes local samples within
        # each waveform frame before token projection.
        self.local_conv = (
            nn.Conv1d(
                self.num_frames,
                self.num_frames,
                kernel_size=local_conv_kernel,
                padding=local_conv_kernel // 2,
                bias=False,
            )
            if embedding_type == "frame_linear"
            else None
        )
        self.local_projection = (
            nn.Linear(frame_length, embed_dim, bias=False)
            if embedding_type == "frame_linear"
            else None
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.position_embedding: nn.Parameter | None
        if learned_position_embedding:
            self.position_embedding = nn.Parameter(torch.zeros(1, self.num_frames, embed_dim))
        else:
            self.register_parameter("position_embedding", None)

        self.local_residual = nn.Conv2d(
            1,
            1,
            kernel_size=residual_conv_kernel,
            padding=residual_conv_kernel // 2,
            bias=False,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        if decoder_type == "overlap_add":
            self.decoder = None if tied_decoder else nn.Linear(embed_dim, frame_length, bias=False)
            self.conv_decoder = None
        else:
            self.decoder = None
            self.conv_decoder = nn.ConvTranspose1d(
                embed_dim,
                1,
                kernel_size=frame_length,
                stride=hop_length,
            )

        self.apply(self._initialize)
        nn.init.normal_(self.mask_token, std=0.02)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def random_mask(
        self,
        batch_size: int,
        ratio: float,
        device: torch.device,
    ) -> torch.Tensor:
        masked_count = int(round(self.num_frames * ratio))
        if masked_count == 0:
            return torch.zeros(batch_size, self.num_frames, dtype=torch.bool, device=device)
        noise = torch.rand(batch_size, self.num_frames, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(batch_size, self.num_frames, dtype=torch.bool, device=device)
        return mask.scatter(1, order[:, :masked_count], True)

    def encode(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.embedding_type == "conv":
            if self.patch_embed is None:
                raise RuntimeError("Conv embedding was not initialized")
            embeddings = self.patch_embed(x).transpose(1, 2)
            embeddings = embeddings.masked_fill(token_mask.unsqueeze(-1), 0.0)
        else:
            if (
                self.token_projection is None
                or self.local_conv is None
                or self.local_projection is None
            ):
                raise RuntimeError("Frame-linear embedding was not initialized")
            frames = frame_signal(x, self.frame_length, self.hop_length)
            visible_frames = frames.masked_fill(token_mask.unsqueeze(-1), 0.0)
            token_features = self.token_projection(visible_frames)
            local_features = self.local_projection(F.gelu(self.local_conv(visible_frames)))
            embeddings = token_features + local_features

        if self.mask_strategy == "mask_token":
            embeddings = torch.where(
                token_mask.unsqueeze(-1),
                self.mask_token.expand_as(embeddings),
                embeddings,
            )
        if self.position_embedding is not None:
            embeddings = embeddings + self.position_embedding

        local_residual = F.gelu(self.local_residual(embeddings.unsqueeze(1))).squeeze(1)
        hidden = self.embedding_dropout(embeddings + local_residual)
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.norm(hidden)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_norm(F.gelu(self.decoder_projection(hidden)))
        if self.decoder_type == "conv_transpose":
            if self.conv_decoder is None:
                raise RuntimeError("ConvTranspose decoder was not initialized")
            return self.conv_decoder(hidden.transpose(1, 2))[..., : self.signal_length]
        if self.token_projection is None:
            raise RuntimeError("Overlap-add decoder requires a token projection")
        if self.decoder is not None:
            return self.decoder(hidden)
        return F.linear(hidden, self.token_projection.weight.transpose(0, 1))

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.signal_length:
            raise ValueError(
                f"Model was configured for length {self.signal_length}, received {x.shape[-1]}"
            )
        ratio = self.default_mask_ratio if mask_ratio is None else mask_ratio
        if not 0.0 <= ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")

        token_mask = self.random_mask(x.shape[0], ratio, x.device)
        decoded = self.decode(self.encode(x, token_mask))
        if self.decoder_type == "conv_transpose":
            reconstruction = decoded
        else:
            reconstruction = overlap_add(
                decoded,
                output_length=self.signal_length,
                hop_length=self.hop_length,
            )
        return reconstruction, token_mask
