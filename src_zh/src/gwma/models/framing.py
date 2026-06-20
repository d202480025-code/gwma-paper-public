from __future__ import annotations

import torch
import torch.nn.functional as F


def frame_signal(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """将 [B, 1, L] 波形转换为 [B, N, frame_length] 的重叠帧。"""
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected [B, 1, L], received {tuple(x.shape)}")
    if x.shape[-1] < frame_length:
        raise ValueError("Signal is shorter than one frame")
    remainder = (x.shape[-1] - frame_length) % hop_length
    if remainder:
        raise ValueError(
            "Signal length must satisfy (length - frame_length) % hop_length == 0; "
            f"received length={x.shape[-1]}, frame_length={frame_length}, "
            f"hop_length={hop_length}"
        )
    return x.unfold(-1, frame_length, hop_length).squeeze(1)


def overlap_add(
    frames: torch.Tensor,
    output_length: int,
    hop_length: int,
) -> torch.Tensor:
    """使用带精确重叠归一化的 overlap-add 重建 [B, 1, L]。"""
    if frames.ndim != 3:
        raise ValueError(f"Expected [B, N, W], received {tuple(frames.shape)}")

    batch_size, _, frame_length = frames.shape
    columns = frames.transpose(1, 2)
    output = F.fold(
        columns,
        output_size=(1, output_length),
        kernel_size=(1, frame_length),
        stride=(1, hop_length),
    )
    weights = F.fold(
        torch.ones_like(columns),
        output_size=(1, output_length),
        kernel_size=(1, frame_length),
        stride=(1, hop_length),
    )
    output = output / weights.clamp_min(torch.finfo(output.dtype).eps)
    return output.reshape(batch_size, 1, output_length)


def number_of_frames(signal_length: int, frame_length: int, hop_length: int) -> int:
    if signal_length < frame_length:
        raise ValueError("Signal is shorter than one frame")
    if (signal_length - frame_length) % hop_length:
        raise ValueError("Signal length is not exactly covered by the framing parameters")
    return 1 + (signal_length - frame_length) // hop_length
