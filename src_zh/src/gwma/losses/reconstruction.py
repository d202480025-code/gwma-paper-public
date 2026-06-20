from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from gwma.models.framing import overlap_add


def hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    spectrum = torch.fft.fft(fft_input, dim=-1)
    length = x.shape[-1]
    multiplier = torch.zeros(length, device=x.device, dtype=fft_input.dtype)
    multiplier[0] = 1.0
    if length % 2 == 0:
        multiplier[length // 2] = 1.0
        multiplier[1 : length // 2] = 2.0
    else:
        multiplier[1 : (length + 1) // 2] = 2.0
    analytic = torch.fft.ifft(spectrum * multiplier, dim=-1)
    return analytic.abs()


def token_mask_to_sample_weight(
    token_mask: torch.Tensor,
    frame_length: int,
    hop_length: int,
    output_length: int,
) -> torch.Tensor:
    frames = token_mask.to(torch.float32).unsqueeze(-1).expand(-1, -1, frame_length)
    return overlap_add(frames, output_length, hop_length)


def approximate_physics_weight(
    target: torch.Tensor,
    sample_rate: float,
    mass1: torch.Tensor | None = None,
    mass2: torch.Tensor | None = None,
    low_frequency_cutoff: float = 20.0,
    outside_weight: float = 1.0 / 6.0,
    final_spin: float = 0.686,
) -> torch.Tensor:
    """近似目标波形附近的 inspiral 与 ringdown 损失支撑区间。"""
    batch, _, length = target.shape
    merger = target.abs().argmax(dim=-1, keepdim=True)
    if mass1 is None or mass2 is None:
        envelope = hilbert_envelope(target)
        threshold = 0.01 * envelope.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        active = envelope >= threshold
        return torch.where(active, 1.0, outside_weight).to(target.dtype)

    solar_mass_seconds = 4.925490947e-6
    m1 = mass1.to(target.device, target.dtype)
    m2 = mass2.to(target.device, target.dtype)
    invalid_mass = (m1 <= 0) | (m2 <= 0)
    if bool(invalid_mass.any()):
        envelope_weight = approximate_physics_weight(
            target,
            sample_rate=sample_rate,
            mass1=None,
            mass2=None,
            low_frequency_cutoff=low_frequency_cutoff,
            outside_weight=outside_weight,
            final_spin=final_spin,
        )
        if bool(invalid_mass.all()):
            return envelope_weight
        m1 = torch.where(invalid_mass, torch.ones_like(m1), m1)
        m2 = torch.where(invalid_mass, torch.ones_like(m2), m2)
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    inspiral_seconds = (
        5.0
        / 256.0
        * (chirp_mass * solar_mass_seconds) ** (-5.0 / 3.0)
        * (math.pi * low_frequency_cutoff) ** (-8.0 / 3.0)
    )
    final_mass_seconds = 0.95 * (m1 + m2) * solar_mass_seconds
    q_factor = 2.0 * (1.0 - final_spin) ** (-0.45)
    ring_frequency = (1.0 - 0.63 * (1.0 - final_spin) ** 0.3) / (2.0 * math.pi * final_mass_seconds)
    damping_seconds = q_factor / (math.pi * ring_frequency)

    left = merger.squeeze(-1).squeeze(-1) - (inspiral_seconds * sample_rate).long()
    right = merger.squeeze(-1).squeeze(-1) + (10.0 * damping_seconds * sample_rate).long()
    positions = torch.arange(length, device=target.device).view(1, 1, length)
    active = (positions >= left.view(batch, 1, 1)) & (positions <= right.view(batch, 1, 1))
    physics_weight = torch.where(active, 1.0, outside_weight).to(target.dtype)
    if "envelope_weight" in locals():
        return torch.where(invalid_mass.view(batch, 1, 1), envelope_weight, physics_weight)
    return physics_weight


class CompositeReconstructionLoss(nn.Module):
    """MSE 加显式包络一致性和对数幅度谱一致性。"""

    def __init__(
        self,
        mse_weight: float = 1.0,
        envelope_weight: float = 0.0,
        spectral_weight: float = 0.0,
        spectral_epsilon: float = 1e-6,
        target_envelope_mse_alpha: float = 0.0,
        pure_noise_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.envelope_weight = envelope_weight
        self.spectral_weight = spectral_weight
        self.spectral_epsilon = spectral_epsilon
        self.target_envelope_mse_alpha = target_envelope_mse_alpha
        self.pure_noise_weight = pure_noise_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        squared_error = (prediction - target).square()
        weights = sample_weight
        if self.target_envelope_mse_alpha > 0.0:
            with torch.no_grad():
                target_envelope = hilbert_envelope(target)
                envelope_max = target_envelope.amax(dim=-1, keepdim=True).clamp_min(1e-8)
                envelope_weight = 1.0 + self.target_envelope_mse_alpha * (
                    target_envelope / envelope_max
                )
                target_energy = target.square().sum(dim=-1, keepdim=True)
                noise_weight = (
                    self.pure_noise_weight
                    if self.pure_noise_weight > 0.0
                    else 10.0 * (1.0 + self.target_envelope_mse_alpha)
                )
                envelope_weight = torch.where(
                    target_energy < 1e-8,
                    torch.full_like(envelope_weight, noise_weight),
                    envelope_weight,
                )
            weights = envelope_weight if weights is None else weights * envelope_weight

        if weights is None:
            mse = squared_error.mean()
        else:
            weights = weights.expand_as(squared_error).to(squared_error.dtype)
            mse = (squared_error * weights).sum() / weights.sum().clamp_min(1.0)

        envelope = (hilbert_envelope(prediction) - hilbert_envelope(target)).square().mean()
        predicted_spectrum = torch.fft.rfft(prediction.float(), dim=-1).abs()
        target_spectrum = torch.fft.rfft(target.float(), dim=-1).abs()
        spectral = (
            (
                torch.log(predicted_spectrum + self.spectral_epsilon)
                - torch.log(target_spectrum + self.spectral_epsilon)
            )
            .square()
            .mean()
        )

        total = (
            self.mse_weight * mse
            + self.envelope_weight * envelope
            + self.spectral_weight * spectral
        )
        return total, {
            "loss": total.detach(),
            "mse": mse.detach(),
            "envelope": envelope.detach(),
            "spectral": spectral.detach(),
        }


def build_loss(config: dict[str, Any]) -> CompositeReconstructionLoss:
    return CompositeReconstructionLoss(**config)
