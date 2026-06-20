from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

GlitchKind = Literal["blip", "sine_gaussian", "ringdown", "gaussian_pulse"]
GlitchRelation = Literal[
    "pre_merger",
    "partial_overlap",
    "full_overlap",
    "post_merger",
    "separated",
]


@dataclass(frozen=True)
class GlitchParameters:
    kind: GlitchKind
    center_time: float
    amplitude: float
    frequency: float = 100.0
    quality: float = 8.0
    duration: float = 0.05
    phase: float = 0.0
    chirp_rate: float = 0.0
    relation: GlitchRelation = "full_overlap"

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def generate_glitch(
    time: np.ndarray,
    parameters: GlitchParameters,
) -> np.ndarray:
    """生成一个显式参数化的解析瞬态干扰。"""
    tau = time - parameters.center_time
    amplitude = float(parameters.amplitude)

    if parameters.kind == "gaussian_pulse":
        sigma = max(parameters.duration / 6.0, np.finfo(float).eps)
        glitch = np.exp(-0.5 * (tau / sigma) ** 2)
    elif parameters.kind == "sine_gaussian":
        sigma = parameters.quality / (2.0 * np.pi * max(parameters.frequency, np.finfo(float).eps))
        phase = 2.0 * np.pi * parameters.frequency * tau + parameters.phase
        glitch = np.exp(-0.5 * (tau / sigma) ** 2) * np.sin(phase)
    elif parameters.kind == "ringdown":
        decay = max(parameters.duration, np.finfo(float).eps)
        phase = 2.0 * np.pi * parameters.frequency * tau + parameters.phase
        glitch = np.where(tau >= 0.0, np.exp(-tau / decay) * np.sin(phase), 0.0)
    elif parameters.kind == "blip":
        # 解析 blip 近似：使用高斯窗包络的线性啁啾。
        # 这里不声称复现完整的 Gravity Spy blip 分布。
        sigma = max(parameters.duration / 6.0, np.finfo(float).eps)
        phase = (
            2.0 * np.pi * parameters.frequency * tau
            + np.pi * parameters.chirp_rate * tau**2
            + parameters.phase
        )
        glitch = np.exp(-0.5 * (tau / sigma) ** 2) * np.cos(phase)
    else:
        raise ValueError(f"Unsupported glitch kind: {parameters.kind}")

    peak = np.max(np.abs(glitch))
    if peak > 0:
        glitch = glitch / peak
    return (amplitude * glitch).astype(np.float32)


def sample_glitch_parameters(
    rng: np.random.Generator,
    kind: GlitchKind,
    relation: GlitchRelation,
    merger_time: float,
    signal_duration: float,
    amplitude: float,
) -> GlitchParameters:
    center = sample_center_time(rng, relation, merger_time, signal_duration)
    if kind == "blip":
        return GlitchParameters(
            kind=kind,
            center_time=center,
            amplitude=amplitude,
            frequency=float(rng.uniform(30.0, 250.0)),
            duration=float(rng.uniform(0.02, 0.12)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
            chirp_rate=float(rng.uniform(-2000.0, 2000.0)),
            relation=relation,
        )
    if kind == "sine_gaussian":
        return GlitchParameters(
            kind=kind,
            center_time=center,
            amplitude=amplitude,
            frequency=float(rng.uniform(40.0, 500.0)),
            quality=float(rng.uniform(3.0, 30.0)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
            relation=relation,
        )
    if kind == "ringdown":
        return GlitchParameters(
            kind=kind,
            center_time=center,
            amplitude=amplitude,
            frequency=float(rng.uniform(50.0, 500.0)),
            duration=float(rng.uniform(0.01, 0.15)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
            relation=relation,
        )
    return GlitchParameters(
        kind=kind,
        center_time=center,
        amplitude=amplitude,
        duration=float(rng.uniform(0.005, 0.08)),
        relation=relation,
    )


def sample_center_time(
    rng: np.random.Generator,
    relation: GlitchRelation,
    merger_time: float,
    signal_duration: float,
) -> float:
    if relation == "full_overlap":
        return float(merger_time + rng.uniform(-0.01, 0.01))
    if relation == "partial_overlap":
        return float(merger_time + rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.12))
    if relation == "pre_merger":
        return float(merger_time - rng.uniform(0.15, 0.35))
    if relation == "post_merger":
        return float(merger_time + rng.uniform(0.15, 0.35))
    if relation == "separated":
        edge = rng.choice([0.1, 0.9]) * signal_duration
        return float(edge + rng.uniform(-0.03, 0.03))
    raise ValueError(f"Unsupported relation: {relation}")
