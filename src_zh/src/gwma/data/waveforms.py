from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal.windows import tukey

from gwma.metrics.overlap import noise_weighted_inner_product


def generate_imrphenomd(
    parameters: dict[str, Any],
    sample_rate: float,
    duration: float,
    merger_fraction: float = 0.5,
    low_frequency_cutoff: float = 20.0,
) -> np.ndarray:
    """生成并裁剪一个 IMRPhenomD plus 极化波形。"""
    try:
        from pycbc.waveform import get_td_waveform
    except ImportError as error:
        raise RuntimeError(
            "PyCBC is required for waveform generation. Install the 'gw' extra."
        ) from error

    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=parameters["mass1"],
        mass2=parameters["mass2"],
        spin1z=parameters.get("spin1z", 0.0),
        spin2z=parameters.get("spin2z", 0.0),
        distance=parameters.get("distance", 1000.0),
        inclination=parameters.get("inclination", 0.0),
        coa_phase=parameters.get("coa_phase", 0.0),
        delta_t=1.0 / sample_rate,
        f_lower=low_frequency_cutoff,
    )
    raw = np.asarray(hp, dtype=np.float64)
    target_length = int(round(sample_rate * duration))
    merger_index = int(np.argmax(np.abs(raw)))
    desired_merger_index = int(round(target_length * merger_fraction))
    start = merger_index - desired_merger_index
    end = start + target_length
    pad_left = max(0, -start)
    pad_right = max(0, end - len(raw))
    padded = np.pad(raw, (pad_left, pad_right))
    start += pad_left
    cropped = padded[start : start + target_length]
    return (cropped * tukey(target_length, alpha=0.1)).astype(np.float32)


def scale_to_optimal_snr(
    signal: np.ndarray,
    target_snr: float,
    psd: np.ndarray,
    sample_rate: float,
    low_frequency_cutoff: float = 20.0,
) -> tuple[np.ndarray, float]:
    current_squared = noise_weighted_inner_product(
        signal,
        signal,
        psd=psd,
        sample_rate=sample_rate,
        low_frequency_cutoff=low_frequency_cutoff,
    )
    current_snr = float(np.sqrt(max(current_squared, 0.0)))
    if current_snr == 0.0:
        return np.zeros_like(signal), 0.0
    scaled = signal * (target_snr / current_snr)
    measured = target_snr
    return scaled.astype(np.float32), measured


def aligo_design_psd(
    signal_length: int,
    sample_rate: float,
    low_frequency_cutoff: float = 20.0,
) -> np.ndarray:
    try:
        from pycbc.psd import aLIGOZeroDetHighPower
    except ImportError as error:
        raise RuntimeError(
            "PyCBC is required for the Advanced LIGO design PSD. Install the 'gw' extra."
        ) from error
    delta_f = sample_rate / signal_length
    series = aLIGOZeroDetHighPower(
        signal_length // 2 + 1,
        delta_f,
        low_frequency_cutoff,
    )
    values = np.asarray(series, dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0)]
    floor = finite.min() if finite.size else 1.0
    return np.nan_to_num(values, nan=np.inf, posinf=np.inf, neginf=floor)


def gaussian_noise_from_psd(
    signal_length: int,
    sample_rate: float,
    psd: np.ndarray,
    seed: int,
) -> np.ndarray:
    try:
        from pycbc.noise import noise_from_psd
        from pycbc.types import FrequencySeries
    except ImportError as error:
        raise RuntimeError(
            "PyCBC is required for PSD-colored Gaussian noise. Install the 'gw' extra."
        ) from error
    delta_f = sample_rate / signal_length
    psd_series = FrequencySeries(np.asarray(psd), delta_f=delta_f)
    noise = noise_from_psd(signal_length, 1.0 / sample_rate, psd_series, seed=seed)
    return np.asarray(noise, dtype=np.float32)
