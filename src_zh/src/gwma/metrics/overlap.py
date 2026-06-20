from __future__ import annotations

import numpy as np


def _frequency_data(
    x: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    delta_t = 1.0 / sample_rate
    spectrum = np.fft.rfft(x) * delta_t
    frequencies = np.fft.rfftfreq(len(x), d=delta_t)
    return spectrum, frequencies, sample_rate / len(x)


def noise_weighted_inner_product(
    a: np.ndarray,
    b: np.ndarray,
    psd: np.ndarray,
    sample_rate: float,
    low_frequency_cutoff: float = 20.0,
    high_frequency_cutoff: float | None = None,
) -> float:
    spectrum_a, frequencies, delta_f = _frequency_data(a, sample_rate)
    spectrum_b, _, _ = _frequency_data(b, sample_rate)
    psd = np.asarray(psd, dtype=np.float64)
    if psd.shape != spectrum_a.shape:
        raise ValueError(f"PSD shape {psd.shape} does not match spectrum {spectrum_a.shape}")
    high = sample_rate / 2.0 if high_frequency_cutoff is None else high_frequency_cutoff
    valid = (
        (frequencies >= low_frequency_cutoff)
        & (frequencies <= high)
        & np.isfinite(psd)
        & (psd > 0.0)
    )
    value = 4.0 * delta_f * np.sum(spectrum_a[valid].conj() * spectrum_b[valid] / psd[valid])
    return float(np.real(value))


def psd_weighted_overlap(
    reference: np.ndarray,
    prediction: np.ndarray,
    psd: np.ndarray,
    sample_rate: float,
    low_frequency_cutoff: float = 20.0,
    high_frequency_cutoff: float | None = None,
    maximize_time_phase: bool = True,
) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")

    norm_reference = noise_weighted_inner_product(
        reference,
        reference,
        psd,
        sample_rate,
        low_frequency_cutoff,
        high_frequency_cutoff,
    )
    norm_prediction = noise_weighted_inner_product(
        prediction,
        prediction,
        psd,
        sample_rate,
        low_frequency_cutoff,
        high_frequency_cutoff,
    )
    denominator = np.sqrt(max(norm_reference * norm_prediction, 0.0))
    if denominator == 0.0:
        return 0.0
    if not maximize_time_phase:
        numerator = noise_weighted_inner_product(
            reference,
            prediction,
            psd,
            sample_rate,
            low_frequency_cutoff,
            high_frequency_cutoff,
        )
        return float(np.clip(numerator / denominator, -1.0, 1.0))

    spectrum_a, frequencies, delta_f = _frequency_data(reference, sample_rate)
    spectrum_b, _, _ = _frequency_data(prediction, sample_rate)
    high = sample_rate / 2.0 if high_frequency_cutoff is None else high_frequency_cutoff
    valid = (
        (frequencies >= low_frequency_cutoff)
        & (frequencies <= high)
        & np.isfinite(psd)
        & (psd > 0.0)
    )
    cross = np.zeros(len(reference), dtype=np.complex128)
    cross[: len(spectrum_a)][valid] = spectrum_a[valid].conj() * spectrum_b[valid] / psd[valid]
    correlation = 4.0 * delta_f * np.fft.ifft(cross) * len(reference)
    return float(np.clip(np.max(np.abs(correlation)) / denominator, 0.0, 1.0))
