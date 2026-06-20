#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from gwma.config import load_config
from gwma.data.glitches import generate_glitch, sample_glitch_parameters
from gwma.data.waveforms import (
    aligo_design_psd,
    gaussian_noise_from_psd,
    generate_imrphenomd,
    scale_to_optimal_snr,
)


PARAMETER_NAMES = [
    "mass1",
    "mass2",
    "spin1z",
    "spin2z",
    "distance",
    "inclination",
    "coa_phase",
    "optimal_snr",
    "sample_seed",
]


def whiten(x: np.ndarray, psd: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    valid_psd = np.where(np.isfinite(psd) & (psd > 0.0), psd, np.inf)
    return np.fft.irfft(spectrum / np.sqrt(valid_psd), n=len(x)).astype(np.float32)


def sample_source(rng: np.random.Generator, config: dict) -> dict[str, float]:
    mass1 = float(rng.uniform(*config["mass1"]))
    mass2 = float(rng.uniform(config["mass2"][0], min(config["mass2"][1], mass1)))
    return {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": float(rng.uniform(*config["spin"])),
        "spin2z": float(rng.uniform(*config["spin"])),
        "distance": float(config.get("distance", 1000.0)),
        "inclination": float(np.arccos(rng.uniform(-1.0, 1.0))),
        "coa_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
    }


def generate_split(config: dict, split: str, count: int, seed: int) -> None:
    sample_rate = float(config["sample_rate"])
    duration = float(config["duration"])
    signal_length = int(round(sample_rate * duration))
    output = Path(config["output_root"]) / split / "data.h5"
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    psd = aligo_design_psd(signal_length, sample_rate)

    noisy_samples: list[np.ndarray] = []
    clean_samples: list[np.ndarray] = []
    noise_samples: list[np.ndarray] = []
    glitch_samples: list[np.ndarray] = []
    parameter_rows: list[list[float]] = []
    metadata_rows: list[str] = []

    for sample_index in range(count):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        sample_rng = np.random.default_rng(sample_seed)
        pure_glitch = config.get("mode") == "pure_glitch"

        source = sample_source(sample_rng, config["source"])
        target_snr = 0.0 if pure_glitch else float(sample_rng.uniform(*config["snr"]))
        if pure_glitch:
            clean_raw = np.zeros(signal_length, dtype=np.float32)
        else:
            clean_raw = generate_imrphenomd(
                source,
                sample_rate,
                duration,
                merger_fraction=float(config.get("merger_fraction", 0.5)),
            )
            clean_raw, target_snr = scale_to_optimal_snr(
                clean_raw,
                target_snr,
                psd,
                sample_rate,
            )

        include_noise = bool(config.get("include_gaussian_noise", True))
        noise_raw = (
            gaussian_noise_from_psd(signal_length, sample_rate, psd, sample_seed)
            if include_noise
            else np.zeros(signal_length, dtype=np.float32)
        )
        white_clean = whiten(clean_raw, psd)
        white_noise = whiten(noise_raw, psd)
        noise_scale = max(float(np.std(white_noise)), np.finfo(np.float32).eps)
        clean = white_clean / noise_scale
        noise = white_noise / noise_scale
        noisy = clean + noise
        glitch = np.zeros_like(clean, dtype=np.float32)

        metadata: dict[str, object] = {
            "sample_id": sample_index,
            "sample_seed": sample_seed,
            "source": source if not pure_glitch else None,
            "optimal_snr": target_snr,
            "glitch": None,
        }
        if config.get("mode") in {"glitch", "pure_glitch"}:
            kind = str(sample_rng.choice(config["glitch"]["kinds"]))
            relation = str(sample_rng.choice(config["glitch"]["relations"]))
            factor = float(sample_rng.choice(config["glitch"]["amplitudes"]))
            reference_peak = max(float(np.max(np.abs(clean))), 1.0)
            parameters = sample_glitch_parameters(
                sample_rng,
                kind=kind,
                relation=relation,
                merger_time=duration * float(config.get("merger_fraction", 0.5)),
                signal_duration=duration,
                amplitude=factor * reference_peak,
            )
            time = np.arange(signal_length) / sample_rate
            glitch = generate_glitch(time, parameters)
            noisy = noisy + glitch
            metadata["glitch"] = {**parameters.to_dict(), "amplitude_factor": factor}

        noisy_samples.append(noisy.astype(np.float32))
        clean_samples.append(clean.astype(np.float32))
        noise_samples.append(noise.astype(np.float32))
        glitch_samples.append(glitch.astype(np.float32))
        parameter_rows.append(
            [
                source["mass1"] if not pure_glitch else 0.0,
                source["mass2"] if not pure_glitch else 0.0,
                source["spin1z"] if not pure_glitch else 0.0,
                source["spin2z"] if not pure_glitch else 0.0,
                source["distance"] if not pure_glitch else 0.0,
                source["inclination"] if not pure_glitch else 0.0,
                source["coa_phase"] if not pure_glitch else 0.0,
                target_snr,
                float(sample_seed),
            ]
        )
        metadata_rows.append(json.dumps(metadata, sort_keys=True))

    with h5py.File(output, "w") as handle:
        handle.create_dataset("noisy", data=np.stack(noisy_samples), compression="gzip")
        handle.create_dataset("clean", data=np.stack(clean_samples), compression="gzip")
        handle.create_dataset("noise", data=np.stack(noise_samples), compression="gzip")
        handle.create_dataset("glitch", data=np.stack(glitch_samples), compression="gzip")
        handle.create_dataset("params", data=np.asarray(parameter_rows, dtype=np.float64))
        handle.create_dataset(
            "metadata_json",
            data=np.asarray(metadata_rows, dtype=h5py.string_dtype("utf-8")),
        )
        handle.attrs["sample_rate"] = sample_rate
        handle.attrs["duration"] = duration
        handle.attrs["representation"] = "whitened_and_noise_std_normalized"
        handle.attrs["param_names"] = json.dumps(PARAMETER_NAMES)
        handle.attrs["config_json"] = json.dumps(config, sort_keys=True)
    print(f"Wrote {count} samples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    base_seed = int(config.get("seed", 42))
    for offset, (split, count) in enumerate(config["splits"].items()):
        generate_split(config, split, int(count), base_seed + offset)


if __name__ == "__main__":
    main()
