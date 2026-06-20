#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from gwma.models.factory import build_model
from gwma.metrics.overlap import psd_weighted_overlap
from gwma.metrics.statistics import summarize
from gwma.training.checkpoints import load_model_state
from gwma.utils import sha256_file


def grouped_summaries(
    rows: list[dict[str, object]],
    key: str,
) -> dict[str, dict[str, object]]:
    values: dict[str, list[float]] = {}
    for row in rows:
        group = str(row[key])
        if not group:
            continue
        overlap = float(row["overlap"])
        if np.isfinite(overlap):
            values.setdefault(group, []).append(overlap)
    return {
        group: summarize(np.asarray(group_values)) for group, group_values in sorted(values.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    checkpoint = load_model_state(args.checkpoint, device)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    checkpoint_hash = sha256_file(args.checkpoint)
    model_name = str(checkpoint["config"]["model"]["name"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with h5py.File(args.data, "r") as handle:
        sample_rate = float(handle.attrs["sample_rate"])
        for index in range(len(handle["noisy"])):
            noisy = np.asarray(handle["noisy"][index], dtype=np.float32)
            clean = np.asarray(handle["clean"][index], dtype=np.float32)
            tensor = torch.from_numpy(noisy).view(1, 1, -1).to(device)
            with torch.no_grad():
                prediction, _ = model(tensor, mask_ratio=0.0)
            reconstructed = prediction.cpu().numpy()[0, 0]
            clean_energy = float(np.sum(clean**2))
            input_energy = float(np.sum(noisy**2))
            output_energy = float(np.sum(reconstructed**2))
            overlap = (
                psd_weighted_overlap(
                    clean,
                    reconstructed,
                    psd=np.ones(len(clean) // 2 + 1),
                    sample_rate=sample_rate,
                )
                if clean_energy > 0
                else np.nan
            )
            metadata = (
                json.loads(handle["metadata_json"][index])
                if "metadata_json" in handle
                else {"sample_id": index}
            )
            rows.append(
                {
                    "sample_id": metadata.get("sample_id", index),
                    "sample_seed": metadata.get("sample_seed", ""),
                    "model_name": model_name,
                    "checkpoint_sha256": checkpoint_hash,
                    "optimal_snr": metadata.get("optimal_snr", ""),
                    "glitch_kind": (metadata.get("glitch") or {}).get("kind", ""),
                    "glitch_relation": (metadata.get("glitch") or {}).get("relation", ""),
                    "glitch_amplitude_factor": (metadata.get("glitch") or {}).get(
                        "amplitude_factor", ""
                    ),
                    "overlap_psd": "unit_whitened_psd",
                    "overlap": overlap,
                    "mse": float(np.mean((reconstructed - clean) ** 2)),
                    "input_energy": input_energy,
                    "output_energy": output_energy,
                    "output_input_energy_ratio": output_energy / max(input_energy, 1e-30),
                }
            )

    with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    overlaps = np.asarray([row["overlap"] for row in rows], dtype=float)
    energy_ratios = np.asarray(
        [row["output_input_energy_ratio"] for row in rows],
        dtype=float,
    )
    summary = {
        "overlap": summarize(overlaps[np.isfinite(overlaps)])
        if np.isfinite(overlaps).any()
        else None,
        "output_input_energy_ratio": summarize(energy_ratios),
        "overlap_by_glitch_kind": grouped_summaries(rows, "glitch_kind"),
        "overlap_by_glitch_relation": grouped_summaries(rows, "glitch_relation"),
        "overlap_by_glitch_amplitude": grouped_summaries(
            rows,
            "glitch_amplitude_factor",
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
