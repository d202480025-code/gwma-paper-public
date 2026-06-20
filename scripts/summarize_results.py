#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

from gwma.evaluation.plan import load_evaluation_jobs


MANUSCRIPT_MODELS = [
    ("gwma", "GWMA"),
    ("bilstm", "BiLSTM"),
    ("unet1d", "U-Net"),
]


def _dataset_family(dataset_key: str) -> str:
    if dataset_key.startswith("pure_glitch"):
        return "pure_glitch"
    if dataset_key.startswith("gaussian"):
        return "gaussian"
    if dataset_key.startswith("glitch"):
        return "glitch"
    return dataset_key


def _read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_row(
    model: str,
    dataset: str,
    metric: str,
    summary: dict,
) -> dict[str, object]:
    stats = summary.get(metric)
    if stats is None:
        return {
            "model": model,
            "dataset": dataset,
            "metric": metric,
            "count": 0,
        }
    return {
        "model": model,
        "dataset": dataset,
        "metric": metric,
        "count": stats["count"],
        "mean": stats["mean"],
        "std": stats["std"],
        "median": stats["median"],
        "min": stats["min"],
        "max": stats["max"],
        "q05": stats["q05"],
        "q25": stats["q25"],
        "q75": stats["q75"],
        "q95": stats["q95"],
        "mean_ci95_low": stats["mean_ci95"][0],
        "mean_ci95_high": stats["mean_ci95"][1],
    }


def _read_per_sample(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_overlap(
    rows: list[dict[str, str]],
    model: str,
    predicate: Callable[[dict[str, str]], bool],
) -> tuple[int, float | None]:
    values: list[float] = []
    for row in rows:
        if row["model_key"] != model or not predicate(row):
            continue
        overlap = float(row["overlap"])
        if overlap == overlap:
            values.append(overlap)
    if not values:
        return 0, None
    return len(values), sum(values) / len(values)


def _format_value(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def _write_condition_overlap_tables(
    jobs,
    output_dir: Path,
) -> None:
    rows: list[dict[str, str]] = []
    for job in jobs:
        per_sample_path = job.output_dir / "per_sample.csv"
        if not per_sample_path.exists():
            continue
        for row in _read_per_sample(per_sample_path):
            row["model_key"] = job.model_key
            row["dataset_key"] = job.dataset_key
            rows.append(row)

    conditions: list[tuple[str, str, Callable[[dict[str, str]], bool]]] = [
        (
            "Gaussian (Baseline)",
            "--",
            lambda row: _dataset_family(row["dataset_key"]) == "gaussian",
        ),
    ]
    for amplitude in (1.0, 2.0, 3.0):
        conditions.append(
            (
                "All Glitches Avg",
                f"{amplitude:.1f}x",
                lambda row, amp=amplitude: (
                    _dataset_family(row["dataset_key"]) == "glitch"
                    and row["glitch_amplitude_factor"]
                    and float(row["glitch_amplitude_factor"]) == amp
                ),
            )
        )
    glitch_labels = {
        "blip": "Blip",
        "sine_gaussian": "Sine Gaussian",
        "ringdown": "Ringdown",
        "gaussian_pulse": "Gaussian Pulse",
    }
    for glitch_kind, label in glitch_labels.items():
        conditions.append(
            (
                label,
                "2.0x",
                lambda row, kind=glitch_kind: (
                    _dataset_family(row["dataset_key"]) == "glitch"
                    and row["glitch_kind"] == kind
                    and row["glitch_amplitude_factor"]
                    and float(row["glitch_amplitude_factor"]) == 2.0
                ),
            )
        )

    condition_rows: list[dict[str, object]] = []
    for condition, amplitude, predicate in conditions:
        record: dict[str, object] = {
            "test_condition": condition,
            "amplitude": amplitude,
        }
        for model_key, label in MANUSCRIPT_MODELS:
            count, mean = _mean_overlap(rows, model_key, predicate)
            record[f"{model_key}_n"] = count
            record[label] = mean
        condition_rows.append(record)

    csv_fields = [
        "test_condition",
        "amplitude",
        "GWMA",
        "gwma_n",
        "BiLSTM",
        "bilstm_n",
        "U-Net",
        "unet1d_n",
    ]
    with (output_dir / "condition_overlap.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(condition_rows)

    with (output_dir / "condition_overlap.md").open("w", encoding="utf-8") as handle:
        handle.write("| Test Condition | Amplitude | GWMA | BiLSTM | U-Net |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in condition_rows:
            handle.write(
                f"| {row['test_condition']} | {row['amplitude']} | "
                f"{_format_value(row['GWMA'])} | "
                f"{_format_value(row['BiLSTM'])} | "
                f"{_format_value(row['U-Net'])} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/paper")
    parser.add_argument("--output-dir", default="results/paper/tables")
    parser.add_argument("--paper-config", default="configs/experiment/paper_main.yaml")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    jobs = load_evaluation_jobs(args.paper_config, results_root.parent)
    for job in jobs:
        summary_path = job.output_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _read_summary(summary_path)
        rows.append(_metric_row(job.model_key, job.dataset_key, "overlap", summary))
        rows.append(
            _metric_row(job.model_key, job.dataset_key, "output_input_energy_ratio", summary)
        )

    if not rows:
        raise FileNotFoundError(f"No configured summary.json files found below {results_root}")

    fieldnames = list(rows[0])
    with (output_dir / "main_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "main_metrics.md").open("w", encoding="utf-8") as handle:
        handle.write("| Model | Dataset | Metric | N | Mean | Std | Median | 95% CI |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            if int(row.get("count", 0)) == 0:
                continue
            handle.write(
                f"| {row['model']} | {row['dataset']} | {row['metric']} | "
                f"{row['count']} | {float(row['mean']):.6f} | "
                f"{float(row['std']):.6f} | {float(row['median']):.6f} | "
                f"[{float(row['mean_ci95_low']):.6f}, "
                f"{float(row['mean_ci95_high']):.6f}] |\n"
            )
    print(f"Wrote tables to {output_dir}")
    _write_condition_overlap_tables(jobs, output_dir)


if __name__ == "__main__":
    main()
