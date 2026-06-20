#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from gwma.config import load_config
from gwma.evaluation.plan import load_evaluation_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["data", "train", "evaluate", "tables"],
        required=True,
    )
    parser.add_argument("--paper-config", default="configs/experiment/paper_main.yaml")
    args = parser.parse_args()
    paper_config_path = Path(args.paper_config)
    paper_config = load_config(paper_config_path)
    paper_config_base = paper_config_path.parent

    if args.stage == "data":
        commands = []
        seen: set[str] = set()
        for dataset_key, data_config in paper_config["datasets"].items():
            resolved_path = (paper_config_base / data_config).resolve()
            resolved = str(resolved_path)
            if resolved in seen:
                continue
            seen.add(resolved)
            commands.append([sys.executable, "scripts/generate_data.py", "--config", resolved])
    elif args.stage == "train":
        sequence = paper_config.get("training_sequence")
        if sequence is None:
            sequence = list(dict.fromkeys(paper_config["models"].values()))
        commands = [
            [
                sys.executable,
                "scripts/train.py",
                "--config",
                str((paper_config_base / item).resolve()),
            ]
            for item in sequence
        ]
    elif args.stage == "evaluate":
        commands = []
        for job in load_evaluation_jobs(args.paper_config):
            commands.append(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--checkpoint",
                    str(job.checkpoint),
                    "--data",
                    str(job.data),
                    "--output-dir",
                    str(job.output_dir),
                ]
            )
    elif args.stage == "tables":
        tables_output_dir = (
            Path("results/paper/tables")
            if paper_config_path.stem == "paper_main"
            else Path("results/paper") / f"tables_{paper_config_path.stem}"
        )
        commands = [
            [
                sys.executable,
                "scripts/summarize_results.py",
                "--paper-config",
                args.paper_config,
                "--output-dir",
                str(tables_output_dir),
            ]
        ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
