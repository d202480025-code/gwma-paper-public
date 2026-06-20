from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gwma.config import load_config


@dataclass(frozen=True)
class EvaluationJob:
    model_key: str
    dataset_key: str
    checkpoint: Path
    data: Path
    output_dir: Path


def load_evaluation_jobs(
    config_path: str | Path,
    results_root: str | Path = "results",
) -> list[EvaluationJob]:
    config_path = Path(config_path)
    config = load_config(config_path)
    base = config_path.parent
    results_root = Path(results_root)

    models = config["models"]
    datasets = config["datasets"]
    jobs: list[EvaluationJob] = []
    for model_key, experiment_config in models.items():
        experiment = load_config((base / experiment_config).resolve())
        checkpoint = Path(experiment["training"]["output_dir"]) / "best.pt"
        for dataset_key, data_config in datasets.items():
            dataset = load_config((base / data_config).resolve())
            split = "test"
            data_path = Path(dataset["output_root"]) / split / "data.h5"
            output_dir = results_root / "paper" / f"{model_key}_{dataset_key}"
            jobs.append(
                EvaluationJob(
                    model_key=str(model_key),
                    dataset_key=str(dataset_key),
                    checkpoint=checkpoint,
                    data=data_path,
                    output_dir=output_dir,
                )
            )
    return jobs
