from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from gwma.data.datasets import H5WaveformDataset
from gwma.losses.reconstruction import (
    approximate_physics_weight,
    build_loss,
    token_mask_to_sample_weight,
)
from gwma.models.factory import build_model
from gwma.training.checkpoints import load_model_state, save_checkpoint
from gwma.utils import seed_everything, sha256_file


def _device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _loader(
    path: str,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = H5WaveformDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _sample_weight(
    mode: str,
    batch: dict[str, Any],
    token_mask: torch.Tensor | None,
    model: torch.nn.Module,
    target: torch.Tensor,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "masked":
        if token_mask is None:
            raise ValueError("Masked loss weighting requires a model token mask")
        return token_mask_to_sample_weight(
            token_mask,
            frame_length=model.frame_length,
            hop_length=model.hop_length,
            output_length=target.shape[-1],
        )
    if mode == "physics":
        params = batch.get("params")
        mass1 = params[:, 0] if params is not None else None
        mass2 = params[:, 1] if params is not None else None
        return approximate_physics_weight(
            target,
            sample_rate=float(batch["sample_rate"][0]),
            mass1=mass1,
            mass2=mass2,
        )
    raise ValueError(f"Unknown sample weighting mode: {mode}")


def _epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    stage: str,
    mask_ratio: float,
    weighting: str,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    gradient_accumulation_steps: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    optimizer_steps = 0
    context = torch.enable_grad if training else torch.no_grad
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and device.type == "cuda",
        init_scale=256.0,
    )
    if training:
        optimizer.zero_grad(set_to_none=True)

    with context():
        for batch_index, batch in enumerate(
            tqdm(loader, leave=False, desc="train" if training else "valid"),
            start=1,
        ):
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            model_input = clean if stage == "pretrain" else noisy
            effective_mask_ratio = mask_ratio if stage == "pretrain" else 0.0

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                prediction, token_mask = model(model_input, mask_ratio=effective_mask_ratio)
                weights = _sample_weight(weighting, batch, token_mask, model, clean)
                if weights is not None:
                    weights = weights.to(device)
                loss, components = criterion(prediction, clean, weights)
                scaled_loss = loss / max(gradient_accumulation_steps, 1)

            if training:
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                    if batch_index % gradient_accumulation_steps == 0 or batch_index == len(loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= scale_before:
                            optimizer_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                else:
                    scaled_loss.backward()
                    if batch_index % gradient_accumulation_steps == 0 or batch_index == len(loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer_steps += 1
                        optimizer.zero_grad(set_to_none=True)

            batches += 1
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value)

    metrics = {key: value / max(batches, 1) for key, value in totals.items()}
    if training:
        metrics["optimizer_steps"] = float(optimizer_steps)
    return metrics


def run_training(config: dict[str, Any]) -> Path:
    seed_everything(int(config.get("seed", 42)))
    training = config["training"]
    device = _device_from_config(training.get("device", "auto"))
    output_dir = Path(training["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    model = build_model(config["model"]).to(device)
    if training.get("initial_checkpoint"):
        checkpoint = load_model_state(training["initial_checkpoint"], device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    criterion = build_loss(config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.05)),
    )
    epochs = int(training["epochs"])
    gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = _loader(
        config["data"]["train"],
        int(training["batch_size"]),
        int(training.get("workers", 0)),
        True,
    )
    valid_loader = _loader(
        config["data"]["valid"],
        int(training["batch_size"]),
        int(training.get("workers", 0)),
        False,
    )

    log_path = output_dir / "metrics.csv"
    best_loss = float("inf")
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_mse",
        "train_envelope",
        "train_spectral",
        "train_optimizer_steps",
        "valid_loss",
        "valid_mse",
        "valid_envelope",
        "valid_spectral",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_metrics = _epoch(
                model,
                train_loader,
                criterion,
                device,
                config["stage"],
                float(training.get("mask_ratio", 0.75)),
                training.get("sample_weighting", "none"),
                optimizer,
                bool(training.get("amp", True)),
                gradient_accumulation_steps,
            )
            valid_metrics = _epoch(
                model,
                valid_loader,
                criterion,
                device,
                config["stage"],
                float(training.get("mask_ratio", 0.75)),
                training.get("sample_weighting", "none"),
                None,
                bool(training.get("amp", True)),
                1,
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"valid_{key}": value for key, value in valid_metrics.items()},
                }
            )
            handle.flush()
            if train_metrics.get("optimizer_steps", 0.0) > 0.0:
                scheduler.step()
            if valid_metrics["loss"] < best_loss:
                best_loss = valid_metrics["loss"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    best_loss,
                )

    checksum = sha256_file(output_dir / "best.pt")
    (output_dir / "checksums.sha256").write_text(
        f"{checksum}  best.pt\n",
        encoding="utf-8",
    )
    return output_dir
