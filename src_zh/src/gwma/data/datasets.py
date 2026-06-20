from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5WaveformDataset(Dataset[dict[str, Any]]):
    """惰性 HDF5 加载器，每个 worker 进程持有一个文件句柄。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            self.length = len(handle["clean"])
            if len(handle["noisy"]) != self.length:
                raise ValueError("The noisy and clean datasets have different lengths")
            self.param_names = _read_param_names(handle)
            self.sample_rate = float(handle.attrs.get("sample_rate", 4096.0))

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        handle = self._handle()
        noisy = torch.from_numpy(np.asarray(handle["noisy"][index], dtype=np.float32))
        clean = torch.from_numpy(np.asarray(handle["clean"][index], dtype=np.float32))
        if noisy.ndim == 1:
            noisy = noisy.unsqueeze(0)
        if clean.ndim == 1:
            clean = clean.unsqueeze(0)

        sample: dict[str, Any] = {
            "sample_id": index,
            "noisy": noisy,
            "clean": clean,
            "sample_rate": self.sample_rate,
        }
        if "params" in handle:
            values = np.asarray(handle["params"][index])
            sample["params"] = torch.from_numpy(values.astype(np.float32))
        if "psd" in handle:
            sample["psd"] = torch.from_numpy(np.asarray(handle["psd"][index], dtype=np.float32))
        return sample

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()


def _read_param_names(handle: h5py.File) -> list[str]:
    value = handle.attrs.get("param_names", handle.attrs.get("param_keys", []))
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return [str(item) for item in decoded]
        except json.JSONDecodeError:
            return [value]
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value]
