"""Manifest-backed dataset for preprocessed ExFMECG training inputs."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_SEX_CODES = {
    "unknown": 0,
    "u": 0,
    "female": 1,
    "f": 1,
    "male": 2,
    "m": 2,
}


class ExFMECGManifestDataset(Dataset):
    """Load preprocessed `(12, 1024)` ECG arrays described by JSON or JSONL."""

    def __init__(self, manifest, ecg_root="."):
        self.manifest = Path(manifest)
        self.ecg_root = Path(ecg_root)
        if self.manifest.suffix == ".jsonl":
            self.records = None
            self.offsets = self._index_jsonl()
        else:
            with self.manifest.open(encoding="utf-8") as handle:
                self.records = json.load(handle)
            self.offsets = None

    def _index_jsonl(self):
        offsets = []
        with self.manifest.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def __len__(self):
        return len(self.offsets) if self.offsets is not None else len(self.records)

    def _record(self, index):
        if self.records is not None:
            return self.records[index]
        with self.manifest.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    @staticmethod
    def _labels(record, key):
        value = record.get(key, "")
        if isinstance(value, str):
            return value
        return "#".join(str(item) for item in value)

    @staticmethod
    def _sex(value):
        if isinstance(value, str):
            return _SEX_CODES.get(value.lower(), 0)
        return int(value) if value in (0, 1, 2) else 0

    def __getitem__(self, index):
        record = self._record(index)
        relative_path = record.get("ecg", record.get("image"))
        if relative_path is None:
            raise KeyError("Each manifest record requires an 'ecg' path")
        path = Path(relative_path)
        if not path.is_absolute():
            path = self.ecg_root / path

        signal = np.load(path, allow_pickle=False)
        if signal.shape != (12, 1024):
            raise ValueError(f"Expected (12, 1024), found {signal.shape} at {path}")
        signal = np.nan_to_num(signal, nan=0.0, posinf=5.0, neginf=-5.0)
        labels = self._labels(record, "label_list") or self._labels(record, "labels")

        return {
            "ecg": torch.from_numpy(np.clip(signal, -5.0, 5.0).astype(np.float32)),
            "age": float(record.get("age", 0) or 0),
            "gender": self._sex(record.get("gender", record.get("sex", 0))),
            "dataset_type": int(record["dataset_type"]),
            "label_list": labels,
            "evaluable_labels": self._labels(record, "evaluable_labels"),
            "report": str(record.get("report") or labels.replace("#", "; ")),
            "study_id": str(record.get("study_id", record.get("id", index))),
        }
