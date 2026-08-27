"""Streaming loader for the public EchoNext waveform and metadata partitions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_echonext_endpoints(path=None):
    path = path or Path(__file__).resolve().parents[3] / "assets/echonext_endpoints.json"
    with Path(path).open(encoding="utf-8") as handle:
        endpoints = json.load(handle)
    required = {"flag", "name", "zero_shot_prompt"}
    if not endpoints or any(required - set(endpoint) for endpoint in endpoints):
        raise ValueError("EchoNext endpoint definitions are incomplete")
    return endpoints


def _sex_code(value):
    normalized = str(value).strip().casefold()
    if normalized.startswith("m") or normalized in {"1", "2"}:
        return 2.0
    if normalized.startswith("f") or normalized == "0":
        return 1.0
    return 0.0


def prepare_echonext_waveform(waveform, scale=0.3):
    """Convert one released EchoNext waveform to ExFMECG input format."""

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim != 2:
        raise ValueError(f"Expected a two-dimensional waveform, found {waveform.shape}")
    if waveform.shape[0] == 12:
        signal = waveform
    elif waveform.shape[1] == 12:
        signal = waveform.T
    else:
        raise ValueError(f"Could not identify 12 leads in waveform shape {waveform.shape}")
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = F.interpolate(
        torch.from_numpy(signal).unsqueeze(0),
        size=1024,
        mode="linear",
        align_corners=False,
    ).squeeze(0)
    return torch.clamp(signal * float(scale), -3.5, 3.5)


class EchoNextDataset(Dataset):
    """Pair one public EchoNext split with its memory-mapped waveform array."""

    def __init__(self, metadata, waveforms, split, endpoints=None, scale=0.3):
        self.metadata_path = Path(metadata)
        self.waveform_path = Path(waveforms)
        self.endpoints = endpoints or load_echonext_endpoints()
        metadata_frame = pd.read_csv(self.metadata_path)
        if "split" not in metadata_frame:
            raise ValueError("EchoNext metadata requires a split column")
        self.metadata = metadata_frame.loc[
            metadata_frame["split"].astype(str).str.casefold() == split.casefold()
        ].reset_index(drop=True)
        missing = {
            "age_at_ecg",
            "sex",
            *(endpoint["flag"] for endpoint in self.endpoints),
        } - set(self.metadata.columns)
        if missing:
            raise ValueError(f"EchoNext metadata is missing columns: {sorted(missing)}")
        self.waveforms = np.load(self.waveform_path, mmap_mode="r", allow_pickle=False)
        if len(self.waveforms) != len(self.metadata):
            raise ValueError(
                f"Metadata/waveform mismatch for {split}: "
                f"{len(self.metadata)} != {len(self.waveforms)}"
            )
        self.scale = float(scale)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        labels = np.asarray(
            [int(row[endpoint["flag"]]) for endpoint in self.endpoints],
            dtype=np.int64,
        )
        identifier = row.get("ecg_key", row.get("study_id", index))
        return {
            "ecg": prepare_echonext_waveform(self.waveforms[index], self.scale),
            "age": np.float32(row["age_at_ecg"] if pd.notna(row["age_at_ecg"]) else 0),
            "gender": np.float32(_sex_code(row["sex"])),
            "labels": torch.from_numpy(labels),
            "study_id": str(identifier),
        }
