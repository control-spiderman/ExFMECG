"""Waveform loading and preprocessing shared by public data-entry scripts."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

_UNIT_TO_MV = {
    "mv": 1.0,
    "uv": 1e-3,
    "microvolt": 1e-3,
    "microvolts": 1e-3,
    "v": 1e3,
}


def load_waveform(path, array_key=None):
    """Load a 12-lead waveform from NumPy, MATLAB, or WFDB storage."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = array_key or (archive.files[0] if len(archive.files) == 1 else None)
            if key is None:
                raise ValueError(f"Specify --array-key for multi-array archive: {path}")
            return archive[key]
    if suffix == ".mat":
        content = loadmat(path)
        key = array_key or ("val" if "val" in content else None)
        if key is None or key not in content:
            raise ValueError(f"Specify --array-key for MATLAB file: {path}")
        return content[key]

    try:
        import wfdb
    except ImportError as error:
        raise ValueError(
            f"Unsupported waveform file {path}; install wfdb for WFDB records"
        ) from error
    record_path = path.with_suffix("") if suffix in {".hea", ".dat"} else path
    signal, _ = wfdb.rdsamp(str(record_path))
    return signal


def standardize_waveform(
    signal,
    sampling_rate,
    unit="mV",
    duration_seconds=10.0,
    target_length=1024,
):
    """Convert one recording to the `(12, target_length)` model input."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected a two-dimensional waveform, found {signal.shape}")
    if signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T
    if signal.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, found {signal.shape}")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    unit_key = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    if unit_key not in _UNIT_TO_MV:
        raise ValueError(f"Unsupported voltage unit: {unit}")
    signal = signal * _UNIT_TO_MV[unit_key]
    signal = np.nan_to_num(signal, nan=0.0, posinf=5.0, neginf=-5.0)

    source_length = round(float(sampling_rate) * float(duration_seconds))
    if signal.shape[1] > source_length:
        start = (signal.shape[1] - source_length) // 2
        signal = signal[:, start : start + source_length]
    elif signal.shape[1] < source_length:
        missing = source_length - signal.shape[1]
        left = missing // 2
        signal = np.pad(signal, ((0, 0), (left, missing - left)))

    if signal.shape[1] != target_length:
        signal = resample(signal, target_length, axis=1)
    return np.clip(signal, -5.0, 5.0).astype(np.float32)
