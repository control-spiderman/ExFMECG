"""Convert the public Georgia E07303 record to ExFMECG model input."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample


def main():
    example_dir = Path(__file__).resolve().parent
    raw_adc = loadmat(example_dir / "E07303.mat")["val"]
    signal_mv = np.asarray(raw_adc, dtype=np.float32) / 1000.0
    signal = resample(signal_mv, 1024, axis=1)
    signal = np.clip(signal, -5.0, 5.0).astype(np.float32)
    output = example_dir / "E07303.npy"
    np.save(output, signal, allow_pickle=False)
    print(f"Wrote {output} with shape {signal.shape}")


if __name__ == "__main__":
    main()
