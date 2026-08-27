"""Memory-mapped concept targets used during ExFMECG training."""

import json
from pathlib import Path

import numpy as np
import torch


_ARRAYS = {
    "binary_targets": ("concept_binary", np.int64),
    "continuous_targets": ("concept_continuous", np.float32),
    "categorical_targets": ("concept_categorical", np.int64),
    "binary_mask": ("concept_binary_mask", np.float32),
    "regression_mask": ("concept_regression_mask", np.float32),
    "categorical_mask": ("concept_categorical_mask", np.float32),
}


class ConceptMatrixLoader:
    """Retrieve one set of concept targets without loading a matrix into memory."""

    def __init__(self, matrix_dir, n_concepts=570):
        self.matrix_dir = Path(matrix_dir) if matrix_dir else None
        self.n_concepts = n_concepts
        self.arrays = {}
        self.row_index = {}
        if self.matrix_dir and self.matrix_dir.is_dir():
            self.arrays = {
                name: np.load(self.matrix_dir / f"{name}.npy", mmap_mode="r")
                for name in _ARRAYS
            }
            with (self.matrix_dir / "ecg_id_to_row.json").open(encoding="utf-8") as handle:
                self.row_index = json.load(handle)
            widths = {array.shape[1] for array in self.arrays.values()}
            if len(widths) != 1 or next(iter(widths)) != n_concepts:
                raise ValueError(f"Invalid concept matrix width in {self.matrix_dir}")

    def _fallback(self):
        return {
            "concept_binary": torch.zeros(self.n_concepts, dtype=torch.long),
            "concept_continuous": torch.full((self.n_concepts,), float("nan")),
            "concept_categorical": torch.full((self.n_concepts,), -1, dtype=torch.long),
            "concept_binary_mask": torch.zeros(self.n_concepts),
            "concept_regression_mask": torch.zeros(self.n_concepts),
            "concept_categorical_mask": torch.zeros(self.n_concepts),
        }

    def fetch(self, ecg_id):
        row = self.row_index.get(str(ecg_id))
        if row is None:
            return self._fallback()
        output = {}
        for name, (output_name, dtype) in _ARRAYS.items():
            output[output_name] = torch.from_numpy(
                np.asarray(self.arrays[name][row], dtype=dtype).copy()
            )
        return output
