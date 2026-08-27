"""Prepare waveform arrays and an ExFMECG JSONL manifest."""

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from scripts.data.waveform import load_waveform, standardize_waveform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--waveform-root", type=Path, default=Path("."))
    parser.add_argument("--path-column", default="ecg_path")
    parser.add_argument("--id-column", default="study_id")
    parser.add_argument("--age-column", default="age")
    parser.add_argument("--sex-column", default="sex")
    parser.add_argument("--label-column", default="label_list")
    parser.add_argument("--evaluable-label-column", default="evaluable_labels")
    parser.add_argument("--report-column", default="report")
    parser.add_argument("--dataset-type-column", default="dataset_type")
    parser.add_argument("--sampling-rate-column", default="sampling_rate")
    parser.add_argument("--unit-column", default="unit")
    parser.add_argument("--default-dataset-type", type=int, default=0)
    parser.add_argument("--default-sampling-rate", type=float)
    parser.add_argument("--default-unit", default="mV")
    parser.add_argument("--array-key")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--target-length", type=int, default=1024)
    return parser.parse_args()


def iter_records(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError("JSON metadata must contain a list of records")
        yield from records
        return
    raise ValueError("Metadata must be CSV, JSON, or JSONL")


def parse_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in re.split(r"[#|;]", text) if item.strip()]


def safe_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    if not name:
        raise ValueError("Each record requires a non-empty identifier")
    return name


def value_or_default(record, column, default=None):
    value = record.get(column)
    return default if value is None or value == "" else value


def normalize_sex(value):
    if isinstance(value, bool):
        raise TypeError("Boolean sex values are ambiguous; use female, male or unknown")
    if isinstance(value, (int, float)) and value in (0, 1, 2):
        return {0: "unknown", 1: "female", 2: "male"}[int(value)]
    normalized = str(value).strip().lower()
    aliases = {
        "f": "female",
        "female": "female",
        "m": "male",
        "male": "male",
        "u": "unknown",
        "unknown": "unknown",
        "": "unknown",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported sex value: {value}")
    return aliases[normalized]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    waveform_dir = args.output_dir / "ecg"
    waveform_dir.mkdir(exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"

    count = 0
    identifiers = set()
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row_number, record in enumerate(iter_records(args.metadata), start=1):
            identifier = value_or_default(record, args.id_column, row_number)
            filename = safe_name(identifier)
            if filename in identifiers:
                raise ValueError(
                    f"Duplicate identifier after filename normalization: {identifier}"
                )
            identifiers.add(filename)

            relative_path = record.get(args.path_column)
            if not relative_path:
                raise ValueError(
                    f"Missing {args.path_column!r} in metadata row {row_number}"
                )
            waveform_path = Path(relative_path)
            if not waveform_path.is_absolute():
                waveform_path = args.waveform_root / waveform_path

            sampling_rate = value_or_default(
                record,
                args.sampling_rate_column,
                args.default_sampling_rate,
            )
            if sampling_rate is None:
                raise ValueError(
                    f"Missing sampling rate for {identifier}; provide a column or default"
                )
            unit = value_or_default(record, args.unit_column, args.default_unit)
            signal = standardize_waveform(
                load_waveform(waveform_path, args.array_key),
                sampling_rate=float(sampling_rate),
                unit=unit,
                duration_seconds=args.duration_seconds,
                target_length=args.target_length,
            )
            output_path = waveform_dir / f"{filename}.npy"
            np.save(output_path, signal, allow_pickle=False)

            labels = parse_list(record.get(args.label_column))
            evaluable = parse_list(record.get(args.evaluable_label_column))
            age = float(value_or_default(record, args.age_column, 0))
            if not math.isfinite(age):
                age = 0.0
            output = {
                "study_id": str(identifier),
                "ecg": str(output_path.relative_to(args.output_dir)),
                "age": age,
                "sex": normalize_sex(
                    value_or_default(record, args.sex_column, "unknown")
                ),
                "dataset_type": int(
                    value_or_default(
                        record,
                        args.dataset_type_column,
                        args.default_dataset_type,
                    )
                ),
                "label_list": labels,
                "report": str(value_or_default(record, args.report_column, "")),
            }
            if evaluable:
                output["evaluable_labels"] = evaluable
            if "patient_id" in record and record["patient_id"] not in (None, ""):
                output["patient_id"] = str(record["patient_id"])
            manifest.write(json.dumps(output, ensure_ascii=False) + "\n")
            count += 1

    print(f"Prepared {count} ECGs")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
