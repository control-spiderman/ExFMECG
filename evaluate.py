"""Evaluate ExFMECG queries on a manifest of preprocessed ECGs."""

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.datasets.datasets.exfmecg_dataset import ExFMECGManifestDataset
from scripts.evaluation.metrics import (
    bootstrap_intervals,
    discrimination_metrics,
    operating_metrics,
    select_mcc_threshold,
    select_sensitivity_threshold,
)
from scripts.inference import encode_queries, load_model, predict_queries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ecg-root", type=Path, default=Path("."))
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold-manifest", type=Path)
    parser.add_argument("--threshold-ecg-root", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sensitivity-targets",
        default="0.6,0.7,0.8,0.9",
        help="Comma-separated targets estimated on the threshold-selection set",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_queries(path):
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)
    if not isinstance(content, list) or not content:
        raise ValueError("Query file must contain a non-empty JSON list")
    output = []
    for item in content:
        if isinstance(item, str):
            output.append({"name": item, "prompt": item})
        else:
            name = item.get("name")
            prompt = item.get("prompt", name)
            if not name or not prompt:
                raise ValueError("Each query requires a name and prompt")
            output.append({"name": str(name), "prompt": str(prompt)})
    names = [item["name"].casefold() for item in output]
    if len(names) != len(set(names)):
        raise ValueError("Query names must be unique")
    return output


def split_labels(value):
    return {item.strip().casefold() for item in str(value).split("#") if item.strip()}


def run_inference(model, query_features, dataset, args, cache_dir, prefix):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    query_names = [item["name"].casefold() for item in args.query_spec]
    shape = (len(dataset), len(query_names))
    scores = np.lib.format.open_memmap(
        cache_dir / f"{prefix}_scores.npy", mode="w+", dtype=np.float32, shape=shape
    )
    labels = np.lib.format.open_memmap(
        cache_dir / f"{prefix}_labels.npy", mode="w+", dtype=np.uint8, shape=shape
    )
    masks = np.lib.format.open_memmap(
        cache_dir / f"{prefix}_masks.npy", mode="w+", dtype=bool, shape=shape
    )
    identifier_path = cache_dir / f"{prefix}_identifiers.txt"
    offset = 0

    with identifier_path.open("w", encoding="utf-8") as identifier_file:
        for batch in loader:
            samples = {
                "ecg": batch["ecg"].to(args.device, non_blocking=True),
                "age": batch["age"].to(args.device, non_blocking=True),
                "gender": batch["gender"].to(args.device, non_blocking=True),
            }
            probabilities = (
                predict_queries(
                    model,
                    samples,
                    query_features,
                    query_batch_size=args.query_batch_size,
                )
                .cpu()
                .numpy()
            )
            stop = offset + probabilities.shape[0]
            scores[offset:stop] = probabilities
            for local_index, (positive_text, evaluable_text, identifier) in enumerate(
                zip(
                    batch["label_list"],
                    batch["evaluable_labels"],
                    batch["study_id"],
                )
            ):
                positive = split_labels(positive_text)
                evaluable = split_labels(evaluable_text)
                labels[offset + local_index] = [
                    int(name in positive) for name in query_names
                ]
                masks[offset + local_index] = [
                    not evaluable or name in evaluable for name in query_names
                ]
                identifier_file.write(str(identifier).replace("\n", " ") + "\n")
            offset = stop
    scores.flush()
    labels.flush()
    masks.flush()
    return scores, labels, masks, identifier_path


def load_thresholds(path):
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    return {
        str(key).casefold(): float(value)
        for key, value in values.items()
        if value is not None
    }


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    args.query_spec = load_queries(args.queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device)
    query_features = encode_queries(model, [item["prompt"] for item in args.query_spec])

    dataset = ExFMECGManifestDataset(args.manifest, args.ecg_root)
    if len(dataset) == 0:
        raise ValueError("Evaluation manifest is empty")
    with tempfile.TemporaryDirectory(
        prefix="exfmecg_eval_", dir=args.output_dir
    ) as temporary:
        cache_dir = Path(temporary)
        scores, labels, masks, identifier_path = run_inference(
            model, query_features, dataset, args, cache_dir, "evaluation"
        )

        thresholds = load_thresholds(args.thresholds)
        sensitivity_thresholds = {}
        if args.threshold_manifest:
            threshold_root = args.threshold_ecg_root or args.ecg_root
            threshold_dataset = ExFMECGManifestDataset(
                args.threshold_manifest, threshold_root
            )
            threshold_scores, threshold_labels, threshold_masks, _ = run_inference(
                model,
                query_features,
                threshold_dataset,
                args,
                cache_dir,
                "threshold",
            )
            sensitivity_targets = [
                float(value) for value in args.sensitivity_targets.split(",") if value
            ]
            for index, query in enumerate(args.query_spec):
                keep = threshold_masks[:, index]
                name = query["name"].casefold()
                thresholds[name] = select_mcc_threshold(
                    threshold_labels[keep, index], threshold_scores[keep, index]
                )
                sensitivity_thresholds[name] = {
                    str(target): select_sensitivity_threshold(
                        threshold_labels[keep, index],
                        threshold_scores[keep, index],
                        target,
                    )
                    for target in sensitivity_targets
                }

        rows = []
        detailed = {}
        for index, query in enumerate(args.query_spec):
            keep = masks[:, index]
            query_labels = labels[keep, index]
            query_scores = scores[keep, index]
            metrics = discrimination_metrics(query_labels, query_scores)
            metrics.update(
                bootstrap_intervals(
                    query_labels,
                    query_scores,
                    repetitions=args.bootstrap,
                    seed=args.seed + index,
                )
            )
            threshold = thresholds.get(query["name"].casefold())
            if threshold is not None and np.isfinite(threshold):
                metrics.update(operating_metrics(query_labels, query_scores, threshold))
            if query["name"].casefold() in sensitivity_thresholds:
                metrics["sensitivity_anchored_thresholds"] = sensitivity_thresholds[
                    query["name"].casefold()
                ]
            detailed[query["name"]] = {
                key: finite_or_none(value) for key, value in metrics.items()
            }
            rows.append(
                {
                    "query": query["name"],
                    **{
                        key: finite_or_none(value)
                        for key, value in metrics.items()
                        if not isinstance(value, dict)
                    },
                }
            )

        with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(detailed, handle, indent=2, ensure_ascii=False)
        fieldnames = sorted({key for row in rows for key in row})
        with (args.output_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with (args.output_dir / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle, identifier_path.open(encoding="utf-8") as identifier_file:
            fieldnames = ["study_id"] + [item["name"] for item in args.query_spec]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row_index, identifier in enumerate(identifier_file):
                writer.writerow(
                    {
                        "study_id": identifier.rstrip("\n"),
                        **{
                            query["name"]: float(scores[row_index, query_index])
                            for query_index, query in enumerate(args.query_spec)
                        },
                    }
                )
        if thresholds:
            with (args.output_dir / "thresholds.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        query["name"]: finite_or_none(
                            thresholds.get(query["name"].casefold(), math.nan)
                        )
                        for query in args.query_spec
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
    print(f"Evaluated {len(dataset)} ECGs across {len(args.query_spec)} queries")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
