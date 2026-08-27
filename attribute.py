"""Export disease-conditioned ECG concept attribution for one held-out centre."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import load_queries, load_thresholds, split_labels
from scripts.datasets.datasets.exfmecg_dataset import ExFMECGManifestDataset
from scripts.inference import encode_queries, load_model, predict_queries
from scripts.interpretability import active_binary_concepts, disease_concept_attribution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ecg-root", type=Path, default=Path("."))
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--disease-thresholds", required=True, type=Path)
    parser.add_argument("--centre", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _selected_examples(model, dataset, query_spec, query_features, thresholds, args):
    names = [item["name"].casefold() for item in query_spec]
    missing = [item["name"] for item in query_spec if item["name"].casefold() not in thresholds]
    if missing:
        raise ValueError(f"Disease thresholds are missing for: {missing[:5]}")

    loader = DataLoader(
        dataset,
        batch_size=args.selection_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    selected = []
    offset = 0
    for batch in loader:
        samples = {
            "ecg": batch["ecg"].to(args.device, non_blocking=True),
            "age": batch["age"].to(args.device, non_blocking=True),
            "gender": batch["gender"].to(args.device, non_blocking=True),
        }
        probabilities = predict_queries(
            model,
            samples,
            query_features,
            query_batch_size=len(query_spec),
        ).cpu().numpy()
        for local_index, (positive_text, evaluable_text) in enumerate(
            zip(batch["label_list"], batch["evaluable_labels"])
        ):
            positive = split_labels(positive_text)
            evaluable = split_labels(evaluable_text)
            query_indices = []
            for query_index, name in enumerate(names):
                if evaluable and name not in evaluable:
                    continue
                if (
                    name in positive
                    and probabilities[local_index, query_index] >= thresholds[name]
                ):
                    query_indices.append(query_index)
            if query_indices:
                selected.append((offset + local_index, query_indices))
        offset += len(batch["study_id"])
    return selected


def main():
    args = parse_args()
    query_spec = load_queries(args.queries)
    thresholds = load_thresholds(args.disease_thresholds)
    dataset = ExFMECGManifestDataset(args.manifest, args.ecg_root)
    if not len(dataset):
        raise ValueError("Attribution manifest is empty")

    model = load_model(args.checkpoint, args.device)
    query_features = encode_queries(
        model,
        [item["prompt"] for item in query_spec],
    )
    concept_names, active_indices = active_binary_concepts(model)
    selected = _selected_examples(
        model,
        dataset,
        query_spec,
        query_features,
        thresholds,
        args,
    )

    sums = defaultdict(lambda: np.zeros(len(concept_names), dtype=np.float64))
    counts = defaultdict(int)
    for dataset_index, query_indices in selected:
        record = dataset[dataset_index]
        samples = {
            "ecg": record["ecg"].unsqueeze(0).to(args.device),
            "age": torch.tensor([record["age"]], device=args.device),
            "gender": torch.tensor([record["gender"]], device=args.device),
        }
        scores = disease_concept_attribution(
            model,
            samples,
            query_features,
            query_indices,
            active_indices=active_indices,
        ).cpu().numpy()
        for row, query_index in enumerate(query_indices):
            disease = query_spec[query_index]["name"]
            sums[disease] += scores[row]
            counts[disease] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "centre",
            "disease",
            "concept",
            "mean_attribution",
            "n_attributed_ecgs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query in query_spec:
            disease = query["name"]
            if counts[disease] == 0:
                continue
            means = sums[disease] / counts[disease]
            for concept, value in zip(concept_names, means):
                writer.writerow(
                    {
                        "centre": args.centre,
                        "disease": disease,
                        "concept": concept,
                        "mean_attribution": float(value),
                        "n_attributed_ecgs": counts[disease],
                    }
                )

    summary = {
        "centre": args.centre,
        "manifest_ecgs": len(dataset),
        "diseases_requested": len(query_spec),
        "diseases_with_attribution": sum(value > 0 for value in counts.values()),
        "attributed_ecgs_by_disease": dict(counts),
        "active_binary_concepts": len(concept_names),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Wrote attribution for {summary['diseases_with_attribution']} diseases "
        f"and {len(concept_names)} concepts to {args.output}"
    )


if __name__ == "__main__":
    main()
