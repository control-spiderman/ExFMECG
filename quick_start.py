"""Run query-based ExFMECG inference on one preprocessed ECG."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.inference import encode_queries, load_model, predict_queries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ecg", required=True, type=Path)
    parser.add_argument("--age", required=True, type=float)
    parser.add_argument("--sex", required=True, choices=("female", "male", "unknown"))
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument(
        "--top-concepts",
        type=int,
        default=0,
        help="Also return the highest-probability binary ECG concepts",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_ecg(path):
    signal = np.load(path, allow_pickle=False)
    if signal.shape != (12, 1024):
        raise ValueError("ECG input must have shape (12, 1024)")
    signal = np.nan_to_num(signal, nan=0.0, posinf=5.0, neginf=-5.0)
    return torch.from_numpy(np.clip(signal, -5.0, 5.0).astype(np.float32))


def main():
    args = parse_args()
    model = load_model(args.checkpoint, args.device)
    query_features = encode_queries(model, args.query)
    sex = {"unknown": 0, "female": 1, "male": 2}[args.sex]
    samples = {
        "ecg": load_ecg(args.ecg).unsqueeze(0).to(args.device),
        "age": torch.tensor([args.age], device=args.device),
        "gender": torch.tensor([sex], device=args.device),
    }
    with torch.inference_mode():
        probabilities = (
            predict_queries(model, samples, query_features)[0].cpu().tolist()
        )
        query_output = dict(zip(args.query, probabilities))
        if args.top_concepts > 0:
            concept_probabilities = model.predict_concepts(samples)["binary"][0].cpu()
            concept_count = min(args.top_concepts, concept_probabilities.numel())
            values, indices = torch.topk(concept_probabilities, concept_count)
            concept_ids = model.sfp_concept_list
            output = {
                "queries": query_output,
                "top_concepts": [
                    {
                        "concept": concept_ids[index],
                        "probability": float(value),
                    }
                    for value, index in zip(values.tolist(), indices.tolist())
                ],
            }
        else:
            output = query_output
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
