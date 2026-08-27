"""Zero-shot evaluation and task-specific adaptation on EchoNext data."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.datasets.datasets.echonext import EchoNextDataset, load_echonext_endpoints
from scripts.evaluation.metrics import bootstrap_intervals, discrimination_metrics
from scripts.inference import encode_queries, load_model, predict_queries


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run zero-shot EchoNext evaluation")
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--metadata", required=True, type=Path)
    evaluate.add_argument("--waveforms", required=True, type=Path)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--endpoint-config", type=Path)
    evaluate.add_argument("--waveform-scale", type=float, default=0.3)
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    finetune = subparsers.add_parser(
        "finetune", help="Fine-tune ExFMECG jointly on the 12 EchoNext outputs"
    )
    finetune.add_argument("--checkpoint", required=True, type=Path)
    finetune.add_argument("--metadata", required=True, type=Path)
    finetune.add_argument("--train-waveforms", required=True, type=Path)
    finetune.add_argument("--val-waveforms", required=True, type=Path)
    finetune.add_argument("--test-waveforms", required=True, type=Path)
    finetune.add_argument("--output-dir", required=True, type=Path)
    finetune.add_argument("--endpoint-config", type=Path)
    finetune.add_argument("--waveform-scale", type=float, default=0.3)
    finetune.add_argument("--epochs", type=int, default=10)
    finetune.add_argument("--warmup-epochs", type=int, default=1)
    finetune.add_argument("--head-lr", type=float, default=1e-5)
    finetune.add_argument("--encoder-lr", type=float, default=1e-6)
    finetune.add_argument("--weight-decay", type=float, default=1e-4)
    finetune.add_argument("--noise-std", type=float, default=0.02)
    finetune.add_argument("--batch-size", type=int, default=32)
    finetune.add_argument("--num-workers", type=int, default=0)
    finetune.add_argument("--bootstrap", type=int, default=1000)
    finetune.add_argument("--seed", type=int, default=42)
    finetune.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _loader(dataset, batch_size, num_workers, device, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )


def _samples(batch, device):
    return {
        "ecg": batch["ecg"].to(device, non_blocking=True),
        "age": batch["age"].to(device, non_blocking=True),
        "gender": batch["gender"].to(device, non_blocking=True),
    }


def _write_results(output_dir, endpoints, labels, scores, identifiers, model_name, bootstrap, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, endpoint in enumerate(endpoints):
        metrics = discrimination_metrics(labels[:, index], scores[:, index])
        metrics.update(
            bootstrap_intervals(
                labels[:, index],
                scores[:, index],
                repetitions=bootstrap,
                seed=seed + index,
            )
        )
        rows.append(
            {
                "model": model_name,
                "endpoint": endpoint["name"],
                **metrics,
            }
        )
    fieldnames = list(rows[0])
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        labels=labels.astype(np.int8),
        predictions=scores.astype(np.float32),
        endpoint_names=np.asarray([endpoint["name"] for endpoint in endpoints]),
        study_ids=np.asarray(identifiers),
    )


def run_zero_shot(args, endpoints):
    dataset = EchoNextDataset(
        args.metadata,
        args.waveforms,
        args.split,
        endpoints=endpoints,
        scale=args.waveform_scale,
    )
    model = load_model(args.checkpoint, args.device)
    scores = np.empty((len(dataset), len(endpoints)), dtype=np.float32)
    labels = None
    identifiers = None
    for endpoint_index, endpoint in enumerate(endpoints):
        query_features = encode_queries(model, [endpoint["zero_shot_prompt"]])
        loader = _loader(
            dataset,
            args.batch_size,
            args.num_workers,
            args.device,
        )
        label_parts = []
        identifier_parts = []
        offset = 0
        for batch in loader:
            probabilities = predict_queries(
                model,
                _samples(batch, args.device),
                query_features,
                query_batch_size=1,
            )[:, 0]
            stop = offset + len(probabilities)
            scores[offset:stop, endpoint_index] = probabilities.cpu().numpy()
            if endpoint_index == 0:
                label_parts.append(batch["labels"].numpy())
                identifier_parts.extend(batch["study_id"])
            offset = stop
        if endpoint_index == 0:
            labels = np.concatenate(label_parts)
            identifiers = identifier_parts
        print(f"[{endpoint_index + 1}/{len(endpoints)}] {endpoint['name']}")
    _write_results(
        args.output_dir,
        endpoints,
        labels,
        scores,
        identifiers,
        "ExFMECG zero-shot",
        args.bootstrap,
        args.seed,
    )
    print(f"Zero-shot EchoNext results: {args.output_dir}")


def _raw_text_features(model, endpoints):
    with torch.no_grad():
        return model.get_text_features(
            model.knowledge_encoder,
            [endpoint["name"] for endpoint in endpoints],
            model.tokenizer,
            model.device,
            model.max_length,
        ).detach()


def _joint_logits(model, batch, raw_text_features, device, noise_std=0.0):
    signal = batch["ecg"].to(device, non_blocking=True).float()
    if noise_std:
        signal = signal + float(noise_std) * torch.randn_like(signal)
    metadata_dtype = torch.float16 if signal.is_cuda else torch.float32
    age = batch["age"].to(device, non_blocking=True, dtype=metadata_dtype).view(-1, 1)
    gender = batch["gender"].to(
        device, non_blocking=True, dtype=metadata_dtype
    ).view(-1, 1)
    with model.maybe_autocast():
        features = model.ecg_model(signal, torch.cat([age, gender], dim=1))
        query_features = model.mlp_embed(raw_text_features)
        return model.llm_proj(
            model.tqn_model(features.transpose(1, 2), query_features)
        )


@torch.no_grad()
def _evaluate_joint(model, loader, raw_text_features, device):
    model.eval()
    scores = []
    labels = []
    identifiers = []
    total_loss = 0.0
    batches = 0
    for batch in loader:
        logits = _joint_logits(model, batch, raw_text_features, device)
        target = batch["labels"].to(device, non_blocking=True).long()
        loss = sum(
            F.cross_entropy(logits[:, index, :], target[:, index])
            for index in range(target.shape[1])
        ) / target.shape[1]
        total_loss += float(loss)
        batches += 1
        scores.append(torch.softmax(logits, dim=-1)[..., 1].float().cpu().numpy())
        labels.append(batch["labels"].numpy())
        identifiers.extend(batch["study_id"])
    return (
        total_loss / max(batches, 1),
        np.concatenate(labels),
        np.concatenate(scores),
        identifiers,
    )


def run_finetune(args, endpoints):
    if not 1 <= args.epochs <= 10:
        raise ValueError("EchoNext fine-tuning uses between 1 and 10 epochs")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs must be smaller than epochs")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    datasets = {
        "train": EchoNextDataset(
            args.metadata,
            args.train_waveforms,
            "train",
            endpoints=endpoints,
            scale=args.waveform_scale,
        ),
        "val": EchoNextDataset(
            args.metadata,
            args.val_waveforms,
            "val",
            endpoints=endpoints,
            scale=args.waveform_scale,
        ),
        "test": EchoNextDataset(
            args.metadata,
            args.test_waveforms,
            "test",
            endpoints=endpoints,
            scale=args.waveform_scale,
        ),
    }
    loaders = {
        name: _loader(
            dataset,
            args.batch_size if name == "train" else max(args.batch_size, 64),
            args.num_workers,
            args.device,
            shuffle=name == "train",
        )
        for name, dataset in datasets.items()
    }

    model = load_model(args.checkpoint, args.device)
    raw_text_features = _raw_text_features(model, endpoints)
    for parameter in model.parameters():
        parameter.requires_grad = False
    head_parameters = []
    for module in (model.tqn_model, model.mlp_embed, model.llm_proj):
        for parameter in module.parameters():
            parameter.requires_grad = True
            head_parameters.append(parameter)
    encoder_parameters = list(model.ecg_model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.head_lr},
            {"params": encoder_parameters, "lr": args.encoder_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    training_labels = np.stack(
        [
            datasets["train"].metadata[endpoint["flag"]].to_numpy(dtype=np.float64)
            for endpoint in endpoints
        ],
        axis=1,
    )
    prevalence = training_labels.mean(axis=0)
    endpoint_weight = (1 - prevalence) / np.maximum(prevalence, 1e-3)
    endpoint_weight = torch.tensor(
        endpoint_weight / endpoint_weight.mean(),
        dtype=torch.float32,
        device=args.device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "echonext_exfmecg_best.pth"
    best_loss = math.inf
    history = []
    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            for parameter in encoder_parameters:
                parameter.requires_grad = True
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = _joint_logits(
                model,
                batch,
                raw_text_features,
                args.device,
                noise_std=args.noise_std,
            )
            target = batch["labels"].to(args.device, non_blocking=True).long()
            loss = sum(
                endpoint_weight[index]
                * F.cross_entropy(logits[:, index, :], target[:, index])
                for index in range(len(endpoints))
            ) / len(endpoints)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite EchoNext fine-tuning loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss)
            train_batches += 1

        validation_loss, validation_labels, validation_scores, _ = _evaluate_joint(
            model,
            loaders["val"],
            raw_text_features,
            args.device,
        )
        validation_aurocs = [
            discrimination_metrics(validation_labels[:, index], validation_scores[:, index])[
                "auroc"
            ]
            for index in range(len(endpoints))
        ]
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_batches, 1),
            "validation_loss": validation_loss,
            "validation_mean_auroc": float(np.nanmean(validation_aurocs)),
        }
        history.append(record)
        print(json.dumps(record))
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "endpoint_names": [endpoint["name"] for endpoint in endpoints],
                    "epoch": epoch + 1,
                    "validation_loss": validation_loss,
                },
                best_path,
            )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    model.to(args.device).eval()
    _, test_labels, test_scores, identifiers = _evaluate_joint(
        model,
        loaders["test"],
        raw_text_features,
        args.device,
    )
    _write_results(
        args.output_dir,
        endpoints,
        test_labels,
        test_scores,
        identifiers,
        "ExFMECG fine-tuned",
        args.bootstrap,
        args.seed,
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Fine-tuned EchoNext results: {args.output_dir}")


def main():
    args = parse_args()
    endpoints = load_echonext_endpoints(args.endpoint_config)
    if len(endpoints) != 12:
        raise ValueError("The released EchoNext task definition contains 12 outputs")
    if args.command == "evaluate":
        run_zero_shot(args, endpoints)
    else:
        run_finetune(args, endpoints)


if __name__ == "__main__":
    main()
