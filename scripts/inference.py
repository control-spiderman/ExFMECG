"""Checkpoint loading and query-based inference for ExFMECG."""

from pathlib import Path

import torch
import yaml

from scripts.models.exfmecg_v13 import ExFMECGV13


def load_model(checkpoint, device="cpu"):
    config_path = Path(__file__).resolve().parent / "config/model/exfmecg_v13.yaml"
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)["model"]
    cfg["mode"] = "train"
    model = ExFMECGV13.from_config(cfg).cpu().eval()
    model.materialize_dynamic_modules()

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data.get("model", checkpoint_data)
    result = model.load_state_dict(state, strict=False)
    parameters = dict(model.named_parameters())
    missing_trainable = [
        key
        for key in result.missing_keys
        if key in parameters and parameters[key].requires_grad
    ]
    if result.unexpected_keys or missing_trainable:
        raise RuntimeError(
            f"Checkpoint mismatch: unexpected={result.unexpected_keys}, "
            f"missing_trainable={missing_trainable}"
        )
    return model.to(device).eval()


@torch.no_grad()
def encode_queries(model, queries):
    text_features = model.get_text_features(
        model.knowledge_encoder,
        queries,
        model.tokenizer,
        model.device,
        model.max_length,
    )
    return model.mlp_embed(text_features)


@torch.no_grad()
def predict_queries(model, samples, query_features, query_batch_size=128):
    signal = samples["ecg"].float()
    metadata_dtype = torch.float16 if signal.is_cuda else torch.float32
    age = torch.as_tensor(
        samples["age"], device=signal.device, dtype=metadata_dtype
    ).view(-1, 1)
    gender = torch.as_tensor(
        samples["gender"], device=signal.device, dtype=metadata_dtype
    ).view(-1, 1)
    with model.maybe_autocast():
        features = model.ecg_model(signal, torch.cat([age, gender], dim=1))
        features = features.transpose(1, 2)

    predictions = []
    for start in range(0, query_features.shape[0], query_batch_size):
        stop = min(start + query_batch_size, query_features.shape[0])
        with model.maybe_autocast():
            logits = model.llm_proj(
                model.tqn_model(features, query_features[start:stop])
            )
        predictions.append(torch.softmax(logits, dim=-1)[..., 1].float())
    return torch.cat(predictions, dim=1)
