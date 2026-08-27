"""Query-conditioned ECG concept attribution utilities."""

import json

import torch


def active_binary_concepts(model):
    """Return active binary concept names and their positions in the 739-output head."""

    path = model.asset_root / "concepts/schema/active_binary_concept_ids.json"
    with path.open(encoding="utf-8") as handle:
        names = json.load(handle)
    lookup = {name: index for index, name in enumerate(model.sfp_concept_list)}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"Active concepts are absent from the model schema: {missing[:3]}")
    indices = torch.tensor(
        [lookup[name] for name in names],
        device=model.device,
        dtype=torch.long,
    )
    return names, indices


def _gradient_weighted_attention(attention, gradients, batch_size):
    if attention is None or gradients is None:
        raise RuntimeError("Attention gradients were not retained")
    if attention.shape[0] % batch_size:
        raise ValueError("Attention tensor cannot be separated into batches and heads")
    heads = attention.shape[0] // batch_size
    attention = attention.reshape(batch_size, heads, *attention.shape[-2:])
    gradients = gradients.reshape(batch_size, heads, *gradients.shape[-2:])
    return (attention * gradients).clamp_min(0).mean(dim=1)


def _normalize_residual(matrix):
    size = matrix.shape[-1]
    identity = torch.eye(
        size,
        device=matrix.device,
        dtype=matrix.dtype,
    ).unsqueeze(0)
    residual = (matrix - identity).clamp_min(0)
    residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return residual + identity


def _temporal_relevance(model, features, query_count, query_index):
    batch_size, _, source_length = features.shape
    if batch_size != 1:
        raise ValueError("Concept attribution is evaluated one ECG at a time")

    r_source = torch.eye(
        source_length,
        device=features.device,
        dtype=features.dtype,
    ).unsqueeze(0)
    r_query = torch.eye(
        query_count,
        device=features.device,
        dtype=features.dtype,
    ).unsqueeze(0)
    r_cross = torch.zeros(
        batch_size,
        query_count,
        source_length,
        device=features.device,
        dtype=features.dtype,
    )

    for layer in model.tqn_model.decoder.layers:
        self_attention = _gradient_weighted_attention(
            layer.self_attn.attn,
            layer.self_attn.attn_gradients,
            batch_size,
        ).to(r_query.dtype)
        r_cross = r_cross + torch.bmm(self_attention, r_cross)
        r_query = r_query + torch.bmm(self_attention, r_query)

        cross_attention = _gradient_weighted_attention(
            layer.multihead_attn.attn,
            layer.multihead_attn.attn_gradients,
            batch_size,
        ).to(r_query.dtype)
        addition = torch.bmm(
            _normalize_residual(r_query).transpose(1, 2),
            cross_attention,
        )
        r_cross = r_cross + torch.bmm(addition, _normalize_residual(r_source))

    relevance = r_cross[:, query_index : query_index + 1]
    minimum = relevance.amin(dim=-1, keepdim=True)
    maximum = relevance.amax(dim=-1, keepdim=True)
    return torch.where(
        maximum > minimum,
        (relevance - minimum) / (maximum - minimum).clamp_min(1e-12),
        torch.zeros_like(relevance),
    )


def disease_concept_attribution(
    model,
    samples,
    query_features,
    query_indices,
    active_indices=None,
):
    """Project disease-conditioned temporal relevance into the binary concept space.

    All disease queries are passed through the query decoder together. ``samples``
    must contain one ECG. The returned tensor has
    shape ``(len(query_indices), number_of_active_binary_concepts)``.
    """

    signal = samples["ecg"].float()
    if signal.shape[0] != 1:
        raise ValueError("Disease-to-concept attribution requires batch size 1")
    if not query_indices:
        return torch.empty((0, 0), device=signal.device)
    if min(query_indices) < 0 or max(query_indices) >= query_features.shape[0]:
        raise IndexError("A query index is outside the supplied query feature matrix")

    for layer in model.tqn_model.decoder.layers:
        layer.self_attn.mode = "explain"
        layer.multihead_attn.mode = "explain"

    metadata_dtype = torch.float16 if signal.is_cuda else torch.float32
    age = torch.as_tensor(
        samples["age"], device=signal.device, dtype=metadata_dtype
    ).view(-1, 1)
    gender = torch.as_tensor(
        samples["gender"], device=signal.device, dtype=metadata_dtype
    ).view(-1, 1)

    model.zero_grad(set_to_none=True)
    with model.maybe_autocast():
        feature_map = model.ecg_model(signal, torch.cat([age, gender], dim=1))
        query_logits = model.llm_proj(
            model.tqn_model(feature_map.transpose(1, 2), query_features)
        )

    if active_indices is None:
        _, active_indices = active_binary_concepts(model)
    output = []
    for position, query_index in enumerate(query_indices):
        model.zero_grad(set_to_none=True)
        query_logits[0, query_index, 1].backward(
            retain_graph=position + 1 < len(query_indices)
        )
        relevance = _temporal_relevance(
            model,
            feature_map,
            query_features.shape[0],
            query_index,
        )
        concept_scores = model.concept_projector.concept_from_cam(
            feature_map,
            relevance,
        )
        output.append(concept_scores.index_select(1, active_indices).float())
    return torch.cat(output, dim=0).detach()
