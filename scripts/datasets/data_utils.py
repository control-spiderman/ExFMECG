"""Tensor transfer helpers."""

import torch


def prepare_sample(sample, device):
    def move(value):
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(move(item) for item in value)
        return value

    return move(sample)
