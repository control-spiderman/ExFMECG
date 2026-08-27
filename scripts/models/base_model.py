"""Base model utilities for ExFMECG."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.common.dist_utils import download_cached_file
from scripts.common.utils import get_abs_path, is_url


class BaseModel(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def load_checkpoint(self, path_or_url):
        path = download_cached_file(path_or_url) if is_url(path_or_url) else path_or_url
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        result = self.load_state_dict(state_dict, strict=False)
        logging.info("Loaded checkpoint from %s", path)
        return result

    @classmethod
    def from_pretrained(cls, model_type="mlp"):
        cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        return cls.from_config(cfg)

    @classmethod
    def default_config_path(cls, model_type):
        if model_type not in cls.PRETRAINED_MODEL_CONFIG_DICT:
            raise KeyError(f"Unknown model type: {model_type}")
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg):
        if cfg.get("load_pretrained", False):
            self.load_checkpoint(cfg.pretrained)
        if cfg.get("load_finetuned", False):
            self.load_checkpoint(cfg.finetuned)

    def get_optimizer_params(self, weight_decay, lr_scale=1.0):
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            target = no_decay if parameter.ndim < 2 or any(
                token in name for token in ("bias", "ln", "bn")
            ) else decay
            target.append(parameter)
        return [
            {"params": decay, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": no_decay, "weight_decay": 0.0, "lr_scale": lr_scale},
        ]
