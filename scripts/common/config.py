"""Configuration loading for the ExFMECG release."""

import json

from omegaconf import OmegaConf

from scripts.common.registry import registry


class Config:
    def __init__(self, args):
        self.args = args
        user_options = OmegaConf.from_dotlist(self._option_list(args.options))
        experiment = OmegaConf.load(args.cfg_path)

        model_cls = registry.get_model_class(experiment.model.arch)
        if model_cls is None:
            raise KeyError(f"Unknown model architecture: {experiment.model.arch}")
        model_type = experiment.model.get("model_type", "mlp")
        model_defaults = OmegaConf.load(model_cls.default_config_path(model_type))
        self.config = OmegaConf.merge(model_defaults, experiment, user_options)
        registry.register("configuration", self)

    @staticmethod
    def _option_list(options):
        if not options:
            return []
        if all("=" in item for item in options):
            return options
        if len(options) % 2:
            raise ValueError("--options must contain key=value entries or key value pairs")
        return [f"{key}={value}" for key, value in zip(options[::2], options[1::2])]

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model

    def to_dict(self):
        return OmegaConf.to_container(self.config, resolve=True)

    def pretty_print(self):
        print(json.dumps(self.to_dict(), indent=2, sort_keys=True))
