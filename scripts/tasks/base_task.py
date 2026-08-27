"""Task interface for ExFMECG training."""

from scripts.common.registry import registry


class BaseTask:
    @classmethod
    def setup_task(cls, cfg):
        return cls()

    def build_model(self, cfg):
        model_cls = registry.get_model_class(cfg.model_cfg.arch)
        return model_cls.from_config(cfg.model_cfg)

    def build_datasets(self, cfg):
        datasets = {}
        for name, dataset_cfg in cfg.datasets_cfg.items():
            builder_cls = registry.get_builder_class(name)
            if builder_cls is None:
                raise KeyError(f"Unknown dataset builder: {name}")
            datasets[name] = builder_cls(dataset_cfg).build_datasets()
        return datasets

    def train_step(self, model, samples):
        output = model(samples)
        return output["loss"], {
            key: value for key, value in output.items() if "loss" in key
        }
