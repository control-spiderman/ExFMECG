from scripts.common.registry import registry
from scripts.tasks.base_task import BaseTask
from scripts.tasks.exfmecg import ExFMECGTrainingTask


def setup_task(cfg):
    task_cls = registry.get_task_class(cfg.run_cfg.task)
    if task_cls is None:
        raise KeyError(f"Unknown task: {cfg.run_cfg.task}")
    return task_cls.setup_task(cfg)


__all__ = ["BaseTask", "ExFMECGTrainingTask", "setup_task"]
