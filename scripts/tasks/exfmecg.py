from scripts.common.registry import registry
from scripts.tasks.base_task import BaseTask


@registry.register_task("exfmecg_training")
class ExFMECGTrainingTask(BaseTask):
    pass
