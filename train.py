"""Train ExFMECG from a manifest of preprocessed ECGs."""

import argparse
import random

import numpy as np
import torch

from scripts import tasks
from scripts.common.config import Config
from scripts.common.dist_utils import get_rank, init_distributed_mode
from scripts.common.logger import setup_logger
from scripts.common.registry import registry
from scripts.common.utils import now


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        default="scripts/config/train/exfmecg_v13.yaml",
    )
    parser.add_argument("--options", nargs="*")
    return parser.parse_args()


def set_seed(seed):
    seed += get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_logger()
    set_seed(int(cfg.run_cfg.seed))
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    runner = runner_cls(
        cfg=cfg,
        job_id=now(),
        task=task,
        model=task.build_model(cfg),
        datasets=task.build_datasets(cfg),
    )
    runner.train()


if __name__ == "__main__":
    main()
