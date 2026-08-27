"""Epoch-based distributed runner used for ExFMECG training."""

import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from scripts.common.dist_utils import get_rank, get_world_size, is_main_process
from scripts.common.registry import registry
from scripts.datasets.data_utils import prepare_sample


@registry.register_runner("runner_base")
class RunnerBase:
    def __init__(self, cfg, task, model, datasets, job_id):
        self.cfg = cfg
        self.task = task
        self.raw_model = model
        self.datasets = datasets
        self.device = torch.device(cfg.run_cfg.device)
        self.output_dir = Path(cfg.run_cfg.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0

        train_sets = [splits["train"] for splits in datasets.values()]
        train_dataset = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
        self.sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
            )
            if cfg.run_cfg.distributed
            else None
        )
        self.loader = DataLoader(
            train_dataset,
            batch_size=int(cfg.run_cfg.batch_size_train),
            sampler=self.sampler,
            shuffle=self.sampler is None,
            num_workers=int(cfg.run_cfg.num_workers),
            pin_memory=True,
            drop_last=True,
        )

        self.raw_model.to(self.device)
        self.model = self.raw_model
        if cfg.run_cfg.distributed:
            self.model = DistributedDataParallel(
                self.raw_model,
                device_ids=[cfg.run_cfg.gpu],
                find_unused_parameters=bool(
                    cfg.run_cfg.get("find_unused_parameters", False)
                ),
            )

        params = self.raw_model.get_optimizer_params(
            float(cfg.run_cfg.weight_decay)
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(cfg.run_cfg.init_lr),
            betas=(0.9, float(cfg.run_cfg.get("beta2", 0.999))),
        )
        scheduler_cls = registry.get_lr_scheduler_class(cfg.run_cfg.lr_sched)
        self.scheduler = scheduler_cls(
            optimizer=self.optimizer,
            max_epoch=int(cfg.run_cfg.max_epoch),
            min_lr=float(cfg.run_cfg.min_lr),
            init_lr=float(cfg.run_cfg.init_lr),
            warmup_start_lr=float(cfg.run_cfg.warmup_lr),
            warmup_steps=int(cfg.run_cfg.warmup_steps),
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(cfg.run_cfg.amp)
        )
        if cfg.run_cfg.get("resume_ckpt_path"):
            self._resume(cfg.run_cfg.resume_ckpt_path)

    def train(self):
        self._write_config()
        epochs = int(self.cfg.run_cfg.max_epoch)
        accumulation = int(self.cfg.run_cfg.get("accum_grad_iters", 1))
        log_frequency = int(self.cfg.run_cfg.get("log_freq", 50))

        for epoch in range(self.start_epoch, epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            for step, samples in enumerate(self.loader):
                samples = prepare_sample(samples, self.device)
                self.scheduler.step(cur_epoch=epoch, cur_step=step)
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    loss, loss_dict = self.task.train_step(self.model, samples)
                    scaled_loss = loss / accumulation
                self.scaler.scale(scaled_loss).backward()

                if (step + 1) % accumulation == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                if is_main_process() and step % log_frequency == 0:
                    values = {
                        key: float(value.detach()) if torch.is_tensor(value) else float(value)
                        for key, value in loss_dict.items()
                    }
                    logging.info("epoch=%d step=%d losses=%s", epoch, step, values)

            self._save(epoch)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    def _save(self, epoch):
        if not is_main_process():
            return
        trainable = {
            name: parameter.requires_grad
            for name, parameter in self.raw_model.named_parameters()
        }
        state = self.raw_model.state_dict()
        state = {
            key: value
            for key, value in state.items()
            if key not in trainable or trainable[key]
        }
        torch.save(
            {
                "model": state,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": self.cfg.to_dict(),
                "epoch": epoch,
            },
            self.output_dir / f"checkpoint_{epoch}.pth",
        )

    def _resume(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(checkpoint["model"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1

    def _write_config(self):
        if is_main_process():
            with (self.output_dir / "config.json").open("w", encoding="utf-8") as handle:
                json.dump(self.cfg.to_dict(), handle, indent=2)
