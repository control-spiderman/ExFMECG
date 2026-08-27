"""Distributed-training helpers."""

import datetime
import os
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def _limit_print_to_main_process(is_master):
    import builtins

    original_print = builtins.print

    def distributed_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            original_print(*args, **kwargs)

    builtins.print = distributed_print


def init_distributed_mode(args):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        return

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(days=1),
    )
    dist.barrier()
    _limit_print_to_main_process(args.rank == 0)


def download_cached_file(url, check_hash=False, progress=True):
    """Download a checkpoint through PyTorch's shared hub cache."""
    filename = Path(urlparse(url).path).name
    destination = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if is_main_process() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, destination, progress=progress)
    if is_dist_avail_and_initialized():
        dist.barrier()
    return str(destination)
