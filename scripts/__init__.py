"""Register the components used by the ExFMECG release."""

from pathlib import Path

from scripts.common.registry import registry

_ROOT = Path(__file__).resolve().parent
registry.register_path("library_root", str(_ROOT))
registry.register_path("repo_root", str(_ROOT.parent))
registry.register_path("cache_root", str(_ROOT.parent / "cache"))

from scripts.common import optims  # noqa: E402,F401
from scripts.datasets import builders  # noqa: E402,F401
from scripts import models, runners, tasks  # noqa: E402,F401
