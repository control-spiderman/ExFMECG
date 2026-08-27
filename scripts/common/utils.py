"""Small path and URL helpers used by the release code."""

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from scripts.common.registry import registry


def now():
    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def is_url(value):
    return urlparse(str(value)).scheme in {"http", "https"}


def get_abs_path(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(Path(registry.get_path("library_root")) / path)
