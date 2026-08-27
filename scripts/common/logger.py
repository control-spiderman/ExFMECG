"""Logging setup."""

import logging

from scripts.common.dist_utils import is_main_process


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
