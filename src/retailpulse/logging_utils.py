"""Uniform logging setup."""
from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("retailpulse")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        root.propagate = False
    return logging.getLogger(name if name.startswith("retailpulse") else f"retailpulse.{name}")
