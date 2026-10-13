"""Loads config/params.yaml and resolves paths relative to the repository root."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config" / "params.yaml"


@lru_cache(maxsize=1)
def load(path: str | None = None) -> dict:
    with open(path or CONFIG_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(*keys: str) -> Path:
    """Return an absolute path from the `paths` section, e.g. path('raw', 'whatsapp')."""
    node = load()["paths"]
    for k in keys:
        node = node[k]
    return ROOT / node


def results_dir(*parts: str) -> Path:
    out = path("results").joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def processed_dir() -> Path:
    out = path("processed")
    out.mkdir(parents=True, exist_ok=True)
    return out


def dataset_name(platform: str) -> str:
    return load()["datasets"][platform]["name"]
