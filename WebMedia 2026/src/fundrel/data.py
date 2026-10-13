"""Builds the FW.Br and FT.Br datasets (Section 3.1 of the paper).

Steps, in order:
1. load the raw BATMAN export (Git LFS file in data/raw/);
2. drop messages from the excluded member IDs listed in config/params.yaml;
3. drop messages without text and "trava-zap" messages;
4. preprocess the text (see preprocessing.py);
5. inclusion filter: keep messages containing >= 1 religious keyword;
6. exclusion filter: drop messages containing any election-campaign keyword.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import config
from .preprocessing import preprocess_text

log = logging.getLogger(__name__)

TEXT_RAW = "text_content_anonymous"
TEXT = "text_processed"


class LFSPointerError(RuntimeError):
    pass


def _check_not_lfs_pointer(path: Path) -> None:
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise LFSPointerError(
            f"{path} is a Git LFS pointer, not the data. Run `git lfs install && "
            "git lfs pull` (see data/README.md)."
        )


def load_raw(platform: str) -> pd.DataFrame:
    path = config.path("raw", platform)
    _check_not_lfs_pointer(path)
    return pd.read_csv(path, low_memory=False)


def contains_any(text: str, keywords: list[str], match: str = "substring") -> bool:
    """Keyword test used by both filters.

    `substring` reproduces the paper run (e.g. "pt" also matches "aceito",
    "reino" also matches "treino"); `word` requires whole-word matches.
    """
    text = text.lower()
    if match == "substring":
        return any(k in text for k in keywords)
    if match == "word":
        return any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text) for k in keywords)
    raise ValueError(f"unknown match mode: {match}")


def clean(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    excluded = set(config.load()["datasets"][platform]["excluded_member_ids"])
    df = df[~df["id_member_anonymous"].isin(excluded)]
    df = df.dropna(subset=[TEXT_RAW])
    return df[df["trava_zap"] == False]  # noqa: E712 (column may be object dtype)


def build(platform: str, save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return the filtered dataset and a dict with the size after each step."""
    cfg = config.load()["filters"]
    raw = load_raw(platform)
    steps = {
        "raw_messages": len(raw),
        "raw_users": int(raw["id_member_anonymous"].nunique()),
        "raw_groups": int(raw["id_group_anonymous"].nunique()),
        "first_message": str(raw["date_message"].min()),
        "last_message": str(raw["date_message"].max()),
    }
    df = clean(raw, platform)
    steps["after_cleaning"] = len(df)
    df = df.copy()
    df[TEXT] = df[TEXT_RAW].apply(preprocess_text)
    rel = df[df[TEXT].apply(contains_any, args=(cfg["religious_keywords"], cfg["match"]))]
    steps["after_inclusion_filter"] = len(rel)
    out = rel[~rel[TEXT].apply(contains_any, args=(cfg["political_keywords"], cfg["match"]))]
    steps["after_exclusion_filter"] = len(out)
    out = out.copy()
    out["platform"] = platform
    log.info("%s: %s", config.dataset_name(platform), steps)
    if save:
        save_dataset(out, platform)
    return out, steps


def processed_path(platform: str) -> Path:
    return config.processed_dir() / f"{config.dataset_name(platform).replace('.', '_').lower()}.csv.gz"


def save_dataset(df: pd.DataFrame, platform: str) -> None:
    df.to_csv(processed_path(platform), index=False)
    # Shareable list of selected messages (no text): lets others verify the
    # selection against the public raw data without redistributing content.
    ids = df[["id_message", "id_group_anonymous", "date_message"]]
    ids.to_csv(config.processed_dir() / f"{config.dataset_name(platform)}_message_ids.csv", index=False)


def load(platform: str) -> pd.DataFrame:
    """Load the processed dataset, building it first if needed."""
    path = processed_path(platform)
    if not path.exists():
        build(platform)
    return pd.read_csv(path, low_memory=False, keep_default_na=True).assign(
        **{TEXT: lambda d: d[TEXT].fillna("")})
