"""Dataset statistics (Table 1) and messages-per-day figure (Figure 2)."""
from __future__ import annotations

import json

import pandas as pd

from . import PLATFORMS, config, data


def basic_statistics(df: pd.DataFrame) -> dict:
    """Table 1. Unique messages are distinct raw texts; tokens are whitespace
    tokens of the preprocessed text of those unique messages."""
    unique = df.drop_duplicates(subset=[data.TEXT_RAW])
    n_tokens = unique[data.TEXT].str.split().str.len()
    return {
        "messages": int(len(df)),
        "unique_messages": int(len(unique)),
        "users": int(df["id_member_anonymous"].nunique()),
        "groups": int(df["id_group_anonymous"].nunique()),
        "tokens_mean": round(float(n_tokens.mean()), 2),
        "tokens_std": round(float(n_tokens.std()), 2),
        "tokens_min": int(n_tokens.min()),
        "tokens_max": int(n_tokens.max()),
    }


def messages_per_day(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["date_message"], errors="coerce")
    return dates.dt.floor("D").value_counts().sort_index()


def plot_messages_per_day(frames: dict[str, pd.DataFrame], out_png) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {p: messages_per_day(df) for p, df in frames.items()}
    idx = pd.date_range(min(s.index.min() for s in series.values()),
                        max(s.index.max() for s in series.values()), freq="D")
    table = pd.DataFrame({p: s.reindex(idx, fill_value=0) for p, s in series.items()})
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(table.index, table["whatsapp"], marker="o", linestyle="-", label="WhatsApp")
    ax.plot(table.index, table["telegram"], marker="x", linestyle="--", label="Telegram")
    ax.set(title="Number of Messages per Day (WhatsApp vs. Telegram)",
           xlabel="Date", ylabel="Number of Messages")
    ax.legend(); ax.grid(True); plt.xticks(rotation=45); fig.tight_layout()
    fig.savefig(out_png, dpi=200); plt.close(fig)
    return table


def run() -> dict:
    out = config.results_dir("01_datasets")
    report, frames = {}, {}
    for p in PLATFORMS:
        df, steps = data.build(p)
        frames[p] = df
        report[config.dataset_name(p)] = {"filtering_steps": steps, **basic_statistics(df)}
    (out / "dataset_statistics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    table = plot_messages_per_day(frames, out / "messages_per_day.png")
    table.to_csv(out / "messages_per_day.csv", index_label="date")
    return report
