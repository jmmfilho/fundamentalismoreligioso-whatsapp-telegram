"""Polarity analysis (Section 4.3, Table 3).

Uses the LeIA sentiment score stored in `score_sentiment` (computed by the
collection platform at ingestion). Reproduces the analysis behind the paper:

* descriptives per dataset; categories negative (< -0.05), neutral
  ([-0.05, 0.05]) and positive (> 0.05);
* Mann-Whitney U (two-sided) and Cliff's delta (FW.Br vs FT.Br; positive means
  FW.Br more positive) with the 95% CI from Cliff's (1993) consistent variance
  estimator;
* chi-square on the 2 x 3 category table with Cramer's V, and the difference in
  the share of negative messages with its 95% CI;
* robustness: group as unit of analysis (median score per group, groups with
  at least N messages) and unique messages only.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from . import config, data

Z = 1.959963985


def cliffs_delta(x, y) -> tuple[float, float, tuple[float, float]]:
    """delta = P(X > Y) - P(X < Y) in O(n log n), with Cliff's (1993) asymmetric CI.

    Returns (delta, standard error, (low, high)).
    """
    x = np.sort(np.asarray(x, float))
    y = np.sort(np.asarray(y, float))
    n, m = len(x), len(y)
    less = np.searchsorted(y, x, "left")
    lesseq = np.searchsorted(y, x, "right")
    di = (less - (m - lesseq)) / m
    delta = di.mean()
    xless = np.searchsorted(x, y, "left")
    xlesseq = np.searchsorted(x, y, "right")
    dj = ((n - xlesseq) - xless) / n
    ties = int((lesseq - less).sum())
    var = (m * m * di.var(ddof=1) * (n - 1) + n * n * dj.var(ddof=1) * (m - 1)
           + ((n * m - ties) - n * m * delta ** 2)) / (n * m * (n - 1) * (m - 1))
    sd = float(np.sqrt(max(var, 0.0)))
    den = 1 - delta ** 2 + Z ** 2 * var
    root = Z * sd * np.sqrt((1 - delta ** 2) ** 2 + Z ** 2 * var)
    return float(delta), sd, (float((delta - delta ** 3 - root) / den),
                              float((delta - delta ** 3 + root) / den))


def categories(s: np.ndarray, band: float) -> np.ndarray:
    return np.array([(s < -band).sum(), ((s >= -band) & (s <= band)).sum(), (s > band).sum()])


def describe(s: np.ndarray, band: float) -> dict:
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    neg, neu, pos = categories(s, band) / len(s)
    return {"n": int(len(s)), "mean": float(s.mean()), "sd": float(s.std(ddof=1)),
            "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "pct_negative": round(100 * neg, 1), "pct_neutral": round(100 * neu, 1),
            "pct_positive": round(100 * pos, 1)}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    mw = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    delta, se, ci = cliffs_delta(a, b)
    return {"n_FW.Br": int(len(a)), "n_FT.Br": int(len(b)), "U": float(mw.statistic),
            "p": float(mw.pvalue), "cliffs_delta": delta, "cliffs_delta_se": se,
            "cliffs_delta_ci95": list(ci)}


def run() -> dict:
    cfg = config.load()["polarity"]
    band, col = cfg["neutral_band"], cfg["column"]
    wa, tg = data.load("whatsapp"), data.load("telegram")
    a, b = wa[col].dropna().to_numpy(), tg[col].dropna().to_numpy()
    report = {"missing_scores": {"FW.Br": int(wa[col].isna().sum()), "FT.Br": int(tg[col].isna().sum())},
              "FW.Br": describe(a, band), "FT.Br": describe(b, band),
              "message_level": compare(a, b)}

    tab = np.vstack([categories(a, band), categories(b, band)])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    p1, p2 = tab[0, 0] / tab[0].sum(), tab[1, 0] / tab[1].sum()
    se = np.sqrt(p1 * (1 - p1) / tab[0].sum() + p2 * (1 - p2) / tab[1].sum())
    report["categories"] = {
        "table": {"FW.Br": tab[0].tolist(), "FT.Br": tab[1].tolist(),
                  "columns": ["negative", "neutral", "positive"]},
        "chi2": float(chi2), "dof": int(dof), "p": float(p),
        "cramers_v": float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))),
        "negative_diff_pp_FT_minus_FW": float(100 * (p2 - p1)),
        "negative_diff_ci95_pp": [float(100 * (p2 - p1 - Z * se)), float(100 * (p2 - p1 + Z * se))],
    }

    report["unique_messages"] = compare(
        wa.drop_duplicates(subset=[data.TEXT_RAW])[col].dropna().to_numpy(),
        tg.drop_duplicates(subset=[data.TEXT_RAW])[col].dropna().to_numpy())

    report["group_level"] = {}
    for gmin in cfg["group_min_messages"]:
        def medians(df):
            g = df.dropna(subset=[col]).groupby("id_group_anonymous")[col].agg(["median", "size"])
            return g.loc[g["size"] >= gmin, "median"].to_numpy()
        report["group_level"][f"min_{gmin}"] = compare(medians(wa), medians(tg))

    out = config.results_dir("04_polarity")
    (out / "polarity_statistics.json").write_text(json.dumps(report, indent=2))
    return report
