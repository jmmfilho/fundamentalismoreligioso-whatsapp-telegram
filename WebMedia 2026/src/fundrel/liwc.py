"""Psycholinguistic analysis with the Brazilian Portuguese LIWC 2007 dictionary
(Section 4.5).

This module reproduces the analysis that produced the values in the paper
(63 categories, 49 significant, 40 after Holm-Bonferroni, phi <= 0.033):

* input: the raw text (`text_content_anonymous`) of FW.Br and FT.Br;
* tokens: NLTK `word_tokenize(text.lower(), language="portuguese")`, dropping
  NLTK's Portuguese stopwords and single punctuation characters;
* dictionary: category header read from the block between the `%` lines;
  entries lowercased; wildcard entries (`abafad*`) matched by prefix, longest
  prefix first; exact entries take precedence;
* per category: share of tokens in each dataset, 2x2 chi-square test without
  continuity correction (category vs. other tokens x platform), phi as effect
  size, 95% CI of the difference in proportions and Holm-Bonferroni p-values.

The legacy notebook (`notebooks/legacy/LIWC.ipynb`) used a different and
incorrect procedure; see docs/REPRODUCIBILITY_NOTES.md, section 6.
"""
from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import PLATFORMS, config, data
from .preprocessing import nltk_stopwords


@dataclass
class Dictionary:
    categories: dict[str, str]             # code -> name
    exact: dict[str, list[str]]            # word -> category names
    prefixes: list[tuple[str, list[str]]]  # (prefix, names), longest first

    def lookup(self, token: str) -> list[str]:
        if token in self.exact:
            return self.exact[token]
        for prefix, names in self.prefixes:
            if token.startswith(prefix):
                return names
        return []


def load_dictionary(path=None) -> Dictionary:
    path = path or config.path("liwc_dictionary")
    with open(path, encoding="latin1") as fh:
        lines = [l.rstrip("\n\r") for l in fh]
    marks = [i for i, l in enumerate(lines) if l.strip() == "%"]
    cats = {}
    for l in lines[marks[0] + 1:marks[1]]:
        parts = l.strip().split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            cats[parts[0]] = parts[1]
    exact, prefixes = {}, []
    for l in lines[marks[1] + 1:]:
        if not l.strip():
            continue
        word, *codes = l.strip().split("\t")
        names = [cats.get(c, "?") for c in codes if c.isdigit()]
        word = word.lower()
        if word.endswith("*"):
            prefixes.append((word[:-1], names))
        else:
            exact[word] = names
    prefixes.sort(key=lambda x: -len(x[0]))
    return Dictionary(cats, exact, prefixes)


def tokenize(text: str) -> list[str]:
    from nltk.tokenize import word_tokenize
    return word_tokenize(text.lower(), language="portuguese")


def count(texts, dic: Dictionary) -> tuple[Counter, int]:
    sw = nltk_stopwords()
    counts, total = Counter(), 0
    for text in texts:
        if not isinstance(text, str):
            continue
        toks = [t for t in tokenize(text) if t not in sw and t not in string.punctuation]
        total += len(toks)
        for tok in toks:
            counts.update(dic.lookup(tok))
    return counts, total


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(c_w: Counter, n_w: int, c_t: Counter, n_t: int, alpha: float) -> pd.DataFrame:
    rows = []
    z = stats.norm.ppf(1 - alpha / 2)
    for cat in sorted(set(c_w) | set(c_t)):
        a, b = c_w.get(cat, 0), c_t.get(cat, 0)
        table = np.array([[a, n_w - a], [b, n_t - b]])
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        pw, pt = a / n_w, b / n_t
        se = np.sqrt(pw * (1 - pw) / n_w + pt * (1 - pt) / n_t)
        diff = pt - pw
        rows.append({"category": cat, "pct_FW.Br": 100 * pw, "pct_FT.Br": 100 * pt,
                     "diff_pp_FT_minus_FW": 100 * diff,
                     "ci95_low_pp": 100 * (diff - z * se), "ci95_high_pp": 100 * (diff + z * se),
                     "ratio_FT_FW": pt / pw if pw else np.nan,
                     "chi2": chi2, "p": p, "phi": float(np.sqrt(chi2 / (n_w + n_t)))})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"].to_numpy())
    df["significant"] = df["p"] <= alpha
    df["significant_holm"] = df["p_holm"] <= alpha
    return df.sort_values("diff_pp_FT_minus_FW", key=abs, ascending=False)


def run() -> pd.DataFrame:
    cfg = config.load()["liwc"]
    dic = load_dictionary()
    (c_w, n_w), (c_t, n_t) = (count(data.load(p)[data.TEXT_RAW], dic) for p in PLATFORMS)
    df = compare(c_w, n_w, c_t, n_t, cfg["alpha"])
    out = config.results_dir("06_liwc")
    df.to_csv(out / "liwc_comparison.csv", index=False)
    summary = {"categories_compared": int(len(df)), "tokens_FW.Br": n_w, "tokens_FT.Br": n_t,
               "significant": int(df["significant"].sum()),
               "significant_holm": int(df["significant_holm"].sum()),
               "phi_max": float(df["phi"].max()), "phi_median": float(df["phi"].median())}
    pd.Series(summary).to_json(out / "liwc_summary.json", indent=2)
    return df
