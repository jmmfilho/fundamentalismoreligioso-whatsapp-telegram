"""Compare generated results (results/generated) with the values in the paper."""
from __future__ import annotations

import json

import pandas as pd

from . import config
from .reference import PAPER


def _row(section, item, paper, got, tol):
    ok = None if got is None else abs(float(got) - float(paper)) <= tol
    return {"section": section, "item": item, "paper": paper,
            "generated": None if got is None else round(float(got), 4),
            "status": "missing" if ok is None else ("ok" if ok else "DIFFERENT")}


def run() -> pd.DataFrame:
    res = config.path("results")
    rows = []
    f = res / "01_datasets" / "dataset_statistics.json"
    if f.exists():
        got = json.loads(f.read_text())
        for ds, ref in PAPER["datasets"].items():
            flat = {**got.get(ds, {}), **got.get(ds, {}).get("filtering_steps", {})}
            for k, v in ref.items():
                rows.append(_row("Table 1 / Sec. 3.1", f"{ds} {k}", v, flat.get(k), 0.005))
    f = res / "04_polarity" / "polarity_statistics.json"
    if f.exists():
        got = json.loads(f.read_text())
        for ds in ("FW.Br", "FT.Br"):
            for k, v in PAPER["polarity"][ds].items():
                rows.append(_row("Table 3", f"{ds} {k}", v, got[ds].get(k), 0.05 if k.startswith("pct") else 0.0005))
        p = PAPER["polarity"]
        ml, cat = got["message_level"], got["categories"]
        rows += [_row("Sec. 4.3", "Mann-Whitney U", p["mann_whitney_U"], ml["U"], 0.5),
                 _row("Sec. 4.3", "Mann-Whitney p", p["mann_whitney_p"], ml["p"], 0.05e-4),
                 _row("Sec. 4.3", "Cliff's delta", p["cliffs_delta"], ml["cliffs_delta"], 0.0005),
                 _row("Sec. 4.3", "Cliff's delta CI low", p["cliffs_delta_ci"][0], ml["cliffs_delta_ci95"][0], 0.0005),
                 _row("Sec. 4.3", "Cliff's delta CI high", p["cliffs_delta_ci"][1], ml["cliffs_delta_ci95"][1], 0.0005),
                 _row("Sec. 4.3", "chi2(2)", p["chi2"], cat["chi2"], 0.005),
                 _row("Sec. 4.3", "Cramer's V", p["cramers_v"], cat["cramers_v"], 0.0005),
                 _row("Sec. 4.3", "negative share diff (pp)", p["negative_diff_pp"],
                      cat["negative_diff_pp_FT_minus_FW"], 0.05),
                 _row("Sec. 4.3 fn.", "group-level p (>=10 msgs)", p["group_level_p"],
                      got["group_level"]["min_10"]["p"], 0.005)]
    f = res / "05_topics" / "topic_quality.csv"
    if f.exists():
        got = pd.read_csv(f).set_index(["model", "dataset"])
        for key, ref in PAPER["topics"].items():
            g = got.loc[key] if key in got.index else {}
            for k, v in ref.items():
                col = {"irbo": "irbo_truncated" if key[0] == "bertopic" else "irbo_rbo_package"}.get(k, k)
                rows.append(_row("Table 5", f"{key[0]} {key[1]} {k}", v, g.get(col) if len(g) else None,
                                 0 if k == "k" else 0.0051))
    f = res / "06_liwc" / "liwc_summary.json"
    if f.exists():
        got = json.loads(f.read_text())
        for k in ("categories_compared", "significant", "significant_holm", "phi_max", "phi_median"):
            rows.append(_row("Sec. 4.5", k, PAPER["liwc"][k], got.get(k), 0.0005))
        cmp = pd.read_csv(res / "06_liwc" / "liwc_comparison.csv").set_index("category")
        for cat, (fw, ft) in PAPER["liwc"]["pct"].items():
            rows.append(_row("Sec. 4.5", f"{cat} % FW.Br", fw, cmp["pct_FW.Br"].get(cat), 0.005))
            rows.append(_row("Sec. 4.5", f"{cat} % FT.Br", ft, cmp["pct_FT.Br"].get(cat), 0.005))
    report = pd.DataFrame(rows)
    report.to_csv(res / "verification_report.csv", index=False)
    return report
