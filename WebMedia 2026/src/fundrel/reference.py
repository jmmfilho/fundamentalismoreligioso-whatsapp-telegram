"""Values reported in the WebMedia 2026 paper, used by `verify` to compare a
fresh run against the published numbers."""

PAPER = {
    "datasets": {
        "FW.Br": {"raw_messages": 598971, "raw_users": 15286, "raw_groups": 240,
                  "after_exclusion_filter": 5167, "unique_messages": 4329,
                  "tokens_mean": 82.34, "tokens_std": 99.64, "tokens_min": 3, "tokens_max": 1683},
        "FT.Br": {"raw_messages": 557586, "raw_users": 14810, "raw_groups": 178,
                  "after_exclusion_filter": 8526, "unique_messages": 6462,
                  "tokens_mean": 88.57, "tokens_std": 116.87, "tokens_min": 3, "tokens_max": 1549},
    },
    "polarity": {
        "FW.Br": {"n": 5167, "mean": 0.270, "sd": 0.551, "median": 0.365, "q1": -0.052, "q3": 0.746,
                  "pct_negative": 25.4, "pct_neutral": 9.9, "pct_positive": 64.8},
        "FT.Br": {"n": 8526, "mean": 0.230, "sd": 0.570, "median": 0.302, "q1": -0.178, "q3": 0.731,
                  "pct_negative": 28.4, "pct_neutral": 10.8, "pct_positive": 60.8},
        "mann_whitney_U": 22820276, "mann_whitney_p": 4.0e-4, "cliffs_delta": 0.036,
        "cliffs_delta_ci": (0.016, 0.056), "chi2": 21.28, "cramers_v": 0.039,
        "negative_diff_pp": 3.0, "group_level_p": 0.61,
    },
    # (model, dataset): k, C_NPMI, C_V, IRBO, diversity
    "topics": {
        ("lda", "FW.Br"): {"k": 6, "c_npmi": 0.002, "c_v": 0.530, "irbo": 0.41, "diversity": 0.48},
        ("lda", "FT.Br"): {"k": 2, "c_npmi": -0.008, "c_v": 0.514, "irbo": 0.58, "diversity": 0.80},
        ("ptm", "FW.Br"): {"k": 2, "c_npmi": 0.061, "c_v": 0.610, "irbo": 0.70, "diversity": 0.95},
        ("ptm", "FT.Br"): {"k": 28, "c_npmi": 0.024, "c_v": 0.563, "irbo": 0.94, "diversity": 0.73},
        ("bertopic", "FW.Br"): {"k": 5, "c_npmi": 0.231, "c_v": 0.659, "irbo": 0.98, "diversity": 0.84},
        ("bertopic", "FT.Br"): {"k": 26, "c_npmi": 0.155, "c_v": 0.641, "irbo": 0.99, "diversity": 0.78},
    },
    "liwc": {
        "categories_compared": 63, "significant": 49, "significant_holm": 40, "phi_max": 0.033,
        "phi_median": 0.008,
        "pct": {"time": (6.87, 8.77), "relig": (12.20, 10.11), "humans": (13.92, 11.85),
                "money": (2.50, 3.38), "we": (0.74, 0.41)},
    },
}
