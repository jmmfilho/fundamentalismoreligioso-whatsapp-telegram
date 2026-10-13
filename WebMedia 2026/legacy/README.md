# Legacy material

Kept for the record; **not needed to reproduce the paper** (use the notebooks in `notebooks/`).

* `notebooks/` — the original analysis notebooks, unchanged except for a phone number
  redacted from printed outputs. Each topic/LIWC notebook runs three variants (no filter,
  religious filter, religious + exclusion filter); only the last one is used in the paper.
  `pipeline_sbert_topics.ipynb` is an unfinished experiment.
* `results/paper_run/` — outputs of those notebooks for the configuration in the paper.
* `results/unused_experiments/` — results not reported in the paper: variants without
  filters or with the religious filter only, Perspective API toxicity scores, association
  rules and sentiment histograms.

See `docs/REPRODUCIBILITY_NOTES.md` for how these relate to the values in the paper.
