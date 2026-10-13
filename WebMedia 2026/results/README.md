# Results

* `legacy_paper_run/` — outputs saved by the original notebooks for the
  configuration used in the paper (religious inclusion filter + exclusion
  filter): topic words for every coherence-selected model, the LIWC comparison
  file and the messages-per-day figure. The models reported in the paper are
  the `c_v` files for LDA and PTM and the `*_filtro_politico.csv` files for
  BERTopic. See `docs/REPRODUCIBILITY_NOTES.md` §6 about the LIWC file.
* `generated/` — created by `python -m fundrel <step>` (git-ignored).
  `verification_report.csv` compares every number with the paper.
