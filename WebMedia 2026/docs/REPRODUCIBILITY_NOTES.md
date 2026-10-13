# Reproducibility notes

This document records how the code in this repository relates to the numbers
published in the WebMedia 2026 paper, including every place where the original
analysis notebooks (kept unchanged in `notebooks/legacy/`) differ from the
paper text or from each other. It was written while consolidating the
notebooks into the `fundrel` package and should be kept up to date.

Status legend: **verified** = value appears in a saved notebook output;
**code available, not verified** = the package implements the analysis but no
saved output exists to compare; **not in repository** = no code or output for
this result was found.

## 1. Overview

| Paper result | Status | Where |
|---|---|---|
| Raw dataset sizes (598,971 / 557,586 messages) | verified | `notebooks/legacy/LIWC.ipynb` |
| Collection period (27 Sep – 14 Nov 2022) | verified (WhatsApp) | `notebooks/legacy/LDA_wpp.ipynb` |
| Users and groups (15,286 / 240; 14,810 / 178) | code available, not verified | `fundrel.data.build` |
| Filtered datasets (5,167 / 8,526) | verified | `notebooks/legacy/analise_dos_dados.ipynb` |
| Table 1: unique messages and token statistics | code available, not verified | `fundrel.descriptive` |
| Figure 2: messages per day | verified | `results/legacy_paper_run/figures/` |
| Table 2: named entities | verified (see §4) | `notebooks/legacy/Reconhecimento_entidade.ipynb` |
| Sec. 4.2: retrieval with Elasticsearch | not in repository (re-implemented) | `fundrel.search` |
| Table 3 and Sec. 4.3: polarity descriptives and tests | ported from the revision analysis (see §7) | `fundrel.polarity` |
| Table 4: number of topics | verified | LDA/PTM/BERTopic notebooks |
| Table 5: C_V, diversity, IRBO | verified | idem |
| Table 5: C_NPMI for LDA and PTM | **inconsistent** (see §5) | idem |
| Tables 6–7: topic words | verified | `results/legacy_paper_run/` |
| Sec. 4.5: LIWC (49/63, Holm, φ, category %) | ported from the revision analysis; **not** what the legacy notebook computes (see §6) | `fundrel.liwc` |

## 2. Dataset construction

The following steps are applied before the keyword filters. They are part of
the paper run but are **not described in the paper**:

1. **Excluded accounts.** 11 hashed member IDs (WhatsApp) and 25 (Telegram) are
   removed. They are listed in `config/params.yaml`. The reason for excluding
   them (e.g. bots or the research team's own monitoring accounts) should be
   documented here by the authors.
2. **Messages without text** (`text_content_anonymous` missing) are dropped.
3. **"Trava-zap" messages** (`trava_zap == True`) are dropped.

After these steps 254,480 WhatsApp and 443,759 Telegram messages remain; the
keyword filters are then applied to the preprocessed text.

**Keyword matching is substring-based.** A message is kept/dropped if a keyword
occurs anywhere in the preprocessed text, not as a whole word. Short exclusion
terms therefore also match inside other words (`pt` in `adapto`, `pl` in
`exemplo`, `13` in `2013`), and `reino` matches `treino`. This is the behaviour
that produced 5,167 and 8,526 messages and is the default
(`filters.match: substring`). Setting `match: word` gives whole-word matching
and different counts.

**Preprocessing happens before filtering** (accents and stopwords removed), so
keywords are written without accents (`misericordia`, `salomao`). `carismatic`
is a stem that matches *carismático/carismática*.

## 3. Statistics in Table 1

The definitions of "unique messages" and of the token counts are not stated in
the notebooks. `fundrel.descriptive` uses: unique = distinct raw texts; tokens =
whitespace tokens of the preprocessed text of the unique messages. Confirm
against the paper values with `python -m fundrel verify`.

## 4. Named entity recognition

* NER runs on the **preprocessed** text (lowercase, no accents, no stopwords).
  This is why XLM-RoBERTa produces fragments such as `je` / `sus` and why the
  spaCy list contains common nouns (`misericordia`, `familia`, `acredito`).
* The transformer model is `Davlan/xlm-roberta-base-ner-hrl` (the paper only
  says "XLM-RoBERTa"; consider naming the checkpoint).
* The two XLM-RoBERTa columns in Table 2 come from different settings: FW.Br
  texts were truncated to 1,000 characters, FT.Br texts were not. Both settings
  are reproduced through `ner.max_chars` in the config.
* Table 2 translates `acredito` ("I believe") as *belief*.

## 5. Topic modeling

* `k` is swept over 2, 4, …, 28 (`range(2, 30, 2)`), not every value from 2 to 30.
* **C_NPMI is not always from the reported model.** The notebooks select a
  separate model for each coherence measure. Table 5 reports C_V, diversity
  and IRBO of the C_V-selected model but, for three cells, the C_NPMI of the
  NPMI-selected model:

  | Model · dataset | k (C_V) | C_NPMI in paper | C_NPMI of the k chosen by C_V (notebook output) |
  |---|---|---|---|
  | LDA · FW.Br | 6 | 0.002 (k = 2) | −0.0074 |
  | LDA · FT.Br | 2 | −0.008 (k = 20) | −0.0845 |
  | PTM · FT.Br | 28 | 0.024 (k = 6) | 0.0208 |

  The conclusions (BERTopic has the highest coherence) do not change. The
  package reports every metric for the C_V-selected model.
* **Random seeds.** LDA was trained without `random_state`; the package now
  sets `seed: 42`, so LDA results will differ slightly from the paper.
  PTM used `seed=42` but multiple workers, which tomotopy warns is not
  deterministic; the package trains with one worker. BERTopic fixes the UMAP
  seed.
* **Coherence top-n differs across models.** LDA coherence used gensim's
  default (top 20 words); PTM and BERTopic used the top 10 words.
* **Two IRBO implementations.** LDA and PTM used the `rbo` package; BERTopic
  used a truncated RBO written in the notebook (p = 0.9). The package computes
  both for every model (`irbo_rbo_package`, `irbo_truncated`) and `verify`
  compares each model with the one used in the paper.
* The paper says the metrics were computed "as implemented in the OCTIS
  library"; the notebooks use gensim's `CoherenceModel` and the implementations
  above, not OCTIS.

## 6. LIWC

The values in Section 4.5 (63 categories, 49 significant, 40 after
Holm-Bonferroni, φ ≤ 0.033 with median 0.008, *relig* 12.20% vs 10.11%, *anger*
p = 0.41, *anx* p = 0.46) were produced during the reviewer-response phase
(August 2026) by a separate script run on FW.Br and FT.Br, not by
`notebooks/legacy/LIWC.ipynb`. That script is now `fundrel.liwc`:

* text: raw `text_content_anonymous` (accents kept);
* tokens: NLTK `word_tokenize(text.lower(), language="portuguese")`, minus
  NLTK's Portuguese stopwords and single punctuation characters (the project's
  custom stopwords are **not** removed here); totals ≈ 104k tokens (FW.Br) and
  190k (FT.Br);
* dictionary: header parsed separately, entries lowercased, wildcard entries
  matched by longest prefix, exact entries first;
* per category: 2x2 chi-square without continuity correction, φ = √(χ²/N),
  95% CI of the difference in proportions, Holm-Bonferroni adjustment;
  categories are the union of those observed in either dataset.

The absolute shares (e.g. *relig* and *swear* around 12%) are high because
wildcard prefixes are broad; when the analysis was run, this was flagged as a
reason to interpret magnitudes and ratios rather than absolute values.

The legacy notebook, kept for the record, differs from the analysis above and
has these problems:

1. In the section "Filtro religioso com exclusão de palavras políticas", the
   WhatsApp data receives only the religious filter (cell 53), while Telegram
   receives both filters. The file
   `results/legacy_paper_run/liwc/comparacao_proporcoes_categorias_filtroPolitico.csv`
   therefore compares FT.Br with a WhatsApp set that is **not** FW.Br.
2. The dictionary parser treats the category header (lines between `%`) as
   words, producing spurious categories (`1`, `121`, `358`, `article`, `Unknown`).
3. Tokens are matched exactly after accent removal, so the ~9,000 wildcard
   entries (`abafad*`) and the ~31,000 accented entries never match.
4. The per-dataset "tests" (cells 18, 37, 41, 51, 55) build the table
   `[[p, 100-p], [100-p, p]]`, which does not test anything meaningful; only
   the cross-platform comparisons (cells 43, 45, 57, 59) are valid tests.

The earlier version of the paper reported "62 of 64 categories"; that figure
came from this notebook and was replaced by the values above.

## 7. Polarity

`score_sentiment` is the LeIA score stored in the BATMAN export (computed at
ingestion), not recomputed here. The tests in Section 4.3 were produced in the
same reviewer-response analysis and are ported to `fundrel.polarity`:

* categories: negative < −0.05, neutral [−0.05, 0.05], positive > 0.05;
* Mann-Whitney U (two-sided, asymptotic) and Cliff's δ, FW.Br vs FT.Br, with
  the 95% CI from Cliff's (1993) consistent variance estimator (not bootstrap);
  the reported U (22,820,276) is the statistic for FW.Br, consistent with
  δ = 0.036 for n = 5,167 and 8,526 (no missing scores);
* χ² on the 2 × 3 category table and Cramér's V; difference in the share of
  negative messages: +3.0 pp in FT.Br (95% CI [1.5, 4.5]);
* robustness (footnote): median score per group, groups with ≥ 10 messages
  (p = 0.61) and ≥ 30 messages (p = 0.70); unique messages only.

## 8. Data protection

* **`id_persona` contains a WhatsApp phone number** (the account used for
  collection) in clear text in `fakeWhatsApp.BR_2022.csv`; the Telegram file
  contains a persona name. This conflicts with the statement in the paper that
  phone numbers were pseudonymised. The number has been redacted from the
  saved notebook outputs, but it is still in the dataset and in the Git (LFS)
  history. Recommended: publish a new version of both CSV files with
  `id_persona` hashed or dropped and remove the old LFS objects from the
  repository history.
* Files produced by `python -m fundrel retrieval` contain message text,
  including the "exposed" message with a private person's business details.
  They are written to `results/generated/` (git-ignored) and must not be
  committed.
* `data/processed/*_message_ids.csv` lists the selected messages without text
  so that the selection can be shared and audited.

## 9. Third-party resources

`data/external/LIWC2007_Portugues_win.dic` is the Brazilian Portuguese LIWC
2007 dictionary (Balage Filho et al., 2013). Check whether its licence allows
redistribution; if not, remove it from the repository and replace it with
download instructions in `data/README.md`.

## 10. Environment

The exact package versions used in the paper run were not recorded. If the
original environment still exists, export it (`pip freeze >
requirements-paper.txt`) and commit it next to `requirements.txt`.
