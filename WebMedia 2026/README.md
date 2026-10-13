# Religious Fundamentalism and Far-Right Ideology in Brazilian WhatsApp and Telegram Groups

Code and data for the paper

> F. Nascimento, M. S. Felipe, L. Torres, I. Claudino, J. M. Monteiro, J. Machado.
> **Religious Fundamentalism and Far-Right Ideology in Brazilian WhatsApp and Telegram Groups: A Multidimensional Computational Analysis.**
> Proceedings of the Brazilian Symposium on Multimedia and the Web (WebMedia 2026), Lavras/MG, Brazil.

The study builds two datasets of religious messages shared in far-right public
groups — **FW.Br** (WhatsApp) and **FT.Br** (Telegram), collected from
27 Sep to 14 Nov 2022 — and analyses them with named entity recognition,
message retrieval, polarity analysis, three topic models (LDA, PTM, BERTopic)
and LIWC.

![Methodology overview](docs/figures/methodology_overview.png)

## Repository layout

```
config/params.yaml        every parameter used in the paper (keywords, excluded IDs, models, k range, seeds)
src/fundrel/              the pipeline, one module per analysis
  data.py                 dataset construction (Sec. 3.1)
  preprocessing.py        text preprocessing shared by all analyses
  descriptive.py          Table 1, Figure 2
  ner.py                  Table 2
  search.py               message retrieval for manual analysis (Sec. 4.2)
  polarity.py             Table 3 and statistical tests (Sec. 4.3)
  topics.py               Tables 4–7 (LDA, PTM, BERTopic)
  liwc.py                 psycholinguistic analysis (Sec. 4.5)
  verify.py, reference.py compare a fresh run with the numbers in the paper
data/raw/                 BATMAN exports of WhatsApp and Telegram groups (Git LFS)
data/external/            LIWC 2007 Brazilian Portuguese dictionary
data/processed/           FW.Br / FT.Br produced by the pipeline (only message-ID lists are versioned)
results/legacy_paper_run/ outputs of the original notebooks that feed the paper
results/generated/        outputs of `python -m fundrel ...` (git-ignored)
notebooks/legacy/         original analysis notebooks, kept unchanged for traceability
archive/                  earlier experiments that are not part of the paper
docs/                     reproducibility notes and figures
tests/                    unit tests (run without the data)
```

## Quick start

```bash
git clone https://github.com/jmmfilho/fundamentalismoreligioso-whatsapp-telegram
cd fundamentalismoreligioso-whatsapp-telegram
git lfs install && git lfs pull        # downloads data/raw/*.csv (~520 MB)

python -m venv .venv && source .venv/bin/activate
make setup                             # dependencies, spaCy model, NLTK data
make all                               # runs every step, then `verify`
```

Each step can also be run on its own (`make datasets`, `make topics`, …) or as
`python -m fundrel <step>`. Steps reuse the processed datasets, so
`datasets` must run first. Topic modeling and NER are the slow steps; a GPU
speeds up NER and BERTopic.

## From the paper to the code

| Paper | Command | Output in `results/generated/` |
|---|---|---|
| Sec. 3.1, Table 1, Figure 2 | `python -m fundrel datasets` | `01_datasets/dataset_statistics.json`, `messages_per_day.png` |
| Sec. 4.1, Table 2 | `python -m fundrel ner` | `02_ner/top_entities.csv` |
| Sec. 4.2 | `python -m fundrel retrieval` | `03_retrieval/*_most_shared.csv`, `*_retrieved_by_entity.csv` |
| Sec. 4.3, Table 3 | `python -m fundrel polarity` | `04_polarity/polarity_statistics.json` |
| Sec. 4.4, Tables 4–7 | `python -m fundrel topics` | `05_topics/topic_quality.csv`, `*_topics.csv`, `*_k_sweep.csv` |
| Sec. 4.5 | `python -m fundrel liwc` | `06_liwc/liwc_comparison.csv`, `liwc_summary.json` |
| all of the above | `python -m fundrel verify` | `verification_report.csv` (paper value vs. generated value) |

The original notebooks that produced the published numbers are in
`notebooks/legacy/`, and their saved outputs are in
`results/legacy_paper_run/`. **Read [`docs/REPRODUCIBILITY_NOTES.md`](docs/REPRODUCIBILITY_NOTES.md)
before comparing results**: it lists which numbers were verified against saved
outputs, which steps are not described in the paper, and the known
differences between the notebooks and the paper.

## Dataset construction in short

1. Remove messages from the excluded member IDs in `config/params.yaml`, messages without text and "trava-zap" messages.
2. Preprocess: URLs → domain, lowercase, remove accents, punctuation and stopwords, normalise laughter and repeated emojis.
3. Keep messages that contain at least one of 18 religious keywords (intersection of lists from three professionals with degrees in Theology).
4. Drop messages that contain any of 25 election-campaign keywords.

| | Raw messages | Users | Groups | After filters | Unique |
|---|---|---|---|---|---|
| FW.Br (WhatsApp) | 598,971 | 15,286 | 240 | 5,167 | 4,329 |
| FT.Br (Telegram) | 557,586 | 14,810 | 178 | 8,526 | 6,462 |

## Data and ethics

The datasets contain messages from **public** WhatsApp and Telegram groups,
collected with the BATMAN platform, with member and group identifiers
pseudonymised by cryptographic hashing (LGPD, Law No. 13,709/2018; CNS
Resolutions No. 510/2016, 466/2012 and 674/2022). Use them for research only,
do not attempt to re-identify individuals and do not republish message text
that identifies private persons. See `data/README.md` for the data dictionary
and `docs/REPRODUCIBILITY_NOTES.md` §8 for open data-protection items.

## Tests

```bash
make test        # runs on synthetic data; does not need the datasets
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## Acknowledgments

Lenovo and the National Institute of Science and Technology for Combating
Prejudice (INCT Preconceitos).
