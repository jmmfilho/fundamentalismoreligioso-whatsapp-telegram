# Notebooks

`legacy/` holds the original notebooks used to produce the paper, **kept
unchanged** (only a phone number in printed outputs was redacted) so the
published numbers can be traced to their source. Their logic now lives in
`src/fundrel/`; use the package to re-run the analyses.

| Notebook | Paper section | Package module |
|---|---|---|
| `analise_dos_dados.ipynb` | 3.1 (filters, Figure 2) | `data`, `descriptive` |
| `Reconhecimento_entidade.ipynb` | 4.1 | `ner` |
| `LDA_wpp.ipynb`, `LDA_telegram.ipynb` | 4.4 | `topics` (LDA) |
| `pseudo_wpp.ipynb`, `pseudo_telegram.ipynb` | 4.4 | `topics` (PTM) |
| `bertopic_wpp.ipynb`, `bertopic_telegram.ipynb` | 4.4 | `topics` (BERTopic) |
| `LIWC.ipynb` | 4.5 | `liwc` (see REPRODUCIBILITY_NOTES §6) |

Each topic and LIWC notebook runs three variants: no filter, religious filter
only, and religious filter + exclusion filter. **Only the last variant is
used in the paper.**
