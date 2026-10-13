# Data

## raw/ (Git LFS)

| File | Platform | Messages | Period |
|---|---|---|---|
| `fakeWhatsApp.BR_2022.csv` | WhatsApp | 598,971 | 27 Sep – 14 Nov 2022 |
| `fakeTelegram.BR_2022.csv` | Telegram | 557,586 | Sep – Nov 2022 |

Both are exports of the BATMAN monitoring platform (de Sá et al., 2023) for
public groups associated with the Brazilian far right. Download them with
`git lfs install && git lfs pull`; without LFS the files are 3-line pointers
and the pipeline stops with an explicit error.

Columns (20):

| Column | Description |
|---|---|
| `date_message` | timestamp of the message |
| `id_member_anonymous` | hashed author ID |
| `id_group_anonymous` | hashed group ID |
| `text_content_anonymous` | message text |
| `media`, `media_type`, `media_url`, `media_name`, `media_md5` | attached media metadata |
| `has_media`, `has_media_url` | media flags |
| `trava_zap` | message flagged as a "trava-zap" (crash/flood message) |
| `score_sentiment` | LeIA sentiment score in [−1, 1] computed at ingestion (used in Sec. 4.3) |
| `score_misinformation` | misinformation score from the platform (not used in the paper) |
| `id_message` | message ID |
| `id_persona` | collection account (**see note below**) |
| `message_type`, `messenger`, `dataset_info_id`, `date_system` | platform metadata |

> **Note.** In the WhatsApp file `id_persona` holds the phone number of the
> collection account in clear text. See `docs/REPRODUCIBILITY_NOTES.md` §8.

## external/

`LIWC2007_Portugues_win.dic`: Brazilian Portuguese LIWC 2007 dictionary
(Balage Filho, Pardo and Aluísio, 2013), Latin-1 encoded. Verify the licence
terms before redistributing.

## processed/

Created by `python -m fundrel datasets`:

* `fw_br.csv.gz`, `ft_br.csv.gz` — the filtered datasets with the
  `text_processed` column (git-ignored, they contain message text);
* `FW.Br_message_ids.csv`, `FT.Br_message_ids.csv` — IDs, group hash and date
  of every selected message (safe to share; lets others check the selection).
