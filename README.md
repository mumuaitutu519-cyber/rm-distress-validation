# RM distress validation — public code, schema and aggregate results

Han Sun, Jing Chen and Doojin Ryu, *Behavioral validation of generative corporate distress scores: A seven-dimensional protocol*. Manuscript: 2026.11.09 sun21.

This public repository contains the current analysis and rebuild code, field dictionary, benchmark configuration, model revision/weight metadata, table/figure mappings, and the supplied manuscript PDF in `paper/`. The PDF is copied unchanged. Code and numerical results have not been revised for this release; no model inference or statistical computation was rerun.

This public repository now includes aggregate numerical values for all manuscript tables and figures, together with cell-by-cell verification results: **561/568 table-body statistics match** in sun21; differences and coverage limits are documented in [MISMATCH_REPORT_sun21.md](verify/MISMATCH_REPORT_sun21.md). Use [the sun21 table/figure map](TABLE_FIGURE_MAP_sun21.csv), [table CSV/TeX files](output/tables/), [figures](output/figures/), [screened aggregate analysis CSVs](output/analysis/) and [verification CSVs by table](verify/VERIFICATION_sun21/) to check each result. The 793-row verification source is partitioned to keep each CSV at or below 500 rows; six individual-example score/probability records and the prompt-text CSV for non-numerical Table A7 are excluded. The mismatch report is a documented public excerpt. Company-level data are not public because of licensing restrictions; authorized review access may be requested from the corresponding author. All retained result values are unchanged; no statistical computation or inference was rerun.

## Data access and omitted files

CSMAR and other authorized databases restrict redistribution. All company-level original or derived financial/control/return/status/regulatory values and company-level input files are omitted. Company prompts, original replies and parsed scores, predictions, frozen bootstrap draws and case audit records are also omitted. Only screened aggregate output snapshots and verification records are included. Logs, restricted case material, prompt text, extracted manuscript artifacts and private inventories remain excluded; the filename/column and 500-row filters also conservatively exclude some aggregate CSVs. [EXCLUDED_FILES.md](EXCLUDED_FILES.md) lists every omitted source path, and [DATA_REVIEW.json](DATA_REVIEW.json) records the public-edition selection.

Researchers may contact the corresponding author through the contact information in the manuscript to request review access to restricted replication materials, subject to the relevant database authorization and institutional rights. This is not permission to redistribute licensed data. The complete review repository remains private; this public repository has a separate, newly initialized Git history.

## Reproduction requirements

This edition alone cannot reproduce the manuscript tables or run the complete offline audit: required licensed inputs, archived inference records, frozen draws and reference artifacts are absent. No substitute observations are supplied; the published aggregate snapshots support inspection, while complete recomputation requires the restricted inputs. An authorized researcher must restore the restricted materials described by `data/DATA_DICTIONARY.csv`, the source scripts and `TABLE_FIGURE_MAP_sun21.csv` before using the unchanged runner. Obtain source observations under your institution's database rights.

After those materials have been restored, the original Python 3.12 entry point is:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
```

The archived source documentation and code refer to paths in the full private package; those references do not indicate that the files are distributed here. `raw_rebuild/` preserves source construction and inference reference code. Model weights and serialized fitted models are not distributed. `MANIFEST.csv` and `CHECKSUMS.sha256` describe this public tree, excluding the two self-referential inventory files.
