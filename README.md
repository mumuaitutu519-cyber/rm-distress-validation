# RM distress validation — public code and schema

Han Sun, Jing Chen and Doojin Ryu, *Behavioral validation of generative corporate distress scores: A seven-dimensional protocol*. Manuscript: 2026.11.09 sun21.

This public repository contains the current analysis and rebuild code, field dictionary, benchmark configuration, model revision/weight metadata, table/figure mappings, and the supplied manuscript PDF in `paper/`. The PDF is copied unchanged. Code and numerical results have not been revised for this release; no model inference or statistical computation was rerun.

## Data access and omitted files

CSMAR and other authorized databases restrict redistribution. All company-level original or derived financial/control/return/status/regulatory values and company-level input files are omitted. Company prompts, original replies and parsed scores, predictions, frozen bootstrap draws and case audit records are also omitted. Numerical output snapshots, logs, extracted manuscript/audit artifacts and private inventories are conservatively excluded. [EXCLUDED_FILES.md](EXCLUDED_FILES.md) lists every omitted source path, and [DATA_REVIEW.json](DATA_REVIEW.json) records the public-edition selection.

Researchers may contact the corresponding author through the contact information in the manuscript to request review access to restricted replication materials, subject to the relevant database authorization and institutional rights. This is not permission to redistribute licensed data. The complete review repository remains private; this public repository has a separate, newly initialized Git history.

## Reproduction requirements

This edition alone cannot reproduce the manuscript tables or run the complete offline audit: required licensed inputs, archived inference records, frozen draws and reference artifacts are absent. No substitute observations or result snapshots are supplied. An authorized researcher must restore the restricted materials described by `data/DATA_DICTIONARY.csv`, the source scripts and `TABLE_FIGURE_MAP_sun21.csv` before using the unchanged runner. Obtain source observations under your institution's database rights.

After those materials have been restored, the original Python 3.12 entry point is:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
```

The archived source documentation and code refer to paths in the full private package; those references do not indicate that the files are distributed here. `raw_rebuild/` preserves source construction and inference reference code. Model weights and serialized fitted models are not distributed. `MANIFEST.csv` and `CHECKSUMS.sha256` describe this public tree, excluding the two self-referential inventory files.
