#!/usr/bin/env python3
"""Build target-aligned validation profiles for the four frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "output" / "analysis"
LOGS = ROOT / "output" / "logs"
VARIANTS = ["strict_canonical", "strict_order_reversed", "strict_unit_decimal"]
FILES = {
    "Qwen3-4B 4-bit": "strict_timing_core_qwen3_4b_4bit_batch1.csv",
    "Qwen3-4B bfloat16": "strict_timing_core_qwen3_4b_fp16_batch1.csv",
    "Qwen3-8B 4-bit": "strict_timing_core_qwen3_8b_4bit_batch1.csv",
    "Mistral-NeMo-12B 4-bit": "strict_timing_core_mistral_nemo_12b_4bit_batch1.csv",
}


def profile(name: str, filename: str) -> tuple[dict[str, object], pd.DataFrame]:
    data = pd.read_csv(ANALYSIS / filename)
    if len(data) != 192 or not data["parse_pd"].all():
        raise AssertionError(f"Incomplete artifact audit: {name}")
    scores = data.pivot(index="case_id", columns="variant", values="pd_percent")[VARIANTS]
    y = data.drop_duplicates("case_id").set_index("case_id")["distress_f1"].loc[scores.index]
    ranges = scores.max(axis=1) - scores.min(axis=1)
    rows = []
    for variant in VARIANTS:
        rows.append(
            {
                "artifact": name,
                "variant": variant,
                "mean_score": float(scores[variant].mean()),
                "roc_auc": float(roc_auc_score(y, scores[variant])),
                "pr_auc": float(average_precision_score(y, scores[variant])),
                "unique_scores": int(scores[variant].nunique()),
            }
        )
    summary = {
        "artifact": name,
        "cases": len(scores),
        "events": int(y.sum()),
        "parse_rate": float(data["parse_pd"].mean()),
        "canonical_auc": float(roc_auc_score(y, scores["strict_canonical"])),
        "canonical_pr_auc": float(average_precision_score(y, scores["strict_canonical"])),
        "median_representation_range_pp": float(ranges.median()),
        "mean_representation_range_pp": float(ranges.mean()),
        "any_representation_change_rate": float(ranges.gt(0).mean()),
        "maximum_representation_range_pp": float(ranges.max()),
    }
    case_output = scores.reset_index()
    case_output["artifact"] = name
    case_output["outcome"] = y.to_numpy(int)
    case_output["representation_range_pp"] = ranges.to_numpy(float)
    return summary, pd.DataFrame(rows), case_output


def main() -> None:
    summaries = []
    variants = []
    case_rows = []
    for name, filename in FILES.items():
        summary, variant, case_output = profile(name, filename)
        summaries.append(summary)
        variants.append(variant)
        case_rows.append(case_output)
    profiles = pd.DataFrame(summaries)
    variant_metrics = pd.concat(variants, ignore_index=True)
    cases = pd.concat(case_rows, ignore_index=True)

    q4 = pd.read_csv(ANALYSIS / FILES["Qwen3-4B 4-bit"])
    fp = pd.read_csv(ANALYSIS / FILES["Qwen3-4B bfloat16"])
    quant = q4.merge(fp, on="run_id", suffixes=("_4bit", "_bfloat16"), validate="one_to_one")
    quant["difference_pp"] = quant["pd_percent_4bit"] - quant["pd_percent_bfloat16"]
    quant_summary = pd.DataFrame(
        [
            {
                "calls": len(quant),
                "changed_rate": float(quant["difference_pp"].ne(0).mean()),
                "mean_difference_pp": float(quant["difference_pp"].mean()),
                "mean_absolute_difference_pp": float(quant["difference_pp"].abs().mean()),
                "maximum_absolute_difference_pp": float(quant["difference_pp"].abs().max()),
            }
        ]
    )

    b1 = pd.read_csv(ANALYSIS / FILES["Qwen3-4B 4-bit"])
    primary = pd.read_csv(ANALYSIS / "strict_timing_primary_qwen3_4b_4bit.csv")
    primary = primary.loc[primary["case_id"].isin(set(b1["case_id"]))]
    batch = b1.merge(primary, on="run_id", suffixes=("_batch1", "_batch8"), validate="one_to_one")
    batch["difference_pp"] = batch["pd_percent_batch1"] - batch["pd_percent_batch8"]
    batch_summary = pd.DataFrame(
        [
            {
                "calls": len(batch),
                "changed_rate": float(batch["difference_pp"].ne(0).mean()),
                "mean_absolute_difference_pp": float(batch["difference_pp"].abs().mean()),
                "maximum_absolute_difference_pp": float(batch["difference_pp"].abs().max()),
            }
        ]
    )

    reload = pd.read_csv(ANALYSIS / "strict_timing_core_qwen3_4b_4bit_batch1_reload.csv")
    repeated = b1.merge(reload, on="run_id", suffixes=("_first", "_reload"), validate="one_to_one")
    repeat_summary = pd.DataFrame(
        [
            {
                "calls": len(repeated),
                "raw_output_exact_match_rate": float(
                    repeated["raw_output_first"].eq(repeated["raw_output_reload"]).mean()
                ),
                "score_changed_rate": float(
                    repeated["pd_percent_first"].ne(repeated["pd_percent_reload"]).mean()
                ),
                "maximum_absolute_difference_pp": float(
                    (repeated["pd_percent_first"] - repeated["pd_percent_reload"]).abs().max()
                ),
            }
        ]
    )

    profiles.to_csv(ANALYSIS / "strict_timing_artifact_profiles.csv", index=False)
    variant_metrics.to_csv(ANALYSIS / "strict_timing_artifact_variant_metrics.csv", index=False)
    cases.to_parquet(ANALYSIS / "strict_timing_artifact_case_ranges.parquet", index=False)
    quant_summary.to_csv(ANALYSIS / "strict_timing_quantization_sensitivity.csv", index=False)
    batch_summary.to_csv(ANALYSIS / "strict_timing_batch_sensitivity.csv", index=False)
    repeat_summary.to_csv(ANALYSIS / "strict_timing_reload_repeatability.csv", index=False)
    log = {
        "status": "PASS",
        "artifacts": len(profiles),
        "core_calls": int(len(cases) * 3),
        "batch_calls": int(len(batch)),
        "quantization_calls": int(len(quant)),
        "reload_calls": int(len(repeated)),
        "reload_exact_match_rate": float(repeat_summary["raw_output_exact_match_rate"].iloc[0]),
    }
    (LOGS / "strict_timing_artifact_analysis.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
