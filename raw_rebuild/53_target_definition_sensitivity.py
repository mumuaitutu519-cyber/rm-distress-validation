#!/usr/bin/env python3
"""Re-estimate probability diagnostics under narrower ST/*ST target definitions.

The probability-audit sample and inverse-inclusion weights remain fixed. Events
with excluded reason codes become certainty-stratum non-events with weight one,
so the weights still recover the 25,390-record holdout frame. The script reports
point estimates; it does not replace the bootstrap inference for the primary
outcome.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "output" / "analysis"
DERIVED = ROOT / "data" / "derived"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    timing = load_module("strict_timing_analysis", Path(__file__).with_name("39_analyze_strict_timing.py"))
    regulatory = load_module(
        "regulatory_validation", Path(__file__).with_name("49_analyze_regulatory_validation.py")
    )

    predictions = pd.read_parquet(ANALYSIS / "strict_timing_probability_predictions.parquet")
    panel = pd.read_parquet(DERIVED / "strict_timing_panel_2011_2024.parquet")
    status = pd.read_parquet(DERIVED / "strict_timing_st_changes.parquet")
    reasons = regulatory.outcome_reason_frame(panel, status)
    reason_columns = ["reason_01", "reason_02", "reason_03", "reason_06", "reason_11"]
    frame = predictions.merge(reasons[["row_id", *reason_columns]], on="row_id", how="left")
    for column in reason_columns:
        frame[column] = frame[column].fillna(False).astype(bool)

    frame["primary"] = frame["outcome"].eq(1)
    frame["excluding_disclosure"] = frame["outcome"].eq(1) & ~frame["reason_11"]
    frame["financial_01_03"] = frame["outcome"].eq(1) & frame[
        ["reason_01", "reason_02", "reason_03"]
    ].any(axis=1)
    frame["financial_operational_01_03_06"] = frame["outcome"].eq(1) & frame[
        ["reason_01", "reason_02", "reason_03", "reason_06"]
    ].any(axis=1)

    models = {
        "Qwen3-4B": "qwen_canonical_naive_raw",
        "Logit": "logit_raw",
        "XGBoost": "xgboost_raw",
    }
    target_labels = {
        "primary": "Primary ST/*ST",
        "excluding_disclosure": "Primary excluding reason 11",
        "financial_01_03": "Financial trigger, codes 01–03",
        "financial_operational_01_03_06": "Financial/operational trigger, codes 01–03 and 06",
    }
    weight = frame["sampling_weight"].to_numpy(float)
    folds = frame["grouped_fold"].to_numpy(int)
    rows: list[dict[str, object]] = []

    for target, target_label in target_labels.items():
        y = frame[target].astype(int).to_numpy()
        for model, score_column in models.items():
            score = frame[score_column].to_numpy(float)
            grouped = np.empty(len(frame), dtype=float)
            for fold in range(1, 6):
                train = folds != fold
                test = folds == fold
                beta = timing.fit_platt(score[train], y[train], weight[train])
                grouped[test] = timing.apply_platt(score[test], beta)
            grouped_metrics = timing.probability_metrics(y, grouped, weight)

            forward = np.full(len(frame), np.nan)
            historical = np.full(len(frame), np.nan)
            for year in (2022, 2023, 2024):
                train = frame["fiscal_year"].lt(year).to_numpy()
                test = frame["fiscal_year"].eq(year).to_numpy()
                beta = timing.fit_platt(score[train], y[train], weight[train])
                forward[test] = timing.apply_platt(score[test], beta)
                historical[test] = np.average(y[train], weights=weight[train])
            evaluation = np.isfinite(forward)
            evaluation_weight = weight[evaluation]
            evaluation_weight = evaluation_weight / evaluation_weight.sum()
            forward_brier = float(
                np.sum(evaluation_weight * (y[evaluation] - forward[evaluation]) ** 2)
            )
            reference_brier = float(
                np.sum(evaluation_weight * (y[evaluation] - historical[evaluation]) ** 2)
            )

            rows.append(
                {
                    "target": target,
                    "target_label": target_label,
                    "model": model,
                    "sample_events": int(y.sum()),
                    "weighted_event_rate": float(np.average(y, weights=weight)),
                    "raw_roc_auc": float(roc_auc_score(y, score, sample_weight=weight)),
                    "raw_pr_auc": float(average_precision_score(y, score, sample_weight=weight)),
                    "retrospective_grouped_bss": grouped_metrics["brier_skill_score"],
                    "forward_bss_2022_2024": float(1 - forward_brier / reference_brier),
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(ANALYSIS / "strict_timing_target_definition_sensitivity.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
