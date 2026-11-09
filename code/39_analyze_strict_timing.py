#!/usr/bin/env python3
"""Analyze the strict-timing primary and probability audits.

The primary 256-firm sample is a paired behavioral stress test.  The 1,500-case
probability audit contains every primary ST/*ST event and design-weighted
non-events.  Probability statements therefore use Horvitz--Thompson weights;
roster statements remain explicitly conditional on the enriched paired audit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


SEED = 20260814
EPS = 1e-6
ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
ANALYSIS = ROOT / "output" / "analysis"
LOGS = ROOT / "output" / "logs"
CORE = ["strict_canonical", "strict_order_reversed", "strict_unit_decimal"]


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -35, 35)
    return 1 / (1 + np.exp(-value))


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(score: np.ndarray, y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fit a weighted intercept-and-slope logistic calibration map."""
    x = np.column_stack([np.ones(len(score)), logit(score)])
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weight = weight / weight.sum()
    prevalence = float(np.sum(weight * y))
    beta = np.array([math.log(prevalence / (1 - prevalence)), 0.0])
    ridge = np.diag([1e-10, 1e-8])

    def objective(candidate: np.ndarray) -> float:
        eta = np.clip(x @ candidate, -35, 35)
        return float(
            np.sum(weight * (y * eta - np.logaddexp(0.0, eta)))
            - 0.5 * candidate @ ridge @ candidate
        )

    for _ in range(200):
        eta = np.clip(x @ beta, -35, 35)
        mu = sigmoid(eta)
        gradient = x.T @ (weight * (y - mu)) - ridge @ beta
        information = x.T @ ((weight * mu * (1 - mu))[:, None] * x) + ridge
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        current = objective(beta)
        scale = 1.0
        candidate = beta + step
        while objective(candidate) < current and scale > 2**-24:
            scale *= 0.5
            candidate = beta + scale * step
        change = candidate - beta
        beta = candidate
        if float(np.max(np.abs(change))) < 1e-10:
            break
    return beta


def apply_platt(score: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return sigmoid(beta[0] + beta[1] * logit(score))


def probability_metrics(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    weight = np.asarray(weight, dtype=float)
    weight = weight / weight.sum()
    prevalence = float(np.sum(weight * y))
    brier = float(np.sum(weight * (y - p) ** 2))
    constant_brier = prevalence * (1 - prevalence)
    return {
        "weighted_event_rate": prevalence,
        "weighted_mean_probability": float(np.sum(weight * p)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weight)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=weight)),
        "brier": brier,
        "brier_skill_score": float(1 - brier / constant_brier),
        "log_loss": float(
            -np.sum(weight * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        ),
    }


def primary_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ANALYSIS / "strict_timing_primary_qwen3_4b_4bit.csv")
    if len(raw) != 768 or not raw["parse_pd"].all():
        raise AssertionError("Primary LLM audit is incomplete or contains parse failures.")
    scores = raw.pivot(index="case_id", columns="variant", values="pd_percent")[CORE]
    cases = pd.read_parquet(DERIVED / "strict_timing_primary_cases_256.parquet").set_index("case_id")
    cases = cases.loc[scores.index]
    y = cases["outcome"].to_numpy(int)
    metric_rows = []
    for variant in CORE:
        value = scores[variant].to_numpy(float)
        metric_rows.append(
            {
                "variant": variant,
                "n": len(value),
                "events": int(y.sum()),
                "mean_score": float(np.mean(value)),
                "median_score": float(np.median(value)),
                "unique_scores": int(pd.Series(value).nunique()),
                "integer_output_rate": float(np.mean(np.isclose(value, np.round(value)))),
                "roc_auc": float(roc_auc_score(y, value)),
                "pr_auc": float(average_precision_score(y, value)),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    ranges = pd.DataFrame(index=scores.index)
    ranges["score_min"] = scores.min(axis=1)
    ranges["score_max"] = scores.max(axis=1)
    ranges["score_range"] = ranges["score_max"] - ranges["score_min"]
    ranges["outcome"] = cases["outcome"]
    range_summary = pd.DataFrame(
        [
            {
                "n": len(ranges),
                "median_range_pp": float(ranges["score_range"].median()),
                "mean_range_pp": float(ranges["score_range"].mean()),
                "p90_range_pp": float(ranges["score_range"].quantile(0.9)),
                "maximum_range_pp": float(ranges["score_range"].max()),
                "any_change_rate": float(ranges["score_range"].gt(0).mean()),
            }
        ]
    )

    roster_rows = []
    score_frame = scores.reset_index()
    y_by_case = cases["outcome"]
    for capacity in (0.05, 0.10):
        top_n = int(math.ceil(capacity * len(score_frame)))
        rosters: dict[str, set[str]] = {}
        for variant in CORE:
            selected = score_frame.sort_values(
                [variant, "case_id"], ascending=[False, True]
            ).head(top_n)
            rosters[variant] = set(selected["case_id"])
            events = int(y_by_case.loc[list(rosters[variant])].sum())
            roster_rows.append(
                {
                    "capacity": capacity,
                    "variant": variant,
                    "top_n": top_n,
                    "selected_events": events,
                    "event_recall": events / int(y_by_case.sum()),
                    "intersection_all": np.nan,
                    "union_all": np.nan,
                    "jaccard_all": np.nan,
                }
            )
        intersection = set.intersection(*rosters.values())
        union = set.union(*rosters.values())
        roster_rows.append(
            {
                "capacity": capacity,
                "variant": "three_representation_summary",
                "top_n": top_n,
                "selected_events": np.nan,
                "event_recall": np.nan,
                "intersection_all": len(intersection),
                "union_all": len(union),
                "jaccard_all": len(intersection) / len(union),
            }
        )
    rosters = pd.DataFrame(roster_rows)
    ranges.reset_index().to_parquet(ANALYSIS / "strict_timing_primary_case_ranges.parquet", index=False)
    metrics.to_csv(ANALYSIS / "strict_timing_primary_metrics.csv", index=False)
    range_summary.to_csv(ANALYSIS / "strict_timing_primary_range_summary.csv", index=False)
    rosters.to_csv(ANALYSIS / "strict_timing_primary_rosters.csv", index=False)
    return metrics, range_summary, rosters


def probability_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    raw = pd.read_csv(ANALYSIS / "strict_timing_probability_qwen3_4b_4bit.csv")
    if len(raw) != 4500 or not raw["parse_pd"].all():
        raise AssertionError("Probability LLM audit is incomplete or contains parse failures.")
    pivot = raw.pivot(index="case_id", columns="variant", values="pd_percent")[CORE] / 100
    cases = pd.read_parquet(DERIVED / "strict_timing_probability_cases_1500.parquet")
    cases = cases.set_index("case_id").loc[pivot.index].reset_index()
    benchmark = pd.read_parquet(ANALYSIS / "strict_timing_benchmark_holdout_predictions.parquet")
    keep = [
        "row_id",
        "logit_probability",
        "hist_gradient_boosting_probability",
        "xgboost_probability",
    ]
    cases = cases.merge(benchmark[keep], on="row_id", how="left", validate="one_to_one")
    if cases[keep[1:]].isna().any().any():
        raise AssertionError("Structured predictions failed to merge into probability audit.")
    scores = {
        "qwen_canonical_naive": pivot.loc[cases["case_id"], "strict_canonical"].to_numpy(),
        "qwen_reverse_naive": pivot.loc[cases["case_id"], "strict_order_reversed"].to_numpy(),
        "qwen_decimal_naive": pivot.loc[cases["case_id"], "strict_unit_decimal"].to_numpy(),
        "logit": cases["logit_probability"].to_numpy(float),
        "hist_gradient_boosting": cases["hist_gradient_boosting_probability"].to_numpy(float),
        "xgboost": cases["xgboost_probability"].to_numpy(float),
    }
    return cases, scores


def assign_grouped_folds(cases: pd.DataFrame) -> np.ndarray:
    strata = cases["fiscal_year"].astype(str) + "_" + cases["outcome"].astype(str)
    groups = cases["firm_id"].astype(str).str.zfill(6)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 101)
    folds = np.empty(len(cases), dtype=int)
    dummy = np.zeros((len(cases), 1))
    for fold, (_, test) in enumerate(splitter.split(dummy, strata, groups=groups), start=1):
        folds[test] = fold
    for fold in range(1, 6):
        if set(groups[folds == fold]) & set(groups[folds != fold]):
            raise AssertionError("A firm leaked across a grouped calibration fold.")
    return folds


def probability_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases, scores = probability_data()
    y = cases["outcome"].to_numpy(int)
    weight = cases["sampling_weight"].to_numpy(float)
    folds = assign_grouped_folds(cases)
    prediction_output = cases[
        ["case_id", "row_id", "firm_id", "fiscal_year", "outcome", "sampling_weight"]
    ].copy()
    prediction_output["grouped_fold"] = folds
    metric_rows = []
    coefficient_rows = []
    canonical_qwen = scores["qwen_canonical_naive"]
    for name, score in scores.items():
        metric_rows.append({"score": name, "evaluation": "raw", **probability_metrics(y, score, weight)})
        calibrated = np.empty(len(score), dtype=float)
        for fold in range(1, 6):
            train = folds != fold
            test = folds == fold
            calibration_score = canonical_qwen if name.startswith("qwen_") else score
            beta = fit_platt(calibration_score[train], y[train], weight[train])
            calibrated[test] = apply_platt(score[test], beta)
            coefficient_rows.append(
                {
                    "score": name,
                    "evaluation": "five_fold_grouped_crossfit",
                    "fold": fold,
                    "intercept": beta[0],
                    "slope": beta[1],
                    "calibration_n": int(train.sum()),
                    "evaluation_n": int(test.sum()),
                    "calibration_source": (
                        "qwen_canonical_naive" if name.startswith("qwen_") else name
                    ),
                }
            )
        prediction_output[f"{name}_raw"] = score
        prediction_output[f"{name}_grouped_platt"] = calibrated
        metric_rows.append(
            {
                "score": name,
                "evaluation": "five_fold_grouped_crossfit_platt",
                **probability_metrics(y, calibrated, weight),
            }
        )

    # Genuine forward-time calibration: each evaluation year uses prior years only.
    temporal_rows = []
    for name, score in scores.items():
        temporal = np.full(len(score), np.nan)
        for year in (2022, 2023, 2024):
            train = cases["fiscal_year"].lt(year).to_numpy()
            test = cases["fiscal_year"].eq(year).to_numpy()
            calibration_score = canonical_qwen if name.startswith("qwen_") else score
            beta = fit_platt(calibration_score[train], y[train], weight[train])
            temporal[test] = apply_platt(score[test], beta)
            coefficient_rows.append(
                {
                    "score": name,
                    "evaluation": f"forward_time_{year}",
                    "fold": year,
                    "intercept": beta[0],
                    "slope": beta[1],
                    "calibration_n": int(train.sum()),
                    "evaluation_n": int(test.sum()),
                    "calibration_source": (
                        "qwen_canonical_naive" if name.startswith("qwen_") else name
                    ),
                }
            )
        eligible = ~np.isnan(temporal)
        prediction_output[f"{name}_forward_platt"] = temporal
        temporal_rows.append(
            {
                "score": name,
                "evaluation": "forward_time_platt_2022_2024",
                **probability_metrics(y[eligible], temporal[eligible], weight[eligible]),
            }
        )
    metrics = pd.DataFrame([*metric_rows, *temporal_rows])
    coefficients = pd.DataFrame(coefficient_rows)

    period_rows = []
    for label, years in {
        "pre_rule_2019_2022": range(2019, 2023),
        "rule_exposed_2023_2024": range(2023, 2025),
    }.items():
        mask = cases["fiscal_year"].isin(years).to_numpy()
        for name, score in scores.items():
            period_rows.append(
                {"period": label, "score": name, **probability_metrics(y[mask], score[mask], weight[mask])}
            )
    periods = pd.DataFrame(period_rows)
    metrics.to_csv(ANALYSIS / "strict_timing_probability_metrics.csv", index=False)
    coefficients.to_csv(ANALYSIS / "strict_timing_calibration_coefficients.csv", index=False)
    periods.to_csv(ANALYSIS / "strict_timing_probability_periods.csv", index=False)
    prediction_output.to_parquet(ANALYSIS / "strict_timing_probability_predictions.parquet", index=False)
    return metrics, periods, prediction_output


def threshold_crossings(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    representations = {
        "canonical": "qwen_canonical_naive_grouped_platt",
        "reverse_order": "qwen_reverse_naive_grouped_platt",
        "decimal_units": "qwen_decimal_naive_grouped_platt",
    }
    weight = predictions["sampling_weight"].to_numpy(float)
    weight = weight / weight.sum()
    values = predictions[list(representations.values())]
    for threshold in np.arange(0.0025, 0.0501, 0.0025):
        flags = values.ge(threshold)
        any_cross = flags.any(axis=1) & ~flags.all(axis=1)
        rows.append(
            {
                "threshold": threshold,
                "unweighted_crossing_rate": float(any_cross.mean()),
                "design_weighted_crossing_rate": float(np.sum(weight * any_cross)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_threshold_crossing_curve.csv", index=False)
    return out


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    primary_metrics, range_summary, rosters = primary_analysis()
    probability_metrics_frame, periods, predictions = probability_analysis()
    crossings = threshold_crossings(predictions)
    log = {
        "status": "PASS",
        "primary": {
            "canonical_auc": float(
                primary_metrics.loc[primary_metrics["variant"].eq("strict_canonical"), "roc_auc"].iloc[0]
            ),
            "median_representation_range_pp": float(range_summary["median_range_pp"].iloc[0]),
            "five_percent_roster_jaccard": float(
                rosters.loc[
                    rosters["capacity"].eq(0.05)
                    & rosters["variant"].eq("three_representation_summary"),
                    "jaccard_all",
                ].iloc[0]
            ),
        },
        "probability_rows": len(predictions),
        "period_rows": len(periods),
        "threshold_points": len(crossings),
    }
    (LOGS / "strict_timing_analysis.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
