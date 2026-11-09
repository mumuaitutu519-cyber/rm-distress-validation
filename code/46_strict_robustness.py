#!/usr/bin/env python3
"""Calibration, outcome-definition, and temporal robustness checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


SEED = 20260814
EPS = 1e-6
BOOTSTRAPS = 1000
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "output" / "analysis"
LOGS = ROOT / "output" / "logs"


def load_analysis_module():
    path = ROOT / "code" / "39_analyze_strict_timing.py"
    spec = importlib.util.spec_from_file_location("strict_timing_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load strict analysis module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_intercept_only(module, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    offset = module.logit(p)
    w = w / w.sum()
    alpha = 0.0
    for _ in range(100):
        mu = module.sigmoid(alpha + offset)
        gradient = np.sum(w * (y - mu))
        information = np.sum(w * mu * (1 - mu)) + 1e-10
        step = gradient / information
        alpha += step
        if abs(step) < 1e-10:
            break
    return float(alpha)


def fit_general_logistic(module, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    prevalence = float(np.sum(w * y))
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(prevalence / (1 - prevalence))
    ridge = np.eye(x.shape[1]) * 1e-8
    ridge[0, 0] = 1e-10

    def objective(candidate: np.ndarray) -> float:
        eta = x @ candidate
        likelihood = np.sum(w * (y * eta - np.logaddexp(0.0, eta)))
        penalty = 0.5 * float(candidate @ ridge @ candidate)
        return float(likelihood - penalty)

    for _ in range(200):
        mu = module.sigmoid(x @ beta)
        gradient = x.T @ (w * (y - mu)) - ridge @ beta
        info = x.T @ ((w * mu * (1 - mu))[:, None] * x) + ridge
        step = np.linalg.pinv(info) @ gradient
        current = objective(beta)
        scale = 1.0
        candidate = beta + step
        while objective(candidate) < current and scale > 1e-8:
            scale *= 0.5
            candidate = beta + scale * step
        beta = candidate
        if np.max(np.abs(scale * step)) < 1e-10:
            break
    return beta


def fit_method(module, method: str, p: np.ndarray, y: np.ndarray, w: np.ndarray):
    p = np.clip(p, EPS, 1 - EPS)
    if method == "intercept_only":
        return method, fit_intercept_only(module, p, y, w)
    if method == "platt":
        return method, module.fit_platt(p, y, w)
    if method == "beta":
        x = np.column_stack([np.ones(len(p)), np.log(p), np.log(1 - p)])
        return method, fit_general_logistic(module, x, y, w)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
        model.fit(p, y, sample_weight=w)
        return method, model
    raise ValueError(method)


def apply_method(module, fitted, p: np.ndarray) -> np.ndarray:
    method, object_ = fitted
    p = np.clip(p, EPS, 1 - EPS)
    if method == "intercept_only":
        return module.sigmoid(object_ + module.logit(p))
    if method == "platt":
        return module.apply_platt(p, object_)
    if method == "beta":
        x = np.column_stack([np.ones(len(p)), np.log(p), np.log(1 - p)])
        return module.sigmoid(x @ object_)
    if method == "isotonic":
        return np.clip(object_.predict(p), EPS, 1 - EPS)
    raise ValueError(method)


def alternative_calibration(module, cases: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    predictions = pd.read_parquet(ANALYSIS / "strict_timing_probability_predictions.parquet")
    folds = predictions["grouped_fold"].to_numpy(int)
    y = cases["outcome"].to_numpy(int)
    w = cases["sampling_weight"].to_numpy(float)
    rows = []
    selected_scores = {
        "qwen_canonical_naive": scores["qwen_canonical_naive"],
        "logit": scores["logit"],
        "hist_gradient_boosting": scores["hist_gradient_boosting"],
        "xgboost": scores["xgboost"],
    }
    for name, score in selected_scores.items():
        for method in ("intercept_only", "platt", "beta", "isotonic"):
            fitted = np.empty(len(score))
            for fold in range(1, 6):
                train = folds != fold
                test = folds == fold
                calibration = fit_method(module, method, score[train], y[train], w[train])
                fitted[test] = apply_method(module, calibration, score[test])
            metrics = module.probability_metrics(y, fitted, w)
            diagnostic = module.fit_platt(fitted, y, w)
            rows.append(
                {
                    "score": name,
                    "calibration": method,
                    **metrics,
                    "pooled_calibration_intercept": float(diagnostic[0]),
                    "pooled_calibration_slope": float(diagnostic[1]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_calibration_robustness.csv", index=False)
    return out


def outcome_robustness(module, cases: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    outcomes = {
        "ST_or_starST_primary_30_day_risk_set": "outcome",
        "composite_primary_30_day_risk_set": "outcome_composite",
    }
    w = cases["sampling_weight"].to_numpy(float)
    rows = []
    selected_scores = {
        "qwen_canonical": scores["qwen_canonical_naive"],
        "logit": scores["logit"],
        "xgboost": scores["xgboost"],
    }
    for label, column in outcomes.items():
        y = cases[column].to_numpy(int)
        for name, score in selected_scores.items():
            metrics = module.probability_metrics(y, score, w)
            rows.append(
                {
                    "outcome_definition": label,
                    "score": name,
                    "design_weighted_events": float(np.sum(w * y)),
                    "design_weighted_event_rate": float(np.average(y, weights=w)),
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_outcome_robustness.csv", index=False)
    return out


def auc_differences() -> pd.DataFrame:
    draws = pd.read_parquet(ANALYSIS / "strict_timing_probability_bootstrap_draws.parquet")
    pivot = draws.pivot(index="bootstrap", columns="score", values="raw_roc_auc")
    rows = []
    for other in ("logit", "hist_gradient_boosting", "xgboost"):
        difference = pivot["qwen_canonical_naive"] - pivot[other]
        rows.append(
            {
                "contrast": f"qwen_minus_{other}",
                "bootstrap_draws": len(difference),
                "mean_difference": float(difference.mean()),
                "ci_lower": float(difference.quantile(0.025)),
                "ci_upper": float(difference.quantile(0.975)),
                "share_positive": float(difference.gt(0).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_auc_difference_ci.csv", index=False)
    return out


def resample_companies(cases: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    firms = cases["firm_id"].astype(str).str.zfill(6).to_numpy()
    unique = np.unique(firms)
    selected = rng.choice(unique, size=len(unique), replace=True)
    indices = []
    groups = []
    for group, firm in enumerate(selected):
        part = np.flatnonzero(firms == firm)
        indices.append(part)
        groups.append(np.full(len(part), group))
    return np.concatenate(indices), np.concatenate(groups)


def temporal_bootstrap(module, cases: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 7000)
    selected_scores = {
        "qwen_canonical_naive": scores["qwen_canonical_naive"],
        "logit": scores["logit"],
        "hist_gradient_boosting": scores["hist_gradient_boosting"],
        "xgboost": scores["xgboost"],
    }
    rows = []
    for draw in range(BOOTSTRAPS):
        index, _ = resample_companies(cases, rng)
        sample = cases.iloc[index].reset_index(drop=True)
        y = sample["outcome"].to_numpy(int)
        w = sample["sampling_weight"].to_numpy(float)
        for name, full_score in selected_scores.items():
            score = full_score[index]
            temporal = np.full(len(score), np.nan)
            historical = np.full(len(score), np.nan)
            for year in (2022, 2023, 2024):
                train = sample["fiscal_year"].lt(year).to_numpy()
                test = sample["fiscal_year"].eq(year).to_numpy()
                beta = module.fit_platt(score[train], y[train], w[train])
                temporal[test] = module.apply_platt(score[test], beta)
                historical[test] = np.average(y[train], weights=w[train])
            eligible = ~np.isnan(temporal)
            ew = w[eligible] / w[eligible].sum()
            model_brier = float(np.sum(ew * (y[eligible] - temporal[eligible]) ** 2))
            history_brier = float(np.sum(ew * (y[eligible] - historical[eligible]) ** 2))
            metrics = module.probability_metrics(y[eligible], temporal[eligible], w[eligible])
            rows.append(
                {
                    "bootstrap": draw + 1,
                    "score": name,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "bss_vs_historical_rate": 1 - model_brier / history_brier,
                }
            )
    draws = pd.DataFrame(rows)
    summary_rows = []
    for score, group in draws.groupby("score"):
        for metric in ("roc_auc", "pr_auc", "bss_vs_historical_rate"):
            summary_rows.append(
                {
                    "score": score,
                    "metric": metric,
                    "bootstrap_draws": len(group),
                    "mean": float(group[metric].mean()),
                    "ci_lower": float(group[metric].quantile(0.025)),
                    "ci_upper": float(group[metric].quantile(0.975)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    draws.to_parquet(ANALYSIS / "strict_timing_temporal_bootstrap_draws.parquet", index=False)
    summary.to_csv(ANALYSIS / "strict_timing_temporal_bootstrap_ci.csv", index=False)
    return summary


def main() -> None:
    module = load_analysis_module()
    cases, scores = module.probability_data()
    calibration = alternative_calibration(module, cases, scores)
    outcomes = outcome_robustness(module, cases, scores)
    differences = auc_differences()
    temporal = temporal_bootstrap(module, cases, scores)
    log = {
        "status": "PASS",
        "calibration_rows": len(calibration),
        "outcome_rows": len(outcomes),
        "auc_contrasts": len(differences),
        "temporal_bootstrap_draws": BOOTSTRAPS,
        "temporal_summary_rows": len(temporal),
    }
    (LOGS / "strict_timing_robustness.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
