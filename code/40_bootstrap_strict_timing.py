#!/usr/bin/env python3
"""Company-cluster bootstrap for the strict-timing validation results.

Every probability bootstrap draw rebuilds grouped calibration folds and refits
the Platt maps.  This propagates calibration-parameter uncertainty rather than
conditioning inference on a single set of out-of-fold predictions.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


SEED = 20260814
BOOTSTRAPS = 1000
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "output" / "analysis"
LOGS = ROOT / "output" / "logs"


def load_analysis_module():
    path = ROOT / "code" / "39_analyze_strict_timing.py"
    spec = importlib.util.spec_from_file_location("strict_timing_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load strict-timing analysis functions.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile_summary(draws: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in draws.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        prefix = dict(zip(keys, values))
        for metric in [column for column in draws if column not in [*keys, "bootstrap"]]:
            data = group[metric].dropna().to_numpy(float)
            if len(data) == 0:
                continue
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "bootstrap_draws": len(data),
                    "bootstrap_mean": float(np.mean(data)),
                    "ci_lower": float(np.quantile(data, 0.025)),
                    "ci_upper": float(np.quantile(data, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def primary_bootstrap(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ANALYSIS / "strict_timing_primary_qwen3_4b_4bit.csv")
    scores = raw.pivot(index="case_id", columns="variant", values="pd_percent")
    y = raw.drop_duplicates("case_id").set_index("case_id")["distress_f1"].loc[scores.index]
    ranges = scores.max(axis=1) - scores.min(axis=1)
    strata = {value: np.flatnonzero(y.to_numpy(int) == value) for value in (0, 1)}
    rows = []
    for draw in range(BOOTSTRAPS):
        index = np.concatenate(
            [rng.choice(part, size=len(part), replace=True) for part in strata.values()]
        )
        draw_y = y.to_numpy(int)[index]
        canonical = scores["strict_canonical"].to_numpy(float)[index]
        rows.append(
            {
                "bootstrap": draw + 1,
                "canonical_auc": roc_auc_score(draw_y, canonical),
                "canonical_pr_auc": average_precision_score(draw_y, canonical),
                "median_representation_range_pp": float(np.median(ranges.to_numpy()[index])),
                "mean_representation_range_pp": float(np.mean(ranges.to_numpy()[index])),
            }
        )
    draws = pd.DataFrame(rows)
    summary = percentile_summary(draws.assign(result="primary"), ["result"])
    return draws, summary


def resample_companies(cases: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    firm = cases["firm_id"].astype(str).str.zfill(6).to_numpy()
    unique_firms = np.unique(firm)
    selected = rng.choice(unique_firms, size=len(unique_firms), replace=True)
    indices = []
    groups = []
    for draw, selected_firm in enumerate(selected):
        firm_rows = np.flatnonzero(firm == selected_firm)
        indices.append(firm_rows)
        groups.append(np.full(len(firm_rows), draw, dtype=int))
    return np.concatenate(indices), np.concatenate(groups)


def grouped_folds(
    cases: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int
) -> np.ndarray:
    strata = cases["fiscal_year"].astype(str).to_numpy() + "_" + y.astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.empty(len(cases), dtype=int)
    dummy = np.zeros((len(cases), 1))
    for fold, (_, test) in enumerate(splitter.split(dummy, strata, groups=groups), start=1):
        folds[test] = fold
    return folds


def crossfit(
    module, calibration_score: np.ndarray, application_score: np.ndarray,
    y: np.ndarray, weight: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    fitted = np.empty(len(y), dtype=float)
    for fold in range(1, 6):
        train = folds != fold
        test = folds == fold
        beta = module.fit_platt(calibration_score[train], y[train], weight[train])
        fitted[test] = module.apply_platt(application_score[test], beta)
    return fitted


def probability_bootstrap(
    module, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases, scores = module.probability_data()
    score_names = [
        "qwen_canonical_naive",
        "logit",
        "hist_gradient_boosting",
        "xgboost",
    ]
    rows = []
    for draw in range(BOOTSTRAPS):
        index, groups = resample_companies(cases, rng)
        sample = cases.iloc[index].reset_index(drop=True)
        y = sample["outcome"].to_numpy(int)
        weight = sample["sampling_weight"].to_numpy(float)
        folds = grouped_folds(sample, y, groups, SEED + 1000 + draw)
        canonical = scores["qwen_canonical_naive"][index]
        reverse = scores["qwen_reverse_naive"][index]
        decimal = scores["qwen_decimal_naive"][index]
        calibrated_representations = {
            "canonical": crossfit(module, canonical, canonical, y, weight, folds),
            "reverse": crossfit(module, canonical, reverse, y, weight, folds),
            "decimal": crossfit(module, canonical, decimal, y, weight, folds),
        }
        normalized_weight = weight / weight.sum()
        crossing_values = np.column_stack(list(calibrated_representations.values()))
        crossing_results = {}
        for threshold in (0.01, 0.02):
            flags = crossing_values >= threshold
            crossed = flags.any(axis=1) & ~flags.all(axis=1)
            crossing_results[f"crossing_{int(100 * threshold)}pct"] = float(
                np.sum(normalized_weight * crossed)
            )
        for name in score_names:
            raw = scores[name][index]
            calibrated = (
                calibrated_representations["canonical"]
                if name == "qwen_canonical_naive"
                else crossfit(module, raw, raw, y, weight, folds)
            )
            raw_metrics = module.probability_metrics(y, raw, weight)
            calibrated_metrics = module.probability_metrics(y, calibrated, weight)
            rows.append(
                {
                    "bootstrap": draw + 1,
                    "score": name,
                    "raw_roc_auc": raw_metrics["roc_auc"],
                    "raw_pr_auc": raw_metrics["pr_auc"],
                    "calibrated_mean_probability": calibrated_metrics["weighted_mean_probability"],
                    "calibrated_brier": calibrated_metrics["brier"],
                    "calibrated_brier_skill_score": calibrated_metrics["brier_skill_score"],
                    "calibrated_log_loss": calibrated_metrics["log_loss"],
                    **(crossing_results if name == "qwen_canonical_naive" else {
                        "crossing_1pct": np.nan,
                        "crossing_2pct": np.nan,
                    }),
                }
            )
        if (draw + 1) % 100 == 0:
            print(json.dumps({"bootstrap_completed": draw + 1, "total": BOOTSTRAPS}))
    draws = pd.DataFrame(rows)
    summary = percentile_summary(draws, ["score"])
    return draws, summary


def main() -> None:
    module = load_analysis_module()
    rng = np.random.default_rng(SEED + 900)
    primary_draws, primary_summary = primary_bootstrap(rng)
    probability_draws, probability_summary = probability_bootstrap(module, rng)
    primary_draws.to_parquet(ANALYSIS / "strict_timing_primary_bootstrap_draws.parquet", index=False)
    probability_draws.to_parquet(
        ANALYSIS / "strict_timing_probability_bootstrap_draws.parquet", index=False
    )
    primary_summary.to_csv(ANALYSIS / "strict_timing_primary_bootstrap_ci.csv", index=False)
    probability_summary.to_csv(
        ANALYSIS / "strict_timing_probability_bootstrap_ci.csv", index=False
    )
    log = {
        "status": "PASS",
        "bootstrap_draws": BOOTSTRAPS,
        "primary_rows": len(primary_draws),
        "probability_rows": len(probability_draws),
        "calibration_refit_within_each_draw": True,
        "cluster_unit": "firm",
    }
    (LOGS / "strict_timing_bootstrap.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
