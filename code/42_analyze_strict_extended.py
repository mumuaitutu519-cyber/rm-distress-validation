#!/usr/bin/env python3
"""Analyze strict-timing representation, economic, and information audits."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


SEED = 20260814
BOOTSTRAPS = 2000
ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
ANALYSIS = ROOT / "output" / "analysis"
LOGS = ROOT / "output" / "logs"

RANDOM_ORDERS = [f"strict_random_order_{index:02d}" for index in range(1, 6)]
FORMAT_VARIANTS = [
    "strict_json",
    "strict_table",
    "strict_sentences",
    "strict_unit_basis_points",
    "strict_precision_trailing_zero",
    "strict_signed_language",
]
ECONOMIC_EXPECTED = {
    "strict_deteriorated": 1,
    "strict_improved": -1,
    "strict_lev_up_2pp": 1,
    "strict_lev_down_2pp": -1,
    "strict_roa_down_1pp": 1,
    "strict_roa_up_1pp": -1,
    "strict_growth_down_5pp": 0,
    "strict_growth_up_5pp": 0,
    "strict_cash_down_2pp": 1,
    "strict_cash_up_2pp": -1,
}
DIAGONAL = {
    "strict_favorable_before": ("favorable", 1, "before"),
    "strict_favorable_after": ("favorable", 2, "after"),
    "strict_favorable_system": ("favorable", 3, "system"),
    "strict_neutral_before": ("neutral", 1, "before"),
    "strict_neutral_after": ("neutral", 2, "after"),
    "strict_neutral_system": ("neutral", 3, "system"),
    "strict_adverse_before": ("adverse", 1, "before"),
    "strict_adverse_after": ("adverse", 2, "after"),
    "strict_adverse_system": ("adverse", 3, "system"),
}


def load_prompt_module():
    path = ROOT / "raw_rebuild" / "37_prepare_strict_timing_audits.py"
    spec = importlib.util.spec_from_file_location("strict_prompt_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load strict prompt builder.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scores() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    extended = pd.read_csv(ANALYSIS / "strict_timing_extended_qwen3_4b_4bit.csv")
    if len(extended) != 1920 or not extended["parse_pd"].all():
        raise AssertionError("Extended audit is incomplete.")
    primary = pd.read_csv(ANALYSIS / "strict_timing_primary_qwen3_4b_4bit.csv")
    nested = pd.read_parquet(DERIVED / "strict_timing_nested_cases_64.parquet")
    case_ids = set(nested["case_id"])
    canonical = primary.loc[
        primary["variant"].eq("strict_canonical") & primary["case_id"].isin(case_ids)
    ].set_index("case_id")["pd_percent"]
    pivot = extended.pivot(index="case_id", columns="variant", values="pd_percent")
    canonical = canonical.loc[pivot.index]
    if len(pivot) != 64 or pivot.isna().any().any():
        raise AssertionError("Extended audit must be a complete 64-case panel.")
    return pivot, canonical, nested.set_index("case_id").loc[pivot.index]


def paired_summary(pivot: pd.DataFrame, canonical: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 100)
    rows = []
    for variant in pivot.columns:
        delta = (pivot[variant] - canonical).to_numpy(float)
        means = np.empty(BOOTSTRAPS)
        absolute_means = np.empty(BOOTSTRAPS)
        for draw in range(BOOTSTRAPS):
            index = rng.integers(0, len(delta), size=len(delta))
            means[draw] = np.mean(delta[index])
            absolute_means[draw] = np.mean(np.abs(delta[index]))
        rows.append(
            {
                "variant": variant,
                "n": len(delta),
                "mean_delta_pp": float(np.mean(delta)),
                "mean_delta_ci_lower": float(np.quantile(means, 0.025)),
                "mean_delta_ci_upper": float(np.quantile(means, 0.975)),
                "median_delta_pp": float(np.median(delta)),
                "mean_absolute_delta_pp": float(np.mean(np.abs(delta))),
                "mean_absolute_delta_ci_lower": float(np.quantile(absolute_means, 0.025)),
                "mean_absolute_delta_ci_upper": float(np.quantile(absolute_means, 0.975)),
                "changed_rate": float(np.mean(delta != 0)),
                "maximum_absolute_delta_pp": float(np.max(np.abs(delta))),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_extended_paired_summary.csv", index=False)
    return out


def order_summary(pivot: pd.DataFrame, canonical: pd.Series) -> pd.DataFrame:
    forms = pd.concat([canonical.rename("strict_canonical"), pivot[RANDOM_ORDERS]], axis=1)
    values = forms.to_numpy(float)
    pairwise = []
    for i in range(values.shape[1]):
        for j in range(i + 1, values.shape[1]):
            pairwise.append(np.abs(values[:, i] - values[:, j]))
    pairwise = np.column_stack(pairwise)
    out = pd.DataFrame(
        {
            "case_id": forms.index,
            "order_range_pp": np.ptp(values, axis=1),
            "mean_pairwise_order_difference_pp": np.mean(pairwise, axis=1),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "forms": values.shape[1],
                "median_order_range_pp": float(out["order_range_pp"].median()),
                "mean_order_range_pp": float(out["order_range_pp"].mean()),
                "mean_pairwise_order_difference_pp": float(
                    out["mean_pairwise_order_difference_pp"].mean()
                ),
            }
        ]
    )
    out.to_parquet(ANALYSIS / "strict_timing_order_case_metrics.parquet", index=False)
    summary.to_csv(ANALYSIS / "strict_timing_order_summary.csv", index=False)
    return summary


def economic_summary(
    pivot: pd.DataFrame, canonical: pd.Series, cases: pd.DataFrame
) -> pd.DataFrame:
    module = load_prompt_module()
    panel = pd.read_parquet(DERIVED / "strict_timing_panel_2011_2024.parquet")
    training = panel.loc[panel["fiscal_year"].le(2018)]
    bounds = {
        feature: (float(training[feature].min()), float(training[feature].max()))
        for feature in module.FEATURES
    }
    rows = []
    for variant, expected in ECONOMIC_EXPECTED.items():
        delta = (pivot[variant] - canonical).to_numpy(float)
        changed = delta != 0
        if expected == 0:
            violation = np.full(len(delta), np.nan)
        else:
            violation = expected * delta < 0
        support = []
        for _, case in cases.iterrows():
            original = module.values_from_row(case)
            changed_values = module.perturb(original, variant)
            changed_features = [
                feature
                for feature in module.FEATURES
                if not np.isclose(original[feature], changed_values[feature])
            ]
            support.append(
                all(
                    bounds[feature][0] <= changed_values[feature] <= bounds[feature][1]
                    for feature in changed_features
                )
            )
        rows.append(
            {
                "variant": variant,
                "expected_direction": expected,
                "mean_delta_pp": float(np.mean(delta)),
                "median_delta_pp": float(np.median(delta)),
                "changed_rate": float(np.mean(changed)),
                "mean_delta_when_changed_pp": float(np.mean(delta[changed])) if changed.any() else 0.0,
                "direction_violation_rate_all": (
                    float(np.mean(violation)) if expected != 0 else np.nan
                ),
                "direction_violation_rate_when_changed": (
                    float(np.mean(violation[changed]))
                    if expected != 0 and changed.any()
                    else np.nan
                ),
                "maximum_reverse_change_pp": (
                    float(np.max(np.maximum(-expected * delta, 0))) if expected != 0 else np.nan
                ),
                "marginal_training_support_rate": float(np.mean(support)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_economic_response.csv", index=False)
    return out


def information_panel(canonical: pd.Series) -> pd.DataFrame:
    extended = pd.read_csv(ANALYSIS / "strict_timing_extended_qwen3_4b_4bit.csv")
    rows = []
    for variant, (valence, wording, placement) in DIAGONAL.items():
        part = extended.loc[extended["variant"].eq(variant), ["case_id", "pd_percent"]].copy()
        part["valence"] = valence
        part["wording"] = wording
        part["placement"] = placement
        rows.append(part)
    missing = pd.read_csv(
        ANALYSIS / "strict_timing_information_factorial_missing_qwen3_4b_4bit.csv"
    )
    if len(missing) != 1152 or not missing["parse_pd"].all():
        raise AssertionError("Information-factorial missing cells are incomplete.")
    missing = missing[["case_id", "variant", "pd_percent"]].copy()
    extracted = missing["variant"].str.extract(
        r"strict_info_(favorable|neutral|adverse)_w([123])_(before|after|system)"
    )
    missing["valence"] = extracted[0]
    missing["wording"] = extracted[1].astype(int)
    missing["placement"] = extracted[2]
    rows.append(missing[["case_id", "pd_percent", "valence", "wording", "placement"]])
    panel = pd.concat(rows, ignore_index=True)
    if len(panel) != 64 * 27 or panel.duplicated(
        ["case_id", "valence", "wording", "placement"]
    ).any():
        raise AssertionError("Information audit is not a balanced 64 x 27 factorial.")
    panel["canonical_score"] = panel["case_id"].map(canonical)
    panel["delta_pp"] = panel["pd_percent"] - panel["canonical_score"]
    return panel


def information_analysis(canonical: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = information_panel(canonical)
    marginal = (
        panel.groupby(["valence", "placement"], as_index=False)
        .agg(
            n=("delta_pp", "size"),
            mean_delta_pp=("delta_pp", "mean"),
            median_delta_pp=("delta_pp", "median"),
            mean_absolute_delta_pp=("delta_pp", lambda values: np.mean(np.abs(values))),
            changed_rate=("delta_pp", lambda values: np.mean(values != 0)),
        )
    )
    # Balanced design: case FE absorbs baseline risk; wording and placement FE
    # prevent the valence contrasts from inheriting prompt-location effects.
    model = smf.ols(
        "delta_pp ~ C(case_id) + C(valence, Treatment(reference='neutral')) + "
        "C(wording) + C(placement, Treatment(reference='before'))",
        data=panel,
    ).fit(cov_type="cluster", cov_kwds={"groups": panel["case_id"]})
    terms = []
    for name, value in model.params.items():
        if "C(valence" in name or "C(placement" in name or "C(wording" in name:
            terms.append(
                {
                    "term": name,
                    "estimate_pp": float(value),
                    "standard_error": float(model.bse[name]),
                    "p_value": float(model.pvalues[name]),
                    "ci_lower": float(model.conf_int().loc[name, 0]),
                    "ci_upper": float(model.conf_int().loc[name, 1]),
                }
            )
    coefficients = pd.DataFrame(terms)
    panel.to_parquet(ANALYSIS / "strict_timing_information_factorial_panel.parquet", index=False)
    marginal.to_csv(ANALYSIS / "strict_timing_information_margins.csv", index=False)
    coefficients.to_csv(ANALYSIS / "strict_timing_information_fe_coefficients.csv", index=False)
    return marginal, coefficients


def precision_summary(pivot: pd.DataFrame, canonical: pd.Series) -> pd.DataFrame:
    forms = pd.concat([canonical.rename("strict_canonical"), pivot], axis=1)
    rows = []
    for variant in forms:
        values = forms[variant].to_numpy(float)
        rows.append(
            {
                "variant": variant,
                "n": len(values),
                "unique_scores": int(pd.Series(values).nunique()),
                "integer_output_rate": float(np.mean(np.isclose(values, np.round(values)))),
                "minimum_score": float(np.min(values)),
                "maximum_score": float(np.max(values)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_output_precision.csv", index=False)
    return out


def main() -> None:
    pivot, canonical, cases = load_scores()
    paired = paired_summary(pivot, canonical)
    order = order_summary(pivot, canonical)
    economic = economic_summary(pivot, canonical, cases)
    margins, coefficients = information_analysis(canonical)
    precision = precision_summary(pivot, canonical)
    log = {
        "status": "PASS",
        "paired_variants": len(paired),
        "order_forms": 6,
        "economic_variants": len(economic),
        "information_cells": 27,
        "information_rows": 64 * 27,
        "information_fe_terms": len(coefficients),
        "precision_variants": len(precision),
        "median_random_order_range_pp": float(order["median_order_range_pp"].iloc[0]),
    }
    (LOGS / "strict_timing_extended_analysis.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
