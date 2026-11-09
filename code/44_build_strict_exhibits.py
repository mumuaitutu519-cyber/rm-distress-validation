#!/usr/bin/env python3
"""Build strict-timing descriptives, temporal diagnostics, and paper figures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
ANALYSIS = ROOT / "output" / "analysis"
FIGURES = ROOT / "output" / "figures" / "strict_timing"
LOGS = ROOT / "output" / "logs"
FEATURES = [
    "Size", "Lev", "ROA", "Growth", "Cash", "Tangibility", "TobinQ",
    "FirmAge", "Loss", "return_volatility",
]
COLORS = {
    "qwen": "#C44E52",
    "logit": "#4C72B0",
    "hgb": "#55A868",
    "xgboost": "#8172B2",
    "gray": "#6B7280",
}


def load_analysis_module():
    path = ROOT / "code" / "39_analyze_strict_timing.py"
    spec = importlib.util.spec_from_file_location("strict_timing_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load strict analysis module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(float), weights=weights.to_numpy(float)))


def sample_table() -> pd.DataFrame:
    panel = pd.read_parquet(DERIVED / "strict_timing_panel_2011_2024.parquet")
    primary = pd.read_parquet(DERIVED / "strict_timing_primary_cases_256.parquet")
    probability = pd.read_parquet(DERIVED / "strict_timing_probability_cases_1500.parquet")
    rows = []
    for label, frame, weight in [
        ("Training frame, 2011--2018", panel.loc[panel["fiscal_year"].le(2018)], None),
        ("Holdout frame, 2019--2024", panel.loc[panel["fiscal_year"].ge(2019)], None),
        ("Paired behavioral audit", primary, None),
        ("Probability audit, unweighted", probability, None),
        ("Probability audit, design-weighted", probability, "sampling_weight"),
    ]:
        if weight is None:
            n = len(frame)
            events = float(frame["outcome"].sum())
            event_rate = float(frame["outcome"].mean())
        else:
            n = float(frame[weight].sum())
            events = float(np.sum(frame[weight] * frame["outcome"]))
            event_rate = events / n
        rows.append(
            {
                "sample": label,
                "records_or_weighted_records": n,
                "firms": int(frame["firm_id"].nunique()),
                "events_or_weighted_events": events,
                "event_rate": event_rate,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_sample_summary.csv", index=False)
    return out


def annual_table() -> pd.DataFrame:
    panel = pd.read_parquet(DERIVED / "strict_timing_panel_2011_2024.parquet")
    holdout = panel.loc[panel["fiscal_year"].between(2019, 2024)].copy()
    benchmark = pd.read_parquet(ANALYSIS / "strict_timing_benchmark_holdout_predictions.parquet")
    out = (
        holdout.groupby("fiscal_year", as_index=False)
        .agg(records=("row_id", "size"), firms=("firm_id", "nunique"), events=("outcome", "sum"))
    )
    out["event_rate"] = out["events"] / out["records"]
    for name, column in {
        "logit": "logit_probability",
        "hist_gradient_boosting": "hist_gradient_boosting_probability",
        "xgboost": "xgboost_probability",
    }.items():
        metrics = []
        for year, part in benchmark.groupby("fiscal_year"):
            metrics.append(
                {
                    "fiscal_year": year,
                    f"{name}_auc": roc_auc_score(part["outcome"], part[column]),
                    f"{name}_pr_auc": average_precision_score(part["outcome"], part[column]),
                }
            )
        out = out.merge(pd.DataFrame(metrics), on="fiscal_year", how="left")
    out["regime"] = np.where(out["fiscal_year"].ge(2023), "rule-exposed", "pre-rule")
    out.to_csv(ANALYSIS / "strict_timing_annual_frame.csv", index=False)
    return out


def descriptives() -> pd.DataFrame:
    panel = pd.read_parquet(DERIVED / "strict_timing_panel_2011_2024.parquet")
    holdout = panel.loc[panel["fiscal_year"].between(2019, 2024)]
    primary = pd.read_parquet(DERIVED / "strict_timing_primary_cases_256.parquet")
    probability = pd.read_parquet(DERIVED / "strict_timing_probability_cases_1500.parquet")
    rows = []
    for feature in FEATURES:
        for label, frame, weight in [
            ("holdout", holdout, None),
            ("paired_audit", primary, None),
            ("probability_unweighted", probability, None),
            ("probability_design_weighted", probability, "sampling_weight"),
        ]:
            values = frame[feature].astype(float)
            rows.append(
                {
                    "variable": feature,
                    "sample": label,
                    "mean": weighted_mean(values, frame[weight]) if weight else float(values.mean()),
                    "standard_deviation": float(values.std()),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_descriptives.csv", index=False)
    return out


def temporal_historical_bss() -> pd.DataFrame:
    module = load_analysis_module()
    cases, scores = module.probability_data()
    predictions = pd.read_parquet(ANALYSIS / "strict_timing_probability_predictions.parquet")
    y = cases["outcome"].to_numpy(int)
    weight = cases["sampling_weight"].to_numpy(float)
    rows = []
    for name in scores:
        temporal = predictions[f"{name}_forward_platt"].to_numpy(float)
        for year in (2022, 2023, 2024):
            train = cases["fiscal_year"].lt(year).to_numpy()
            test = cases["fiscal_year"].eq(year).to_numpy()
            p = temporal[test]
            w_test = weight[test] / weight[test].sum()
            historical_rate = float(np.average(y[train], weights=weight[train]))
            current_rate = float(np.average(y[test], weights=weight[test]))
            model_brier = float(np.sum(w_test * (y[test] - p) ** 2))
            historical_brier = float(np.sum(w_test * (y[test] - historical_rate) ** 2))
            current_brier = current_rate * (1 - current_rate)
            rows.append(
                {
                    "score": name,
                    "evaluation_year": year,
                    "evaluation_n": int(test.sum()),
                    "evaluation_events": int(y[test].sum()),
                    "historical_event_rate": historical_rate,
                    "evaluation_event_rate": current_rate,
                    "mean_prediction": float(np.sum(w_test * p)),
                    "roc_auc": roc_auc_score(y[test], p, sample_weight=w_test),
                    "pr_auc": average_precision_score(y[test], p, sample_weight=w_test),
                    "model_brier": model_brier,
                    "bss_vs_historical_rate": 1 - model_brier / historical_brier,
                    "bss_vs_evaluation_rate": 1 - model_brier / current_brier,
                }
            )
        eligible = cases["fiscal_year"].ge(2022).to_numpy()
        w_test = weight[eligible] / weight[eligible].sum()
        p = temporal[eligible]
        # The deployable yearly constants vary by year; score them row by row.
        history_reference = np.empty(eligible.sum(), dtype=float)
        eligible_years = cases.loc[eligible, "fiscal_year"].to_numpy(int)
        for year in (2022, 2023, 2024):
            train = cases["fiscal_year"].lt(year).to_numpy()
            history_reference[eligible_years == year] = np.average(y[train], weights=weight[train])
        model_brier = float(np.sum(w_test * (y[eligible] - p) ** 2))
        history_brier = float(np.sum(w_test * (y[eligible] - history_reference) ** 2))
        rows.append(
            {
                "score": name,
                "evaluation_year": "2022--2024 pooled",
                "evaluation_n": int(eligible.sum()),
                "evaluation_events": int(y[eligible].sum()),
                "historical_event_rate": np.nan,
                "evaluation_event_rate": float(np.average(y[eligible], weights=weight[eligible])),
                "mean_prediction": float(np.sum(w_test * p)),
                "roc_auc": roc_auc_score(y[eligible], p, sample_weight=w_test),
                "pr_auc": average_precision_score(y[eligible], p, sample_weight=w_test),
                "model_brier": model_brier,
                "bss_vs_historical_rate": 1 - model_brier / history_brier,
                "bss_vs_evaluation_rate": 1 - model_brier / (
                    np.average(y[eligible], weights=weight[eligible])
                    * (1 - np.average(y[eligible], weights=weight[eligible]))
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_forward_time_historical_bss.csv", index=False)
    return out


def reliability_bins() -> pd.DataFrame:
    predictions = pd.read_parquet(ANALYSIS / "strict_timing_probability_predictions.parquet")
    probability_columns = {
        "Qwen3-4B": "qwen_canonical_naive_grouped_platt",
        "Logit": "logit_grouped_platt",
        "XGBoost": "xgboost_grouped_platt",
    }
    rows = []
    for name, column in probability_columns.items():
        data = predictions[["outcome", "sampling_weight", column]].copy()
        data["bin"] = pd.qcut(data[column].rank(method="first"), 10, labels=False)
        for bin_number, part in data.groupby("bin"):
            w = part["sampling_weight"]
            rows.append(
                {
                    "score": name,
                    "bin": int(bin_number) + 1,
                    "records": len(part),
                    "unweighted_events": int(part["outcome"].sum()),
                    "weighted_records": float(w.sum()),
                    "mean_prediction": weighted_mean(part[column], w),
                    "observed_rate": weighted_mean(part["outcome"], w),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "strict_timing_reliability_bins.csv", index=False)
    return out


def make_figures(annual: pd.DataFrame, reliability: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()

    ranges = pd.read_parquet(ANALYSIS / "strict_timing_primary_case_ranges.parquet")
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    values = np.sort(ranges["score_range"].to_numpy(float))
    ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=COLORS["qwen"], lw=2)
    ax.axvline(np.median(values), color="#222222", ls="--", lw=1, label=f"Median = {np.median(values):.0f} pp")
    ax.set(xlabel="Within-firm range across three equivalent representations (percentage points)", ylabel="Cumulative share of firms", xlim=(0, 80), ylim=(0, 1.01))
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIGURES / "figure_representation_range.pdf")
    fig.savefig(FIGURES / "figure_representation_range.png")
    plt.close(fig)

    crossing = pd.read_csv(ANALYSIS / "strict_timing_threshold_crossing_curve.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(100 * crossing["threshold"], 100 * crossing["design_weighted_crossing_rate"], color=COLORS["qwen"], marker="o", ms=3, lw=1.8, label="Design-weighted")
    ax.plot(100 * crossing["threshold"], 100 * crossing["unweighted_crossing_rate"], color=COLORS["gray"], ls="--", lw=1.4, label="Audit-sample unweighted")
    ax.set(xlabel="External calibrated-probability threshold (%)", ylabel="Cases crossing across representations (%)", xlim=(0.25, 5), ylim=(0, None))
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "figure_threshold_crossing.pdf")
    fig.savefig(FIGURES / "figure_threshold_crossing.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), sharex=True, sharey=True)
    for ax, (name, part) in zip(axes, reliability.groupby("score", sort=False)):
        ax.scatter(100 * part["mean_prediction"], 100 * part["observed_rate"], s=24, color=(COLORS["qwen"] if name.startswith("Qwen") else COLORS["logit"] if name == "Logit" else COLORS["xgboost"]), zorder=3)
        upper = max(6, 100 * max(part["mean_prediction"].max(), part["observed_rate"].max()))
        ax.plot([0, upper], [0, upper], color="#777777", ls="--", lw=1)
        for _, row in part.iterrows():
            ax.annotate(str(int(row["unweighted_events"])), (100 * row["mean_prediction"], 100 * row["observed_rate"]), xytext=(2, 2), textcoords="offset points", fontsize=6, color="#444444")
        ax.set_title(name)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("Mean prediction (%)")
    axes[0].set_ylabel("Design-weighted event rate (%)")
    fig.savefig(FIGURES / "figure_reliability.pdf")
    fig.savefig(FIGURES / "figure_reliability.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    ax1.bar(annual["fiscal_year"], 100 * annual["event_rate"], color="#A7C5EB", width=0.65, label="Event rate")
    ax1.set(xlabel="Fiscal year", ylabel="One-year ST/*ST event rate (%)")
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(annual["fiscal_year"], annual["logit_auc"], marker="o", color=COLORS["logit"], label="Logit AUC")
    ax2.plot(annual["fiscal_year"], annual["xgboost_auc"], marker="s", color=COLORS["xgboost"], label="XGBoost AUC")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_ylim(0.55, 0.95)
    ax1.axvline(2022.5, color="#444444", ls="--", lw=1)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=3, loc="upper left", fontsize=8)
    fig.savefig(FIGURES / "figure_annual_regime.pdf")
    fig.savefig(FIGURES / "figure_annual_regime.png")
    plt.close(fig)

    artifacts = pd.read_csv(ANALYSIS / "strict_timing_artifact_profiles.csv")
    fig, ax = plt.subplots(figsize=(6.7, 3.6))
    x = np.arange(len(artifacts))
    ax.bar(x - 0.18, artifacts["median_representation_range_pp"], width=0.36, color="#DD8452", label="Median range")
    ax.bar(x + 0.18, artifacts["mean_representation_range_pp"], width=0.36, color="#4C72B0", label="Mean range")
    ax.set_xticks(x, ["Qwen 4B\n4-bit", "Qwen 4B\nbfloat16", "Qwen 8B\n4-bit", "Mistral 12B\n4-bit"])
    ax.set_ylabel("Representation range (percentage points)")
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "figure_artifact_profiles.pdf")
    fig.savefig(FIGURES / "figure_artifact_profiles.png")
    plt.close(fig)


def main() -> None:
    sample = sample_table()
    annual = annual_table()
    description = descriptives()
    forward = temporal_historical_bss()
    reliability = reliability_bins()
    make_figures(annual, reliability)
    log = {
        "status": "PASS",
        "sample_rows": len(sample),
        "annual_rows": len(annual),
        "descriptive_rows": len(description),
        "forward_time_rows": len(forward),
        "reliability_rows": len(reliability),
        "figure_files": len(list(FIGURES.glob("*.pdf"))),
    }
    (LOGS / "strict_timing_exhibits.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
