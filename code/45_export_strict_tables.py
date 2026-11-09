#!/usr/bin/env python3
"""Export compact LaTeX tables for the strict-timing manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "output" / "analysis"
TABLES = ROOT / "output" / "tables"
LOGS = ROOT / "output" / "logs"


def write(name: str, text: str) -> None:
    (TABLES / name).write_text(text.strip() + "\n", encoding="utf-8")


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}\\%"


def ci(point: float, lower: float, upper: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def tex(value: object) -> str:
    return str(value).replace("_", r"\_")


def main_sample() -> None:
    annual = pd.read_csv(ANALYSIS / "strict_timing_annual_frame.csv")
    lines = []
    for _, row in annual.iterrows():
        lines.append(
            f"{int(row.fiscal_year)} & {int(row.records):,} & {int(row.events)} & "
            f"{100 * row.event_rate:.2f} & {row.regime.replace('-', '--')} \\\\"
        )
    write(
        "table1_sample.tex",
        r"""
\begin{table}[!htbp]
\centering
\caption{Strict One-Year Holdout Risk Set}
\label{tab:sample}
\begin{threeparttable}
\begin{tabular}{lrrrr}
\toprule
Fiscal year & Records & Events & Event rate (\%) & Regime \\
\midrule
""" + "\n".join(lines) + r"""
\midrule
Total & 25,390 & 474 & 1.87 & --- \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item Notes: Each record is a nonfinancial firm in normal status 30 calendar days after its annual-report announcement. The event is the first transition to ST or *ST during the following calendar year. Fiscal years 2023--2024 are reported separately as rule-exposed because the 2024 listing-rule changes can affect the outcome definition.
\end{tablenotes}
\end{threeparttable}
\end{table}
""",
    )


def main_performance() -> None:
    metrics = pd.read_csv(ANALYSIS / "strict_timing_probability_metrics.csv")
    boot = pd.read_csv(ANALYSIS / "strict_timing_probability_bootstrap_ci.csv")
    labels = {
        "qwen_canonical_naive": "Qwen3-4B score",
        "logit": "Regularized logit",
        "hist_gradient_boosting": "Histogram gradient boosting",
        "xgboost": "XGBoost",
    }
    lines = []
    for score, label in labels.items():
        raw = metrics[(metrics.score == score) & (metrics.evaluation == "raw")].iloc[0]
        cal = metrics[(metrics.score == score) & (metrics.evaluation == "five_fold_grouped_crossfit_platt")].iloc[0]
        auc_ci = boot[(boot.score == score) & (boot.metric == "raw_roc_auc")].iloc[0]
        bss_ci = boot[(boot.score == score) & (boot.metric == "calibrated_brier_skill_score")].iloc[0]
        lines.append(
            f"{label} & {ci(raw.roc_auc, auc_ci.ci_lower, auc_ci.ci_upper)} & {raw.pr_auc:.3f} & "
            f"{100 * cal.weighted_mean_probability:.2f} & {cal.brier:.4f} & "
            f"{ci(cal.brier_skill_score, bss_ci.ci_lower, bss_ci.ci_upper)} & {cal.log_loss:.4f} \\\\"
        )
    write(
        "table2_performance.tex",
        r"""
\begin{table}[!htbp]
\centering
\caption{Design-Weighted Ranking and Probability Quality}
\label{tab:performance}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
Score & Raw ROC-AUC [95\% CI] & Raw PR-AUC & Mean calibrated $p$ (\%) & Brier & BSS [95\% CI] & Log loss \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\begin{minipage}{\textwidth}\footnotesize
\textit{Notes:} $N=1{,}500$ sampled records, including all 474 events; inverse-inclusion weights recover 25,390 holdout records and a 1.87\% event rate. Ranking metrics use raw scores. Probability columns use firm-grouped five-fold Platt calibration. Each Qwen representation receives the map estimated on canonical scores; no representation-specific refitting is allowed. Confidence intervals use 1,000 firm-cluster bootstrap draws and refit calibration within each draw.
\end{minipage}
\end{table}
""",
    )


def main_stability() -> None:
    ranges = pd.read_csv(ANALYSIS / "strict_timing_primary_range_summary.csv").iloc[0]
    pboot = pd.read_csv(ANALYSIS / "strict_timing_primary_bootstrap_ci.csv")
    median_ci = pboot[pboot.metric == "median_representation_range_pp"].iloc[0]
    roster = pd.read_csv(ANALYSIS / "strict_timing_primary_rosters.csv")
    five = roster[(roster.capacity == 0.05) & (roster.variant == "three_representation_summary")].iloc[0]
    ten = roster[(roster.capacity == 0.10) & (roster.variant == "three_representation_summary")].iloc[0]
    crossing = pd.read_csv(ANALYSIS / "strict_timing_threshold_crossing_curve.csv")
    prob_ci = pd.read_csv(ANALYSIS / "strict_timing_probability_bootstrap_ci.csv")
    one = crossing[np.isclose(crossing.threshold, 0.01)].iloc[0]
    two = crossing[np.isclose(crossing.threshold, 0.02)].iloc[0]
    one_ci = prob_ci[(prob_ci.score == "qwen_canonical_naive") & (prob_ci.metric == "crossing_1pct")].iloc[0]
    two_ci = prob_ci[(prob_ci.score == "qwen_canonical_naive") & (prob_ci.metric == "crossing_2pct")].iloc[0]
    write(
        "table3_stability.tex",
        rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Representation and Action Stability}}
\label{{tab:stability}}
\begin{{threeparttable}}
\begin{{tabular}}{{lrr}}
\toprule
Diagnostic & Estimate & Reference \\
\midrule
Raw score range, median & {ranges.median_range_pp:.1f} pp [{median_ci.ci_lower:.1f}, {median_ci.ci_upper:.1f}] & 256-firm audit \\
Raw score range, mean & {ranges.mean_range_pp:.1f} pp & 256-firm audit \\
Any raw score change & {pct(ranges.any_change_rate)} & 256-firm audit \\
1\% calibrated-threshold crossing & {pct(one.design_weighted_crossing_rate)} [{pct(one_ci.ci_lower)}, {pct(one_ci.ci_upper)}] & Weighted frame \\
2\% calibrated-threshold crossing & {pct(two.design_weighted_crossing_rate)} [{pct(two_ci.ci_lower)}, {pct(two_ci.ci_upper)}] & Weighted frame \\
5\% roster intersection / union & {int(five.intersection_all)} / {int(five.union_all)} & $k=13$ per representation \\
5\% roster Jaccard overlap & {five.jaccard_all:.3f} & Enriched audit \\
10\% roster intersection / union & {int(ten.intersection_all)} / {int(ten.union_all)} & $k=26$ per representation \\
10\% roster Jaccard overlap & {ten.jaccard_all:.3f} & Enriched audit \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item Notes: Equivalent representations are canonical bullets, reversed feature order, and decimal rather than percentage units. Threshold rates use design weights and a common canonical Platt map. Roster analyses are stress tests within the deliberately event-enriched 256-firm audit; they are not estimates for the full holdout population. Fixed case identifiers break ties and enforce exact top-$k$ capacity.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""",
    )


def main_behavior() -> None:
    info = pd.read_csv(ANALYSIS / "strict_timing_information_fe_coefficients.csv")
    adverse = info[info.term.str.contains("T.adverse")].iloc[0]
    favorable = info[info.term.str.contains("T.favorable")].iloc[0]
    after = info[info.term.str.contains("T.after")].iloc[0]
    system = info[info.term.str.contains("T.system")].iloc[0]
    order = pd.read_csv(ANALYSIS / "strict_timing_order_summary.csv").iloc[0]
    econ = pd.read_csv(ANALYSIS / "strict_timing_economic_response.csv").set_index("variant")
    write(
        "table4_behavior.tex",
        rf"""
\begin{{table}}[!htbp]
\centering
\caption{{Extended Behavioral Diagnostics on 64 Firms}}
\label{{tab:behavior}}
\begin{{threeparttable}}
\begin{{tabular}}{{lrr}}
\toprule
Diagnostic & Estimate (pp) & 95\% CI / auxiliary statistic \\
\midrule
Six-order within-firm range, median & {order.median_order_range_pp:.2f} & Mean pairwise difference {order.mean_pairwise_order_difference_pp:.2f} \\
Adverse vs neutral unsupported text & {adverse.estimate_pp:.2f} & [{adverse.ci_lower:.2f}, {adverse.ci_upper:.2f}] \\
Favorable vs neutral unsupported text & {favorable.estimate_pp:.2f} & [{favorable.ci_lower:.2f}, {favorable.ci_upper:.2f}] \\
Text after vs before facts & {after.estimate_pp:.2f} & [{after.ci_lower:.2f}, {after.ci_upper:.2f}] \\
Text in system vs before facts & {system.estimate_pp:.2f} & [{system.ci_lower:.2f}, {system.ci_upper:.2f}] \\
Joint deterioration & {econ.loc['strict_deteriorated','mean_delta_pp']:.2f} & Direction violations {pct(econ.loc['strict_deteriorated','direction_violation_rate_when_changed'])} \\
Joint improvement & {econ.loc['strict_improved','mean_delta_pp']:.2f} & Direction violations {pct(econ.loc['strict_improved','direction_violation_rate_when_changed'])} \\
ROA deterioration & {econ.loc['strict_roa_down_1pp','mean_delta_pp']:.2f} & Direction violations {pct(econ.loc['strict_roa_down_1pp','direction_violation_rate_when_changed'])} \\
Cash deterioration & {econ.loc['strict_cash_down_2pp','mean_delta_pp']:.2f} & Direction violations {pct(econ.loc['strict_cash_down_2pp','direction_violation_rate_when_changed'])} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]\footnotesize
\item Notes: The information experiment is a balanced $3\times3\times3$ factorial (valence, wording, placement) on every firm. Coefficients come from case fixed effects with firm-clustered standard errors. Economic direction violations are conditional on a changed output; sales-growth shocks have no imposed global sign and are omitted here.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""",
    )


def main_artifacts() -> None:
    profiles = pd.read_csv(ANALYSIS / "strict_timing_artifact_profiles.csv")
    lines = []
    for _, row in profiles.iterrows():
        lines.append(
            f"{row.artifact} & {row.canonical_auc:.3f} & {row.median_representation_range_pp:.1f} & "
            f"{row.mean_representation_range_pp:.1f} & {pct(row.any_representation_change_rate)} & {row.maximum_representation_range_pp:.1f} \\\\"
        )
    write(
        "table5_artifacts.tex",
        r"""
\begin{table}[!htbp]
\centering
\caption{Target-Aligned Profiles of Four Frozen Artifacts}
\label{tab:artifacts}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Artifact & Canonical AUC & Median range & Mean range & Any change & Maximum range \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\begin{minipage}{\textwidth}\footnotesize
\textit{Notes:} All artifacts receive the same 64 cases, three representations, prompt, batch size one, and temperature zero. AUC values are descriptive within an event-enriched sample. Model family, parameter count, tokenizer, and quantization are not separately identified; rows are artifact profiles, not causal size effects.
\end{minipage}
\end{table}
""",
    )


def main_temporal() -> None:
    temporal = pd.read_csv(ANALYSIS / "strict_timing_forward_time_historical_bss.csv")
    labels = {
        "qwen_canonical_naive": "Qwen3-4B score",
        "logit": "Regularized logit",
        "hist_gradient_boosting": "Histogram gradient boosting",
        "xgboost": "XGBoost",
    }
    rows = []
    for score, label in labels.items():
        row = temporal[(temporal.score == score) & (temporal.evaluation_year == "2022--2024 pooled")].iloc[0]
        rows.append(
            f"{label} & {row.roc_auc:.3f} & {row.pr_auc:.3f} & {100 * row.mean_prediction:.2f} & {row.model_brier:.4f} & {row.bss_vs_historical_rate:.3f} \\\\"
        )
    write(
        "table6_temporal.tex",
        r"""
\begin{table}[!htbp]
\centering
\caption{Forward-Time Calibration, 2022--2024}
\label{tab:temporal}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Score & ROC-AUC & PR-AUC & Mean $p$ (\%) & Brier & BSS vs historical rate \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\begin{minipage}{\textwidth}\footnotesize
\textit{Notes:} The 2022 map uses 2019--2021, the 2023 map uses 2019--2022, and the 2024 map uses 2019--2023. The reference forecast for each year is the design-weighted event rate available from prior years only. Pooled BSS scores these varying historical-rate forecasts on their corresponding evaluation years.
\end{minipage}
\end{table}
""",
    )


def appendix_tables() -> None:
    desc = pd.read_csv(ANALYSIS / "strict_timing_descriptives.csv")
    hold = desc[desc["sample"] == "holdout"].set_index("variable")
    audit = desc[desc["sample"] == "paired_audit"].set_index("variable")
    weighted = desc[desc["sample"] == "probability_design_weighted"].set_index("variable")
    lines = []
    for variable in hold.index:
        lines.append(
            f"{tex(variable)} & {hold.loc[variable,'mean']:.3f} & {hold.loc[variable,'standard_deviation']:.3f} & "
            f"{audit.loc[variable,'mean']:.3f} & {weighted.loc[variable,'mean']:.3f} \\\\"
        )
    write(
        "tableA1_descriptives.tex",
        r"""
\begin{table}[!htbp]\centering
\caption{Input Descriptives Across Samples}\label{tab:a_descriptives}
\begin{threeparttable}\begin{tabular}{lrrrr}\toprule
Variable & Holdout mean & Holdout SD & Paired-audit mean & Weighted-audit mean \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize\item Notes: The weighted probability-audit mean uses inverse inclusion weights. Variables are the training-period-winsorized, prompt-rounded inputs used by all models.\end{tablenotes}
\end{threeparttable}\end{table}
""",
    )

    annual = pd.read_csv(ANALYSIS / "strict_timing_annual_frame.csv")
    lines = [
        f"{int(r.fiscal_year)} & {int(r.records)} & {int(r.events)} & {100*r.event_rate:.2f} & {r.logit_auc:.3f} & {r.xgboost_auc:.3f} \\\\"
        for _, r in annual.iterrows()
    ]
    write(
        "tableA2_annual.tex",
        r"""
\begin{table}[!htbp]\centering
\caption{Annual Holdout Composition and Structured-Model Ranking}\label{tab:a_annual}
\begin{tabular}{lrrrrr}\toprule
Year & Records & Events & Event rate (\%) & Logit AUC & XGBoost AUC \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table}
""",
    )

    extended = pd.read_csv(ANALYSIS / "strict_timing_extended_paired_summary.csv")
    lines = [
        f"{tex(r.variant)} & {r.mean_delta_pp:.2f} & {r.mean_absolute_delta_pp:.2f} & {100*r.changed_rate:.1f} & {r.maximum_absolute_delta_pp:.1f} \\\\"
        for _, r in extended.iterrows()
    ]
    write(
        "tableA3_extended.tex",
        r"""
{\small
\begin{longtable}{@{}p{0.35\textwidth}rrrr@{}}
\caption{All Extended Variants Relative to Canonical}\label{tab:a_extended}\\
\toprule Variant & Mean $\Delta$ & Mean $|\Delta|$ & Changed (\%) & Max $|\Delta|$ \\
\midrule\endfirsthead
\toprule Variant & Mean $\Delta$ & Mean $|\Delta|$ & Changed (\%) & Max $|\Delta|$ \\
\midrule\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}}
""",
    )

    temporal = pd.read_csv(ANALYSIS / "strict_timing_forward_time_historical_bss.csv")
    temporal = temporal[temporal.score.isin(["qwen_canonical_naive", "logit", "xgboost"])]
    lines = []
    for _, r in temporal.iterrows():
        lines.append(
            f"{tex(r.score)} & {r.evaluation_year} & {r.roc_auc:.3f} & {100*r.mean_prediction:.2f} & {r.bss_vs_historical_rate:.3f} \\\\"
        )
    write(
        "tableA4_temporal.tex",
        r"""
{\small\begin{longtable}{@{}p{0.29\textwidth}lrrr@{}}
\caption{Year-Specific Forward-Time Results}\label{tab:a_temporal}\\
\toprule Score & Year & ROC-AUC & Mean $p$ (\%) & Historical BSS \\
\midrule\endfirsthead
\toprule Score & Year & ROC-AUC & Mean $p$ (\%) & Historical BSS \\
\midrule\endhead
""" + "\n".join(lines) + r"""
\bottomrule\end{longtable}}
""",
    )

    calibration = pd.read_csv(ANALYSIS / "strict_timing_calibration_robustness.csv")
    calibration = calibration.loc[
        calibration["score"].eq("qwen_canonical_naive")
        | calibration["calibration"].eq("platt")
    ].copy()
    score_labels = {
        "qwen_canonical_naive": "Qwen3-4B",
        "logit": "Logit",
        "hist_gradient_boosting": "Histogram boosting",
        "xgboost": "XGBoost",
    }
    method_labels = {
        "intercept_only": "Intercept only",
        "platt": "Platt",
        "beta": "Beta",
        "isotonic": "Isotonic",
    }
    lines = []
    for _, r in calibration.iterrows():
        lines.append(
            f"{score_labels[r.score]} & {method_labels[r.calibration]} & "
            f"{100*r.weighted_mean_probability:.2f} & {r.brier:.4f} & "
            f"{r.brier_skill_score:.3f} & {r.log_loss:.4f} & "
            f"{r.pooled_calibration_intercept:.3f} & {r.pooled_calibration_slope:.3f} \\\\"
        )
    write(
        "tableA5_calibration.tex",
        r"""
\begin{table}[!htbp]\centering
\caption{Alternative Grouped Cross-Fit Calibration Maps}\label{tab:a_calibration}
\resizebox{\textwidth}{!}{
\begin{tabular}{llrrrrrr}\toprule
Score & Map & Mean $p$ (\%) & Brier & BSS & Log loss & Pooled intercept & Pooled slope \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}}
\begin{minipage}{\textwidth}\footnotesize\textit{Notes:} All maps use the same firm-grouped folds and design weights. The Qwen rows compare four candidate maps on canonical scores. The remaining rows report the prespecified Platt map for the structured benchmarks. Ranking after pooled cross-fitting can change slightly because held-out folds receive separately estimated maps.\end{minipage}
\end{table}
""",
    )

    washout = pd.read_csv(ANALYSIS / "strict_timing_washout_risksets.csv")
    lines = []
    for _, r in washout.iterrows():
        lines.append(
            f"{int(r.washout_days)} & {r.period.title()} & {int(r.records):,} & "
            f"{int(r.firms):,} & {int(r.st_events)} & {100*r.st_event_rate:.2f} & "
            f"{int(r.composite_events)} & {100*r.composite_event_rate:.2f} \\\\"
        )
    write(
        "tableA6_washout.tex",
        r"""
\begin{table}[!htbp]\centering
\caption{Independently Rebuilt Risk Sets by Washout}\label{tab:a_washout}
\resizebox{\textwidth}{!}{
\begin{tabular}{rlrrrrrr}\toprule
Days & Period & Records & Firms & ST/*ST events & Rate (\%) & Composite events & Rate (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}}
\begin{minipage}{\textwidth}\footnotesize\textit{Notes:} Each row rebuilds eligibility at its own landmark; the zero- and seven-day rows are not conditional on surviving the 30-day risk set. The composite adds transitions to trading-warning and direct-delisting states. Changing the landmark changes both who remains normal and the end of the one-year window, so these rows describe alternative targets rather than a parameter perturbation on a fixed cohort.\end{minipage}
\end{table}
""",
    )

    auc = pd.read_csv(ANALYSIS / "strict_timing_auc_difference_ci.csv")
    contrast_labels = {
        "qwen_minus_logit": "Qwen minus logit",
        "qwen_minus_hist_gradient_boosting": "Qwen minus histogram boosting",
        "qwen_minus_xgboost": "Qwen minus XGBoost",
    }
    lines = [
        f"{contrast_labels[r.contrast]} & {r.mean_difference:.3f} & "
        f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}] & {100*r.share_positive:.1f} \\\\"
        for _, r in auc.iterrows()
    ]
    write(
        "tableA7_auc_differences.tex",
        r"""
\begin{table}[!htbp]\centering
\caption{Paired Design-Weighted AUC Contrasts}\label{tab:a_auc_diff}
\begin{tabular}{lrrr}\toprule
Contrast & Mean difference & 95\% bootstrap CI & Positive draws (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\begin{minipage}{0.92\textwidth}\footnotesize\textit{Notes:} The same 1,500 sampled records and design weights enter both members of each contrast. Intervals use 1,000 firm-cluster bootstrap draws.\end{minipage}
\end{table}
""",
    )

    batch = pd.read_csv(ANALYSIS / "strict_timing_batch_sensitivity.csv").iloc[0]
    quant = pd.read_csv(ANALYSIS / "strict_timing_quantization_sensitivity.csv").iloc[0]
    reload = pd.read_csv(ANALYSIS / "strict_timing_reload_repeatability.csv").iloc[0]
    lines = [
        f"Fresh reload, same artifact and batch & {int(reload.calls)} & {100*reload.score_changed_rate:.2f} & 0.00 & {reload.maximum_absolute_difference_pp:.1f} \\\\ ",
        f"Batch size 1 versus 8 & {int(batch.calls)} & {100*batch.changed_rate:.2f} & {batch.mean_absolute_difference_pp:.2f} & {batch.maximum_absolute_difference_pp:.1f} \\\\ ",
        f"Qwen3-4B 4-bit versus bfloat16 & {int(quant.calls)} & {100*quant.changed_rate:.2f} & {quant.mean_absolute_difference_pp:.2f} & {quant.maximum_absolute_difference_pp:.1f} \\\\ ",
    ]
    write(
        "tableA8_implementation.tex",
        r"""
\begin{table}[!htbp]\centering
\caption{Repeatability and Implementation-Context Diagnostics}\label{tab:a_implementation}
\resizebox{\textwidth}{!}{\begin{tabular}{lrrrr}\toprule
Comparison & Calls & Changed (\%) & Mean absolute change & Maximum change \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}}
\begin{minipage}{0.94\textwidth}\footnotesize\textit{Notes:} All comparisons are prompt-matched. Fresh reload achieves 100\% raw-string equality. Batch and precision comparisons are properties of the tested MLX artifacts, not general engine effects.\end{minipage}
\end{table}
""",
    )

    margins = pd.read_csv(ANALYSIS / "strict_timing_information_margins.csv")
    lines = [
        f"{r.valence.title()} & {r.placement.title()} & {r.mean_delta_pp:.2f} & "
        f"{r.mean_absolute_delta_pp:.2f} & {100*r.changed_rate:.1f} \\\\"
        for _, r in margins.iterrows()
    ]
    write(
        "tableA9_information.tex",
        r"""
\begin{table}[!htbp]\centering
\caption{Information-Factorial Margins Relative to Canonical}\label{tab:a_information}
\begin{tabular}{llrrr}\toprule
Valence & Placement & Mean change & Mean absolute change & Changed (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\begin{minipage}{0.92\textwidth}\footnotesize\textit{Notes:} Each cell pools three wording templates and 64 firms ($n=192$). These raw margins are descriptive; Table 4 reports case fixed-effect contrasts that jointly control valence, wording, and placement.\end{minipage}
\end{table}
""",
    )

    economic = pd.read_csv(ANALYSIS / "strict_timing_economic_response.csv")
    lines = []
    for _, r in economic.iterrows():
        direction = {1: "Increase", -1: "Decrease", 0: "None"}[int(r.expected_direction)]
        violation = "---" if pd.isna(r.direction_violation_rate_when_changed) else f"{100*r.direction_violation_rate_when_changed:.1f}"
        reverse = "---" if pd.isna(r.maximum_reverse_change_pp) else f"{r.maximum_reverse_change_pp:.1f}"
        lines.append(
            f"{tex(r.variant)} & {direction} & {r.mean_delta_pp:.2f} & {100*r.changed_rate:.1f} & "
            f"{violation} & {reverse} & {100*r.marginal_training_support_rate:.1f} \\\\"
        )
    write(
        "tableA10_economic.tex",
        r"""
{\scriptsize\begin{longtable}{@{}p{0.27\textwidth}lrrrrr@{}}
\caption{Economic-Response Diagnostics and Empirical Support}\label{tab:a_economic}\\
\toprule Variant & Expected & Mean $\Delta$ & Changed & Viol. & Max rev. & Support \\
\midrule\endfirsthead
\toprule Variant & Expected & Mean $\Delta$ & Changed & Viol. & Max rev. & Support \\
\midrule\endhead
""" + "\n".join(lines) + r"""
\bottomrule\end{longtable}}
\begin{minipage}{\textwidth}\footnotesize\textit{Notes:} Violation rates condition on a changed output. Growth shocks have no global sign restriction. Support is the share of perturbed marginal values inside the training-period empirical minimum and maximum; it is not a joint-support diagnostic.\end{minipage}
""",
    )

    primary = pd.read_csv(ANALYSIS / "strict_timing_primary_metrics.csv")
    artifact = pd.read_csv(ANALYSIS / "strict_timing_artifact_variant_metrics.csv")
    precision_rows = []
    for _, r in primary.iterrows():
        precision_rows.append(
            f"Qwen3-4B 4-bit, $N=256$ & {tex(r.variant)} & {int(r.unique_scores)} & "
            f"{100*r.integer_output_rate:.1f} \\\\"
        )
    for _, r in artifact[artifact["variant"].eq("strict_canonical")].iterrows():
        precision_rows.append(
            f"{r.artifact}, $N=64$ & canonical & {int(r.unique_scores)} & 100.0 \\\\"
        )
    write(
        "tableA11_precision.tex",
        r"""
\begin{table}[!htbp]\centering
\caption{Output Precision and Score Support}\label{tab:a_precision}
\begin{tabular}{llrr}\toprule
Artifact/sample & Representation & Unique scores & Integer outputs (\%) \\
\midrule
""" + "\n".join(precision_rows) + r"""
\bottomrule\end{tabular}
\begin{minipage}{0.92\textwidth}\footnotesize\textit{Notes:} The output contract permits any number from 0 to 100 but does not require a decimal place. The tested calls nevertheless return integer scores throughout, which limits observable local sensitivity.\end{minipage}
\end{table}
""",
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    main_sample()
    main_performance()
    main_stability()
    main_behavior()
    main_artifacts()
    main_temporal()
    appendix_tables()
    log = {"status": "PASS", "tables": len(list(TABLES.glob("*.tex")))}
    (LOGS / "strict_timing_tables.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
