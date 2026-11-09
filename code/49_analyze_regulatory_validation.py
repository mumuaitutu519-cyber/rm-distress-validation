#!/usr/bin/env python3
"""Use regulatory records as external target-validity evidence.

The ten-variable generative interface and common-input structured benchmarks
remain unchanged.  This script instead asks whether information public by the
landmark predicts later ST/*ST and whether direct status-file reason codes help
describe the 2024-rule-exposed increase in outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260814
BOOTSTRAP_DRAWS = 1_000
ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
ANALYSIS = ROOT / "output" / "analysis"
TABLES = ROOT / "output" / "tables"
LOGS = ROOT / "output" / "logs"

PANEL_PATH = DERIVED / "strict_timing_panel_2011_2024.parquet"
FEATURE_PATH = ANALYSIS / "strict_timing_prelandmark_regulatory_features.parquet"
EVENT_LINK_PATH = ANALYSIS / "strict_timing_regulatory_event_linkage.csv"
STATUS_PATH = DERIVED / "strict_timing_st_changes.parquet"

SIGNALS = {
    "violation_financial_misconduct_3y": "Financial/disclosure violation in prior 3 years",
    "violation_fraud_or_false_statement_3y": "Fraud/false-statement violation in prior 3 years",
    "annual_inquiry_3y": "Annual-report inquiry in prior 3 years",
    "annual_inquiry_current_30d": "Annual-report inquiry by 30-day landmark",
    "annual_inquiry_requires_intermediary_3y": "Prior inquiry requiring intermediary review",
}


def cluster_rate_difference(
    frame: pd.DataFrame, signal: str, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    work = frame[["firm_id", "outcome", signal]].copy()
    work["exposed"] = work[signal].gt(0)
    work["y1"] = work["outcome"] * work["exposed"]
    work["n1"] = work["exposed"].astype(int)
    work["y0"] = work["outcome"] * (~work["exposed"])
    work["n0"] = (~work["exposed"]).astype(int)
    clustered = work.groupby("firm_id")[["y1", "n1", "y0", "n0"]].sum().to_numpy(float)
    rng = np.random.default_rng(SEED + sum(map(ord, signal)))
    differences = np.empty(draws)
    n_firms = len(clustered)
    for draw in range(draws):
        sampled = clustered[rng.integers(0, n_firms, size=n_firms)].sum(axis=0)
        differences[draw] = sampled[0] / sampled[1] - sampled[2] / sampled[3]
    return tuple(np.quantile(differences, [0.025, 0.975]))


def regulatory_signal_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal, label in SIGNALS.items():
        exposed = frame[signal].gt(0)
        event_rate_exposed = float(frame.loc[exposed, "outcome"].mean())
        event_rate_unexposed = float(frame.loc[~exposed, "outcome"].mean())
        difference = event_rate_exposed - event_rate_unexposed
        ci_low, ci_high = cluster_rate_difference(frame, signal)
        rows.append(
            {
                "signal": signal,
                "label": label,
                "exposed_records": int(exposed.sum()),
                "exposed_events": int(frame.loc[exposed, "outcome"].sum()),
                "unexposed_records": int((~exposed).sum()),
                "unexposed_events": int(frame.loc[~exposed, "outcome"].sum()),
                "event_rate_exposed": event_rate_exposed,
                "event_rate_unexposed": event_rate_unexposed,
                "rate_difference": difference,
                "rate_difference_ci_low": float(ci_low),
                "rate_difference_ci_high": float(ci_high),
                "risk_ratio": float(event_rate_exposed / event_rate_unexposed),
            }
        )
    return pd.DataFrame(rows)


def outcome_reason_frame(panel: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    events = panel.loc[
        panel["fiscal_year"].between(2019, 2024) & panel["outcome"].eq(1),
        ["row_id", "firm_id", "fiscal_year", "st_event_date_w30"],
    ].rename(columns={"st_event_date_w30": "execution_date"})
    st = status.loc[
        status["transition"].isin(["AB", "AD"]),
        [
            "firm_id",
            "execution_date",
            "transition",
            "Chgreas",
        ],
    ].copy()
    merged = events.merge(st, on=["firm_id", "execution_date"], how="left", validate="many_to_one")
    if len(merged) != 474 or merged["Chgreas"].isna().any():
        raise AssertionError("Status reason linkage did not recover all 474 holdout outcomes.")
    merged["reason_codes"] = merged["Chgreas"].astype(str).str.replace("，", ",", regex=False)
    for code in ["01", "02", "03", "04", "06", "07", "08", "11", "12", "14"]:
        merged[f"reason_{code}"] = merged["reason_codes"].str.split(",").map(
            lambda values: code in {value.strip() for value in values}
        )
    return merged


def reason_summary(reasons: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    labels = {
        "01": "Repeated losses",
        "02": "Net-asset deterioration",
        "03": "Adverse audit opinion",
        "04": "Other specified condition",
        "06": "Operational impairment",
        "07": "Stalled major matter",
        "08": "Frozen bank account",
        "11": "Information-disclosure violation",
        "12": "Investment risk",
        "14": "Other abnormal condition",
    }
    periods = {
        "2019--2022": reasons["fiscal_year"].between(2019, 2022),
        "2023": reasons["fiscal_year"].eq(2023),
        "2024": reasons["fiscal_year"].eq(2024),
    }
    rows: list[dict[str, object]] = []
    for code, label in labels.items():
        row: dict[str, object] = {"reason_code": code, "reason": label}
        for period, mask in periods.items():
            row[f"count_{period}"] = int(reasons.loc[mask, f"reason_{code}"].sum())
            row[f"share_{period}"] = float(reasons.loc[mask, f"reason_{code}"].mean())
        rows.append(row)

    before_events = int(periods["2019--2022"].sum())
    exposed_events = int((periods["2023"] | periods["2024"]).sum())
    disclosure_before = int(
        reasons.loc[periods["2019--2022"], "reason_11"].sum()
    )
    disclosure_exposed = int(
        reasons.loc[periods["2023"] | periods["2024"], "reason_11"].sum()
    )
    decomposition = {
        "pre_records": 15_873,
        "rule_exposed_records": 9_517,
        "pre_events": before_events,
        "rule_exposed_events": exposed_events,
        "pre_disclosure_reason_events": disclosure_before,
        "rule_exposed_disclosure_reason_events": disclosure_exposed,
        "raw_rate_pre": before_events / 15_873,
        "raw_rate_rule_exposed": exposed_events / 9_517,
        "raw_rate_difference": exposed_events / 9_517 - before_events / 15_873,
        "excluding_disclosure_rate_pre": (before_events - disclosure_before) / 15_873,
        "excluding_disclosure_rate_rule_exposed": (
            exposed_events - disclosure_exposed
        )
        / 9_517,
        "excluding_disclosure_rate_difference": (
            (exposed_events - disclosure_exposed) / 9_517
            - (before_events - disclosure_before) / 15_873
        ),
    }
    return pd.DataFrame(rows), decomposition


def write_latex(
    signal_table: pd.DataFrame,
    reason_table: pd.DataFrame,
    decomposition: dict[str, object],
) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    signal_lines = []
    for row in signal_table.itertuples(index=False):
        signal_lines.append(
            f"{row.label} & {row.exposed_records:,} & {100*row.event_rate_exposed:.2f} & "
            f"{100*row.event_rate_unexposed:.2f} & {100*row.rate_difference:.2f} "
            f"[{100*row.rate_difference_ci_low:.2f}, {100*row.rate_difference_ci_high:.2f}] \\\\"
        )
    (TABLES / "table_regulatory_signal_validation.tex").write_text(
        """\\begin{table}[H]
\\centering
\\caption{Pre-landmark regulatory signals and subsequent ST/*ST outcomes}
\\label{tab:regulatory-signals}
\\small
\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrr}
\\toprule
Signal public by landmark & Exposed $N$ & Exposed rate (\\%) & Other rate (\\%) & Difference, pp [95\\% CI] \\\\
\\midrule
"""
        + "\n".join(signal_lines)
        + """
\\bottomrule
\\end{tabular}}
\\begin{minipage}{0.97\\textwidth}\\footnotesize
\\textit{Notes:} The frame contains 25,390 at-risk records and 474 outcomes. Dates are enforced at the 30-day annual-report landmark. Intervals use 1,000 firm-cluster bootstrap draws. These are descriptive external-validity checks; the regulatory records are not supplied to the generative interface or the common-input structured benchmarks.
\\end{minipage}
\\end{table}
""",
        encoding="utf-8",
    )

    reason_lines = []
    for _, row in reason_table.iterrows():
        reason_lines.append(
            f"{row['reason']} ({row['reason_code']}) & {int(row['count_2019--2022'])} & "
            f"{int(row['count_2023'])} & {int(row['count_2024'])} \\\\"
        )
    (TABLES / "table_status_reason_decomposition.tex").write_text(
        """\\begin{table}[H]
\\centering
\\caption{Direct status-file reasons for holdout outcomes}
\\label{tab:status-reasons}
\\small
\\begin{tabular}{lrrr}
\\toprule
Recorded reason & 2019--2022 & 2023 & 2024 \\\\
\\midrule
"""
        + "\n".join(reason_lines)
        + f"""
\\midrule
Any outcome & {int(decomposition['pre_events'])} & 109 & 119 \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{minipage}}{{0.94\\textwidth}}\\footnotesize
\\textit{{Notes:}} Fiscal year indexes the financial record; the designation occurs after its report landmark. One event may carry multiple reason codes, so columns need not sum to the number of outcomes. Code 11 records an information-disclosure violation. Its count rises from {int(decomposition['pre_disclosure_reason_events'])} of {int(decomposition['pre_events'])} outcomes in 2019--2022 to {int(decomposition['rule_exposed_disclosure_reason_events'])} of {int(decomposition['rule_exposed_events'])} in 2023--2024. This is a descriptive decomposition, not a causal estimate of the rule revision.
\\end{{minipage}}
\\end{{table}}
""",
        encoding="utf-8",
    )


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    holdout = panel.loc[panel["fiscal_year"].between(2019, 2024)].copy()
    features = pd.read_parquet(FEATURE_PATH)
    if len(features) != 25_390:
        raise AssertionError("Unexpected regulatory-feature frame size.")
    frame = holdout.merge(features.drop(columns=["firm_id", "fiscal_year", "outcome"]), on="row_id")
    signal_table = regulatory_signal_table(frame)

    status = pd.read_parquet(STATUS_PATH)
    reasons = outcome_reason_frame(panel, status)
    reason_table, decomposition = reason_summary(reasons)

    links = pd.read_csv(EVENT_LINK_PATH)
    links = links.merge(reasons[["row_id", "reason_11"]], on="row_id", how="left")
    corroboration = (
        links.loc[links["reason_11"]]
        .groupby("fiscal_year", as_index=False)
        .agg(
            disclosure_reason_events=("row_id", "size"),
            financial_violation_near_event=("financial_violation_near_event", "sum"),
            fraud_violation_near_event=("fraud_violation_near_event", "sum"),
        )
    )

    signal_table.to_csv(ANALYSIS / "strict_timing_regulatory_signal_validation.csv", index=False)
    reason_table.to_csv(ANALYSIS / "strict_timing_status_reason_decomposition.csv", index=False)
    write_latex(signal_table, reason_table, decomposition)

    summary = {
        "regulatory_signals": signal_table.to_dict("records"),
        "reason_decomposition": decomposition,
        "disclosure_reason_corroboration": corroboration.to_dict("records"),
        "interpretation": {
            "prediction_use": "Only records public by the landmark enter signal definitions.",
            "interface_use": "External regulatory data are not added to the ten-variable prompt.",
            "regime_use": "Direct status-file reasons describe, but do not causally identify, the rule-exposed increase.",
        },
    }
    (LOGS / "strict_timing_regulatory_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(signal_table.to_string(index=False))
    print(json.dumps(decomposition, ensure_ascii=False, indent=2))
    print(corroboration.to_string(index=False))


if __name__ == "__main__":
    main()
