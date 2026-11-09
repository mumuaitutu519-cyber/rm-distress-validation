#!/usr/bin/env python3
"""Audit newly supplied violation and annual-report inquiry-letter data.

The audit treats public availability as part of the variable definition.  A
regulatory record can be used as a predictor only when its public date is no
later than the annual-report landmark.  Later records may describe outcomes or
mechanisms, but they are never counted as pre-landmark information.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DERIVED = ROOT / "data" / "derived"
ANALYSIS = ROOT / "output" / "analysis"
LOGS = ROOT / "output" / "logs"

VIOLATION_XLSX = (
    WORKSPACE
    / "codex项目/（更新）上市公司-违法行为违规事件内容处罚数据（1993-2025.7）"
    / "违法行为违规事件内容处罚数据.xlsx"
)
OLD_VIOLATION_XLSX = (
    WORKSPACE / "codex项目/上市公司-违规与处罚2024/上市公司违规与处罚.xlsx"
)
INQUIRY_XLSX = WORKSPACE / "codex项目/年报问询函/问询函原始数据.xlsx"
PANEL_PATH = DERIVED / "strict_timing_panel_2011_2024.parquet"

FINANCIAL_MISCONDUCT_CODES = {
    "P2501",  # fabricated profit
    "P2502",  # fabricated assets
    "P2503",  # false or misleading statements
    "P2504",  # delayed disclosure
    "P2505",  # material omission
    "P2506",  # other inaccurate disclosure
    "P2507",  # fraudulent listing
    "P2515",  # improper accounting treatment
}


def normalize_code(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d{1,6})", expand=False)
    return extracted.str.zfill(6)


def parse_exact_date(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = series.astype("string").str.strip()
    partial = raw.str.match(r"^\d{4}-(?:00|\d{2})-(?:00|\d{2})$", na=False) & raw.str.contains(
        "-00", regex=False, na=False
    )
    exact = pd.to_datetime(raw.mask(partial), errors="coerce")
    return exact, partial


def read_violations() -> pd.DataFrame:
    frame = pd.read_excel(VIOLATION_XLSX, engine="openpyxl")
    # The first data row repeats Chinese labels and units rather than an event.
    frame = frame.loc[frame["Symbol"].astype("string").str.fullmatch(r"\d{1,6}", na=False)].copy()
    frame["firm_id"] = normalize_code(frame["Symbol"])
    frame["declare_date"], frame["declare_date_partial"] = parse_exact_date(frame["DeclareDate"])
    frame["disposal_date"], frame["disposal_date_partial"] = parse_exact_date(frame["DisposalDate"])
    frame["company_violated"] = frame["IsViolated"].astype("string").str.upper().eq("Y")
    code_sets = (
        frame["ViolationTypeID"]
        .astype("string")
        .fillna("")
        .map(lambda value: set(re.findall(r"P\d{4}", value)))
    )
    frame["financial_misconduct"] = code_sets.map(
        lambda codes: bool(codes.intersection(FINANCIAL_MISCONDUCT_CODES))
    )
    frame["fraud_or_false_statement"] = code_sets.map(
        lambda codes: bool(codes.intersection({"P2501", "P2502", "P2503", "P2507"}))
    )
    frame["penalty_numeric"] = pd.to_numeric(frame["Penalty"], errors="coerce")
    frame["sum_penalty_numeric"] = pd.to_numeric(frame["SumPenalty"], errors="coerce")
    return frame


def read_old_violations() -> pd.DataFrame:
    frame = pd.read_excel(OLD_VIOLATION_XLSX, engine="openpyxl")
    frame = frame.loc[frame["SCode"].astype("string").str.fullmatch(r"\d{1,6}", na=False)].copy()
    frame["firm_id"] = normalize_code(frame["SCode"])
    frame["announce_date"], frame["announce_date_partial"] = parse_exact_date(frame["AnncDate"])
    return frame


def read_inquiries() -> pd.DataFrame:
    frame = pd.read_excel(INQUIRY_XLSX, engine="openpyxl")
    frame["firm_id"] = normalize_code(frame["证券代码"])
    frame["send_date"], frame["send_date_partial"] = parse_exact_date(frame["发函日期"])
    frame["reply_date"], frame["reply_date_partial"] = parse_exact_date(frame["回函日期"])
    frame["annual_report_inquiry"] = pd.to_numeric(
        frame["问询函类别"], errors="coerce"
    ).eq(1)
    frame["requires_intermediary"] = pd.to_numeric(
        frame["是否需要中介机构回复"], errors="coerce"
    ).eq(0)
    frame["reply_delay_days"] = (frame["reply_date"] - frame["send_date"]).dt.days
    return frame


def duplicate_summary(frame: pd.DataFrame, key: list[str]) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_key_rows": int(frame.duplicated(key, keep=False).sum()),
        "unique_keys": int(frame[key].drop_duplicates().shape[0]),
    }


def date_summary(series: pd.Series) -> dict[str, object]:
    return {
        "nonmissing": int(series.notna().sum()),
        "missing": int(series.isna().sum()),
        "min": None if series.dropna().empty else str(series.min().date()),
        "max": None if series.dropna().empty else str(series.max().date()),
    }


def attach_pre_landmark_counts(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    date_col: str,
    prefix: str,
    additional_flags: list[str],
) -> pd.DataFrame:
    left = panel[["row_id", "firm_id", "report_date", "landmark_date"]].copy()
    use_cols = ["firm_id", date_col, *additional_flags]
    cross = left.merge(events[use_cols], on="firm_id", how="left")
    dated = cross[date_col].notna()
    pre = dated & cross[date_col].le(cross["landmark_date"])
    within_1y = pre & cross[date_col].gt(cross["landmark_date"] - pd.DateOffset(years=1))
    within_3y = pre & cross[date_col].gt(cross["landmark_date"] - pd.DateOffset(years=3))
    current_window = (
        dated
        & cross[date_col].ge(cross["report_date"])
        & cross[date_col].le(cross["landmark_date"])
    )
    cross[f"{prefix}_ever"] = pre.astype(int)
    cross[f"{prefix}_1y"] = within_1y.astype(int)
    cross[f"{prefix}_3y"] = within_3y.astype(int)
    cross[f"{prefix}_current_30d"] = current_window.astype(int)
    for flag in additional_flags:
        flag_values = cross[flag].fillna(False).astype(bool)
        cross[f"{prefix}_{flag}_3y"] = (within_3y & flag_values).astype(int)
    aggregate_cols = [
        f"{prefix}_ever",
        f"{prefix}_1y",
        f"{prefix}_3y",
        f"{prefix}_current_30d",
        *[f"{prefix}_{flag}_3y" for flag in additional_flags],
    ]
    return cross.groupby("row_id", as_index=False)[aggregate_cols].sum()


def event_linkage(panel: pd.DataFrame, violations: pd.DataFrame, inquiries: pd.DataFrame) -> pd.DataFrame:
    cases = panel.loc[
        panel["outcome"].eq(1),
        [
            "row_id",
            "firm_id",
            "fiscal_year",
            "report_date",
            "landmark_date",
            "st_event_date_w30",
            "regime_exposed",
        ],
    ].copy()
    cases = cases.rename(columns={"st_event_date_w30": "st_event_date"})

    company_violations = violations.loc[violations["company_violated"]].copy()
    v = cases.merge(
        company_violations[
            ["firm_id", "declare_date", "financial_misconduct", "fraud_or_false_statement"]
        ],
        on="firm_id",
        how="left",
    )
    v["prior_landmark"] = v["declare_date"].notna() & v["declare_date"].le(v["landmark_date"])
    v["between_landmark_event"] = (
        v["declare_date"].notna()
        & v["declare_date"].gt(v["landmark_date"])
        & v["declare_date"].le(v["st_event_date"])
    )
    v["within_event_year"] = (
        v["declare_date"].notna()
        & v["declare_date"].between(
            v["st_event_date"] - pd.DateOffset(years=1),
            v["st_event_date"] + pd.DateOffset(years=1),
        )
    )
    v["financial_near_event"] = v["within_event_year"] & v["financial_misconduct"].fillna(False)
    v["fraud_near_event"] = v["within_event_year"] & v["fraud_or_false_statement"].fillna(False)
    v_agg = v.groupby("row_id", as_index=False).agg(
        prior_violation=("prior_landmark", "max"),
        violation_between_landmark_event=("between_landmark_event", "max"),
        financial_violation_near_event=("financial_near_event", "max"),
        fraud_violation_near_event=("fraud_near_event", "max"),
    )

    annual = inquiries.loc[inquiries["annual_report_inquiry"]].copy()
    q = cases.merge(annual[["firm_id", "send_date"]], on="firm_id", how="left")
    q["inquiry_prior_landmark"] = q["send_date"].notna() & q["send_date"].le(q["landmark_date"])
    q["inquiry_between_landmark_event"] = (
        q["send_date"].notna()
        & q["send_date"].gt(q["landmark_date"])
        & q["send_date"].le(q["st_event_date"])
    )
    q_agg = q.groupby("row_id", as_index=False).agg(
        inquiry_prior_landmark=("inquiry_prior_landmark", "max"),
        inquiry_between_landmark_event=("inquiry_between_landmark_event", "max"),
    )
    return cases.merge(v_agg, on="row_id", how="left").merge(q_agg, on="row_id", how="left")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    violations = read_violations()
    old_violations = read_old_violations()
    inquiries = read_inquiries()
    annual_inquiries = inquiries.loc[inquiries["annual_report_inquiry"]].copy()
    panel = pd.read_parquet(PANEL_PATH)
    holdout = panel.loc[
        panel["fiscal_year"].between(2019, 2024) & panel["at_risk"].fillna(False)
    ].copy()

    company_violations = violations.loc[violations["company_violated"]].copy()
    v_counts = attach_pre_landmark_counts(
        holdout,
        company_violations,
        "declare_date",
        "violation",
        ["financial_misconduct", "fraud_or_false_statement"],
    )
    q_counts = attach_pre_landmark_counts(
        holdout,
        annual_inquiries,
        "send_date",
        "annual_inquiry",
        ["requires_intermediary"],
    )
    enriched = holdout.merge(v_counts, on="row_id", how="left").merge(
        q_counts, on="row_id", how="left"
    )
    count_cols = [
        column
        for column in enriched.columns
        if column.startswith("violation_") or column.startswith("annual_inquiry_")
    ]
    enriched[count_cols] = enriched[count_cols].fillna(0).astype(int)

    event_links = event_linkage(holdout, violations, inquiries)
    event_links.to_csv(ANALYSIS / "strict_timing_regulatory_event_linkage.csv", index=False)
    enriched[["row_id", "firm_id", "fiscal_year", "outcome", *count_cols]].to_parquet(
        ANALYSIS / "strict_timing_prelandmark_regulatory_features.parquet", index=False
    )

    annual_years = annual_inquiries.assign(year=annual_inquiries["send_date"].dt.year)
    violation_years = violations.assign(year=violations["declare_date"].dt.year)
    old_years = old_violations.assign(year=old_violations["announce_date"].dt.year)

    key_candidates = {
        "violation_id": duplicate_summary(violations, ["ViolationID"]),
        "violation_firm_date_file": duplicate_summary(
            violations, ["firm_id", "DeclareDate", "FileName"]
        ),
        "inquiry_id": duplicate_summary(inquiries, ["问询事件ID"]),
        "annual_inquiry_firm_date_title": duplicate_summary(
            annual_inquiries, ["firm_id", "发函日期", "问询函标题"]
        ),
    }

    regulatory_prevalence = []
    for year, group in enriched.groupby("fiscal_year", sort=True):
        regulatory_prevalence.append(
            {
                "fiscal_year": int(year),
                "rows": int(len(group)),
                "events": int(group["outcome"].sum()),
                "violation_3y_share": float(group["violation_3y"].gt(0).mean()),
                "financial_violation_3y_share": float(
                    group["violation_financial_misconduct_3y"].gt(0).mean()
                ),
                "annual_inquiry_3y_share": float(group["annual_inquiry_3y"].gt(0).mean()),
                "annual_inquiry_current_30d_share": float(
                    group["annual_inquiry_current_30d"].gt(0).mean()
                ),
            }
        )

    summary = {
        "source_files": {
            "violations": str(VIOLATION_XLSX),
            "annual_report_inquiries": str(INQUIRY_XLSX),
            "strict_panel": str(PANEL_PATH),
        },
        "violations": {
            "rows": int(len(violations)),
            "unique_firms": int(violations["firm_id"].nunique()),
            "company_violation_rows": int(violations["company_violated"].sum()),
            "financial_misconduct_rows": int(violations["financial_misconduct"].sum()),
            "fraud_or_false_statement_rows": int(violations["fraud_or_false_statement"].sum()),
            "declare_date": date_summary(violations["declare_date"]),
            "disposal_date": date_summary(violations["disposal_date"]),
            "partial_declare_dates": int(violations["declare_date_partial"].sum()),
            "partial_disposal_dates": int(violations["disposal_date_partial"].sum()),
            "missing_company_flag": int(violations["IsViolated"].isna().sum()),
            "year_counts_2019_plus": {
                str(int(year)): int(count)
                for year, count in violation_years.loc[violation_years["year"].ge(2019)]
                .groupby("year")
                .size()
                .items()
            },
            "old_file_comparison": {
                "old_rows": int(len(old_violations)),
                "old_unique_firms": int(old_violations["firm_id"].nunique()),
                "old_date": date_summary(old_violations["announce_date"]),
                "old_2024_plus_rows": int(old_years["year"].ge(2024).sum()),
                "new_2024_plus_rows": int(violation_years["year"].ge(2024).sum()),
            },
        },
        "inquiries": {
            "all_rows": int(len(inquiries)),
            "annual_report_rows": int(len(annual_inquiries)),
            "annual_report_unique_firms": int(annual_inquiries["firm_id"].nunique()),
            "send_date": date_summary(annual_inquiries["send_date"]),
            "reply_date": date_summary(annual_inquiries["reply_date"]),
            "partial_send_dates": int(annual_inquiries["send_date_partial"].sum()),
            "missing_reply_dates": int(annual_inquiries["reply_date"].isna().sum()),
            "negative_reply_delays": int(annual_inquiries["reply_delay_days"].lt(0).sum()),
            "category_counts": {
                str(int(category)): int(count)
                for category, count in inquiries.groupby("问询函类别").size().items()
            },
            "annual_year_counts": {
                str(int(year)): int(count)
                for year, count in annual_years.groupby("year").size().items()
            },
        },
        "key_candidates": key_candidates,
        "holdout_linkage": {
            "rows": int(len(enriched)),
            "events": int(enriched["outcome"].sum()),
            "firms_with_any_company_violation": int(
                enriched.loc[enriched["violation_ever"].gt(0), "firm_id"].nunique()
            ),
            "records_with_violation_3y": int(enriched["violation_3y"].gt(0).sum()),
            "records_with_financial_violation_3y": int(
                enriched["violation_financial_misconduct_3y"].gt(0).sum()
            ),
            "records_with_fraud_violation_3y": int(
                enriched["violation_fraud_or_false_statement_3y"].gt(0).sum()
            ),
            "records_with_annual_inquiry_3y": int(
                enriched["annual_inquiry_3y"].gt(0).sum()
            ),
            "records_with_current_30d_annual_inquiry": int(
                enriched["annual_inquiry_current_30d"].gt(0).sum()
            ),
            "prevalence_by_fiscal_year": regulatory_prevalence,
        },
        "outcome_event_linkage": {
            "events": int(len(event_links)),
            "prior_violation": int(event_links["prior_violation"].sum()),
            "violation_between_landmark_event": int(
                event_links["violation_between_landmark_event"].sum()
            ),
            "financial_violation_near_event": int(
                event_links["financial_violation_near_event"].sum()
            ),
            "fraud_violation_near_event": int(event_links["fraud_violation_near_event"].sum()),
            "inquiry_prior_landmark": int(event_links["inquiry_prior_landmark"].sum()),
            "inquiry_between_landmark_event": int(
                event_links["inquiry_between_landmark_event"].sum()
            ),
            "by_fiscal_year": event_links.groupby("fiscal_year")[
                [
                    "prior_violation",
                    "violation_between_landmark_event",
                    "financial_violation_near_event",
                    "fraud_violation_near_event",
                    "inquiry_prior_landmark",
                    "inquiry_between_landmark_event",
                ]
            ]
            .sum()
            .astype(int)
            .reset_index()
            .to_dict("records"),
        },
        "quality_findings": {
            "violation_grain": "one regulatory event/document; multiple rows may occur per firm-date",
            "inquiry_grain": "one inquiry letter; annual-report subset is category 1",
            "availability_rule": "DeclareDate/send date must be on or before landmark for predictor use",
            "known_coverage_limit": "violation data end in 2025-07 and cannot fully classify later 2025/2026 outcomes",
        },
    }

    (LOGS / "strict_timing_regulatory_data_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
