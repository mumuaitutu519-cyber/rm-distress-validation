#!/usr/bin/env python3
"""Build a strict annual-report landmark panel through fiscal year 2024.

The scoring landmark is 30 calendar days after the annual-report announcement.
Firms already under a non-normal trading status at the landmark are excluded.
The primary outcome is the first transition from normal status to ST or *ST in
the following calendar year.  Zero- and seven-day landmarks and a composite
ST/*ST/direct-delisting outcome are retained for sensitivity analysis.

All outputs use new ``strict_timing`` names and do not overwrite the earlier
calendar-record design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260814
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DERIVED = ROOT / "data" / "derived"
ANALYSIS = ROOT / "output" / "analysis"
LOGS = ROOT / "output" / "logs"

FINANCIAL = (
    WORKSPACE
    / "codex项目/C17 a股上市公司财务数据合集（90-25年）/"
    "上市公司财务年度数据合并（90-24）.dta"
)
CONTROLS = (
    WORKSPACE
    / "codex项目/【260731V4.1】常用控制变量2000_2025_Ver4.1"
    "[请尊重个人知识产权 禁止二传或转卖]/常用控制变量2000_2025_Ver4.1.dta"
)
RETURNS = WORKSPACE / "codex项目/data/raw/stock_returns_daily.dta"
ST_CHANGES_XLSX = WORKSPACE / "codex项目/特殊处理变动文件215444852.zip"

FEATURES = [
    "Size",
    "Lev",
    "ROA",
    "Growth",
    "Cash",
    "Tangibility",
    "TobinQ",
    "FirmAge",
    "Loss",
    "return_volatility",
]
CONTINUOUS = [feature for feature in FEATURES if feature != "Loss"]
TRAIN_START = 2011
TRAIN_END = 2018
HOLDOUT_START = 2019
HOLDOUT_END = 2024
PRIMARY_WASHOUT_DAYS = 30


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)


def read_st_change_file() -> pd.DataFrame:
    """Read the CSMAR workbook directly from the zip archive."""
    import io
    import zipfile

    with zipfile.ZipFile(ST_CHANGES_XLSX) as archive:
        xlsx_names = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
        if len(xlsx_names) != 1:
            raise AssertionError(f"Expected one xlsx member, found {xlsx_names}")
        payload = io.BytesIO(archive.read(xlsx_names[0]))
    events = pd.read_excel(payload, engine="openpyxl")
    events["firm_id"] = normalize_code(events["Symbol"])
    events["announcement_date"] = pd.to_datetime(events["Annoudt"], errors="coerce")
    events["execution_date"] = pd.to_datetime(events["Execudt"], errors="coerce")
    events["transition"] = events["Chgtype"].astype(str).str.strip().str.upper()
    events = events.loc[
        events["firm_id"].str.fullmatch(r"\d{6}", na=False)
        & events["execution_date"].notna()
        & events["transition"].str.fullmatch(r"[ABDCSTX]{2}", na=False)
    ].copy()
    events["from_state"] = events["transition"].str[0]
    events["to_state"] = events["transition"].str[-1]
    events = events.sort_values(["firm_id", "execution_date", "announcement_date"])
    if events.duplicated(["firm_id", "execution_date", "transition"]).any():
        raise AssertionError("Duplicate firm-date-transition records in ST change file.")
    return events


def read_financial_features() -> pd.DataFrame:
    columns = [
        "Stkcd",
        "Accper",
        "Typrep",
        "Annodt",
        "TotalAssets",
        "AssetLiabilityRatio",
        "ROA",
        "OperatingRevenueGrowth",
        "Cash",
        "TangibleAssetRatio",
    ]
    frame = pd.read_stata(
        FINANCIAL, columns=columns, convert_categoricals=False, preserve_dtypes=False
    )
    frame["firm_id"] = normalize_code(frame["Stkcd"])
    frame["fiscal_year"] = pd.to_datetime(frame["Accper"], errors="coerce").dt.year.astype("Int64")
    frame["report_date"] = pd.to_datetime(frame["Annodt"], errors="coerce")
    frame = frame.loc[frame["Typrep"].astype(str).eq("A")].copy()
    frame = frame.sort_values(["firm_id", "fiscal_year", "report_date"])
    if frame.duplicated(["firm_id", "fiscal_year"]).any():
        duplicates = int(frame.duplicated(["firm_id", "fiscal_year"]).sum())
        raise AssertionError(f"Unexpected duplicate annual financial rows: {duplicates}")
    total_assets = pd.to_numeric(frame["TotalAssets"], errors="coerce")
    frame["Size"] = np.log(total_assets.where(total_assets > 0))
    frame["Lev"] = pd.to_numeric(frame["AssetLiabilityRatio"], errors="coerce")
    frame["ROA"] = pd.to_numeric(frame["ROA"], errors="coerce")
    frame["Growth"] = pd.to_numeric(frame["OperatingRevenueGrowth"], errors="coerce")
    frame["Cash"] = pd.to_numeric(frame["Cash"], errors="coerce") / total_assets
    frame["Tangibility"] = pd.to_numeric(frame["TangibleAssetRatio"], errors="coerce")
    return frame[
        ["firm_id", "fiscal_year", "report_date", "Size", "Lev", "ROA", "Growth", "Cash", "Tangibility"]
    ].copy()


def read_controls() -> pd.DataFrame:
    columns = [
        "stkcode",
        "year",
        "nnindcd",
        "nnindnme",
        "markettype",
        "tobinq_1",
        "lnage",
        "loss",
    ]
    frame = pd.read_stata(
        CONTROLS, columns=columns, convert_categoricals=False, preserve_dtypes=False
    )
    frame["firm_id"] = normalize_code(frame["stkcode"])
    frame["fiscal_year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame = frame.sort_values(["firm_id", "fiscal_year"])
    if frame.duplicated(["firm_id", "fiscal_year"]).any():
        raise AssertionError("Control file is not unique by firm-year.")
    frame["industry"] = frame["nnindcd"].astype(str).str.strip().str.upper()
    frame["industry_name"] = frame["nnindnme"].astype(str).str.strip()
    frame["TobinQ"] = pd.to_numeric(frame["tobinq_1"], errors="coerce")
    frame["FirmAge"] = pd.to_numeric(frame["lnage"], errors="coerce")
    frame["Loss"] = pd.to_numeric(frame["loss"], errors="coerce")
    return frame[
        [
            "firm_id",
            "fiscal_year",
            "industry",
            "industry_name",
            "markettype",
            "TobinQ",
            "FirmAge",
            "Loss",
        ]
    ].copy()


def read_return_volatility() -> pd.DataFrame:
    columns = ["证券代码", "year", "考虑现金红利再投资的日个股回报率"]
    # The raw daily-return file has more than 16 million rows.  Aggregate
    # sufficient statistics by chunks so the build remains reproducible on a
    # 16GB host instead of materializing the entire file in memory.
    parts: list[pd.DataFrame] = []
    reader = pd.read_stata(
        RETURNS,
        columns=columns,
        convert_categoricals=False,
        preserve_dtypes=False,
        chunksize=500_000,
    )
    for frame in reader:
        frame["firm_id"] = normalize_code(frame["证券代码"])
        frame["fiscal_year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
        frame["daily_return"] = pd.to_numeric(
            frame["考虑现金红利再投资的日个股回报率"], errors="coerce"
        )
        frame = frame.loc[
            frame["fiscal_year"].between(TRAIN_START, HOLDOUT_END)
            & frame["daily_return"].notna()
        ].copy()
        frame["return_squared"] = frame["daily_return"] ** 2
        parts.append(
            frame.groupby(["firm_id", "fiscal_year"], as_index=False).agg(
                trading_days=("daily_return", "size"),
                return_sum=("daily_return", "sum"),
                return_squared_sum=("return_squared", "sum"),
            )
        )
    annual = (
        pd.concat(parts, ignore_index=True)
        .groupby(["firm_id", "fiscal_year"], as_index=False)
        .agg(
            trading_days=("trading_days", "sum"),
            return_sum=("return_sum", "sum"),
            return_squared_sum=("return_squared_sum", "sum"),
        )
    )
    numerator = annual["return_squared_sum"] - annual["return_sum"] ** 2 / annual["trading_days"]
    annual["return_volatility"] = np.sqrt(
        np.maximum(numerator, 0) / (annual["trading_days"] - 1)
    )
    annual.loc[annual["trading_days"] < 30, "return_volatility"] = np.nan
    return annual[["firm_id", "fiscal_year", "return_volatility", "trading_days"]]


def state_and_outcome(
    panel: pd.DataFrame, events: pd.DataFrame, washout_days: int
) -> pd.DataFrame:
    groups = {firm: group for firm, group in events.groupby("firm_id", sort=False)}
    rows: list[dict[str, object]] = []
    for record in panel[["firm_id", "report_date"]].itertuples(index=False):
        if pd.isna(record.report_date):
            rows.append(
                {
                    "landmark_date": pd.NaT,
                    "state_at_landmark": pd.NA,
                    "at_risk": False,
                    "st_event": pd.NA,
                    "st_event_date": pd.NaT,
                    "composite_event": pd.NA,
                    "composite_event_date": pd.NaT,
                }
            )
            continue
        landmark = record.report_date + pd.Timedelta(days=washout_days)
        group = groups.get(record.firm_id)
        if group is None:
            state = "A"
            future = None
        else:
            history = group.loc[group["execution_date"].le(landmark)]
            state = str(history.iloc[-1]["to_state"]) if len(history) else "A"
            future = group.loc[
                group["execution_date"].gt(landmark)
                & group["execution_date"].le(landmark + pd.DateOffset(years=1))
            ]
        at_risk = state == "A"
        if at_risk and future is not None:
            st = future.loc[future["transition"].isin(["AB", "AD"])]
            composite = future.loc[future["transition"].isin(["AB", "AD", "AT", "AX"])]
            st_date = st["execution_date"].min() if len(st) else pd.NaT
            composite_date = composite["execution_date"].min() if len(composite) else pd.NaT
        else:
            st_date = pd.NaT
            composite_date = pd.NaT
        rows.append(
            {
                "landmark_date": landmark,
                "state_at_landmark": state,
                "at_risk": at_risk,
                "st_event": int(pd.notna(st_date)) if at_risk else pd.NA,
                "st_event_date": st_date,
                "composite_event": int(pd.notna(composite_date)) if at_risk else pd.NA,
                "composite_event_date": composite_date,
            }
        )
    suffix = f"_w{washout_days}"
    labelled = pd.DataFrame(rows, index=panel.index).add_suffix(suffix)
    return pd.concat([panel, labelled], axis=1)


def winsorize_from_training(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = panel.copy()
    training = out.loc[out["fiscal_year"].between(TRAIN_START, TRAIN_END)]
    cutpoints: dict[str, dict[str, float]] = {}
    for feature in CONTINUOUS:
        lower = float(training[feature].quantile(0.01))
        upper = float(training[feature].quantile(0.99))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise AssertionError(f"Invalid training-period winsor bounds for {feature}")
        cutpoints[feature] = {"p01": lower, "p99": upper}
        out[f"{feature}_raw"] = out[feature]
        out[feature] = out[feature].clip(lower, upper)
    return out, cutpoints


def main() -> None:
    for directory in [DERIVED, ANALYSIS, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)
    financial = read_financial_features()
    controls = read_controls()
    volatility = read_return_volatility()
    events = read_st_change_file()

    panel = financial.merge(
        controls, on=["firm_id", "fiscal_year"], how="left", validate="one_to_one"
    ).merge(
        volatility, on=["firm_id", "fiscal_year"], how="left", validate="one_to_one"
    )
    panel = panel.loc[panel["fiscal_year"].between(TRAIN_START, HOLDOUT_END)].copy()
    panel["industry"] = panel["industry"].replace({"NAN": np.nan, "": np.nan})
    panel["is_financial"] = panel["industry"].str.startswith("J", na=False)
    panel = panel.loc[~panel["is_financial"]].copy()

    for washout in [0, 7, PRIMARY_WASHOUT_DAYS]:
        panel = state_and_outcome(panel, events, washout)

    primary_suffix = f"_w{PRIMARY_WASHOUT_DAYS}"
    panel["outcome"] = panel[f"st_event{primary_suffix}"]
    panel["outcome_composite"] = panel[f"composite_event{primary_suffix}"]
    panel["landmark_date"] = panel[f"landmark_date{primary_suffix}"]
    panel["state_at_landmark"] = panel[f"state_at_landmark{primary_suffix}"]
    panel["at_risk"] = panel[f"at_risk{primary_suffix}"].astype(bool)
    panel["industry_sector"] = panel["industry"].str[0]
    panel["regime_exposed"] = panel["fiscal_year"].ge(2023)

    complete_candidates = panel.loc[
        panel["report_date"].notna()
        & panel["industry"].notna()
        & panel[FEATURES].notna().all(axis=1)
    ].copy()
    complete_candidates = complete_candidates.sort_values(
        ["fiscal_year", "firm_id"]
    ).reset_index(drop=True)
    candidate_path = DERIVED / "strict_timing_complete_candidates_2011_2024.parquet"
    complete_candidates.to_parquet(candidate_path, index=False)

    washout_rows: list[dict[str, object]] = []
    for washout in [0, 7, PRIMARY_WASHOUT_DAYS]:
        at_risk_column = f"at_risk_w{washout}"
        for period, start, end in [
            ("training", TRAIN_START, TRAIN_END),
            ("holdout", HOLDOUT_START, HOLDOUT_END),
        ]:
            risk_set = complete_candidates.loc[
                complete_candidates["fiscal_year"].between(start, end)
                & complete_candidates[at_risk_column].fillna(False)
            ].copy()
            washout_rows.append(
                {
                    "washout_days": washout,
                    "period": period,
                    "records": int(len(risk_set)),
                    "firms": int(risk_set["firm_id"].nunique()),
                    "st_events": int(risk_set[f"st_event_w{washout}"].sum()),
                    "st_event_rate": float(risk_set[f"st_event_w{washout}"].mean()),
                    "composite_events": int(
                        risk_set[f"composite_event_w{washout}"].sum()
                    ),
                    "composite_event_rate": float(
                        risk_set[f"composite_event_w{washout}"].mean()
                    ),
                }
            )
    washout_frame = pd.DataFrame(washout_rows)
    washout_frame.to_csv(
        ANALYSIS / "strict_timing_washout_risksets.csv", index=False
    )

    eligible = panel.loc[
        panel["at_risk"]
        & panel["outcome"].notna()
        & panel["report_date"].notna()
        & panel["industry"].notna()
        & panel[FEATURES].notna().all(axis=1)
    ].copy()
    eligible["outcome"] = eligible["outcome"].astype(int)
    eligible["outcome_composite"] = eligible["outcome_composite"].astype(int)
    eligible, cutpoints = winsorize_from_training(eligible)
    eligible = eligible.sort_values(["fiscal_year", "firm_id"]).reset_index(drop=True)
    eligible["row_id"] = [f"strict_{index:06d}" for index in range(1, len(eligible) + 1)]

    if not eligible["landmark_date"].add(pd.DateOffset(years=1)).le(events["execution_date"].max()).all():
        raise AssertionError("At least one primary outcome window exceeds ST-file coverage.")
    if eligible.duplicated(["firm_id", "fiscal_year"]).any():
        raise AssertionError("Strict panel is not unique by firm-year.")
    if not eligible["fiscal_year"].between(TRAIN_START, HOLDOUT_END).all():
        raise AssertionError("Unexpected fiscal year in strict panel.")

    panel_path = DERIVED / "strict_timing_panel_2011_2024.parquet"
    events_path = DERIVED / "strict_timing_st_changes.parquet"
    eligible.to_parquet(panel_path, index=False)
    events.to_parquet(events_path, index=False)

    yearly = (
        eligible.groupby("fiscal_year", as_index=False)
        .agg(
            observations=("row_id", "size"),
            firms=("firm_id", "nunique"),
            events=("outcome", "sum"),
            composite_events=("outcome_composite", "sum"),
            event_rate=("outcome", "mean"),
            report_date_min=("report_date", "min"),
            report_date_max=("report_date", "max"),
            landmark_date_max=("landmark_date", "max"),
        )
    )
    yearly.to_csv(ANALYSIS / "strict_timing_yearly_sample.csv", index=False)

    source_coverage = {
        "financial_rows_2011_2024_nonfinancial": int(len(panel)),
        "eligible_complete_at_risk_rows": int(len(eligible)),
        "eligible_firms": int(eligible["firm_id"].nunique()),
        "training_rows": int(eligible["fiscal_year"].between(TRAIN_START, TRAIN_END).sum()),
        "training_events": int(
            eligible.loc[eligible["fiscal_year"].between(TRAIN_START, TRAIN_END), "outcome"].sum()
        ),
        "holdout_rows": int(eligible["fiscal_year"].between(HOLDOUT_START, HOLDOUT_END).sum()),
        "holdout_events": int(
            eligible.loc[eligible["fiscal_year"].between(HOLDOUT_START, HOLDOUT_END), "outcome"].sum()
        ),
        "holdout_event_rate": float(
            eligible.loc[eligible["fiscal_year"].between(HOLDOUT_START, HOLDOUT_END), "outcome"].mean()
        ),
        "industry_missing_before_complete_case": int(panel["industry"].isna().sum()),
        "feature_missing_before_complete_case": {
            feature: int(panel[feature].isna().sum()) for feature in FEATURES
        },
        "st_change_rows": int(len(events)),
        "st_change_max_execution_date": str(events["execution_date"].max().date()),
        "primary_washout_days": PRIMARY_WASHOUT_DAYS,
        "primary_horizon": "one calendar year after landmark",
        "winsorization": "training-period 1st/99th percentiles",
        "winsor_cutpoints": cutpoints,
        "yearly": yearly.assign(
            report_date_min=yearly["report_date_min"].astype(str),
            report_date_max=yearly["report_date_max"].astype(str),
            landmark_date_max=yearly["landmark_date_max"].astype(str),
        ).to_dict("records"),
        "washout_risk_sets": washout_frame.to_dict("records"),
        "files": {
            "panel": str(panel_path),
            "complete_candidates": str(candidate_path),
            "events": str(events_path),
        },
    }
    (LOGS / "strict_timing_panel_build.json").write_text(
        json.dumps(source_coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(yearly.to_string(index=False))
    print(json.dumps(source_coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
