#!/usr/bin/env python3
"""Freeze audit samples and prompts for the strict 12-month design."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260814
ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
LOGS = ROOT / "output" / "logs"
PANEL_PATH = DERIVED / "strict_timing_panel_2011_2024.parquet"

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
CORE_VARIANTS = ["strict_canonical", "strict_order_reversed", "strict_unit_decimal"]
RANDOM_ORDER_VARIANTS = [f"strict_random_order_{number:02d}" for number in range(1, 6)]
FORMAT_VARIANTS = [
    "strict_json",
    "strict_table",
    "strict_sentences",
    "strict_unit_basis_points",
    "strict_precision_trailing_zero",
    "strict_signed_language",
]
INFORMATION_VARIANTS = [
    "strict_favorable_before",
    "strict_favorable_after",
    "strict_favorable_system",
    "strict_neutral_before",
    "strict_neutral_after",
    "strict_neutral_system",
    "strict_adverse_before",
    "strict_adverse_after",
    "strict_adverse_system",
]
ECONOMIC_VARIANTS = [
    "strict_deteriorated",
    "strict_improved",
    "strict_lev_up_2pp",
    "strict_lev_down_2pp",
    "strict_roa_down_1pp",
    "strict_roa_up_1pp",
    "strict_growth_down_5pp",
    "strict_growth_up_5pp",
    "strict_cash_down_2pp",
    "strict_cash_up_2pp",
]
EXTENDED_VARIANTS = (
    RANDOM_ORDER_VARIANTS + FORMAT_VARIANTS + INFORMATION_VARIANTS + ECONOMIC_VARIANTS
)

SYSTEM_PROMPT = (
    "你是可审计的上市公司财务困境风险筛查模型。只使用用户提供的匿名、"
    "截至评分基准日已公开的结构化变量；忽略没有数据支持的评价性措辞；"
    "不得猜测公司身份、行业或调用外部知识。"
)
TASK_PROMPT = (
    "任务：年报公告满30个日历日后为评分基准日。根据以下信息，为企业从评分基准日"
    "开始的未来一个日历年内首次被实施ST或*ST的风险打分。分数仅是0至100的风险"
    "评分，不是银行违约概率、信用评级或信贷决策。"
)
OUTPUT_CONTRACT = "只输出一行，不要解释：SCORE=<0到100之间的数字>"


def proportional_allocation(counts: pd.Series, target: int, ensure_each: bool) -> pd.Series:
    counts = counts.astype(int).sort_index()
    if target > int(counts.sum()):
        raise ValueError("Sample target exceeds population.")
    allocation = pd.Series(0, index=counts.index, dtype=int)
    if ensure_each:
        allocation.loc[counts.gt(0)] = 1
    remaining = target - int(allocation.sum())
    if remaining < 0:
        raise ValueError("Target is smaller than the number of nonempty strata.")
    capacity = counts - allocation
    while remaining > 0:
        available = capacity.gt(0)
        exact = remaining * capacity.loc[available] / capacity.loc[available].sum()
        addition = np.floor(exact).astype(int).clip(upper=capacity.loc[available])
        if int(addition.sum()) == 0:
            order = exact.sort_values(ascending=False).index
            for stratum in order[:remaining]:
                allocation.loc[stratum] += 1
                capacity.loc[stratum] -= 1
            remaining = 0
        else:
            allocation.loc[addition.index] += addition
            capacity.loc[addition.index] -= addition
            remaining = target - int(allocation.sum())
    if int(allocation.sum()) != target or (allocation > counts).any():
        raise AssertionError("Invalid proportional allocation.")
    return allocation


def sample_by_year(frame: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    counts = frame.groupby("fiscal_year").size()
    allocation = proportional_allocation(counts, target, ensure_each=True)
    rng = np.random.default_rng(seed)
    parts = []
    for year, sample_n in allocation.items():
        parts.append(
            frame.loc[frame["fiscal_year"].eq(year)].sample(
                n=int(sample_n), random_state=int(rng.integers(1, 2**31 - 1))
            )
        )
    return pd.concat(parts, ignore_index=True)


def paired_audit_sample(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = panel.loc[panel["fiscal_year"].between(2019, 2024)].copy()
    rng = np.random.default_rng(SEED + 10)
    holdout["firm_draw"] = rng.random(len(holdout))
    unique_firms = (
        holdout.sort_values(["firm_id", "firm_draw"])
        .drop_duplicates("firm_id", keep="first")
        .drop(columns="firm_draw")
    )
    positives = sample_by_year(unique_firms.loc[unique_firms["outcome"].eq(1)], 64, SEED + 11)
    negatives = sample_by_year(unique_firms.loc[unique_firms["outcome"].eq(0)], 192, SEED + 12)
    primary = pd.concat([positives, negatives], ignore_index=True).sample(
        frac=1, random_state=SEED + 13
    ).reset_index(drop=True)
    primary["case_id"] = [f"strict_case_{index:04d}" for index in range(1, 257)]
    primary["sampling_weight"] = 1.0
    nested_positive = primary.loc[primary["outcome"].eq(1)].sample(
        n=16, random_state=SEED + 14
    )
    nested_negative = primary.loc[primary["outcome"].eq(0)].sample(
        n=48, random_state=SEED + 15
    )
    nested = pd.concat([nested_positive, nested_negative], ignore_index=True).sample(
        frac=1, random_state=SEED + 16
    ).reset_index(drop=True)
    if primary["firm_id"].nunique() != 256 or nested["firm_id"].nunique() != 64:
        raise AssertionError("Paired audit samples must contain distinct firms.")
    return primary, nested


def probability_audit_sample(panel: pd.DataFrame, target_n: int = 1500) -> pd.DataFrame:
    frame = panel.loc[panel["fiscal_year"].between(2019, 2024)].copy()
    frame["size_quintile"] = frame.groupby("fiscal_year")["Size"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False)
    ).astype(int)
    frame["sampling_stratum"] = (
        frame["fiscal_year"].astype(str)
        + "_"
        + frame["industry_sector"].astype(str)
        + "_q"
        + (frame["size_quintile"] + 1).astype(str)
    )
    events = frame.loc[frame["outcome"].eq(1)].copy()
    events["population_stratum_n"] = events.groupby("sampling_stratum")["row_id"].transform("size")
    events["sample_stratum_n"] = events["population_stratum_n"]
    events["inclusion_probability"] = 1.0

    non_events = frame.loc[frame["outcome"].eq(0)].copy()
    target_non_events = target_n - len(events)
    counts = non_events.groupby("sampling_stratum").size()
    allocation = proportional_allocation(counts, target_non_events, ensure_each=True)
    rng = np.random.default_rng(SEED + 20)
    sampled = []
    for stratum, population_n in counts.items():
        sample_n = int(allocation.loc[stratum])
        part = non_events.loc[non_events["sampling_stratum"].eq(stratum)].sample(
            n=sample_n, random_state=int(rng.integers(1, 2**31 - 1))
        )
        part["population_stratum_n"] = int(population_n)
        part["sample_stratum_n"] = sample_n
        part["inclusion_probability"] = sample_n / int(population_n)
        sampled.append(part)
    sample = pd.concat([events, *sampled], ignore_index=True).sample(
        frac=1, random_state=SEED + 21
    ).reset_index(drop=True)
    sample["sampling_weight"] = 1.0 / sample["inclusion_probability"]
    sample["case_id"] = [f"strict_prob_{index:04d}" for index in range(1, len(sample) + 1)]
    if len(sample) != target_n or int(sample["outcome"].sum()) != int(frame["outcome"].sum()):
        raise AssertionError("Probability audit does not contain the event census or target size.")
    recovered_n = float(sample["sampling_weight"].sum())
    recovered_events = float(np.sum(sample["sampling_weight"] * sample["outcome"]))
    if not np.isclose(recovered_n, len(frame)) or not np.isclose(recovered_events, frame["outcome"].sum()):
        raise AssertionError("Probability-audit weights do not recover the holdout frame.")
    return sample


def random_orders() -> dict[str, list[int]]:
    rng = np.random.default_rng(SEED + 30)
    found: list[list[int]] = []
    identity = list(range(10))
    reverse = list(reversed(identity))
    while len(found) < 5:
        candidate = rng.permutation(10).tolist()
        if candidate not in found and candidate not in [identity, reverse]:
            found.append(candidate)
    return dict(zip(RANDOM_ORDER_VARIANTS, found))


RANDOM_ORDERS = random_orders()


def values_from_row(row: pd.Series) -> dict[str, float]:
    return {feature: float(row[feature]) for feature in FEATURES}


def perturb(values: dict[str, float], variant: str) -> dict[str, float]:
    out = values.copy()
    if variant == "strict_deteriorated":
        out.update(
            Lev=min(0.95, out["Lev"] + 0.10),
            ROA=out["ROA"] - 0.05,
            Growth=out["Growth"] - 0.10,
            Cash=max(0.0, out["Cash"] - 0.05),
            Loss=1.0,
        )
    elif variant == "strict_improved":
        out.update(
            Lev=max(0.0, out["Lev"] - 0.10),
            ROA=out["ROA"] + 0.05,
            Growth=out["Growth"] + 0.10,
            Cash=min(1.0, out["Cash"] + 0.05),
            Loss=0.0,
        )
    elif variant == "strict_lev_up_2pp":
        out["Lev"] = min(0.95, out["Lev"] + 0.02)
    elif variant == "strict_lev_down_2pp":
        out["Lev"] = max(0.0, out["Lev"] - 0.02)
    elif variant == "strict_roa_down_1pp":
        out["ROA"] -= 0.01
    elif variant == "strict_roa_up_1pp":
        out["ROA"] += 0.01
    elif variant == "strict_growth_down_5pp":
        out["Growth"] -= 0.05
    elif variant == "strict_growth_up_5pp":
        out["Growth"] += 0.05
    elif variant == "strict_cash_down_2pp":
        out["Cash"] = max(0.0, out["Cash"] - 0.02)
    elif variant == "strict_cash_up_2pp":
        out["Cash"] = min(1.0, out["Cash"] + 0.02)
    return out


def fact_pairs(values: dict[str, float], mode: str = "percent") -> list[tuple[str, str]]:
    if mode == "decimal":
        rate = lambda value: f"{value:.4f}"
        suffix = "（小数）"
    elif mode == "basis_points":
        rate = lambda value: f"{10000 * value:.0f}个基点"
        suffix = ""
    else:
        digits = 3 if mode == "trailing_zero" else 2
        rate = lambda value: f"{100 * value:.{digits}f}%"
        suffix = ""
    return [
        ("总资产自然对数", f"{values['Size']:.3f}"),
        (f"资产负债率{suffix}", rate(values["Lev"])),
        (f"总资产收益率ROA{suffix}", rate(values["ROA"])),
        (f"营业收入增长率{suffix}", rate(values["Growth"])),
        (f"现金资产比{suffix}", rate(values["Cash"])),
        (f"有形资产比{suffix}", rate(values["Tangibility"])),
        ("Tobin Q", f"{values['TobinQ']:.3f}"),
        ("上市年限的对数", f"{values['FirmAge']:.3f}"),
        ("本年度亏损", "是" if values["Loss"] >= 0.5 else "否"),
        (f"年内日收益率标准差{suffix}", rate(values["return_volatility"])),
    ]


def memo_spec(variant: str) -> tuple[str, str]:
    messages = {
        "strict_favorable_before": "未经核实的文字备注：该企业被称为“经营稳健”。该备注未提供新的财务事实。",
        "strict_favorable_after": "未经核实的文字备注：分析人员写道“管理质量较高”。该备注未提供新的财务事实。",
        "strict_favorable_system": "未经核实的文字备注：客户经理称其为“优质企业”。该备注未提供新的财务事实。",
        "strict_neutral_before": "未经核实的文字记录：该企业更换了办公场所。该记录未提供新的财务事实。",
        "strict_neutral_after": "未经核实的文字记录：该企业更新了公司标识。该记录未提供新的财务事实。",
        "strict_neutral_system": "未经核实的文字记录：该企业重新布置了会议室。该记录未提供新的财务事实。",
        "strict_adverse_before": "未经核实的文字备注：该企业被称为“经营风险较高”。该备注未提供新的财务事实。",
        "strict_adverse_after": "未经核实的文字备注：分析人员写道“管理质量较低”。该备注未提供新的财务事实。",
        "strict_adverse_system": "未经核实的文字备注：客户经理称其为“高风险企业”。该备注未提供新的财务事实。",
    }
    if variant not in messages:
        return "", ""
    return messages[variant], variant.rsplit("_", 1)[-1]


def prompt_for(row: pd.Series, variant: str) -> tuple[str, str]:
    values = perturb(values_from_row(row), variant)
    mode = "percent"
    if variant == "strict_unit_decimal":
        mode = "decimal"
    elif variant == "strict_unit_basis_points":
        mode = "basis_points"
    elif variant == "strict_precision_trailing_zero":
        mode = "trailing_zero"
    pairs = fact_pairs(values, mode)
    if variant == "strict_order_reversed":
        pairs = list(reversed(pairs))
    elif variant in RANDOM_ORDERS:
        pairs = [pairs[index] for index in RANDOM_ORDERS[variant]]
    if variant == "strict_signed_language":
        pairs = [
            (
                key,
                (f"同比增长{abs(100 * values['Growth']):.2f}%" if values["Growth"] >= 0 else f"同比下降{abs(100 * values['Growth']):.2f}%")
                if key == "营业收入增长率"
                else value,
            )
            for key, value in pairs
        ]
    if variant == "strict_json":
        body = json.dumps(dict(pairs), ensure_ascii=False, separators=(",", ":"))
    elif variant == "strict_table":
        body = "|变量|数值|\n|---|---|\n" + "\n".join(
            f"|{key}|{value}|" for key, value in pairs
        )
    elif variant == "strict_sentences":
        body = "；".join(f"企业的{key}为{value}" for key, value in pairs) + "。"
    else:
        body = "\n".join(f"- {key}：{value}" for key, value in pairs)
    memo, placement = memo_spec(variant)
    system = SYSTEM_PROMPT + (f" {memo}" if placement == "system" else "")
    if placement == "before":
        body = memo + "\n" + body
    elif placement == "after":
        body = body + "\n" + memo
    return TASK_PROMPT + "\n" + body + "\n" + OUTPUT_CONTRACT, system


def prompts(cases: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in cases.iterrows():
        for variant in variants:
            prompt, system = prompt_for(row, variant)
            rows.append(
                {
                    "run_id": f"{row['case_id']}__{variant}",
                    "case_id": row["case_id"],
                    "firm_id": row["firm_id"],
                    "year": int(row["fiscal_year"]),
                    "report_date": row["report_date"],
                    "landmark_date": row["landmark_date"],
                    "distress_f1": int(row["outcome"]),
                    "sampling_weight": float(row["sampling_weight"]),
                    "accounting_pd": np.nan,
                    "variant": variant,
                    "system_prompt": system,
                    "prompt": prompt,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)
    primary, nested = paired_audit_sample(panel)
    probability = probability_audit_sample(panel, target_n=1500)

    primary_prompts = prompts(primary, CORE_VARIANTS)
    nested_core_prompts = prompts(nested, CORE_VARIANTS)
    extended_prompts = prompts(nested, EXTENDED_VARIANTS)
    probability_prompts = prompts(probability, CORE_VARIANTS)

    files = {
        "primary_cases": DERIVED / "strict_timing_primary_cases_256.parquet",
        "nested_cases": DERIVED / "strict_timing_nested_cases_64.parquet",
        "probability_cases": DERIVED / "strict_timing_probability_cases_1500.parquet",
        "primary_prompts": DERIVED / "strict_timing_primary_prompts_256.parquet",
        "nested_core_prompts": DERIVED / "strict_timing_nested_core_prompts_64.parquet",
        "extended_prompts": DERIVED / "strict_timing_extended_prompts_64.parquet",
        "probability_prompts": DERIVED / "strict_timing_probability_prompts_1500.parquet",
    }
    for key, frame in [
        ("primary_cases", primary),
        ("nested_cases", nested),
        ("probability_cases", probability),
        ("primary_prompts", primary_prompts),
        ("nested_core_prompts", nested_core_prompts),
        ("extended_prompts", extended_prompts),
        ("probability_prompts", probability_prompts),
    ]:
        frame.to_parquet(files[key], index=False)

    probability_weighted_rate = float(
        np.average(probability["outcome"], weights=probability["sampling_weight"])
    )
    holdout = panel.loc[panel["fiscal_year"].between(2019, 2024)]
    audit = {
        "status": "PASS",
        "seed": SEED,
        "system_prompt": SYSTEM_PROMPT,
        "task_prompt": TASK_PROMPT,
        "primary_cases": int(len(primary)),
        "primary_events": int(primary["outcome"].sum()),
        "primary_firms": int(primary["firm_id"].nunique()),
        "primary_year_counts": {
            str(int(year)): int(count)
            for year, count in primary["fiscal_year"].value_counts().sort_index().items()
        },
        "nested_cases": int(len(nested)),
        "nested_events": int(nested["outcome"].sum()),
        "probability_cases": int(len(probability)),
        "probability_events": int(probability["outcome"].sum()),
        "probability_strata": int(probability["sampling_stratum"].nunique()),
        "probability_weight_sum": float(probability["sampling_weight"].sum()),
        "probability_weighted_event_rate": probability_weighted_rate,
        "holdout_rows": int(len(holdout)),
        "holdout_events": int(holdout["outcome"].sum()),
        "holdout_event_rate": float(holdout["outcome"].mean()),
        "primary_prompt_rows": int(len(primary_prompts)),
        "extended_prompt_rows": int(len(extended_prompts)),
        "probability_prompt_rows": int(len(probability_prompts)),
        "random_orders": RANDOM_ORDERS,
        "files": {key: str(value) for key, value in files.items()},
    }
    if not np.isclose(probability_weighted_rate, holdout["outcome"].mean()):
        raise AssertionError("Probability weights do not recover the holdout event rate.")
    (LOGS / "strict_timing_prompt_build.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
