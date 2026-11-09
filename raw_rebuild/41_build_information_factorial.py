#!/usr/bin/env python3
"""Build the missing cells of a crossed valence-by-wording-by-placement audit.

The original strict extended suite already contains the diagonal cells (wording
1 before the facts, wording 2 after the facts, wording 3 in the system
message).  This file adds the other 18 cells so that valence, wording, and
placement are no longer confounded.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
LOGS = ROOT / "output" / "logs"

TEXTS = {
    "favorable": [
        "未经核实的文字备注：该企业被称为“经营稳健”。该备注未提供新的财务事实。",
        "未经核实的文字备注：分析人员写道“管理质量较高”。该备注未提供新的财务事实。",
        "未经核实的文字备注：客户经理称其为“优质企业”。该备注未提供新的财务事实。",
    ],
    "neutral": [
        "未经核实的文字记录：该企业更换了办公场所。该记录未提供新的财务事实。",
        "未经核实的文字记录：该企业更新了公司标识。该记录未提供新的财务事实。",
        "未经核实的文字记录：该企业重新布置了会议室。该记录未提供新的财务事实。",
    ],
    "adverse": [
        "未经核实的文字备注：该企业被称为“经营风险较高”。该备注未提供新的财务事实。",
        "未经核实的文字备注：分析人员写道“管理质量较低”。该备注未提供新的财务事实。",
        "未经核实的文字备注：客户经理称其为“高风险企业”。该备注未提供新的财务事实。",
    ],
}
PLACEMENTS = ["before", "after", "system"]
EXISTING_DIAGONAL = {(1, "before"), (2, "after"), (3, "system")}


def load_prompt_module():
    path = ROOT / "scripts" / "python" / "37_prepare_strict_timing_audits.py"
    spec = importlib.util.spec_from_file_location("strict_prompt_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load strict prompt builder.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_prompt_module()
    cases = pd.read_parquet(DERIVED / "strict_timing_nested_cases_64.parquet")
    rows = []
    cells = []
    for valence, wordings in TEXTS.items():
        for wording_number, memo in enumerate(wordings, start=1):
            for placement in PLACEMENTS:
                if (wording_number, placement) in EXISTING_DIAGONAL:
                    continue
                variant = f"strict_info_{valence}_w{wording_number}_{placement}"
                cells.append(
                    {
                        "variant": variant,
                        "valence": valence,
                        "wording": wording_number,
                        "placement": placement,
                    }
                )
                for _, case in cases.iterrows():
                    canonical, canonical_system = module.prompt_for(case, "strict_canonical")
                    task_and_body, contract = canonical.rsplit("\n", 1)
                    if placement == "before":
                        task, body = task_and_body.split("\n", 1)
                        prompt = task + "\n" + memo + "\n" + body + "\n" + contract
                        system = canonical_system
                    elif placement == "after":
                        prompt = task_and_body + "\n" + memo + "\n" + contract
                        system = canonical_system
                    else:
                        prompt = canonical
                        system = canonical_system + " " + memo
                    rows.append(
                        {
                            "run_id": f"{case['case_id']}__{variant}",
                            "case_id": case["case_id"],
                            "firm_id": case["firm_id"],
                            "year": int(case["fiscal_year"]),
                            "report_date": case["report_date"],
                            "landmark_date": case["landmark_date"],
                            "distress_f1": int(case["outcome"]),
                            "sampling_weight": 1.0,
                            "accounting_pd": np.nan,
                            "variant": variant,
                            "valence": valence,
                            "wording": wording_number,
                            "placement": placement,
                            "system_prompt": system,
                            "prompt": prompt,
                        }
                    )
    prompts = pd.DataFrame(rows)
    cells_frame = pd.DataFrame(cells)
    if len(cells_frame) != 18 or len(prompts) != 64 * 18:
        raise AssertionError("The missing-cell factorial must contain 18 cells and 1,152 prompts.")
    prompts.to_parquet(DERIVED / "strict_timing_information_factorial_missing_prompts.parquet", index=False)
    cells_frame.to_csv(DERIVED / "strict_timing_information_factorial_missing_cells.csv", index=False)
    log = {
        "status": "PASS",
        "nested_cases": len(cases),
        "missing_cells": len(cells_frame),
        "prompt_rows": len(prompts),
        "existing_diagonal": [
            {"wording": wording, "placement": placement}
            for wording, placement in sorted(EXISTING_DIAGONAL)
        ],
        "design": "3 valences x 3 wordings x 3 placements",
    }
    (LOGS / "strict_timing_information_factorial_build.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
