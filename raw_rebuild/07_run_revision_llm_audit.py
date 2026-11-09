#!/usr/bin/env python3
"""Run a revision prompt suite through a local MLX language model."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler
from huggingface_hub import snapshot_download


SEED = 20260810
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DERIVED = ROOT / "data/derived"
ANALYSIS = ROOT / "output/analysis"
LOGS = ROOT / "output/logs"

MODEL_PATHS = {
    "qwen3_4b_4bit": (
        Path.home()
        / ".cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/"
        "snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
    ),
    "qwen25_15b_4bit": WORKSPACE / "jfe_ai_measurement_20260727/models/qwen25_15b_4bit",
    "qwen25_15b_fp16": (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/"
        "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    ),
    "qwen3_4b_fp16": (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
    ),
    "qwen3_8b_4bit": (
        "mlx-community/Qwen3-8B-4bit",
        "545dc4251c05440727734bcd94334791f6ab0192",
    ),
    "mistral_nemo_12b_4bit": (
        "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "647ca0751669b21a364c86ccc5df54c4d7e4e91c",
    ),
}


def resolve_model_path(model_key: str) -> Path:
    path = MODEL_PATHS[model_key]
    if isinstance(path, tuple):
        repo_id, revision = path
        return Path(snapshot_download(repo_id, revision=revision))
    if model_key == "qwen3_4b_fp16" and path.is_dir():
        snapshots = sorted(p for p in path.iterdir() if p.is_dir())
        if snapshots:
            path = snapshots[-1]
    if not path.exists():
        raise FileNotFoundError(f"Missing model for {model_key}: {path}")
    return path


def format_chat(tokenizer, system_text: str, user_text: str) -> list[int]:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_output(raw: str) -> dict[str, object]:
    # New target-aligned audits request SCORE rather than PD because the
    # listed-firm ST/*ST outcome is not a bank probability of default.  Keep
    # the legacy PD parser so the same runner remains able to resume and audit
    # the frozen earlier experiments.
    pd_match = re.search(
        r"(?:PD|SCORE)\s*=\s*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.I
    )
    grade_match = re.search(r"GRADE\s*=\s*([ABCD])", raw, flags=re.I)
    decision_match = re.search(
        r"DECISION\s*=\s*(APPROVE|REVIEW|DECLINE)", raw, flags=re.I
    )
    pd_value = float(pd_match.group(1)) if pd_match else np.nan
    if pd_match and 0 <= pd_value <= 100:
        parse_pd = True
    else:
        pd_value = np.nan
        parse_pd = False
    has_generated_labels = bool(grade_match and decision_match)
    return {
        "pd_percent": pd_value,
        "grade": grade_match.group(1).upper() if grade_match else "",
        "decision": decision_match.group(1).upper() if decision_match else "",
        "parse_pd": parse_pd,
        "parse_full": bool(parse_pd and (has_generated_labels or "SCORE" in raw.upper())),
        "raw_output": raw.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=MODEL_PATHS, required=True)
    parser.add_argument("--prompts", required=True, help="Parquet filename under data/derived")
    parser.add_argument("--output-name", required=True, help="CSV filename without extension")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variants",
        default="ALL",
        help="Comma-separated variants to retain, or ALL (default).",
    )
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    prompts_path = DERIVED / args.prompts
    prompts = pd.read_parquet(prompts_path)
    if args.variants.upper() != "ALL":
        variants = [value.strip() for value in args.variants.split(",") if value.strip()]
        prompts = prompts.loc[prompts["variant"].isin(variants)].copy()
    if args.limit:
        prompts = prompts.head(args.limit)
    if "run_id" not in prompts:
        prompts["run_id"] = (
            prompts["case_id"].astype(str) + "__" + prompts["variant"].astype(str)
        )

    output_path = ANALYSIS / f"{args.output_name}.csv"
    completed: set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        completed = set(existing["run_id"].astype(str))
    todo = prompts.loc[~prompts["run_id"].isin(completed)].copy()

    model_path = resolve_model_path(args.model_key)
    mx.random.seed(SEED)
    started = time.time()
    tokenizer_config = (
        {"fix_mistral_regex": True}
        if args.model_key == "mistral_nemo_12b_4bit"
        else None
    )
    model, tokenizer = load(
        str(model_path), lazy=True, tokenizer_config=tokenizer_config
    )
    sampler = make_sampler(temp=args.temperature)
    tokenized = [
        format_chat(tokenizer, str(row["system_prompt"]), str(row["prompt"]))
        for _, row in todo.iterrows()
    ]

    buffer: list[dict[str, object]] = []
    for start in range(0, len(todo), args.batch_size):
        batch = todo.iloc[start : start + args.batch_size]
        response = batch_generate(
            model,
            tokenizer,
            tokenized[start : start + args.batch_size],
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        for (_, record), raw in zip(batch.iterrows(), response.texts):
            buffer.append(
                {
                    "run_id": record["run_id"],
                    "case_id": record["case_id"],
                    "model_key": args.model_key,
                    "variant": record["variant"],
                    "year": int(record["year"]),
                    "distress_f1": int(record["distress_f1"]),
                    "sampling_weight": float(record["sampling_weight"]),
                    "accounting_pd": float(record["accounting_pd"]),
                    "repeat_id": record.get("repeat_id", np.nan),
                    **parse_output(raw),
                }
            )
        if len(buffer) >= 64 or start + args.batch_size >= len(todo):
            new = pd.DataFrame(buffer)
            new.to_csv(
                output_path,
                mode="a",
                index=False,
                header=not output_path.exists(),
            )
            buffer.clear()
        completed_now = min(start + args.batch_size, len(todo))
        if completed_now % 64 == 0 or completed_now == len(todo):
            print(
                json.dumps(
                    {
                        "model": args.model_key,
                        "output": args.output_name,
                        "completed_this_run": completed_now,
                        "remaining_this_run": len(todo) - completed_now,
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final = pd.read_csv(output_path).drop_duplicates("run_id", keep="last")
    audit = {
        "status": "PASS",
        "model_key": args.model_key,
        "model_path": str(model_path),
        "prompt_file": str(prompts_path),
        "output_file": str(output_path),
        "seed": SEED,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "tokenizer_config": tokenizer_config or {},
        "rows": int(len(final)),
        "unique_runs": int(final["run_id"].nunique()),
        "pd_parse_rate": float(final["parse_pd"].astype(bool).mean()),
        "decision_parse_rate": float(final["decision"].notna().mean()),
        "full_parse_rate": float(final["parse_full"].astype(bool).mean()),
        "elapsed_seconds_this_run": round(time.time() - started, 2),
    }
    (LOGS / f"{args.output_name}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
