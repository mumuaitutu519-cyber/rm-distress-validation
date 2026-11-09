#!/usr/bin/env python3
"""Fit temporally tuned structured benchmarks for the strict timing panel."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 20260814
ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
ANALYSIS = ROOT / "output" / "analysis"
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
DECIMALS = {
    "Size": 3,
    "Lev": 4,
    "ROA": 4,
    "Growth": 4,
    "Cash": 4,
    "Tangibility": 4,
    "TobinQ": 3,
    "FirmAge": 3,
    "Loss": 0,
    "return_volatility": 4,
}


def rounded_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[FEATURES].copy()
    for column, digits in DECIMALS.items():
        out[column] = out[column].round(digits)
    return out


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, probability))
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": prevalence,
        "mean_probability": float(probability.mean()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": brier,
        "brier_skill_score": float(1 - brier / (prevalence * (1 - prevalence))),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def logit_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=c_value,
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def hgb_model(params: dict[str, float | int]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        random_state=SEED,
    )


def xgb_model(params: dict[str, float | int]) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=float(params["reg_lambda"]),
        random_state=SEED,
        n_jobs=4,
    )


def select_best(tuning: pd.DataFrame) -> pd.Series:
    return tuning.sort_values(
        ["roc_auc", "pr_auc", "brier"], ascending=[False, False, True]
    ).iloc[0]


def main() -> None:
    for directory in [ANALYSIS, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PANEL_PATH)
    x = rounded_inputs(frame)
    y = frame["outcome"].to_numpy(int)
    tune_train = frame["fiscal_year"].between(2011, 2016).to_numpy()
    validation = frame["fiscal_year"].between(2017, 2018).to_numpy()
    full_train = frame["fiscal_year"].between(2011, 2018).to_numpy()
    holdout = frame["fiscal_year"].between(2019, 2024).to_numpy()

    logit_rows = []
    for c_value in [0.02, 0.05, 0.2, 1.0, 5.0]:
        model = logit_model(c_value).fit(x.loc[tune_train], y[tune_train])
        probability = model.predict_proba(x.loc[validation])[:, 1]
        logit_rows.append({"C": c_value, **metrics(y[validation], probability)})
    logit_tuning = pd.DataFrame(logit_rows)
    best_logit = select_best(logit_tuning)

    hgb_rows = []
    for max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization in itertools.product(
        [200, 400], [0.03, 0.05], [7, 15, 31], [20, 50], [0.0, 1.0]
    ):
        params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
        }
        model = hgb_model(params).fit(x.loc[tune_train], y[tune_train])
        probability = model.predict_proba(x.loc[validation])[:, 1]
        hgb_rows.append({**params, **metrics(y[validation], probability)})
    hgb_tuning = pd.DataFrame(hgb_rows)
    best_hgb = select_best(hgb_tuning)

    xgb_rows = []
    for n_estimators, max_depth, learning_rate, min_child_weight, reg_lambda in itertools.product(
        [200, 400], [2, 3, 4], [0.03, 0.05], [5.0, 20.0], [1.0, 10.0]
    ):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
        }
        model = xgb_model(params).fit(x.loc[tune_train], y[tune_train])
        probability = model.predict_proba(x.loc[validation])[:, 1]
        xgb_rows.append({**params, **metrics(y[validation], probability)})
    xgb_tuning = pd.DataFrame(xgb_rows)
    best_xgb = select_best(xgb_tuning)

    selected = {
        "logit": {"C": float(best_logit["C"])},
        "hist_gradient_boosting": {
            key: (int(best_hgb[key]) if key in ["max_iter", "max_leaf_nodes", "min_samples_leaf"] else float(best_hgb[key]))
            for key in ["max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]
        },
        "xgboost": {
            key: (int(best_xgb[key]) if key in ["n_estimators", "max_depth"] else float(best_xgb[key]))
            for key in ["n_estimators", "max_depth", "learning_rate", "min_child_weight", "reg_lambda"]
        },
    }
    models = {
        "logit": logit_model(selected["logit"]["C"]),
        "hist_gradient_boosting": hgb_model(selected["hist_gradient_boosting"]),
        "xgboost": xgb_model(selected["xgboost"]),
    }
    predictions = frame[
        [
            "row_id",
            "firm_id",
            "fiscal_year",
            "report_date",
            "landmark_date",
            "industry",
            "industry_sector",
            "regime_exposed",
            "outcome",
            "outcome_composite",
            *FEATURES,
        ]
    ].copy()
    metric_rows = []
    for name, model in models.items():
        model.fit(x.loc[full_train], y[full_train])
        predictions[f"{name}_probability"] = model.predict_proba(x)[:, 1]
        joblib.dump(model, ANALYSIS / f"strict_timing_{name}.joblib")
        for period, mask in [
            ("validation_2017_2018", validation),
            ("holdout_2019_2024", holdout),
            ("pre_2024_rules_2019_2022", frame["fiscal_year"].between(2019, 2022).to_numpy()),
            ("rule_exposed_2023_2024", frame["fiscal_year"].between(2023, 2024).to_numpy()),
        ]:
            if period == "validation_2017_2018":
                temporary = {
                    "logit": logit_model(selected["logit"]["C"]),
                    "hist_gradient_boosting": hgb_model(selected["hist_gradient_boosting"]),
                    "xgboost": xgb_model(selected["xgboost"]),
                }[name]
                temporary.fit(x.loc[tune_train], y[tune_train])
                probability = temporary.predict_proba(x.loc[mask])[:, 1]
            else:
                probability = predictions.loc[mask, f"{name}_probability"].to_numpy()
            metric_rows.append({"model": name, "period": period, **metrics(y[mask], probability)})
        for year in range(2019, 2025):
            mask = frame["fiscal_year"].eq(year).to_numpy()
            metric_rows.append(
                {
                    "model": name,
                    "period": f"fiscal_year_{year}",
                    **metrics(y[mask], predictions.loc[mask, f"{name}_probability"].to_numpy()),
                }
            )

    predictions.loc[holdout].to_parquet(
        ANALYSIS / "strict_timing_benchmark_holdout_predictions.parquet", index=False
    )
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(ANALYSIS / "strict_timing_benchmark_metrics.csv", index=False)
    logit_tuning.to_csv(ANALYSIS / "strict_timing_logit_tuning.csv", index=False)
    hgb_tuning.to_csv(ANALYSIS / "strict_timing_hgb_tuning.csv", index=False)
    xgb_tuning.to_csv(ANALYSIS / "strict_timing_xgboost_tuning.csv", index=False)

    audit = {
        "status": "PASS",
        "seed": SEED,
        "features": FEATURES,
        "input_precision": "prompt displayed",
        "tuning_train": [2011, 2016],
        "validation": [2017, 2018],
        "final_train": [2011, 2018],
        "holdout": [2019, 2024],
        "selected_parameters": selected,
        "holdout_metrics": metric_table.loc[
            metric_table["period"].eq("holdout_2019_2024")
        ].to_dict("records"),
        "files": {
            "predictions": str(ANALYSIS / "strict_timing_benchmark_holdout_predictions.parquet"),
            "metrics": str(ANALYSIS / "strict_timing_benchmark_metrics.csv"),
        },
    }
    (LOGS / "strict_timing_benchmarks.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metric_table.loc[metric_table["period"].eq("holdout_2019_2024")].to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
