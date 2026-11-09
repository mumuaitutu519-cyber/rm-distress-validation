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
DERIVED = ROOT / 'data' / 'derived'
ANALYSIS = ROOT / 'output' / 'analysis'
LOGS = ROOT / 'output' / 'logs'
PANEL_PATH = DERIVED / 'strict_timing_panel_2011_2024.parquet'
FEATURES = ['Size', 'Lev', 'ROA', 'Growth', 'Cash', 'Tangibility', 'TobinQ', 'FirmAge', 'Loss', 'return_volatility']
DECIMALS = {'Size': 3, 'Lev': 4, 'ROA': 4, 'Growth': 4, 'Cash': 4, 'Tangibility': 4, 'TobinQ': 3, 'FirmAge': 3, 'Loss': 0, 'return_volatility': 4}

def rounded_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[FEATURES].copy()
    for column, digits in DECIMALS.items():
        out[column] = out[column].round(digits)
    return out

def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-08, 1 - 1e-08)
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, probability))
    return {'n': int(len(y)), 'events': int(y.sum()), 'event_rate': prevalence, 'mean_probability': float(probability.mean()), 'roc_auc': float(roc_auc_score(y, probability)), 'pr_auc': float(average_precision_score(y, probability)), 'brier': brier, 'brier_skill_score': float(1 - brier / (prevalence * (1 - prevalence))), 'log_loss': float(log_loss(y, probability, labels=[0, 1]))}

def logit_model(c_value: float) -> Pipeline:
    return Pipeline([('scaler', StandardScaler()), ('logit', LogisticRegression(C=c_value, max_iter=4000, solver='lbfgs', random_state=SEED))])

def hgb_model(params: dict[str, float | int]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=float(params['learning_rate']), max_iter=int(params['max_iter']), max_leaf_nodes=int(params['max_leaf_nodes']), min_samples_leaf=int(params['min_samples_leaf']), l2_regularization=float(params['l2_regularization']), random_state=SEED)

def xgb_model(params: dict[str, float | int]) -> XGBClassifier:
    return XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_estimators=int(params['n_estimators']), max_depth=int(params['max_depth']), learning_rate=float(params['learning_rate']), min_child_weight=float(params['min_child_weight']), subsample=0.9, colsample_bytree=0.9, reg_lambda=float(params['reg_lambda']), random_state=SEED, n_jobs=1)