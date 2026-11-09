from pathlib import Path
import json,pandas as pd,numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from benchmark_models import rounded_inputs,logit_model,hgb_model,xgb_model,FEATURES
ROOT=Path(__file__).resolve().parents[1]
def fit(name):
 a=pd.read_parquet(ROOT/'data/derived/strict_timing_panel_2011_2024.parquet');x=rounded_inputs(a);y=a.outcome.to_numpy(int);c=json.loads((ROOT/'data/benchmark_config.json').read_text())['selected_parameters'][name];m=logit_model(c['C'])if name=='logit'else hgb_model(c)if name=='hist_gradient_boosting'else xgb_model(c);train=a.fiscal_year.between(2011,2018);hold=a.fiscal_year.between(2019,2024);m.fit(x.loc[train],y[train]);return name,m.predict_proba(x.loc[hold])[:,1]
def main():
 a=pd.read_parquet(ROOT/'data/derived/strict_timing_panel_2011_2024.parquet');a=a[a.fiscal_year.between(2019,2024)].copy();old=pd.read_parquet(ROOT/'data/strict_timing_benchmark_holdout_predictions.parquet');q=old.drop(columns=[c for c in old if c.endswith('_probability')]);assert q.row_id.tolist()==a.row_id.tolist()
 with ProcessPoolExecutor(max_workers=3,mp_context=get_context('fork'))as pool:
  for name,p in pool.map(fit,['logit','hist_gradient_boosting','xgboost']):q[name+'_probability']=p
 q.to_parquet(ROOT/'output/analysis/strict_timing_benchmark_holdout_predictions.parquet',index=False,compression='zstd');rows=[]
 for name in ['logit','hist_gradient_boosting','xgboost']:
  c=name+'_probability';rows.append(dict(model=name,maximum_absolute_prediction_difference=float(np.max(abs(q[c].to_numpy()-old[c].to_numpy()))),n=len(q)))
 pd.DataFrame(rows).to_csv(ROOT/'verify/benchmark_refit_comparison.csv',index=False);print('Structured models retrained from company-year observations',rows,flush=True)
if __name__=='__main__':main()
