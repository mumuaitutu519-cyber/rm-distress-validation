from pathlib import Path
import importlib.util,os,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
def load(n):
 p=next((ROOT/'code').glob(n+'_*.py'));s=importlib.util.spec_from_file_location('rm_'+n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main(mode='quick'):
 import pandas as pd,numpy as np
 out=ROOT/'output/analysis'
 for x in ['output/analysis','output/logs','output/tables','output/figures/strict_timing']:(ROOT/x).mkdir(parents=True,exist_ok=True)
 for f in (ROOT/'data/calls').glob('*.csv'):shutil.copy2(f,out/f.name)
 for f in (ROOT/'data').glob('*'):
  if f.is_file() and f.suffix in ['.csv','.parquet']:shutil.copy2(f,out/f.name)
 from refit_benchmarks import main as refit
 refit()
 a=load('39');a.main()
 e=load('42');e.main()
 c=load('43');c.main()
 f=load('44');f.main()
 b=load('40');r=load('46')
 # Quick mode recomputes intervals from the original saved bootstrap replicates.
 # Full mode reruns calibration inside all 1,000 bootstrap samples.
 if mode=='full':b.main();r.main()
 else:
  for p in (ROOT/'data/frozen_draws').glob('*.parquet'):shutil.copy2(p,out/p.name)
  for name in ['primary','probability','temporal']:
   data=pd.read_parquet(out/f'strict_timing_{name}_bootstrap_draws.parquet')
   if name=='primary':data['result']='primary'
   z=b.percentile_summary(data,['result'] if name=='primary' else ['score'])
   if name=='temporal':z=z.rename(columns={'bootstrap_mean':'mean'})
   z.to_csv(out/f'strict_timing_{name}_bootstrap_ci.csv',index=False)
  r.auc_differences();r.alternative_calibration(a,*a.probability_data())
 g=load('49');g.main()
 # Rebuild all three washout risk sets directly from observed candidate records
 # and dated status changes, using the original construction functions.
 spec=importlib.util.spec_from_file_location('riskset',ROOT/'raw_rebuild/36_build_strict_timing_panel.py')
 risk=importlib.util.module_from_spec(spec);spec.loader.exec_module(risk)
 candidates=pd.read_parquet(ROOT/'data/derived/strict_timing_complete_candidates_2011_2024.parquet')
 status=pd.read_parquet(ROOT/'data/derived/strict_timing_st_changes.parquet')
 # Risk-set rebuild is completed by the separate compact routine if its source API changes.
 from rm_tables import risksets
 risksets(candidates,status,risk,out)
 load('45').main()
 # Preserve all calls at observation level in one compressed, documented table.
 calls=[]
 for p in sorted((ROOT/'data/calls').glob('*.csv')):
  q=pd.read_csv(p);q['configuration_file']=p.stem;calls.append(q)
 pd.concat(calls,ignore_index=True).to_parquet(ROOT/'output/model_calls.parquet',index=False,compression='zstd')
 (ROOT/'output/bootstrap_mode.json').write_text(json.dumps({'mode':mode,'quick_intervals':'percentiles of original frozen refit draws','inference_rerun':False,'seed':20260814},indent=2))
if __name__=='__main__':main(os.environ.get('REPLICATION_MODE','quick'))
