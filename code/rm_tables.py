from pathlib import Path
import pandas as pd
def risksets(candidates,status,risk,out):
 rows=[]
 # Drop frozen labels: rebuild them from dates and the transition observations.
 base=candidates[[c for c in candidates if not any(c.endswith('_w'+str(x)) for x in [0,7,30])]].copy()
 for days in [0,7,30]:
  a=risk.state_and_outcome(base,status,days)
  for label,lo,hi in [('training',2011,2018),('holdout',2019,2024)]:
   z=a[a.fiscal_year.between(lo,hi)&a[f'at_risk_w{days}'].fillna(False)]
   rows.append(dict(washout_days=days,period=label,records=len(z),firms=z.firm_id.nunique(),st_events=z[f'st_event_w{days}'].sum(),st_event_rate=z[f'st_event_w{days}'].mean(),composite_events=z[f'composite_event_w{days}'].sum(),composite_event_rate=z[f'composite_event_w{days}'].mean()))
 pd.DataFrame(rows).to_csv(out/'strict_timing_washout_risksets.csv',index=False)

def build_cells():
 import numpy as np,csv,re,json
 from sklearn.metrics import roc_auc_score
 root=Path(__file__).resolve().parents[1];out=root/'output/analysis';rows=[];cache={}
 def rd(name):
  if name not in cache:cache[name]=pd.read_csv(out/f'strict_timing_{name}.csv')
  return cache[name]
 def one(name,**keys):
  z=rd(name)
  for k,v in keys.items():z=z[np.isclose(z[k],v)]if isinstance(v,float)else z[z[k].eq(v)]
  assert len(z)==1,(name,keys,len(z));return z.iloc[0]
 def put(t,r,c,value,source,token=1,reason=''):
  rows.append(dict(object=t,row=r,col=c,token=token,value=float(value),source=source,reason=reason,status=''))
 def seq(t,r,values,source,cols=None):
  for c,v in zip(cols or range(2,len(values)+2),values):put(t,r,c,v,source)
 annual=rd('annual_frame')
 for j,a in annual.iterrows():seq(1,j+2,[a.records,a.events,100*a.event_rate],'output/analysis/strict_timing_annual_frame.csv')
 seq(1,8,[annual.records.sum(),annual.events.sum(),100*annual.events.sum()/annual.records.sum()],'output/analysis/strict_timing_annual_frame.csv (sum)')
 models=['qwen_canonical_naive','logit','hist_gradient_boosting','xgboost']
 for j,m in enumerate(models):
  a=one('probability_metrics',score=m,evaluation='raw');b=one('probability_metrics',score=m,evaluation='five_fold_grouped_crossfit_platt')
  seq(2,j+2,[a.roc_auc,a.pr_auc,100*b.weighted_mean_probability,100*b.brier,b.brier_skill_score,b.log_loss],'output/analysis/strict_timing_probability_metrics.csv score='+m)
 q=rd('primary_range_summary').iloc[0]
 for r,k,m in [(2,'median_range_pp',1),(3,'mean_range_pp',1),(4,'any_change_rate',100)]:put(3,r,2,q[k]*m,'output/analysis/strict_timing_primary_range_summary.csv')
 for r,x in [(5,.01),(6,.02)]:put(3,r,2,100*one('threshold_crossing_curve',threshold=x).design_weighted_crossing_rate,'output/analysis/strict_timing_threshold_crossing_curve.csv')
 for r,x in [(7,.05),(9,.1)]:
  a=one('primary_rosters',capacity=x,variant='three_representation_summary')
  put(3,r,2,a.intersection_all,'output/analysis/strict_timing_primary_rosters.csv',1);put(3,r,2,a.union_all,'output/analysis/strict_timing_primary_rosters.csv',2);put(3,r+1,2,a.jaccard_all,'output/analysis/strict_timing_primary_rosters.csv')
  put(3,r,3,a.top_n,'output/analysis/strict_timing_primary_rosters.csv (fixed-capacity list count)')
 for r in [2,3,4,8,10]:put(3,r,3,q.n,'output/analysis/strict_timing_primary_range_summary.csv (sample size)')
 pred=pd.read_parquet(out/'strict_timing_probability_predictions.parquet');w=pred.sampling_weight.to_numpy(float);y=pred.outcome.to_numpy(float)
 action=[]
 for i,thr in enumerate([.01,.02]):
  for j,m in enumerate(['qwen_canonical_naive','qwen_reverse_naive','qwen_decimal_naive']):
   flag=pred[m+'_grouped_platt'].ge(thr).to_numpy();a=[100*(w*flag).sum()/w.sum(),100*(w*y*flag).sum()/(w*y).sum(),100*(w*y*flag).sum()/(w*flag).sum()]
   seq(4,3+i*3+j,a,'output/analysis/strict_timing_probability_predictions.parquet (weighted threshold)',[3,4,5]);action.append(dict(kind='threshold',score=m,threshold=thr,flagged=a[0],capture=a[1],precision=a[2]))
 def capacity(score,k):
  z=pd.DataFrame(dict(score=score,w=w,e=w*y)).groupby('score',sort=True)[['w','e']].sum().sort_index(ascending=False);before=z.w.cumsum()-z.w
  f=((k*w.sum()-before)/z.w).clip(0,1);return 100*(f*z.e).sum()/(w*y).sum()
 for i,m in enumerate(models):
  auc=one('probability_metrics',score=m,evaluation='raw').roc_auc;v=[auc,capacity(pred[m+'_raw'],.05),capacity(pred[m+'_raw'],.1)]
  seq(4,11+i,v,'output/analysis/strict_timing_probability_predictions.parquet (weighted capacity, fractional score ties)');action.append(dict(kind='capacity',score=m,auc=auc,capture5=v[1],capture10=v[2]))
 pd.DataFrame(action).to_csv(out/'screening_workload.csv',index=False)
 order=rd('order_summary').iloc[0];put(5,2,2,order.median_order_range_pp,'output/analysis/strict_timing_order_summary.csv');put(5,2,3,order.mean_pairwise_order_difference_pp,'output/analysis/strict_timing_order_summary.csv')
 info=rd('information_fe_coefficients')
 for r,pattern in [(3,'[T.adverse]'),(4,'[T.favorable]'),(5,'[T.after]'),(6,'[T.system]')]:
  a=info[info.term.str.contains(pattern,regex=False)].iloc[0];put(5,r,2,a.estimate_pp,'output/analysis/strict_timing_information_fe_coefficients.csv term='+a.term)
 for r,key in [(7,'strict_deteriorated'),(8,'strict_improved'),(9,'strict_roa_down_1pp'),(10,'strict_cash_down_2pp')]:
  a=one('economic_response',variant=key);put(5,r,2,a.mean_delta_pp,'output/analysis/strict_timing_economic_response.csv '+key);put(5,r,3,100*a.direction_violation_rate_when_changed,'output/analysis/strict_timing_economic_response.csv '+key)
 for i,a in rd('artifact_profiles').iterrows():seq(6,i+2,[a.canonical_auc,a.median_representation_range_pp,a.mean_representation_range_pp,100*a.any_representation_change_rate,a.maximum_representation_range_pp],'output/analysis/strict_timing_artifact_profiles.csv '+a.artifact)
 features=['Size','Lev','ROA','Growth','Cash','Tangibility','TobinQ','FirmAge','Loss','return_volatility']
 for i,k in enumerate(features):
  a=one('descriptives',variable=k,sample='holdout');b=one('descriptives',variable=k,sample='paired_audit');c=one('descriptives',variable=k,sample='probability_design_weighted')
  seq(7,i+2,[a['mean'],a.standard_deviation,b['mean'],c['mean']],'output/analysis/strict_timing_descriptives.csv '+k)
 for i,a in rd('washout_risksets').iterrows():seq(8,i+2,[a.records,a.firms,a.st_events,100*a.st_event_rate,a.composite_events,100*a.composite_event_rate],'output/analysis/strict_timing_washout_risksets.csv',range(3,9))
 for i,a in annual.iterrows():seq(9,i+2,[a.records,a.events,100*a.event_rate,a.logit_auc,a.xgboost_auc],'output/analysis/strict_timing_annual_frame.csv')
 for i,a in rd('regulatory_signal_validation').iterrows():seq(10,i+2,[a.exposed_records,100*a.event_rate_exposed,100*a.event_rate_unexposed,100*a.rate_difference],'output/analysis/strict_timing_regulatory_signal_validation.csv '+a.signal)
 reason=rd('status_reason_decomposition')
 for i,a in reason.iterrows():seq(11,i+2,[a['count_2019--2022'],a['count_2023'],a['count_2024']],'output/analysis/strict_timing_status_reason_decomposition.csv')
 seq(11,12,[annual[annual.fiscal_year.le(2022)].events.sum(),annual[annual.fiscal_year.eq(2023)].events.sum(),annual[annual.fiscal_year.eq(2024)].events.sum()],'output/analysis/strict_timing_annual_frame.csv (sum)')
 # A6 is alphabetically sorted by its machine-readable variant keys.
 for i,a in rd('extended_paired_summary').sort_values('variant').iterrows():
  pass
 for i,a in enumerate(rd('extended_paired_summary').sort_values('variant').to_dict('records')):seq(12,i+2,[a['mean_delta_pp'],a['mean_absolute_delta_pp'],100*a['changed_rate'],a['maximum_absolute_delta_pp']],'output/analysis/strict_timing_extended_paired_summary.csv '+a['variant'])
 for i,a in rd('information_margins').iterrows():seq(14,i+2,[a.mean_delta_pp,a.mean_absolute_delta_pp,100*a.changed_rate],'output/analysis/strict_timing_information_margins.csv '+a.valence+'/'+a.placement,[3,4,5])
 for i,a in rd('economic_response').iterrows():
  values=[a.mean_delta_pp,100*a.changed_rate,100*a.direction_violation_rate_when_changed,a.maximum_reverse_change_pp,100*a.marginal_training_support_rate]
  for c,v in zip(range(3,8),values):
   if np.isfinite(v):put(15,i+2,c,v,'output/analysis/strict_timing_economic_response.csv '+a.variant)
 callfiles=['primary_qwen3_4b_4bit','probability_qwen3_4b_4bit','extended_qwen3_4b_4bit','information_factorial_missing_qwen3_4b_4bit','core_qwen3_4b_4bit_batch1','core_qwen3_4b_4bit_batch1_reload','core_qwen3_4b_fp16_batch1','core_qwen3_8b_4bit_batch1','core_mistral_nemo_12b_4bit_batch1'];tot=0
 for i,k in enumerate(callfiles):
  a=pd.read_csv(root/'data/calls'/f'strict_timing_{k}.csv');tot+=len(a);seq(16,i+2,[len(a),8 if i<4 else 1,100*a.parse_pd.mean()],'data/calls/strict_timing_'+k+'.csv')
 seq(16,11,[tot,100],'data/calls/* (sum)',[2,4])
 for i,(m,cal)in enumerate([('qwen_canonical_naive',c)for c in ['intercept_only','platt','beta','isotonic']]+[(m,'platt')for m in models[1:]]):
  a=one('calibration_robustness',score=m,calibration=cal);seq(17,i+2,[100*a.weighted_mean_probability,a.brier,a.brier_skill_score,a.log_loss,a.pooled_calibration_intercept,a.pooled_calibration_slope],'output/analysis/strict_timing_calibration_robustness.csv '+m+'/'+cal,range(3,9))
 for i,a in rd('auc_difference_ci').iterrows():seq(18,i+2,[a.mean_difference,100*a.share_positive],'output/analysis/strict_timing_auc_difference_ci.csv')
 for r,name in [(2,'reload_repeatability'),(3,'batch_sensitivity'),(4,'quantization_sensitivity')]:
  a=rd(name).iloc[0];seq(19,r,[a.calls,100*a.get('score_changed_rate',a.get('changed_rate',0)),a.get('mean_absolute_difference_pp',0),a.maximum_absolute_difference_pp],'output/analysis/strict_timing_'+name+'.csv')
 pd.DataFrame(rows).to_csv(root/'output/computed_cells.csv',index=False)
 # Statistical backing data and original-vintage backing data remain side by side.
 fdir=root/'verify/figures';fdir.mkdir(exist_ok=True)
 for k in ['annual_frame','reliability_bins','primary_case_ranges','threshold_crossing_curve','artifact_profiles']:
  for suffix in ['csv','parquet']:
   f=out/f'strict_timing_{k}.{suffix}'
   if f.exists():
    import shutil
    shutil.copy2(f,fdir/('recomputed_'+f.name));old=root/'verify/source_results'/f.name
    if old.exists():shutil.copy2(old,fdir/('manuscript_vintage_'+f.name))
 print('RM mapped cells',len(rows),flush=True)
if __name__=='__main__':build_cells()
