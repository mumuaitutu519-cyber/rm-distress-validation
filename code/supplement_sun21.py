"""Additional sun21 note/body checks using archived observations only."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import importlib.util, json, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/analysis'

def module(name):
    p=next((ROOT/'code').glob(name+'_*.py'))
    spec=importlib.util.spec_from_file_location('supp_'+name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def weighted_capture(score,weight,event,capacity):
    _,inverse=np.unique(score,return_inverse=True)
    mass=np.bincount(inverse,weights=weight)[::-1]
    events=np.bincount(inverse,weights=weight*event,minlength=len(mass))[::-1]
    before=np.cumsum(mass)-mass
    share=np.divide(capacity*weight.sum()-before,mass,out=np.zeros_like(mass),where=mass>0).clip(0,1)
    return 100*np.dot(share,events)/np.dot(weight,event)

def capture_chunk(task):
    seeds,scores,weight,event,firm_inverse,nfirms=task;rows=[]
    for draw,seed in seeds:
        rng=np.random.default_rng(seed)
        counts=np.bincount(rng.choice(nfirms,nfirms,replace=True),minlength=nfirms)
        w=weight*counts[firm_inverse]
        for cap in [.05,.10]:
            base=weighted_capture(scores[:,0],w,event,cap)
            for j,name in enumerate(['logit','hist_gradient_boosting','xgboost'],1):
                rows.append(dict(draw=draw,model=name,capacity=cap,difference_pp=weighted_capture(scores[:,j],w,event,cap)-base))
    return rows

def main():
    values={}
    def put(key,value,source,**extra):values[key]=dict(value=float(value),source=source,**extra)
    panel=pd.read_parquet(ROOT/'data/derived/strict_timing_panel_2011_2024.parquet')
    for key,value in [('panel_records',len(panel)),('panel_firms',panel.firm_id.nunique())]:put(key,value,'data/derived/strict_timing_panel_2011_2024.parquet')
    for label,lo,hi in [('training',2011,2018),('holdout',2019,2024)]:
        q=panel[panel.fiscal_year.between(lo,hi)]
        for key,value in [('records',len(q)),('events',q.outcome.sum()),('rate_percent',100*q.outcome.mean())]:put(label+'_'+key,value,'data/derived/strict_timing_panel_2011_2024.parquet')
    status=pd.read_parquet(ROOT/'data/derived/strict_timing_st_changes.parquet')
    put('status_records',len(status.drop_duplicates(['firm_id','execution_date','transition'])),'data/derived/strict_timing_st_changes.parquet')
    put('status_firms',status.firm_id.nunique(),'data/derived/strict_timing_st_changes.parquet')
    primary=pd.read_csv(ROOT/'data/calls/strict_timing_primary_qwen3_4b_4bit.csv')
    piv=primary.pivot(index='case_id',columns='variant',values='pd_percent')
    match=piv[(piv.strict_canonical==32)&(piv.strict_order_reversed==12)&(piv.strict_unit_decimal==38)]
    assert len(match)==1
    case=match.index[0]
    a=module('39');cases,scores=a.probability_data();folds=a.assign_grouped_folds(cases)
    original=pd.read_parquet(ROOT/'data/derived/strict_timing_primary_cases_256.parquet').set_index('case_id').loc[case]
    index=np.flatnonzero(cases.row_id.eq(original.row_id).to_numpy())
    if len(index):
        idx=index[0];fold=folds[idx];train=folds!=fold
        beta=a.fit_platt(scores['qwen_canonical_naive'][train],cases.outcome.to_numpy()[train],cases.sampling_weight.to_numpy()[train])
        probability_method=f'Probability audit fold {fold}; baseline Platt estimated without this firm'
    else:
        beta=a.fit_platt(scores['qwen_canonical_naive'],cases.outcome.to_numpy(),cases.sampling_weight.to_numpy())
        probability_method='Full design-weighted baseline Platt fit; example firm absent from probability audit'
    for variant,name in [('strict_canonical','baseline'),('strict_order_reversed','reverse'),('strict_unit_decimal','decimal')]:
        score=match.iloc[0][variant]
        put('intro_score_'+name,score,'data/calls/strict_timing_primary_qwen3_4b_4bit.csv; '+case)
        put('intro_probability_'+name,100*a.apply_platt(np.array([score/100]),beta)[0],probability_method,diagnostic_only=not bool(len(index)))
    coefficients=pd.read_csv(OUT/'strict_timing_calibration_coefficients.csv')
    coefficients=coefficients[(coefficients.score=='qwen_canonical_naive')&(coefficients.evaluation=='five_fold_grouped_crossfit')]
    alternatives=[dict(fold=int(r.fold),probabilities_percent=(100*a.apply_platt(np.array([.32,.12,.38]),np.array([r.intercept,r.slope]))).tolist()) for r in coefficients.itertuples()]
    (ROOT/'verify/INTRO_EXAMPLE_sun21.json').write_text(json.dumps(dict(case_id=case,row_id=original.row_id,method=probability_method,beta=beta.tolist(),matched_probability_audit_records=len(index),archived_fold_alternatives=alternatives,note='Fold 3 reproduces the manuscript rounded probabilities, but no archived assignment connects this out-of-audit example to that map. This is conditional recovery, not verified map provenance.',original_replies=primary[primary.case_id.eq(case)][['variant','raw_output','pd_percent']].to_dict('records')),indent=2)+'\n')
    put('probability_n',len(cases),'data/derived/strict_timing_probability_cases_1500.parquet')
    put('probability_events',cases.outcome.sum(),'data/derived/strict_timing_probability_cases_1500.parquet')
    put('weighted_holdout_records',cases.sampling_weight.sum(),'data/derived/strict_timing_probability_cases_1500.parquet')
    put('weighted_holdout_rate',100*np.average(cases.outcome,weights=cases.sampling_weight),'data/derived/strict_timing_probability_cases_1500.parquet')
    ext=pd.read_csv(ROOT/'data/calls/strict_timing_extended_qwen3_4b_4bit.csv')
    for var,name,sign in [('strict_roa_down_1pp','roa',1),('strict_cash_down_2pp','cash',1)]:
        b=primary[primary.variant.eq('strict_canonical')].set_index('case_id').pd_percent
        q=ext[ext.variant.eq(var)].set_index('case_id').pd_percent
        d=q-b.loc[q.index]
        put(name+'_changed',d.ne(0).sum(),'data/calls/ (paired raw scores)')
        put(name+'_wrong',(d*sign<0).sum(),'data/calls/ (paired raw scores)')
    info=pd.read_parquet(OUT/'strict_timing_information_factorial_panel.parquet')
    sizes=info.groupby(['valence','placement']).size()
    assert sizes.nunique()==1
    put('info_cell_n',sizes.iloc[0],'output/analysis/strict_timing_information_factorial_panel.parquet')
    put('info_firms',info.case_id.nunique(),'output/analysis/strict_timing_information_factorial_panel.parquet')
    reason=pd.read_csv(OUT/'strict_timing_status_reason_decomposition.csv',dtype={'reason_code':str})
    rr=reason[reason.reason_code.eq('11')].iloc[0]
    for k,v in [('reason11_pre',rr['count_2019--2022']),('reason11_post',rr['count_2023']+rr['count_2024']),('pre_events',246),('post_events',228)]:
        if k.endswith('events'):
            years=panel[panel.fiscal_year.between(2019,2022) if k=='pre_events' else panel.fiscal_year.between(2023,2024)]
            v=years.outcome.sum()
        put(k,v,'output/analysis/strict_timing_status_reason_decomposition.csv and rebuilt panel')
    reload=pd.read_csv(ROOT/'data/calls/strict_timing_core_qwen3_4b_4bit_batch1_reload.csv')
    base=pd.read_csv(ROOT/'data/calls/strict_timing_core_qwen3_4b_4bit_batch1.csv')
    z=base.merge(reload,on=['case_id','variant'],suffixes=('_a','_b'),validate='one_to_one')
    put('reload_raw_equality_percent',100*z.raw_output_a.eq(z.raw_output_b).mean(),'data/calls/ matched reload raw strings')
    mistral=pd.read_csv(ROOT/'data/calls/strict_timing_core_mistral_nemo_12b_4bit_batch1.csv')
    for var,name in [('strict_canonical','baseline'),('strict_order_reversed','reverse'),('strict_unit_decimal','decimal')]:put('mistral_mean_'+name,mistral.loc[mistral.variant.eq(var),'pd_percent'].mean(),'data/calls/strict_timing_core_mistral_nemo_12b_4bit_batch1.csv')
    # Check percent/decimal equivalence from the archived prompt strings.
    field_pairs=0;different=0
    from decimal import Decimal
    for name in ['primary_prompts_256','probability_prompts_1500']:
        prompts=pd.read_parquet(ROOT/'raw_rebuild'/f'strict_timing_{name}.parquet')
        for _,g in prompts.groupby('case_id'):
            c=g[g.variant.eq('strict_canonical')].iloc[0].prompt
            d=g[g.variant.eq('strict_unit_decimal')].iloc[0].prompt
            cp=re.findall(r'[-+]?\d+(?:\.\d+)?(?=%)',c)
            dp=[re.search(r'[:：]\s*([-+]?\d+(?:\.\d+)?)',line).group(1) for line in d.splitlines() if ('decimal' in line or '小数' in line) and re.search(r'[:：]\s*([-+]?\d+(?:\.\d+)?)',line)]
            if len(cp)!=len(dp):raise AssertionError((name,len(cp),len(dp)))
            for x,y in zip(cp,dp):field_pairs+=1;different+=Decimal(x)!=100*Decimal(y)
    put('prompt_field_pairs',field_pairs,'raw_rebuild/strict_timing_{primary,probability}_prompts*.parquet')
    put('prompt_inequivalent_pairs',different,'Archived prompt percentages versus decimal strings (exact Decimal)')
    put('all_prompt_field_pairs',(256+1500)*10,'Ten archived fields per primary/probability case, including unchanged non-ratio fields')
    put('all_equal_prompt_field_pairs',(256+1500)*10-different,'All ten fields minus the non-equivalent ratio field pairs')
    # Exact top-k boundary diagnostics; stable case ID is the published rule.
    variants=['strict_canonical','strict_order_reversed','strict_unit_decimal']
    for var,name in [(variants[0],'baseline'),(variants[2],'decimal')]:
        s=piv[var].sort_values(ascending=False,kind='stable');boundary=s.iloc[12]
        put('tie5_'+name+'_selected',13-(s>boundary).sum(),'data/calls/ primary top-13 boundary')
        put('tie5_'+name+'_tied',s.eq(boundary).sum(),'data/calls/ primary top-13 boundary')
    rng=np.random.default_rng(20260814);ties={.05:[],.10:[]}
    for _ in range(1000):
        order=rng.permutation(len(piv));tie_rank=np.empty(len(piv),int);tie_rank[order]=np.arange(len(piv))
        for cap in ties:
            k=int(np.ceil(len(piv)*cap))
            sets=[set(np.lexsort((tie_rank,-piv[v].to_numpy()))[:k]) for v in variants]
            ties[cap].append(len(set.intersection(*sets))/len(set.union(*sets)))
    for cap,draws in ties.items():put(f'random_tie_jaccard_{int(cap*100)}',np.mean(draws),'Independent 1,000 shared-random-tie draws; seed 20260814; original sun21 seed unavailable',diagnostic_only=True)
    # Independently regenerate new capacity-contrast intervals. The manuscript
    # does not archive its draw stream; these remain diagnostic, not exact audits.
    _,inverse=np.unique(cases.firm_id.astype(str),return_inverse=True)
    seeds=np.random.SeedSequence(20260814).spawn(1000)
    tasks=[([(i+1,seeds[i]) for i in range(j,1000,3)],np.column_stack([scores[k] for k in ['qwen_canonical_naive','logit','hist_gradient_boosting','xgboost']]),cases.sampling_weight.to_numpy(float),cases.outcome.to_numpy(float),inverse,inverse.max()+1) for j in range(3)]
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('fork')) as pool:
        draws=pd.DataFrame([r for batch in pool.map(capture_chunk,tasks) for r in batch])
    draws.sort_values(['draw','model','capacity']).to_csv(OUT/'sun21_independent_capacity_bootstrap_draws.csv',index=False)
    for (name,cap),g in draws.groupby(['model','capacity']):
        for side,q in [('low',.025),('high',.975)]:put(f'capacity_ci_{name}_{int(100*cap)}_{side}',g.difference_pp.quantile(q),'Independent 1,000 firm-cluster resamples; seed 20260814 / SeedSequence; original sun21 draw stream unavailable',diagnostic_only=True)
    (OUT/'supplement_values_sun21.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
    print('Supplement computed:',len(values),'values; no generative inference')

if __name__=='__main__':main()
