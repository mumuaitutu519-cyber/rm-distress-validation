"""Audit every sun21 table token, table note and selected body claims."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import csv, json, re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAT=re.compile(r'(?<![\w.])[-−–+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?')

def dec(s):return Decimal(str(s).replace(',','').replace('−','-').replace('–','-'))
def display(want,value):return format(dec(value).quantize(Decimal(1).scaleb(dec(want).as_tuple().exponent),rounding=ROUND_HALF_UP),'f')
def equal(want,value):
    try:return dec(display(want,value))==dec(want)
    except (InvalidOperation,ValueError):return False

def main():
    V=ROOT/'verify';M=V/'manuscript'
    tables=json.loads((M/'extracted_tables_sun21.json').read_text())
    paragraphs=json.loads((M/'paragraphs_sun21.json').read_text())
    blocks=json.loads((M/'blocks_sun21.json').read_text())
    extra=json.loads((ROOT/'output/analysis/supplement_values_sun21.json').read_text())
    mapped={tuple(int(r[k]) for k in ['object','row','col','token']):r for r in csv.DictReader((ROOT/'output/computed_cells.csv').open())}
    contexts={tuple(int(r[k]) for k in ['object','row','col','token']):r for r in csv.DictReader((V/'legacy_sun15/MISMATCH_CONTEXT.csv').open())}
    rows=[]
    def add(table,location,cell,want,value='',source='',reason='',status=None,scope='statistic',kind='numeric',**coords):
        if status is None:status='MATCH' if value!='' and equal(want,value) else 'MISMATCH' if value!='' else 'UNVERIFIED'
        proposal=''
        if value!='' and kind=='numeric':
            try:proposal=display(want,value)
            except:pass
        rows.append(dict(table=table,location=location,kind=kind,scope=scope,status=status,manuscript_value=want,recomputed_value=value,proposed_display_value=proposal,cell_text=cell,source=source,reason=reason,table_object=coords.get('object',''),row=coords.get('row',''),column=coords.get('column',''),token=coords.get('token','')))
    for t in tables:
        for i,row in enumerate(t['rows'],1):
            label=row[0] if row else ''
            for j,cell in enumerate(row,1):
                for token,m in enumerate(PAT.finditer(cell),1):
                    key=(t['object'],i,j,token);r=mapped.get(key);ctx=contexts.get(key,{})
                    kw=dict(object=t['object'],row=i,column=j,token=token)
                    loc=f'object {t["object"]}, row {i}, cell {j}, token {token}'
                    if r:add(t['table'],loc,cell,m.group(),r['value'],r['source'],ctx.get('reason',r.get('reason','')),**kw)
                    else:
                        structural=j==1 or i==1 or label in ['Model','Threshold','Score','Fiscal year'] or re.fullmatch(r'\(\d+\)',cell.strip()) or re.fullmatch(r'\d{1,2}:\d{2}',cell.strip()) or (i<=3 and not label.strip() and re.fullmatch(r'\d+',cell.strip())) or (i<=3 and re.search(r'[a-zA-Z]',cell) and not re.search(r'\d\.\d',cell))
                        add(t['table'],loc,cell,m.group(),status='STRUCTURAL' if structural else 'UNVERIFIED',scope='design_label' if structural else 'statistic',reason='Fixed design, year, window, model name, code or label' if structural else 'No computation mapping',**kw)
    # Significance rule is exactly the sun21 note: ** denotes p<0.05.
    fe=pd.read_csv(ROOT/'output/analysis/strict_timing_information_fe_coefficients.csv')
    for row,term in [(3,'[T.adverse]'),(4,'[T.favorable]'),(5,'[T.after]'),(6,'[T.system]')]:
        p=fe[fe.term.str.contains(term,regex=False)].iloc[0].p_value
        cell=tables[4]['rows'][row-1][1];actual=''.join(re.findall(r'\*+',cell));expected='**' if p<.05 else ''
        add('Table 5',f'object 5, row {row}, cell 2, significance',cell,actual,expected,'output/analysis/strict_timing_information_fe_coefficients.csv',f'Sun21 ** rule: p<0.05; recomputed p={p}',status='MATCH' if actual==expected else 'MISMATCH',scope='significance',kind='stars',object=5,row=row,column=2)
    # Numeric tokens in every table note are exhaustively inventoried. A mapping
    # value of None marks a design constant rather than a statistic.
    note_maps={
      'Table 1':{},
      'Table 2':{1:'probability_n',2:'probability_events',3:'weighted_holdout_records',4:'weighted_holdout_rate'},
      'Table 3':{1:('256','data/derived/strict_timing_primary_cases_256.parquet')},
      'Table 4':{},
      'Table 5':{4:'roa_wrong',5:'roa_changed',6:'cash_wrong',7:'cash_changed'},
      'Table 6':{1:'info_firms'},
      'Table A1':{3:('256','data/derived/strict_timing_primary_cases_256.parquet')},
      'Table A2':{},'Table A3':{},
      'Table A4':{1:'holdout_records',2:'holdout_events'},
      'Table A5':{2:'reason11_pre',3:'pre_events',6:'reason11_post',7:'post_events'},
      'Table A6':{},'Table A7':{},
      'Table A8':{1:'info_firms',2:'info_cell_n'},
      'Table A9':{},'Table A10':{},'Table A11':{},
      'Table A12':{1:'probability_n'},
      'Table A13':{1:'reload_raw_equality_percent'},
    }
    auc=pd.read_csv(ROOT/'output/analysis/strict_timing_auc_difference_ci.csv')
    notes={};current=None
    for block in blocks:
        if block['type']=='table':current=tables[block['object']-1]['table']
        elif block['text'].startswith('Notes.') and current:
            notes[current]=(block['index'],block['text']);current=None
    for table,(pi,s) in notes.items():
        tokens=list(PAT.finditer(s));mapping=note_maps.get(table,{})
        if table=='Table 4':
            bracket=list(re.finditer(r'\[([^\]]+)\]',s));counter=0
            for name in ['logit','hist_gradient_boosting','xgboost']:
                for cap in [5,10]:
                    b=bracket[counter];counter+=1
                    nums=[(n+1,m) for n,m in enumerate(tokens) if b.start()<m.start()<b.end()]
                    for (n,m),side in zip(nums,['low','high']):mapping[n]=f'capacity_ci_{name}_{cap}_{side}'
        if table=='Table A12':
            bracket=list(re.finditer(r'\[([^\]]+)\]',s))
            for idx,b in enumerate(bracket):
                nums=[(n+1,m) for n,m in enumerate(tokens) if b.start()<m.start()<b.end()]
                for (n,m),side in zip(nums,['ci_lower','ci_upper']):mapping[n]=(auc.iloc[idx][side],'output/analysis/strict_timing_auc_difference_ci.csv')
        for n,m in enumerate(tokens,1):
            key=mapping.get(n);loc=f'paragraph {pi}, table note, token {n}'
            if key is None:add(table,loc,s,m.group(),status='STRUCTURAL',scope='design_label',reason='Year, cross-reference, fixed design, capacity, confidence level or number of bootstrap draws',token=n)
            elif isinstance(key,tuple):add(table,loc,s,m.group(),key[0],key[1],token=n)
            else:
                v=extra[key]
                add(table,loc,s,m.group(),v['value'],v['source'],reason='Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction' if v.get('diagnostic_only') else '',status='UNVERIFIED' if v.get('diagnostic_only') else None,token=n)
    # Captions containing empirical sample sizes are also counted.
    add('Table 5','caption sample size',tables[4]['caption'],'64',extra['info_firms']['value'],extra['info_firms']['source'])
    # Body values required by the upload task and additional new sun21 claims.
    body_specs=[
      (26,['32','1.26','12','0.52','38','1.52'],['intro_score_baseline','intro_probability_baseline','intro_score_reverse','intro_probability_reverse','intro_score_decimal','intro_probability_decimal']),
      (None,['46,058','5,215','20,668','346','25,390','474'],['panel_records','panel_firms','training_records','training_events','holdout_records','holdout_events']),
      (None,['4,192','1,231'],['status_records','status_firms']),
      (329,['12','17,548'],['prompt_inequivalent_pairs','prompt_field_pairs']),
      (393,['60.98','57.00','60.27'],['mistral_mean_baseline','mistral_mean_reverse','mistral_mean_decimal']),
      (402,['1','22','9','36','0.283','0.457'],['tie5_baseline_selected','tie5_baseline_tied','tie5_decimal_selected','tie5_decimal_tied','random_tie_jaccard_5','random_tie_jaccard_10'])]
    for pi,wants,keys in body_specs:
        if pi is None:pi=next(i for i,s in enumerate(paragraphs) if all(w in s for w in wants))
        s=paragraphs[pi]
        for want,key in zip(wants,keys):
            v=extra[key];assert want in s,(pi,want)
            why=''
            if v.get('diagnostic_only'):
                why='Original random tie-break seed/draws unavailable; independent seeded diagnostic only'
                if key.startswith('intro_probability'):why='Raw scores verified. Example is absent from the probability audit; archived fold 3 recovers 1.26/0.52/1.52 after rounding, but its assignment to that map is not archived. Pooled-fit alternatives shown only as diagnostics; no proposed manuscript correction.'
            if key=='prompt_field_pairs':why='There are 1,756 cases × 6 percentage/decimal ratio fields = 10,536 pairs, of which 12 differ. Across all ten fields there are 17,560 pairs and 17,548 equal pairs. Thus 17,548 is an equality count, not the denominator. A valid wording is: 12 of 10,536 percentage/decimal ratio-field pairs differ by 0.01 percentage points; or 12 of 17,560 total field pairs.'
            add('Body',f'paragraph {pi}, {key}',s,want,v['value'],v['source'],status='UNVERIFIED' if v.get('diagnostic_only') else None,reason=why)
    s=paragraphs[137]
    for want,claim in [('63','independent rerun count'),('58','raw-string matching reruns')]:
        add('Body',f'paragraph 137, {claim}',s,want,reason='Neither the 63 independently rerun raw replies nor a run-ID comparison ledger is in the 11.06 archive or supplied project. Archived 192-call reload is a different experiment. No inference rerun permitted.')
    fields=list(rows[0])
    with (V/'VERIFICATION_sun21.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    stats=[r for r in rows if r['scope']=='statistic' and r['table']!='Body']
    body=[r for r in rows if r['scope']=='statistic' and r['table']=='Body']
    summary=[]
    for name in [t['table'] for t in tables]+['Body']:
        rr=[r for r in rows if r['table']==name and r['scope']=='statistic']
        counts={status:sum(r['status']==status for r in rr) for status in ['MATCH','MISMATCH','UNVERIFIED']}
        summary.append(dict(table=name,**counts,total=len(rr),agreement=counts['MATCH']/len(rr) if rr else None))
    state=dict(numeric_statistical_tokens=len(stats),matched=sum(r['status']=='MATCH' for r in stats),mismatched=sum(r['status']=='MISMATCH' for r in stats),unverified=sum(r['status']=='UNVERIFIED' for r in stats),stars_total=4,stars_matched=sum(r['kind']=='stars' and r['status']=='MATCH' for r in rows),body_total=len(body),body_matched=sum(r['status']=='MATCH' for r in body),body_mismatched=sum(r['status']=='MISMATCH' for r in body),body_unverified=sum(r['status']=='UNVERIFIED' for r in body),structural=sum(r['status']=='STRUCTURAL' for r in rows),all_statistics_match=all(r['status']=='MATCH' for r in stats+body),per_table=summary)
    (V/'verification_status_sun21.json').write_text(json.dumps(state,indent=2)+'\n')
    lines=['# sun21 逐格核验','', '准绳：2026.11.09 sun21。统计值使用十进制 ROUND_HALF_UP 直接按稿件显示位数舍入；设计数字登记但不计分母。表注样本数及区间端点计入。星号独立核验。正文关键数字单列。未经核实的独立随机诊断不视为稿件错误。','', '| 表 | 一致 | 不一致 | 未覆盖 | 分母 | 一致率 |','|---|---:|---:|---:|---:|---:|']
    for d in summary:lines.append(f"| {d['table']} | {d['MATCH']} | {d['MISMATCH']} | {d['UNVERIFIED']} | {d['total']} | {d['agreement']:.2%} |" if d['total'] else f"| {d['table']} | 0 | 0 | 0 | 0 | 无统计量 |")
    (V/'VERIFICATION_SUMMARY_sun21.md').write_text('\n'.join(lines)+'\n')
    report=['# sun21 不一致与未覆盖报告','',f"表体和表注：{state['matched']}/{len(stats)} 一致；{state['mismatched']} 不一致，{state['unverified']} 未覆盖。星号 {state['stars_matched']}/4。正文：{state['body_matched']}/{len(body)} 一致。",'', '以下为有底层计算支持的条件性稿件替换值；本任务未修改稿件或统计结果。','', '| 表与位置 | 稿件 | 直接舍入应显示 | 原因与证据 |','|---|---|---|---|']
    for r in rows:
        if r['status']=='MISMATCH':report.append(f"| {r['table']} / {r['location']} | {r['manuscript_value']} | {r['proposed_display_value']} | {r['reason']}；{r['source']} |")
    report+=['','## 未覆盖项','', '| 位置 | 稿件值 | 独立诊断值 | 原因 |','|---|---|---|---|']
    for r in rows:
        if r['status']=='UNVERIFIED':report.append(f"| {r['table']} / {r['location']} | {r['manuscript_value']} | {r['proposed_display_value']} | {r['reason']} |")
    report+=['','## 最不利事实','', '1. 原始生成调用未记录最终 token 序列和每次调用时间；当前包不能验证 sun21 新增的 63 次独立重跑、58 次原文一致声明。已存档的 192 次 reload 不能替代该证据。','2. 表 4 新增容量差区间、正文随机 tie-break 均缺原始随机种子和抽样流。已用包内逐公司数据补做独立 1,000 次计算，但保守记为未覆盖，不冒充原计算。','3. 引言分数 32/12/38 已核实。概率可用已存档 fold 3 映射恢复，但该例子不在 1,500 条概率样本内，原包未记录采用该映射的依据；三项概率保守记未覆盖。','4. quick 模式从原始冻结 1,000 次自助抽样重新计算已存档区间；不重新执行全部 calibration bootstrap，也不重新选择基准模型超参数。','5. 私有库保留源自授权数据的逐公司分析数据及提示词；改公开前须按 DATA_REVIEW.json 移除，并另审查聚合结果披露许可。','6. sun21 五幅嵌入图与已核查 sun15 原图逐字节相同；424 个底层数值在既定容差内一致。见 FIGURE_VERIFICATION_sun21.csv。']
    (V/'MISMATCH_REPORT_sun21.md').write_text('\n'.join(report)+'\n')
    import export_tables
    export_tables.main()
    print(json.dumps({k:v for k,v in state.items() if k!='per_table'}),flush=True)

if __name__=='__main__':main()
