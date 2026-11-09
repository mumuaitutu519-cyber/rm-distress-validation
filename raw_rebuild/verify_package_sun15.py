"""Exhaustive DOCX numeric inventory and coordinate-based replication check."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP,InvalidOperation
import csv,json,re,hashlib,zipfile,xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
PAT=re.compile(r'(?<![\w.])[-−–+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?')
def text(el):
 parts=[]
 def walk(x):
  if x.tag.endswith('}del'):return
  if x.tag in ['{'+NS['w']+'}t','{'+NS['m']+'}t']:parts.append(x.text or '')
  for y in x:walk(y)
 walk(el);return ''.join(parts)
def extract():
 doc=next((ROOT/'verify/manuscript').glob('*.docx'));z=zipfile.ZipFile(doc)
 body=E.fromstring(z.read('word/document.xml')).find('w:body',NS)
 tables=[];paragraphs=[];current='Table 1';figs=[]
 for el in body:
  if el.tag.endswith('}p'):
   t=text(el);paragraphs.append(t)
   m=re.match(r'Table\s+(A?\d+)[.．:]?\s*(.*)',t)
   if m:current='Table '+m[1]
   if re.match(r'(?:Figure|Fig\.)\s*A?\d+[.:]',t):figs.append(t)
  elif el.tag.endswith('}tbl'):
   rows=[[text(c)for c in r.findall('w:tc',NS)]for r in el.findall('w:tr',NS)]
   tables.append(dict(object=len(tables)+1,table=current,caption=paragraphs[-1]if paragraphs else'',rows=rows))
 dest=ROOT/'verify/manuscript';(dest/'extracted_tables.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2))
 (dest/'source.json').write_text(json.dumps(dict(filename=doc.name,sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),table_objects=len(tables),figure_captions=figs),ensure_ascii=False,indent=2))
 media=dest/'figures';media.mkdir(exist_ok=True)
 for n in z.namelist():
  if n.startswith('word/media/'):(media/Path(n).name).write_bytes(z.read(n))
 return tables
def decimal(s):return Decimal(str(s).replace(',','').replace('−','-').replace('–','-'))
def equal(want,value):
 try:
  d=decimal(want);v=decimal(value)
  if not v.is_finite():return False
  return v.quantize(Decimal(1).scaleb(d.as_tuple().exponent),rounding=ROUND_HALF_UP)==d
 except (InvalidOperation,ValueError):return False
def run():
 tables=extract();p=ROOT/'output/computed_cells.csv'
 mapped={}
 if p.exists():
  for r in csv.DictReader(p.open()):mapped[tuple(int(r[k])for k in ['object','row','col','token'])]=r
 contexts={}
 cp=ROOT/'verify/MISMATCH_CONTEXT.csv'
 if cp.exists():
  for c in csv.DictReader(cp.open()):contexts[tuple(int(c[k])for k in ['object','row','col','token'])]=c
 rows=[]
 for t in tables:
  for i,row in enumerate(t['rows'],1):
   label=row[0]if row else''
   for j,cell in enumerate(row,1):
    for token,m in enumerate(PAT.finditer(cell),1):
     key=(t['object'],i,j,token);mapp=mapped.get(key);num=m.group();val='';diff='';scope='statistic';cause='';source=''
     if mapp:
      val=mapp['value'];source=mapp.get('source','');cause=mapp.get('reason','')
      status='MATCH'if equal(num,val)else'MISMATCH'
      if mapp.get('status'):status=mapp['status']
      try:diff=str(decimal(val)-decimal(num))
      except:pass
     elif j==1 or i==1 or label in ['Model','Threshold','Score','Fiscal year'] or re.fullmatch(r'\(\d+\)',cell.strip()) or re.fullmatch(r'\d{1,2}:\d{2}',cell.strip()) or(i<=3 and not label.strip()and re.fullmatch(r'\d+',cell.strip()))or (i<=3 and re.search(r'[a-zA-Z]',cell) and not re.search(r'\d\.\d',cell)) or t['caption'].startswith('Panel A.'):status='STRUCTURAL';scope='design_label';cause='Definition, year, window, column number or other fixed design label; not an estimated statistic.'
     else:status='UNVERIFIED';cause='No reviewed coordinate-to-computation mapping; not counted as agreement.'
     ctx=contexts.get(key,{})
     if ctx:cause=ctx.get('reason',cause)
     proposed=''
     if val:
      try:proposed=format(decimal(val).quantize(Decimal(1).scaleb(decimal(num).as_tuple().exponent),rounding=ROUND_HALF_UP),'f')
      except:pass
     rows.append(dict(manuscript_origin_path=ctx.get('manuscript_origin_path',''),proposed_display_value=proposed,table=t['table'],table_object=t['object'],row=i,column=j,token=token,row_label=label,cell_text=cell,manuscript_value=num,recomputed_value=val,difference=diff,consistent='是'if status=='MATCH'else'否'if status=='MISMATCH'else'',status=status,scope=scope,source=source,reason=cause))
 dest=ROOT/'verify';dest.mkdir(exist_ok=True)
 fields=list(rows[0])if rows else[]
 with (dest/'VERIFICATION.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 lines=['# Manuscript comparison','', 'Each numeric token is inventoried at its physical DOCX table/row/cell/token coordinate. Structural design labels are separate. Only recomputed statistics count as agreement. Rounding: decimal ROUND_HALF_UP at the displayed precision.','', '| Table | Match | Mismatch | Unverified | Statistical total | Agreement |','|---|---:|---:|---:|---:|---:|']
 for name in dict.fromkeys(r['table']for r in rows):
  z=[r for r in rows if r['table']==name and r['scope']=='statistic'];counts={s:sum(r['status']==s for r in z)for s in ['MATCH','MISMATCH','UNVERIFIED']};n=len(z);lines.append(f"| {name} | {counts['MATCH']} | {counts['MISMATCH']} | {counts['UNVERIFIED']} | {n} | {counts['MATCH']/n:.1%} |"if n else f'| {name} | 0 | 0 | 0 | 0 | — |')
 (dest/'VERIFICATION_SUMMARY.md').write_text('\n'.join(lines)+'\n')
 bad=[r for r in rows if r['status']=='MISMATCH'];unknown=[r for r in rows if r['status']=='UNVERIFIED']
 report=['# Mismatch and unresolved-cell report','', 'No manuscript or computed result is changed to obtain agreement. A proposed replacement below is conditional on the documented computation; unresolved mappings have no proposed correction.','', '| Table / physical cell | Manuscript | Recomputed replacement | Cause / source |','|---|---:|---:|---|']
 for r in bad:report.append(f"| {r['table']} / object {r['table_object']}, row {r['row']}, cell {r['column']}, token {r['token']} | {r['manuscript_value']} | {r['proposed_display_value']} | {r['reason']} Recomputed: {r['source']}. Manuscript origin: {r['manuscript_origin_path']} |")
 report+=['',f'Unverified numeric tokens: {len(unknown)}. Exact coordinates and reasons are in VERIFICATION.csv. They are not transcription errors by default.']
 (dest/'MISMATCH_REPORT.md').write_text('\n'.join(report)+'\n')
 summary=dict(numeric_tokens=len(rows),matched=sum(r['status']=='MATCH'for r in rows),mismatched=len(bad),unverified=len(unknown),structural=sum(r['status']=='STRUCTURAL'for r in rows),complete=not unknown,all_statistics_match=not bad and not unknown)
 (dest/'verification_status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);return summary
if __name__=='__main__':run()
