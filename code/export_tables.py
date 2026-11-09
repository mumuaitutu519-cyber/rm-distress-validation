from pathlib import Path
import json,csv,os
ROOT=Path(__file__).resolve().parents[1]
def main():
 tables=json.loads((ROOT/'verify/manuscript/extracted_tables_sun21.json').read_text());mapping={}
 p=ROOT/'output/computed_cells.csv'
 if p.exists():
  for r in csv.DictReader(p.open()):mapping[tuple(int(r[k])for k in ['object','row','col','token'])]=r
 out=ROOT/'output/tables';out.mkdir(exist_ok=True)
 for t in tables:
  name=t['table'].replace(' ','_')+'_object_'+str(t['object'])+'.csv';rows=[]
  for i,row in enumerate(t['rows'],1):
   for j,cell in enumerate(row,1):
    keys=[k for k in mapping if k[:3]==(t['object'],i,j)]
    rows.append(dict(replication_mode=os.environ.get('REPLICATION_MODE','quick'),bootstrap_policy='RM: saved 1000 draws quick, regenerate full; sun21 new capacity intervals independently resampled and flagged unresolved',row=i,column=j,row_label=row[0],manuscript_text=cell,recomputed_values='; '.join(str(mapping[k]['value'])for k in sorted(keys)),source='; '.join(mapping[k]['source']for k in sorted(keys)),coverage='mapped'if keys else'label or unverified; see VERIFICATION_sun21.csv'))
  with(out/name).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
