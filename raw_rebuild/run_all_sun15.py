#!/usr/bin/env python3
"""One-command local statistical replication; no upstream raw-data access."""
from pathlib import Path
import os,sys,argparse,time,json,subprocess,platform,threading,traceback,shutil
ROOT=Path(__file__).resolve().parent
THREADS=['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','POLARS_MAX_THREADS']
for k in THREADS:os.environ[k]='1'
os.environ.update(PYTHONHASHSEED='20261106',MPLBACKEND='Agg',MPLCONFIGDIR=str(ROOT/'output/cache/matplotlib'),PYTHONDONTWRITEBYTECODE='1')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['quick','full'],default='quick');ap.add_argument('--strict',action='store_true',help='Exit 2 when any manuscript statistic disagrees or remains unverified.');args=ap.parse_args()
 config=json.loads((ROOT/'PACKAGE.json').read_text());os.environ['REPLICATION_MODE']=args.mode;os.environ['REPLICATION_BOOTSTRAPS']='49'if args.mode=='quick'else'999';os.environ['REPLICATION_WORKERS']='4'
 for n in ['output','logs','verify','output/cache']:(ROOT/n).mkdir(parents=True,exist_ok=True)
 if config.get('clear_generated_caches'):
  for rel in ['output/r3/task_results','output/r4/task_results']:
   d=ROOT/rel
   if d.exists():shutil.rmtree(d)
 start=time.monotonic();log=ROOT/'logs/clean_run.log';records=[]
 with log.open('w')as f:
  header=dict(package=config['name'],mode=args.mode,python=sys.version,platform=platform.platform(),workers=4,numeric_threads=1,network='disabled for computation',memory_limit_bytes=8_500_000_000)
  print(json.dumps(header,ensure_ascii=False),file=f,flush=True);print(json.dumps(header,ensure_ascii=False),flush=True)
  env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'code')+os.pathsep+env.get('PYTHONPATH','')
  # Every subprocess also installs the offline hook before importing analysis code.
  stages=config['stages']
  for stage in stages:
   st=time.monotonic();script=ROOT/stage
   cmd=[sys.executable,'-u',str(ROOT/'code/offline_exec.py'),str(script)]
   print('STAGE '+stage,file=f,flush=True);print('STAGE '+stage,flush=True)
   with subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)as p:
    import psutil
    peak=[0];exceeded=[False];stop=threading.Event()
    def monitor():
     while not stop.wait(.2):
      try:
       parent=psutil.Process(p.pid);processes=[parent]+parent.children(recursive=True);rss=sum(q.memory_info().rss for q in processes if q.is_running());peak[0]=max(peak[0],rss)
       if rss>8_500_000_000:
        exceeded[0]=True
        for q in reversed(processes):
         try:q.terminate()
         except psutil.Error:pass
      except psutil.Error:pass
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    for line in p.stdout:f.write(line);f.flush()
    rc=p.wait();stop.set();watcher.join();rec=dict(stage=stage,returncode=rc,seconds=time.monotonic()-st,peak_rss_bytes=peak[0],memory_limit_triggered=exceeded[0]);records.append(rec);print(json.dumps(rec),file=f,flush=True);print(json.dumps(rec),flush=True)
  elapsed=time.monotonic()-start;ok=all(r['returncode']==0 for r in records)
  status=json.loads((ROOT/'verify/verification_status.json').read_text())if(ROOT/'verify/verification_status.json').exists()else{}
  result=dict(**header,computation_passed=ok,total_seconds=elapsed,stages=records,manuscript_verification=status,offline=True)
  (ROOT/'logs/clean_run_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),file=f,flush=True);print('COMPUTATION '+('PASS'if ok else'FAILED')+f'; {elapsed:.2f} seconds',flush=True)
 # Keep all table objects inspectable, including unverified cells.
 try:
  sys.path.insert(0,str(ROOT/'code'));import export_tables;export_tables.main()
 except Exception as e:print('Table export failed:',repr(e));ok=False
 if not ok:return 1
 if args.strict and not status.get('all_statistics_match'):return 2
 return 0
if __name__=='__main__':raise SystemExit(main())
