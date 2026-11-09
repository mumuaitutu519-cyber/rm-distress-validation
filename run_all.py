#!/usr/bin/env python3
"""Offline statistical reproduction and sun21 audit; never runs LLM inference."""
from pathlib import Path
import argparse, contextlib, json, os, platform, resource, runpy, socket, sys, tempfile, threading, time, traceback

ROOT=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','POLARS_MAX_THREADS','ARROW_NUM_THREADS']:
    os.environ[key]='1'
os.environ.update(PYTHONHASHSEED='20261106',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',REPLICATION_WORKERS='3')
TEMP=Path(os.environ.get('TMPDIR',tempfile.gettempdir()))/'rm-sun21'
TEMP.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(TEMP/'matplotlib')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'code'))

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for f in self.streams:f.write(s);f.flush()
    def flush(self):
        for f in self.streams:f.flush()

def blocked(*args,**kwargs):raise RuntimeError('Network disabled during statistical reproduction')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['quick','full'],default='quick')
    ap.add_argument('--strict',action='store_true')
    args=ap.parse_args();os.environ['REPLICATION_MODE']=args.mode
    # This coordinator and at most three forked computational workers.
    limit=2_400_000_000
    os_limit_supported=True
    try:resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    except (ValueError,OSError):os_limit_supported=False
    import psutil, pyarrow
    pyarrow.set_cpu_count(1);pyarrow.set_io_thread_count(1)
    socket.create_connection=blocked
    def audit(event,values):
        if event in ['socket.connect','socket.getaddrinfo']:blocked()
        if event=='open' and values and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.parts[1:2]==('Volumes',) and not any(p.is_relative_to(x) for x in [ROOT,Path(sys.prefix).resolve(),TEMP.resolve()]):
                raise RuntimeError('External mounted-volume access blocked: '+str(p))
    sys.addaudithook(audit)
    peak=[0];proc_peak=[1];exceeded=[False];stop=threading.Event()
    def monitor():
        while not stop.wait(.05):
            try:
                processes=[psutil.Process()]+psutil.Process().children(recursive=True)
                rss=sum(p.memory_info().rss for p in processes if p.is_running())
                peak[0]=max(peak[0],rss);proc_peak[0]=max(proc_peak[0],len(processes))
                if rss>8_500_000_000 or len(processes)>4:
                    exceeded[0]=True
                    for p in reversed(processes):
                        try:p.kill()
                        except psutil.Error:pass
            except psutil.Error:pass
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    for p in ['logs','verify','output']:(ROOT/p).mkdir(exist_ok=True)
    started=time.monotonic();records=[]
    stages=['code/rm_compute.py','code/rm_tables.py','code/verify_figures.py','code/supplement_sun21.py','code/verify_sun21.py']
    with (ROOT/'logs/clean_run_sun21.log').open('w') as f, contextlib.redirect_stdout(Tee(sys.stdout,f)),contextlib.redirect_stderr(Tee(sys.stderr,f)):
        for stage in stages:
            st=time.monotonic();print('STAGE',stage,flush=True)
            try:runpy.run_path(str(ROOT/stage),run_name='__main__');rc=0
            except Exception:traceback.print_exc();rc=1
            records.append(dict(stage=stage,returncode=rc,seconds=time.monotonic()-st))
            if rc:break
        stop.set();watcher.join()
        status=json.loads((ROOT/'verify/verification_status_sun21.json').read_text()) if (ROOT/'verify/verification_status_sun21.json').exists() else {}
        receipt=dict(manuscript='2026.11.09 sun21',mode=args.mode,python=sys.version,platform=platform.platform(),stages=records,computation_passed=len(records)==len(stages) and not any(r['returncode'] for r in records),total_seconds=time.monotonic()-started,peak_aggregate_rss_bytes=peak[0],max_total_processes=proc_peak[0],numeric_threads=1,os_address_space_limit_supported=os_limit_supported,per_process_address_space_hard_limit_bytes=limit if os_limit_supported else None,aggregate_rss_kill_guard_bytes=8_500_000_000,limit_triggered=exceeded[0],inference_rerun=False,offline=True,verification=status)
        (ROOT/'logs/clean_run_receipt_sun21.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    if not receipt['computation_passed']:return 1
    return 2 if args.strict and not status.get('all_statistics_match') else 0

if __name__=='__main__':raise SystemExit(main())
