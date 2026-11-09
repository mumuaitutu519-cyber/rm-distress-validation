#!/usr/bin/env python3
"""No-value credential scan of distribution files; never reads credential stores."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from zipfile import ZipFile,is_zipfile
import argparse,json,re,subprocess,os,io,hashlib

EXAMPLE_FILE=Path(__file__).resolve().parent/'scanner_reviewed_examples_340.json'
EXAMPLES=json.loads(EXAMPLE_FILE.read_text()) if EXAMPLE_FILE.exists() else {}

PATTERNS={
 'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
 'openai_token':rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b',
 'aws_access_key':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'slack_token':rb'\bxox[baprs]-[A-Za-z0-9-]{20,}\b',
 'google_key':rb'\bAIza[A-Za-z0-9_-]{30,}\b',
 'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----',
 'credential_url':rb'https?://[^\s/:<>"\x27]{1,120}:[^\s/@<>"\x27]{2,200}@',
 'rpc_provider_key':rb'https?://[^\s/"\x27]*(?:alchemy\.com|infura\.io)/v[23]/[A-Za-z0-9_-]{16,}',
 'bearer_literal':rb'(?i)\bbearer\s+[A-Za-z0-9_.-]{24,}',
 'jwt':rb'\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}',
 'secret_literal':rb'(?i)["\x27]?(?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|private[_-]?key)["\x27]?\s*[:=]\s*["\x27]([^"\x27\r\n]{8,200})["\x27]',
}
PATTERNS['internal_file_share']=rb'https?://(?:[A-Za-z0-9._-]+\.)?(?:sharepoint\.com|onedrive\.live\.com|1drv\.ms|dropbox\.com|box\.com|pan\.baidu\.com|pan\.quark\.cn|aliyundrive\.com|drive\.google\.com|docs\.google\.com)/[^\s<>\"\x27]+'
PATTERNS['private_network_url']=rb'https?://(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})(?::[0-9]+)?/'
RULES={k:re.compile(v) for k,v in PATTERNS.items()}
KEYWORDS=re.compile(rb'(?i)secret|token|password|passwd|credential|api[_-]?key|private[_-]?key')
PLACEHOLDER=re.compile(rb'(?i)^(?:none|null|false|password|placeholder|your[_ -]|example|test[_ -]|\$|<|\{|os\.|env\.|process\.|getenv|\*|x{4,}|0{8,})')
def forbidden_name(p):
    n=p.name.lower()
    return n=='.env' or n.startswith('.env.') or n in {'auth.json','credentials.json','id_rsa','id_ed25519','.netrc','.npmrc','.pypirc','hosts.yml'} or p.suffix.lower() in {'.pem','.key','.p12','.pfx'}
def inspect_stream(f):
    found=set();words=False;tail=b''
    for part in iter(lambda:f.read(1024*1024),b''):
        data=tail+part;words=words or bool(KEYWORDS.search(data))
        for name,rule in RULES.items():
            for m in rule.finditer(data):
                if name=='secret_literal' and (PLACEHOLDER.search(m[1]) or b' ' in m[1] or b'/' in m[1] or b'(' in m[1]):continue
                found.add(name)
        tail=data[-4096:]
    return found,words
def scan_one(args):
    root,path=args;p=Path(path);r={'path':str(p.relative_to(root)),'rules':[],'keyword_hit':False,'reviewed_examples':[]}
    if p.is_symlink():r['rules']=['symlink'];return r
    if forbidden_name(p):r['rules']=['credential_filename_not_opened'];return r
    if p.stat().st_size>100_000_000:r['rules'].append('over_100MB')
    with p.open('rb') as f:found,words=inspect_stream(f)
    r['rules']+=sorted(found);r['keyword_hit']=words
    if p.suffix.lower() in {'.whl','.zip','.docx'} and is_zipfile(p):
        with ZipFile(p) as z:
            for x in z.infolist():
                if x.is_dir():continue
                key=p.name+'::'+x.filename
                reviewed=EXAMPLES.get(key)
                if reviewed:
                    h=hashlib.sha256()
                    with z.open(x) as f:
                        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
                    if h.hexdigest()==reviewed['sha256']:
                        r['reviewed_examples'].append({'member':x.filename,'reason':reviewed['reason']})
                        continue
                if forbidden_name(Path(x.filename)):
                    r['rules'].append('embedded_credential_filename');continue
                with z.open(x) as f:found,_=inspect_stream(f)
                r['rules']+=['archive:'+a for a in sorted(found)]
    if p.suffix.lower()=='.pdf':
        cp=subprocess.run(['pdftotext',str(p),'-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if cp.returncode:r['rules'].append('pdf_text_extraction_failed')
        else:
            found,_=inspect_stream(io.BytesIO(cp.stdout));r['rules']+=['pdf:'+a for a in sorted(found)]
    if p.suffix.lower()=='.parquet':
        import pyarrow.parquet as pq, pyarrow as pa
        pa.set_cpu_count(1);pa.set_io_thread_count(1)
        pf=pq.ParquetFile(p)
        cols=[x.name for x in pf.schema_arrow if pa.types.is_string(x.type) or pa.types.is_large_string(x.type)]
        for batch in pf.iter_batches(batch_size=2048,columns=cols,use_threads=False):
            for col in batch.columns:
                for value in col.to_pylist():
                    if value is not None:
                        hits,words=inspect_stream(io.BytesIO(value.encode('utf-8')))
                        r['rules']+=['parquet:'+a for a in sorted(hits)];r['keyword_hit']=r['keyword_hit'] or words
    r['rules']=sorted(set(r['rules']));return r
def scan(root):
    paths=[]
    for dp,ds,fs in os.walk(root):
        ds[:]=[d for d in ds if d not in {'.git','.venv','__pycache__'}]
        paths.extend(Path(dp)/n for n in fs)
    from multiprocessing import get_context
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('fork')) as pool:results=list(pool.map(scan_one,[(root,str(p)) for p in paths],chunksize=32))
    return dict(root=str(root),files_scanned=len(paths),method='3 forked workers + coordinator (4 processes); streamed bytes + nested archives + PDF text + decoded Parquet strings; internal share/network URL rules; credential filenames excluded without opening',
      findings=[x for x in results if x['rules']],keyword_files=[x['path'] for x in results if x['keyword_hit']],
      reviewed_noncredential_examples=[{'path':x['path'],'members':x['reviewed_examples']} for x in results if x['reviewed_examples']],
      credential_values_recorded=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=scan(a.root.resolve());a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {'keyword_files','findings','reviewed_noncredential_examples'}},ensure_ascii=False))
    print('Candidate files:',len(result['findings']),'Keyword-only files:',len(result['keyword_files']))
    for r in result['findings']:print(r['path'],r['rules'])
    return bool(result['findings'])
if __name__=='__main__':raise SystemExit(main())
