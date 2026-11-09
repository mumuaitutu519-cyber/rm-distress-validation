from pathlib import Path
import sys,os,runpy,socket
script=Path(sys.argv[1]).resolve();sys.argv=[str(script)]
ROOT=Path(__file__).resolve().parents[1]
def blocked(*args,**kwargs):raise RuntimeError('Network access is disabled during statistical replication.')
socket.create_connection=blocked
def audit(event,args):
 if event in ['socket.connect','socket.getaddrinfo']:raise RuntimeError('Offline computation: '+event)
 # Data and result paths must stay under this relocated package. Standard
 # library, environment, system fonts and OS configuration are allowed.
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  try:
   p=Path(os.fsdecode(args[0])).resolve()
   if p.parts[1:2]==('Volumes',) and not p.is_relative_to(ROOT) and not p.is_relative_to(Path(sys.prefix)):
    raise RuntimeError('External mounted-volume file access is blocked: '+str(p))
  except (ValueError,OSError):pass
sys.addaudithook(audit);sys.path.insert(0,str(script.parent));runpy.run_path(str(script),run_name='__main__')
