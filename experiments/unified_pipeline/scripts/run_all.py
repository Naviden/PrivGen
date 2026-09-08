import os
import subprocess,os,sys,time
from concurrent.futures import ThreadPoolExecutor
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY=os.environ.get('PRIVGEN_PY', os.path.join(os.environ.get('PRIVGEN_REPO','.'),'.venv_synth','bin','python'))
ds=sys.argv[1]; workers=int(sys.argv[2]) if len(sys.argv)>2 else 3
SY=['arf','tvae','ctgan','adsgan','rtvae','ddpm','dpgan','decaf','pategan']
jobs=[(a,s) for a in ['base','privgen'] for s in SY]
todo=[j for j in jobs if not os.path.exists(f'{H}/res/{ds}__{j[0]}__{j[1]}.csv')]
print(f'{len(todo)}/{len(jobs)} to run',flush=True)
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TQDM_DISABLE='1',
         MPLBACKEND='Agg',PYTHONWARNINGS='ignore')
t0=time.time(); n=[0]
def run(j):
    a,s=j
    try: r=subprocess.run([PY,os.path.join(os.path.dirname(os.path.abspath(__file__)),'evaluate_arm.py'),ds,a,s],capture_output=True,text=True,env=env,timeout=10800)
    except subprocess.TimeoutExpired: r=None
    n[0]+=1
    m=[l for l in (r.stdout.splitlines() if r else []) if l.startswith(f'{ds}|')]
    print(f'[{n[0]}/{len(todo)} {(time.time()-t0)/60:.1f}m] {m[-1] if m else f"{ds}|{a}|{s} -> TIMEOUT"}',flush=True)
with ThreadPoolExecutor(max_workers=workers) as ex: list(ex.map(run,todo))
print('ALL DONE',flush=True)
