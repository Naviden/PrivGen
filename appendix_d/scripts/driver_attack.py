"""Pausable, resumable driver for the Appendix D attribute-inference sweep.

PAUSE : touch appendix_d/PAUSE   -> finishes in-flight jobs, dispatches no more, exits.
RESUME: rm appendix_d/PAUSE      -> rerun the same command; finished jobs are cached.
"""
import subprocess, sys, os, time
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
REPO=os.path.dirname(ROOT)
PY=os.environ.get('PRIVGEN_PY', os.path.join(REPO,'.venv_synth','bin','python'))
PAUSE=f'{ROOT}/PAUSE'
ARMS=['base','privgen']
SYNTH=['arf','tvae','ctgan','adsgan','rtvae','ddpm','dpgan','decaf']   # pategan excluded: non-terminating
datasets=sys.argv[1].split(',') if len(sys.argv)>1 else ['cervical','german','health']
workers=int(sys.argv[2]) if len(sys.argv)>2 else 3

jobs=[(d,a,s) for d in datasets for a in ARMS for s in SYNTH]
todo=[j for j in jobs if not os.path.exists(f'{ROOT}/res/{j[0]}__{j[1]}__{j[2]}.csv')]
print(f'{len(todo)}/{len(jobs)} jobs to run, {workers} workers',flush=True)
print(f'pause with:  touch {PAUSE}',flush=True)
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
         TQDM_DISABLE='1',PYTHONWARNINGS='ignore',MPLBACKEND='Agg')
t0=time.time(); done=[0]; skipped=[0]
def run(j):
    d,a,s=j
    if os.path.exists(PAUSE):
        skipped[0]+=1; return
    try:
        r=subprocess.run([PY,f'{HERE}/run_attack.py',d,a,s],capture_output=True,text=True,env=env,timeout=10800)
        line=[l for l in r.stdout.splitlines() if l.startswith(f'{d}|')]
        msg=line[-1] if line else f'{d}|{a}|{s} -> no output rc={r.returncode}'
    except subprocess.TimeoutExpired:
        open(f'{ROOT}/res/FAIL__{d}__{a}__{s}.txt','w').write('TIMEOUT'); msg=f'{d}|{a}|{s} -> TIMEOUT'
    done[0]+=1; el=time.time()-t0
    print(f'[{done[0]}/{len(todo)} {el/60:.1f}m eta {el/max(done[0],1)*(len(todo)-done[0])/60:.0f}m] {msg}',flush=True)
with ThreadPoolExecutor(max_workers=workers) as ex:
    list(ex.map(run,todo))
if skipped[0]:
    print(f'PAUSED: {done[0]} done, {skipped[0]} not started. Remove {PAUSE} and rerun to resume.',flush=True)
else:
    print(f'ALL DONE in {(time.time()-t0)/60:.1f} min',flush=True)
