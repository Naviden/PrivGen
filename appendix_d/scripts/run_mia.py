"""Appendix D, membership-inference half (DOMIAS).

A clean MIA needs real records the generator never saw. The Appendix B runs fitted every
generator on the whole sanitised table, so no such holdout exists and those synthetic
samples cannot be reused. We therefore re-generate under a member/non-member split:

  R_train (70%) / R_hold (30%)   fixed split of the RAW data, seed 42
  members      = rows of the arm's sanitised table that fall in R_train  -> generator fitted on these
  non-members  = raw rows from R_hold (never seen, and not selected by sanitisation)
  reference    = a disjoint slice of R_hold, used for the density ratio

DOMIAS scores each record by p_synth(x)/p_real(x) and we report attack AUCROC
(0.5 = no membership signal) and accuracy.
"""
import os
os.environ.setdefault("MPLBACKEND","Agg"); os.environ.setdefault("OMP_NUM_THREADS","2")
import matplotlib; matplotlib.use("Agg")
import warnings, sys, time; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_privacy import DomiasMIAKDE

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS=os.path.join(os.path.dirname(ROOT),'appendix_b','arms')
R=f'{ROOT}/mia'; os.makedirs(R,exist_ok=True)
SEED=42
TGT={'cervical':'Biopsy','german':'class','health':'claim'}

def split(base):
    rng=np.random.default_rng(SEED)
    idx=rng.permutation(len(base)); cut=int(0.7*len(base))
    return base.iloc[idx[:cut]].reset_index(drop=True), base.iloc[idx[cut:]].reset_index(drop=True)

def key(df): return df.round(6).astype(str).agg('|'.join,axis=1)

def run(ds,arm,synth):
    out=f'{R}/{ds}__{arm}__{synth}.csv'
    if os.path.exists(out): return 'cached'
    base=pd.read_csv(f'{ARMS}/{ds}__base.csv')
    arm_df=pd.read_csv(f'{ARMS}/{ds}__{arm}.csv')[base.columns]
    tr,ho=split(base)
    # members: rows kept by this arm AND in the training half
    keep=set(key(arm_df)); mem=tr[key(tr).isin(keep)].reset_index(drop=True)
    if len(mem)<50 or len(ho)<40: return 'SKIP too small'
    # DOMIAS-KDE needs a non-singular covariance; constant / duplicated columns in the
    # real data carry no membership signal, so drop them for the attack only.
    nun=base.nunique(); const=[c for c in base.columns if nun[c]<2]
    dup=[]
    num=base.drop(columns=const).select_dtypes(include='number')
    if num.shape[1]>1:
        corr=num.corr().abs()
        for i,c in enumerate(num.columns):
            if any(corr.iloc[i,j]>0.999 for j in range(i)): dup.append(c)
    drop=[c for c in const+dup]
    if drop:
        mem=mem.drop(columns=drop); ho=ho.drop(columns=drop)
    t0=time.time()
    p=Plugins().get(synth)
    tgt=TGT[ds] if TGT[ds] in mem.columns else None
    p.fit(GenericDataLoader(mem,target_column=tgt))
    syn  =p.generate(len(mem),random_state=SEED).dataframe()
    synv =p.generate(len(mem),random_state=SEED+1).dataframe()
    ref=min(len(ho)//2, len(mem), 500)
    res=DomiasMIAKDE(use_cache=False).evaluate(
        GenericDataLoader(ho  ,target_column=tgt),   # X_gt        (non-members + reference)
        GenericDataLoader(syn ,target_column=tgt),   # synth_set
        GenericDataLoader(mem ,target_column=tgt),   # X_train     (members)
        GenericDataLoader(synv,target_column=tgt),   # synth_val_set
        ref)                                             # reference_size
    d=pd.DataFrame([dict(dataset=ds,arm=arm,synth=synth,n_members=len(mem),
                         n_nonmembers=ref,reference_size=ref,n_cols_dropped=len(drop),
                         **{k:round(float(v),4) for k,v in res.items()},
                         secs=round(time.time()-t0,1))])
    d.to_csv(out,index=False)
    return f'{time.time()-t0:.1f}s AUC={res.get("aucroc",float("nan")):.3f}'

if __name__=='__main__':
    ds,arm,synth=sys.argv[1:4]
    try: print(f'{ds}|{arm}|{synth} -> {run(ds,arm,synth)}',flush=True)
    except Exception as e:
        print(f'{ds}|{arm}|{synth} -> FAILED {type(e).__name__}: {str(e)[:200]}',flush=True)
        open(f'{R}/FAIL__{ds}__{arm}__{synth}.txt','w').write(f'{type(e).__name__}: {e}')
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
