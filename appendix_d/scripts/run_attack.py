"""Appendix D, attribute-inference half.

Protocol (as fixed in the paper): predict a sensitive attribute from the remaining
attributes using models fitted on the SYNTHETIC data, then measure inference accuracy
on the REAL data against a majority-class baseline.

We implement the attack directly rather than via synthcity's `data_leakage_*`, which
raises on the cases that matter most here: when sanitisation leaves a sensitive
attribute constant in the synthetic data, the attack is degenerate and that fact is
itself a result, not an error to be swallowed.

Adversary: strong (sees all non-sensitive attributes). Models: logistic regression and
random forest, both fitted on synthetic only. Reported per sensitive attribute:
  acc          - accuracy on the real data
  baseline     - majority-class rate of the real data (what guessing achieves)
  advantage    - acc - baseline  (>0 means the synthetic data leaked something)
  bal_acc      - balanced accuracy (chance = 1/n_classes)
  degenerate   - True if the attribute is constant in the synthetic data
"""
import os
os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("OMP_NUM_THREADS","1")     # xgboost/OpenMP segfaults on this host otherwise
import matplotlib; matplotlib.use("Agg")
import warnings, sys, time, json; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS=os.path.join(os.path.dirname(ROOT),'appendix_b','arms')
R=f'{ROOT}/res'; S=f'{ROOT}/synth'
os.makedirs(R,exist_ok=True); os.makedirs(S,exist_ok=True)
SEED=42
TARGET={'cervical':'Biopsy','german':'class','health':'claim'}
SENS={'cervical':['Biopsy','STDs:HIV'],'german':['class','Attribute9'],'health':['diabetes','smoker']}

def encode(df, ref):
    """Ordinal-encode object columns using categories from ref (real data)."""
    out=df.copy()
    cat=[c for c in ref.columns if ref[c].dtype==object]
    if cat:
        enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
        enc.fit(ref[cat].astype(str))
        out[cat]=enc.transform(out[cat].astype(str))
    return out.astype(float).fillna(-1)

def attack(real, syn, col):
    y_syn_raw=syn[col]; y_real_raw=real[col]
    classes=sorted(set(y_real_raw.dropna().unique()))
    base=y_real_raw.value_counts(normalize=True).max()
    rec=dict(attribute=col, n_classes=len(classes), baseline=round(float(base),4),
             degenerate=bool(y_syn_raw.nunique()<2))
    if rec['degenerate']:
        # the attack cannot be fitted; a real adversary would fall back to guessing
        rec.update(model='n/a', acc=round(float(base),4), bal_acc=round(1/len(classes),4),
                   advantage=0.0)
        return [rec]
    Xs=encode(syn.drop(columns=[col]), real.drop(columns=[col]))
    Xr=encode(real.drop(columns=[col]), real.drop(columns=[col]))
    ys=pd.factorize(y_syn_raw.astype(str))[0]
    lut={v:i for i,v in enumerate(pd.factorize(y_syn_raw.astype(str))[1])}
    yr=y_real_raw.astype(str).map(lut).fillna(-1).astype(int).values
    out=[]
    for name,mdl in [('logreg',LogisticRegression(max_iter=1000)),
                     ('rf',RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=1))]:
        r=dict(rec); r['model']=name
        try:
            mdl.fit(Xs.values, ys); pred=mdl.predict(Xr.values)
            mask=yr>=0                              # real rows whose class the syn data knows
            r['acc']=round(float(accuracy_score(yr[mask],pred[mask])),4) if mask.any() else 0.0
            r['bal_acc']=round(float(balanced_accuracy_score(yr[mask],pred[mask])),4) if mask.any() else 0.0
            r['advantage']=round(r['acc']-r['baseline'],4)
        except Exception as e:
            r.update(model=name,acc=None,bal_acc=None,advantage=None,error=type(e).__name__)
        out.append(r)
    return out

def run(ds,arm,synth):
    out=f'{R}/{ds}__{arm}__{synth}.csv'
    if os.path.exists(out): return 'cached'
    base=pd.read_csv(f'{ARMS}/{ds}__base.csv')
    cache=f'{S}/{ds}__{arm}__{synth}.csv.gz'
    t0=time.time()
    if os.path.exists(cache):
        syn=pd.read_csv(cache)
    else:
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader
        src=pd.read_csv(f'{ARMS}/{ds}__{arm}.csv')[base.columns]
        p=Plugins().get(synth)
        p.fit(GenericDataLoader(src,target_column=TARGET[ds],sensitive_features=SENS[ds]))
        syn=p.generate(len(base),random_state=SEED).dataframe()
        syn.to_csv(cache,index=False,compression='gzip')
    rows=[]
    for col in SENS[ds]:
        if col in syn.columns: rows+=attack(base,syn,col)
    d=pd.DataFrame(rows); d['dataset']=ds; d['arm']=arm; d['synth']=synth
    d['secs']=round(time.time()-t0,1)
    d.to_csv(out,index=False)
    return f'{time.time()-t0:.1f}s'

if __name__=='__main__':
    ds,arm,synth=sys.argv[1:4]
    try: print(f'{ds}|{arm}|{synth} -> {run(ds,arm,synth)}',flush=True)
    except Exception as e:
        print(f'{ds}|{arm}|{synth} -> FAILED {type(e).__name__}: {str(e)[:200]}',flush=True)
        open(f'{R}/FAIL__{ds}__{arm}__{synth}.txt','w').write(f'{type(e).__name__}: {e}')
    finally:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
