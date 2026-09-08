import os
"""Same search, but the quantile rule is applied WITHIN each (cluster, class) stratum.
The trimming stage then removes exactly tau of every class by construction, so any
residual per-class excess is attributable to the DBSCAN stage alone."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU=0.10
raw=pd.read_csv('datasets/cervical-cancer_csv.csv')
D=pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(raw),columns=raw.columns).astype(float)
y=D['Biopsy'].values; X=StandardScaler().fit_transform(D.values)
npos=int((y==1).sum()); nneg=int((y==0).sum())
def geomed(A,eps=1e-5):
    m=A.mean(0)
    for _ in range(500):
        d=np.linalg.norm(A-m,axis=1); nz=d>0
        if not nz.any(): return m
        w=1/d[nz]; w/=w.sum(); n=(w[:,None]*A[nz]).sum(0)
        if np.linalg.norm(n-m)<eps: return n
        m=n
    return m
def stratified(lab):
    keep=np.zeros(len(X),bool); w=np.ones(X.shape[1])/X.shape[1]
    for c in set(lab)-{-1}:
        idx=np.where(lab==c)[0]; mu=geomed(D.values[idx])
        for cl in np.unique(y[idx]):
            s=idx[y[idx]==cl]
            d=np.sqrt((w*(D.values[s]-mu)**2).sum(1))
            keep[s[d<=np.quantile(d,1-TAU)]]=True   # exactly tau of THIS class
    return keep
rows=[]
for eps in [2,3,4,5,6,7,8,9,10,11,12]:
    for mp in [3,5,8,10,15,20,25,30]:
        lab=DBSCAN(eps=eps,min_samples=mp).fit_predict(X)
        k=len(set(lab))-(1 if -1 in lab else 0)
        if k<2: continue
        db_keep=lab!=-1
        keep=stratified(lab)
        if keep.sum()<200: continue
        ol=1-keep.sum()/len(y); pl=1-(y[keep]==1).sum()/npos; nl=1-(y[keep]==0).sum()/nneg
        db_pl=1-(y[db_keep]==1).sum()/npos
        try: sil=silhouette_score(X,lab)
        except Exception: sil=-1
        noise=(lab==-1).mean()
        rows.append(dict(eps=eps,mp=mp,k=k,noise=round(noise,4),sil=round(sil,4),kept=int(keep.sum()),
            overall_loss=round(ol,4),pos_kept=int((y[keep]==1).sum()),pos_loss=round(pl,4),
            neg_loss=round(nl,4),dbscan_pos_loss=round(db_pl,4),excess=round(pl-ol,4),
            cap_ok=bool(pl-ol<=TAU+1e-9),score=round((1-max(sil,0))*abs(noise-TAU),5)))
R=pd.DataFrame(rows); R.to_csv(os.path.join(OUT,'config','cervical_config_search.csv'),index=False)
ok=R[R.cap_ok].sort_values('score')
print(f'evaluated: {len(R)}   satisfying cap: {len(ok)}')
print(ok.head(6).to_string(index=False) if len(ok) else R.sort_values('excess').head(6).to_string(index=False))
if len(ok):
    b=ok.iloc[0]
    json.dump(dict(eps=float(b.eps),min_samples=int(b.mp),tau=TAU,stratified=True),
              open(os.path.join(OUT,'config','cervical_chosen.json'),'w'))
    print(f'\nCHOSEN eps={b.eps} mp={b.mp}: keeps {b.kept}/835, {b.pos_kept}/{npos} positives'
          f' | class loss {b.pos_loss:.3f} vs overall {b.overall_loss:.3f} (cap satisfied)')
