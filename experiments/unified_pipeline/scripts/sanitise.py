import os
"""Apply the UNIFIED pipeline (class-aware cap + stratified (1-tau) quantile rule)
to a dataset, selecting the DBSCAN configuration by ExpertSelect top-1 among the
configurations that satisfy the cap on the final table."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys, json, os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TAU=0.10
TGT={'german':'class','health':'diabetes','cervical':'Biopsy'}
ds=sys.argv[1]; t=TGT[ds]
base=pd.read_csv(f'{H}/arms/{ds}__base.csv')
E=base.copy()
cat=E.select_dtypes(include=['object','category']).columns
if len(cat): E[cat]=OrdinalEncoder().fit_transform(E[cat].astype(str))
E=E.astype(float); y=E[t].values; X=StandardScaler().fit_transform(E.values)
counts={c:int((y==c).sum()) for c in np.unique(y)}
rng=np.random.default_rng(0)
med=np.median(pdist(X[rng.choice(len(X),min(1200,len(X)),replace=False)]))
def geomed(A,eps=1e-5):
    m=A.mean(0)
    for _ in range(400):
        d=np.linalg.norm(A-m,axis=1); nz=d>0
        if not nz.any(): return m
        w=1/d[nz]; w/=w.sum(); n=(w[:,None]*A[nz]).sum(0)
        if np.linalg.norm(n-m)<eps: return n
        m=n
    return m
def stratified(lab):
    keep=np.zeros(len(X),bool); w=np.ones(X.shape[1])/X.shape[1]
    for c in set(lab)-{-1}:
        idx=np.where(lab==c)[0]; mu=geomed(E.values[idx])
        for cl in np.unique(y[idx]):
            sset=idx[y[idx]==cl]
            d=np.sqrt((w*(E.values[sset]-mu)**2).sum(1))
            keep[sset[d<=np.quantile(d,1-TAU)]]=True
    return keep
rows=[]
for fr in [0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,1.0,1.2]:
    for mp in [3,5,8,10,15,20,30,45]:
        lab=DBSCAN(eps=fr*med,min_samples=mp).fit_predict(X)
        k=len(set(lab))-(1 if -1 in lab else 0)
        if k<2: continue
        keep=stratified(lab)
        if keep.sum()<0.4*len(X): continue
        ol=1-keep.sum()/len(X)
        worst=max((1-(y[keep]==c).sum()/n)-ol for c,n in counts.items())
        try: sil=silhouette_score(X,lab)
        except Exception: sil=-1
        noise=(lab==-1).mean()
        rows.append(dict(frac=fr,eps=fr*med,mp=mp,k=k,noise=noise,sil=sil,kept=int(keep.sum()),
                         overall_loss=ol,worst_excess=worst,cap_ok=bool(worst<=TAU+1e-9),
                         score=(1-max(sil,0))*abs(noise-TAU)))
R=pd.DataFrame(rows); R.to_csv(f'{H}/{ds}_config_search.csv',index=False)
ok=R[R.cap_ok].sort_values('score')
print(f'{ds}: evaluated {len(R)}, cap-satisfying {len(ok)}')
if not len(ok): print(R.sort_values("worst_excess").head(4).to_string(index=False)); sys.exit(1)
b=ok.iloc[0]
lab=DBSCAN(eps=b.eps,min_samples=int(b.mp)).fit_predict(X); keep=stratified(lab)
base.loc[keep].reset_index(drop=True).to_csv(f'{H}/arms/{ds}__privgen.csv',index=False)
json.dump(dict(eps=float(b.eps),eps_frac=float(b.frac),min_samples=int(b.mp),tau=TAU,
               k=int(b.k),noise=float(b.noise),sil=float(b.sil),kept=int(keep.sum()),n=len(X),
               overall_loss=float(b.overall_loss),worst_excess=float(b.worst_excess)),
          open(f'{H}/{ds}_chosen.json','w'),indent=1)
print(f'  chosen eps={b.eps:.2f} (={b.frac} x median) minPts={int(b.mp)} k={int(b.k)} sil={b.sil:.3f}')
print(f'  keeps {keep.sum()}/{len(X)} (removal {b.overall_loss:.3f}); worst class excess {b.worst_excess:.3f}')
for c,n in counts.items(): print(f'   class {c}: kept {(y[keep]==c).sum()}/{n}')
