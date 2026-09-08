import os
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, os, json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
R=os.environ.get('PRIVGEN_REPO', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ARM=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'arms')
TEX=os.environ.get('PRIVGEN_TEX', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'tables'))
os.makedirs(f'{TEX}/appendix',exist_ok=True)
DSN={'cervical':'Cervical Cancer','german':'German Credit','health':'Health Insurance'}
TAU=0.10
def load(ds):
    if ds=='cervical': raw=pd.read_csv(f'{R}/datasets/cervical-cancer_csv.csv')
    elif ds=='health': raw=pd.read_csv(f'{R}/datasets/healthinsurance.csv')
    else: return pd.read_csv(f'{R}/data/german_1_encoded_data.csv').astype(float)
    c=raw.columns; raw=pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(raw),columns=c)
    if ds=='health':
        for x in ['age','weight','no_of_dependents','bloodpressure']: raw[x]=raw[x].astype(int)
        raw['bmi']=raw['bmi'].astype(float)
    cat=raw.select_dtypes(include=['object','category']).columns
    if len(cat): raw[cat]=OrdinalEncoder().fit_transform(raw[cat])
    return raw.astype(float)
def geomed(A,eps=1e-5):
    m=A.mean(0)
    for _ in range(400):
        d=np.linalg.norm(A-m,axis=1); nz=d>0
        if not nz.any(): return m
        w=1/d[nz]; w/=w.sum(); n=(w[:,None]*A[nz]).sum(0)
        if np.linalg.norm(n-m)<eps: return n
        m=n
    return m
# ---- diagnostics on a COMMON extended grid (eps as a fraction of median pairwise distance) ----
from scipy.spatial.distance import pdist
rng=np.random.default_rng(0); diag=[]
for ds in ['cervical','german','health']:
    E=load(ds); X=StandardScaler().fit_transform(E.values)
    idx=rng.choice(len(X),min(1200,len(X)),replace=False)
    med=np.median(pdist(X[idx]))
    best_sil=-1; min_noise=1.0
    for fr in [0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.3,1.6,2.0]:
        for mp in [3,5,10,20,30,45]:
            lab=DBSCAN(eps=fr*med,min_samples=mp).fit_predict(X)
            k=len(set(lab))-(1 if -1 in lab else 0)
            if k<2: continue
            try: s=silhouette_score(X,lab)
            except Exception: continue
            best_sil=max(best_sil,s); min_noise=min(min_noise,(lab==-1).mean())
    qi=[c for c in E.columns if E[c].nunique()<=20]
    g=E.groupby(qi,dropna=False).size()
    diag.append(dict(ds=ds,sil=best_sil,noise=min_noise,qi=len(qi),
                     med_eq=float(g.median()),uniq=float(g[g==1].sum()/len(E))))
Dg=pd.DataFrame(diag); print(Dg.round(3).to_string(index=False))
G=[r'\begin{table}[t]\centering\footnotesize',
 r'\caption{Pre-generation diagnostics, computed from the raw data and the clustering grid alone '
 r'(no synthesiser is trained). To make them comparable, columns 2--3 are evaluated on a common grid '
 r'in which \texttt{eps} is expressed as a fraction of each dataset''\'''s median pairwise distance. '
 r'Columns 4--6 give the quasi-identifier (QI) diagnostic that does separate the failure case; the QI '
 r'set is the attributes with at most 20 distinct values.}',r'\label{tab:diagnostics}',
 r'\begin{tabular}{lrr rrr}\toprule',
 r'& \multicolumn{2}{c}{Separation indicators} & \multicolumn{3}{c}{QI diagnostic}\\\cmidrule(lr){2-3}\cmidrule(lr){4-6}',
 r'Dataset & Best sil. & Min.\ noise & \#QI & Med.\ eq.\ class & Unique \\\midrule']
for _,r in Dg.iterrows():
    G.append(f"{DSN[r.ds]} & {r.sil:.3f} & {r.noise:.3f} & {int(r.qi)} & {r.med_eq:.0f} & {r.uniq*100:.1f}\\% \\\\")
G+=[r'\bottomrule\end{tabular}\end{table}']
open(f'{TEX}/appendix/diagnostics.tex','w').write('\n'.join(G))

# ---- per-class removal audit (cervical from the corrected run) ----
TGT={'cervical':('Biopsy',{0.0:'Biopsy$=0$',1.0:'Biopsy$=1$'}),
     'german':('class',{1:'Good risk',2:'Bad risk',1.0:'Good risk',2.0:'Bad risk'}),
     'health':('diabetes',{0.0:'Diabetes$=0$',1.0:'Diabetes$=1$'})}
L=[r'\begin{table}[htbp]\centering\footnotesize',
 r'\caption{Per-class removal audit. For each dataset and target label: records in the raw data, the '
 r'number removed, and the removal rate. Cap OK asks whether the class removal rate exceeds the '
 r'overall removal rate by more than the elicited tail mass $\tau=0.10$.}',r'\label{tab:removal-audit}',
 r'\begin{tabular}{llrrrc}\toprule',
 r'Dataset & Class & $n$ & Removed & Rate & Cap OK? \\\midrule']
for ds,(t,names) in TGT.items():
    if ds=='cervical':
        base=pd.read_csv(f'{ARM}/cervical__base.csv')
        fin =pd.read_csv(f'{ARM}/cervical__privgen.csv')
    elif ds=='german':
        base=pd.read_csv(f'{ARM}/german__base.csv')
        fin =pd.read_csv(f'{ARM}/german__privgen.csv')
    else:
        base=load(ds); fin=pd.read_csv(f'{R}/data/{ds}_5_final_cleaned.csv')
    ol=1-len(fin)/len(base); vc=base[t].value_counts().sort_index()
    for k,(lab,n0) in enumerate(vc.items()):
        nk=int((fin[t]==lab).sum()); rate=(n0-nk)/n0
        ok=r'\checkmark' if rate-ol<=TAU+1e-9 else r'\textbf{no}'
        L.append(f"{DSN[ds] if k==0 else ''} & {names[lab]} & {int(n0)} & {int(n0)-nk} & {rate*100:.1f}\\% & {ok} \\\\")
    L.append(f"\\multicolumn{{6}}{{l}}{{\\quad\\small overall removal {ol*100:.1f}\\%}}\\\\")
    L.append(r'\midrule')
L[-1]=r'\bottomrule'; L+=[r'\end{tabular}\end{table}']
open(f'{TEX}/appendix/removal_audit.tex','w').write('\n'.join(L))
print('\nwrote diagnostics.tex, removal_audit.tex')
