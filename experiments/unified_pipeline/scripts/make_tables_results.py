import os
"""Regenerate every results table using the CORRECTED Cervical Cancer run."""
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')
from scipy import stats
R=os.environ.get('PRIVGEN_REPO', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
TEX=os.environ.get('PRIVGEN_TEX', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'tables','appendix'))
os.makedirs(TEX,exist_ok=True)
DS=['cervical','german','health']
DSN={'cervical':'Cervical Cancer','german':'German Credit','health':'Health Insurance'}
MODELS=['adsgan','pategan','dpgan','decaf','ctgan','tvae','rtvae','ddpm','arf']
MN=dict(adsgan='adsGAN',pategan='PateGAN',dpgan='DPGAN',decaf='Decaf',ctgan='CTGAN',
        tvae='TVAE',rtvae='RTVAE',ddpm='TabDDPM',arf='ARF')
MET={'delta-presence.score':('Delta Presence',r'$\downarrow$'),
 'distinct l-diversity.syn':(r'$l$-diversity',r'$\uparrow$'),
 'k-anonymization.syn':(r'$k$-anonymity',r'$\uparrow$'),'k-map.score':(r'$k$-map',r'$\uparrow$'),
 'close_values_probability.score':('Close values prob.',r'$\uparrow$'),
 'nearest_syn_neighbor_distance.mean':(r'Nearest syn.\ neigh.\ dist.',r'$\downarrow$'),
 'alpha_precision.authenticity_naive':(r'Alpha prec.\ authenticity',r'$\uparrow$'),
 'chi_squared_test.marginal':(r'$\chi^2$ test',r'$\uparrow$'),
 'inv_kl_divergence.marginal':('Inverse KL divergence',r'$\uparrow$'),
 'jensenshannon_dist.marginal':('Jensen--Shannon dist.',r'$\downarrow$'),
 'ks_test.marginal':('KS test',r'$\uparrow$'),
 'max_mean_discrepancy.joint':(r'Max.\ mean discrepancy',r'$\downarrow$'),
 'prdc.coverage':('PRDC coverage',r'$\uparrow$'),'prdc.density':('PRDC density',r'$\uparrow$'),
 'prdc.precision':('PRDC precision',r'$\uparrow$'),'prdc.recall':('PRDC recall',r'$\uparrow$'),
 'wasserstein_dist.joint':('Wasserstein dist.',r'$\downarrow$')}
ORD=list(MET)
DIR={m:('Min' if a==r'$\downarrow$' else 'Max') for m,(n,a) in MET.items()}
raw={}
for ds in DS:
    f=f'{R}/results/raw_results_{ds}_v2.csv' if ds in ('cervical','german') else f'{R}/results/raw_results_{ds}.csv'
    d=pd.read_csv(f); s=np.where(d.metric.map(DIR)=='Max',1.,-1.)
    d['sgn']=np.sign(s*(d.PrivGen-d.Orig))
    with np.errstate(divide='ignore',invalid='ignore'):
        d['dpct']=s*(d.PrivGen-d.Orig)/np.abs(d.Orig)*100
    raw[ds]=d
ALL=pd.concat(raw.values())
def fm(v):
    if not np.isfinite(v): return '--'
    a=abs(v)
    return f'{v:.0f}' if a>=1000 else (f'{v:.1f}' if a>=10 else (f'{v:.3f}' if a>=0.01 else f'{v:.1e}'))
def fp(v): return '--' if not np.isfinite(v) else (f'{v:+.0f}' if abs(v)>=100 else f'{v:+.1f}')

# ---------- summary proportions ----------
bd={ds:{c:(g.sgn>0).sum()/len(g) for c,g in d.groupby('metric_category')} for ds,d in raw.items()}
cg=pd.concat([raw['cervical'],raw['german']])
bm={m:{c:(g.sgn>0).sum()/len(g) for c,g in cg[cg.synthesizer==m].groupby('metric_category')} for m in MODELS}
print('=== BY DATASET (corrected) ==='); print(pd.DataFrame(bd).T.round(3).to_string())
print('\n=== BY MODEL, cervical+german (corrected) ==='); print(pd.DataFrame(bm).T.round(3).to_string())
open(f'{TEX}/summary_numbers.txt','w').write(pd.DataFrame(bd).T.round(2).to_string()+'\n\n'+pd.DataFrame(bm).T.round(2).to_string())

# ---------- significance ----------
rows=[]
for ds in DS:
    for fam,g in raw[ds].groupby('metric_category'):
        w=int((g.sgn>0).sum()); l=int((g.sgn<0).sum()); t=len(g)-w-l
        ps=stats.binomtest(w,w+l,.5).pvalue if w+l else np.nan
        dp=g.dpct.replace([np.inf,-np.inf],np.nan).dropna(); dp=dp[dp!=0]
        pw=stats.wilcoxon(dp).pvalue if len(dp)>5 else np.nan
        rk=stats.rankdata(np.abs(dp)); rbc=(rk[dp>0].sum()-rk[dp<0].sum())/rk.sum()
        n=w+l; ph=w/n; z=1.96; den=1+z*z/n
        ctr=(ph+z*z/(2*n))/den; hw=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        rows.append(dict(ds=ds,fam=fam,n=len(g),w=w,l=l,t=t,lo=ctr-hw,hi=ctr+hw,
                         med=g.dpct.replace([np.inf,-np.inf],np.nan).median(),ps=ps,pw=pw,rbc=rbc))
T=pd.DataFrame(rows)
for c,nc in [('ps','psh'),('pw','pwh')]:
    T[nc]=np.nan
    for ds in DS:
        i=T.index[T.ds==ds]; p=T.loc[i,c].values; o=np.argsort(p); m=len(p); a=np.empty(m); run=0
        for k,j in enumerate(o): run=max(run,(m-k)*p[j]); a[j]=min(1.,run)
        T.loc[i,nc]=a
print('\n=== SIGNIFICANCE (corrected) ===')
print(T[['ds','fam','n','w','l','t','med','psh','pwh','rbc']].round(4).to_string(index=False))
def pf(p): return r'$<$0.001' if p<0.001 else f'{p:.3f}'
S=[r'\begin{table}[t]\centering\footnotesize',
 r'\caption{Significance and effect size per (dataset, metric family). Imp./Deg./Tie counts '
 r'(dataset, model, metric) triples. $p_{\mathrm{sign}}$ is an exact two-sided sign test on the '
 r'non-tied triples; $p_{\mathrm{W}}$ a Wilcoxon signed-rank test on the direction-adjusted relative '
 r'changes $\Delta$; both Holm--Bonferroni corrected across the three families within a dataset. '
 r'$r_{\mathrm{rb}}$ is the matched-pairs rank-biserial correlation (positive favours \privgenv). '
 r'CI is a 95\% Wilson interval. $^{*}$: $p<0.05$ after correction.}',r'\label{tab:significance}',
 r'\begin{tabular}{llrrrrr}\toprule',
 r'Dataset & Family & Imp./Deg./Tie & Prop. [95\% CI] & Med.\ $\Delta$\% & $p_{\mathrm{sign}}$ & $p_{\mathrm{W}}$ ($r_{\mathrm{rb}}$)\\\midrule']
for ds in DS:
    for k,(_,r) in enumerate(T[T.ds==ds].iterrows()):
        S.append(f"{DSN[ds] if k==0 else ''} & {r.fam.capitalize()} & {r.w}/{r.l}/{r.t} & "
                 f"{r.w/r.n:.2f} [{r.lo:.2f},{r.hi:.2f}] & {r.med:+.1f} & "
                 f"{pf(r.psh)}{'$^{*}$' if r.psh<0.05 else ''} & {pf(r.pwh)}{'$^{*}$' if r.pwh<0.05 else ''} ({r.rbc:+.2f})\\\\")
    if ds!=DS[-1]: S.append(r'\midrule')
S+=[r'\bottomrule\end{tabular}\end{table}']
open(f'{TEX}/significance.tex','w').write('\n'.join(S))

# ---------- numeric longtable ----------
L=[r'% AUTO-GENERATED', r'\begin{small}',r'\begin{longtable}{llrrr}',
 r'\caption{Numeric metric values with and without \privgen for every (dataset, model, metric) triple. '
 r'$\Delta$ is the change in the metric''\'''s preferred direction (arrow beside the metric name); positive '
 r'$\Delta$ always means \privgen improved it.}\label{tab:results-numeric}\\',r'\toprule',
 r'Metric & Model & Without \privgenv & With \privgenv & $\Delta$ (\%) \\',r'\midrule\endfirsthead',
 r'\toprule Metric & Model & Without \privgenv & With \privgenv & $\Delta$ (\%) \\\midrule\endhead',
 r'\midrule\multicolumn{5}{r}{\small\itshape continued}\\\endfoot',r'\bottomrule\endlastfoot']
for ds in DS:
    L.append(r'\multicolumn{5}{c}{\textbf{'+DSN[ds]+r'}}\\ \midrule')
    d=raw[ds].set_index(['metric','synthesizer'])
    for m in ORD:
        nm,ar=MET[m]
        for j,mo in enumerate(MODELS):
            if (m,mo) not in d.index: continue
            r=d.loc[(m,mo)]
            L.append(f"{(nm+' '+ar) if j==0 else ''} & {MN[mo]} & {fm(r.Orig)} & {fm(r.PrivGen)} & {fp(r.dpct)} \\\\")
        L.append(r'\cmidrule(l){1-5}')
    L.append(r'\midrule')
L+=[r'\end{longtable}',r'\end{small}']
open(f'{TEX}/numeric_results.tex','w').write('\n'.join(L))
print('\nwrote significance.tex, numeric_results.tex, summary_numbers.txt')
