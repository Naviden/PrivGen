import os
"""Appendix B downstream half: train each synthesiser on one sanitisation arm and
evaluate against the ORIGINAL data, exactly as in evaluate_privegen_AWS_*.py."""
import os
os.environ.setdefault("MPLBACKEND","Agg")
import matplotlib
matplotlib.use("Agg")
import warnings, os, sys, json, time; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics import Metrics

A=os.path.join(os.path.dirname(os.path.abspath(__file__)),'arms')
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'res')
os.makedirs(R,exist_ok=True)
SEED=42
TARGET={'cervical':'Biopsy','german':'class','health':'claim'}
SENS  ={'cervical':[],'german':['Attribute10'],'health':['diabetes']}
METRICS={'sanity':['data_mismatch','common_rows_proportion','nearest_syn_neighbor_distance','close_values_probability'],
 'stats':['jensenshannon_dist','chi_squared_test','feature_corr','inv_kl_divergence','ks_test',
          'max_mean_discrepancy','wasserstein_dist','prdc','alpha_precision'],
 'privacy':['delta-presence','k-anonymization','k-map','distinct l-diversity','identifiability_score']}

def run(ds, arm, synth):
    out=f'{R}/{ds}__{arm}__{synth}.csv'
    if os.path.exists(out): return 'cached'
    base=pd.read_csv(f'{A}/{ds}__base.csv')
    src =pd.read_csv(f'{A}/{ds}__{arm}.csv')[base.columns]
    t0=time.time()
    plugin=Plugins().get(synth)
    loader=GenericDataLoader(src, target_column=TARGET[ds], sensitive_columns=SENS[ds])
    plugin.fit(loader)
    syn=plugin.generate(len(base), random_state=SEED).dataframe()
    score=Metrics.evaluate(X_gt=base, X_syn=syn, metrics=METRICS,
                           task_type='regression', random_state=SEED)
    score=score.reset_index(); score['dataset']=ds; score['arm']=arm; score['synth']=synth
    score['secs']=round(time.time()-t0,1)
    score.to_csv(out,index=False)
    return f'{time.time()-t0:.1f}s'

if __name__=='__main__':
    ds,arm,synth=sys.argv[1],sys.argv[2],sys.argv[3]
    try:
        print(f'{ds}|{arm}|{synth} -> {run(ds,arm,synth)}',flush=True)
    except Exception as e:
        print(f'{ds}|{arm}|{synth} -> FAILED {type(e).__name__}: {str(e)[:300]}',flush=True)
        open(f'{R}/FAIL__{ds}__{arm}__{synth}.txt','w').write(f'{type(e).__name__}: {e}')
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        # geomloss/KeOps installs an atexit handler that can hang for tens of minutes
        # after the work is done; skip interpreter shutdown entirely.
        os._exit(0)
