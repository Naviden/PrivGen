# PrivGen — reproduction package

Class-aware, cluster-local outlier sanitisation applied before training a synthetic
tabular data generator. This repository contains everything needed to reproduce the
paper: the method, the exact sanitised tables it produced, all per-run metric outputs,
and the scripts that turn those into the paper's tables.

> **Headline results are mixed and reported as such.** Privacy proxies improve on German
> Credit, do not improve on Cervical Cancer, and degrade on Health Insurance. See
> *Findings*.

## Quick start

```bash
python3.11 -m venv .venv_synth
.venv_synth/bin/pip install -r requirements.txt
export PRIVGEN_REPO=$PWD PRIVGEN_PY=$PWD/.venv_synth/bin/python
```
`MPLBACKEND=Agg` is required for every run, or each worker opens a GUI window.

## Where each number in the paper comes from

| Paper element | Artefact | Regenerate with |
|---|---|---|
| Table 3 (summary proportions) | `results/raw_results_{cervical,german}_v2.csv`, `results/raw_results_health.csv` | `experiments/unified_pipeline/scripts/make_tables_results.py` |
| Table A1 (sign table) | same | same |
| Table A2 (significance) | same | same |
| Table C4 (per-triple values) | same | same |
| Table 2 (diagnostics) | raw datasets + clustering grid | `scripts/make_tables_audit.py` |
| Table B3 (per-class removal audit) | `experiments/unified_pipeline/arms/` | `scripts/make_tables_audit.py` |
| §4.3.1 Cervical config | `experiments/unified_pipeline/config/cervical_*.{json,csv}` | `scripts/select_config_cervical.py` |
| §4.3.3 German config | `experiments/unified_pipeline/config/german_*.{json,csv}` | `scripts/sanitise.py german` |
| Appendix D (attribute inference) | `appendix_d/` | `appendix_d/scripts/driver_attack.py` |

**Which runs the paper reports.** Cervical Cancer and German Credit use the unified
pipeline in `experiments/unified_pipeline/` (`*_v2.csv` results). Health Insurance uses
the earlier hand-thresholded pipeline in `notebooks/` + `data/`
(`results/raw_results_health.csv`); this is stated in the paper and is a known
inconsistency, not an oversight.

## Reproducing the two pipelines

**Unified pipeline (Cervical Cancer, German Credit).** Selects a DBSCAN configuration
subject to a per-class cap, then trims the $(1-\tau)$ distance tail within each
(cluster, class) stratum:

```bash
cd experiments/unified_pipeline
MPLBACKEND=Agg $PRIVGEN_PY scripts/select_config_cervical.py    # -> config/cervical_chosen.json
MPLBACKEND=Agg $PRIVGEN_PY scripts/sanitise.py german           # -> arms/german__privgen.csv
MPLBACKEND=Agg $PRIVGEN_PY scripts/run_all.py cervical 3        # -> results/  (~12 min)
MPLBACKEND=Agg $PRIVGEN_PY scripts/run_all.py german 3          # -> results/  (~12 min)
$PRIVGEN_PY scripts/make_tables_results.py                      # -> tables/
$PRIVGEN_PY scripts/make_tables_audit.py
```

**Original pipeline (Health Insurance).** `notebooks/privgen_example_health.ipynb`
produces `data/health_{1..8}_*.csv`; `notebooks/evaluate_privegen_AWS_health.py` then
trains the nine synthesisers. Heavy — the original was run on an AWS `g6.4xlarge`.

## Layout

| Path | Contents |
|---|---|
| `utils/` | the method: encoding, DBSCAN, weighted distances, trimming |
| `experiments/unified_pipeline/` | the runs the paper reports for CC and GC: scripts, sanitised tables (`arms/`), per-run metrics (`results/`), chosen configs and full config searches (`config/`) |
| `appendix_d/` | attribute-inference attacks, saved synthetic samples, MIA harness |
| `datasets/` | raw inputs |
| `data/` | per-stage artefacts of the original pipeline (`_1` encoded … `_8` decoded) |
| `results/` | per-triple metric values behind the paper's tables |
| `notebooks/` | per-dataset pipeline notebooks and the AWS evaluation scripts |

## Findings

- **German Credit** is the only dataset with a significant privacy gain (61% of
  model–metric combinations, Wilcoxon *p* = 0.009, median +84%).
- **Cervical Cancer** privacy proxies do not improve (39%: 14 improvements, 19
  degradations). A cap-free configuration reports 64% *only by deleting all 54
  biopsy-positive records*; no configuration on the original search grid retains one.
  See `experiments/unified_pipeline/CERVICAL_CLASS_COLLAPSE.md`.
- **Health Insurance** privacy degrades (6%). Separation-based diagnostics fail to
  predict this; a quasi-identifier statistic on the raw data succeeds.
- Two metric directions were corrected against synthcity's implementations
  (`inv_kl_divergence` and `ks_test` are higher-is-better).
- Identifiability Score is computed but not reported: it measures record-level
  uniqueness, whereas the paper's claims concern group-based proxies. Values and
  rationale in `experiments/unified_pipeline/METRICS_SCOPE.md`.
- **Every cell is a single seed.** Two arms whose removal sets agree at Jaccard 0.97
  differ by 0.11 in improvement rate, so differences below ~0.1 should not be read as
  real.

## Known gaps

- Health Insurance is not re-run under the unified pipeline.
- No matched-budget comparison against automatic outlier detectors.
- Membership inference is scaffolded (`appendix_d/scripts/run_mia.py`, BNAF variant
  works) but not swept.
- No licence file yet — add one before making the repository public.
