# Cervical Cancer re-run: class-aware cap + normative quantile rule

## Pipeline changes
- Config selected under a per-class cap on the FINAL table, ranked by ExpertSelect top-1
  (no human adjudication): DBSCAN eps=9, min_samples=10 -> 2 clusters, noise 3.1%.
- Quantile rule ACTUALLY APPLIED (tau=0.10), i.e. the rule the paper specifies, instead
  of hand-set raw distance thresholds.
- Quantile applied within each (cluster, class) stratum, so the trimming stage removes
  exactly tau of every class by construction.

## Effect on the sanitised table
| | paper run | corrected run |
|---|---|---|
| records kept | 496 / 835 (59.4%) | 726 / 835 (86.9%) |
| biopsy-positive kept | **0 / 54** | **42 / 54** |
| silhouette of the config | 0.060 | 0.558 |
| per-class cap | violated (class lost 100%) | satisfied (22.2% vs 13.1% overall) |

## Effect on the results
| family | paper run | corrected run |
|---|---|---|
| privacy | 0.639 | **0.389**  (median -16.7%, Wilcoxon p=0.41, n.s.) |
| sanity  | 0.389 | 0.722 |
| stats   | 0.374 | 0.455 |

## Reading
The 64% privacy improvement reported for Cervical Cancer does not survive a pipeline that
keeps the minority class. Once the class is preserved, privacy proxies no longer improve
(0.39, below break-even and not significant) while utility-oriented families improve.
The earlier figure was produced by a table containing only Biopsy=0 patients.

Two changes are confounded and we do not separate them here: the corrected run both
preserves the class AND removes far less data (13.1% vs 40.6%). Either could drive the
drop; an intermediate arm would be needed to attribute it.

Consequence: Cervical Cancer is not a clean success. German Credit is the only dataset
where privacy improves with statistics indistinguishable from no change.
