# Which metrics the paper reports, and which it does not

The per-run files in `results/` are the raw output of `evaluate_arm.py`, which requests
everything synthcity computes for the sanity, statistical and privacy families. The
paper reports a subset: 17 metrics, four of them privacy.

## Identifiability Score — computed, not reported

`privacy.identifiability_score.score` (and `..._OC`) appear in all 36 Cervical Cancer
and German Credit runs. They are deliberately excluded from every proportion, test and
table in the paper, for a reason of scope rather than outcome:

Identifiability Score measures **record-level uniqueness** — how distinguishable an
individual synthetic record is from the real data. The paper's privacy claims are
explicitly about **group-based proxies** (k-anonymity, k-map, l-diversity,
delta-presence), and its threat model states the mechanism accordingly: PrivGen works
by enlarging equivalence classes in quasi-identifier space. The two families answer
different questions and we did not want to average them into one number.

For completeness, the values go mildly against the method:

| dataset | identifiability improves | median base -> PrivGen | Wilcoxon |
|---|---|---|---|
| Cervical Cancer | 3/9 | 0.168 -> 0.195 | p = 0.16 |
| German Credit   | 4/9 | 0.388 -> 0.368 | p = 0.36 |

Neither is significant, and the effects are small (mostly +/-0.01-0.03; the exception is
PateGAN on German Credit, 0.063 -> 0.328). Read against the threat model this is
coherent rather than contradictory: enlarging equivalence classes does not, on its own,
make individual records less unique. It is also a reason not to over-read the
group-based improvements as evidence about re-identification risk, which is why the
paper's Limitations call for attack-based evaluation instead of more proxies.

Health Insurance has no identifiability values: its reported numbers come from the
original AWS run, whose aggregation dropped the metric. Re-running it would add them —
see `RERUN_HEALTH.md`.

Anyone recomputing from this repository will find these values; nothing in the paper
depends on their absence.
