# Appendix D: attack-based evaluation

## Attribute inference (complete, 48 runs)
A classifier is fitted on the SYNTHETIC data to predict a sensitive attribute from the
remaining columns, then scored on the REAL data against a majority-class baseline.

    acc / baseline / advantage(= acc - baseline) / bal_acc / degenerate

`degenerate` means the attribute is constant in the synthetic data, so no attack can be
fitted. This happens for every Cervical Cancer PrivGen run, because sanitisation removes
all 54 biopsy-positive records.

Implemented directly (`scripts/run_attack.py`) rather than via synthcity's
`data_leakage_*`, which raises on exactly those degenerate cases and whose exception is
then swallowed by `Metrics.evaluate`, yielding a silently empty result.

## Membership inference (scaffolded, not swept)
`scripts/run_mia.py` runs DOMIAS under a member/non-member split (70/30 of the raw data,
seed 42; members are the arm's rows falling in the training half). The KDE and prior
variants fail on Cervical Cancer with a singular covariance; the BNAF variant runs.

## Layout
    arms/    base + privgen tables per dataset (inputs; self-contained)
    res/     one CSV per (dataset, arm, synthesiser)
    synth/   the generated samples, gzipped
    mia/     membership-inference output
    out_all.csv   all attribute-inference rows concatenated

## Reproduce
    python3.11 -m venv .venv_synth
    .venv_synth/bin/pip install synthcity 'opacus==1.4.0'
    MPLBACKEND=Agg .venv_synth/bin/python appendix_d/scripts/driver_attack.py cervical,german,health 3

Pause with `touch appendix_d/PAUSE`; delete it and rerun to resume (finished jobs are
cached). `MPLBACKEND=Agg` is required or every worker opens a GUI window. PATE-GAN is
excluded: it fails to terminate on some sanitised tables.
