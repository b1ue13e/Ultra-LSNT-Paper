# Repository Structure

This document maps the public Ultra-LSNT repository so new readers can quickly find the right code, data, and outputs.

## Top level

```text
Ultra-LSNT/
|- README.md
|- QUICKSTART.md
|- STRUCTURE.md
|- EXPERIMENTS.md
|- DATA.md
|- SUMMARY.md
|- requirements.txt
|- LICENSE
|- src/
|- scripts/
|- data/
|- results/
|- docs/
|- split_manifest_80_20.json
|- split_manifest_80_20_unified.json
`- root-level compatibility CSV files
```

## Key directories

### `src/`

The main codebase.

- `src/models/`
  Core Ultra-LSNT model definitions and time-series forecasting utilities.
- `src/baselines/`
  Baseline implementations used in the paper and related comparisons.
- `src/experiments/`
  Experiment drivers for training, robustness, ablation, efficiency, dispatch analysis, and figure generation.
- Root-level modules inside `src/`
  Shared utilities such as preprocessing, split handling, dispatch mapping, and Wind (CN) audit helpers.

### `data/`

Canonical data location for the public release.

- `data/raw/`
  Raw released Wind (CN) time-series file used for the main pipeline.
- `data/processed/`
  Processed tables used directly by the forecasting and diagnostic scripts.

### `results/`

Repository outputs that support the paper.

- `results/figures/`
  PDF figures used in the manuscript and companion analyses.
- `results/tables/`
  CSV tables summarizing reported results and supporting runs.

### `scripts/`

Batch shell scripts for running larger experiment bundles or plot-generation workflows.

### `docs/`

Documentation that is useful but not important enough to keep in the repository root.

- `docs/archive/`
  Historical reports and inventory snapshots.
- `docs/maintainers/`
  Notes for keeping the public GitHub release tidy.

## Why there are CSV files in the repository root

The canonical data files are under `data/`, but several older scripts in the repository were originally written to load files such as `wind_final.csv` from the root directory. To avoid breaking those scripts, a small set of root-level compatibility copies is intentionally retained:

- `wind_main.csv`
- `wind_final.csv`
- `processed_wind.csv`
- `wind_us.csv`
- `air_quality_ready.csv`
- `gefcom_ready.csv`

These are compatibility artifacts, not a second data organization scheme.

## Main entry points

If you are new to the repository, start here:

- `src/experiments/train_ultra_lsnt_stable.py`
- `src/experiments/run_universal_robustness.py`
- `src/experiments/run_efficiency_benchmark.py`
- `src/experiments/run_comprehensive_ablation.py`
- `src/experiments/run_multi_domain_baselines.py`

## Suggested reading order

1. `README.md`
2. `QUICKSTART.md`
3. `DATA.md`
4. `EXPERIMENTS.md`
5. `docs/README.md`
