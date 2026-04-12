# Data Notes

This repository exposes the data artifacts referenced by the manuscript and keeps the layout explicit for reproducibility.

## Canonical file locations

### Raw Wind (CN)

- `data/raw/wind_main.csv`

This file is the released raw SCADA-like time series used as the upstream source for the main Wind (CN) study.

### Processed datasets

- `data/processed/wind_final.csv`
- `data/processed/processed_wind.csv`
- `data/processed/wind_us.csv`
- `data/processed/air_quality_ready.csv`
- `data/processed/gefcom_ready.csv`
- `data/processed/wind_expert_activation.csv`
- `data/processed/air_quality_expert_activation.csv`

## Split manifests

The public chronological split definitions used in the paper are stored at the repository root:

- `split_manifest_80_20.json`
- `split_manifest_80_20_unified.json`

These manifests record the train/test index ranges used by the released experiments.

## Why some CSV files also appear at the repository root

Several legacy experiment scripts in `src/` expect paths such as `wind_final.csv` or `wind_us.csv` relative to the repository root. To preserve backward compatibility, matching root-level copies are retained for:

- `wind_main.csv`
- `wind_final.csv`
- `processed_wind.csv`
- `wind_us.csv`
- `air_quality_ready.csv`
- `gefcom_ready.csv`

The canonical source of truth is still `data/`.

## Preprocessing

The main preprocessing entry point is:

- `src/data_preprocess.py`

The repository therefore contains both:

- the raw Wind (CN) input file needed to reproduce preprocessing, and
- the processed tables used directly by the forecasting scripts.

## Split policy

Unless a script explicitly states otherwise, the released experiments follow a chronological 80/20 split:

- first 80% of samples for training
- last 20% of samples for testing

This avoids look-ahead leakage in time-series evaluation.
