# Quick Start

This guide is the fastest way to get the public Ultra-LSNT repository running on a fresh machine.

## 1. Create an environment

### `venv`

```bash
python -m venv .venv
.venv\Scripts\activate
```

### `conda`

```bash
conda create -n ultra_lsnt python=3.9
conda activate ultra_lsnt
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Check that the repository is readable

```bash
python src/experiments/run_multi_domain_baselines.py --help
```

If that command prints the argument list, the repository layout and compatibility wrappers are working as expected.

## 4. Main entry points

### Train the main Wind (CN) model

```bash
python src/experiments/train_ultra_lsnt_stable.py
```

### Run the corrupted-SCADA robustness evaluation

```bash
python src/experiments/run_universal_robustness.py
```

### Run the batch-1 efficiency benchmark

```bash
python src/experiments/run_efficiency_benchmark.py
```

### Run the ablation study

```bash
python src/experiments/run_comprehensive_ablation.py
```

### Run the cross-domain diagnostics

```bash
python src/experiments/run_multi_domain_baselines.py
```

## 5. Where the main data files live

Canonical data files:

- `data/raw/wind_main.csv`
- `data/processed/wind_final.csv`
- `data/processed/processed_wind.csv`
- `data/processed/wind_us.csv`
- `data/processed/air_quality_ready.csv`
- `data/processed/gefcom_ready.csv`

Legacy root-level copies are also kept so older scripts can run without path changes.

## 6. Common notes

- The repository uses chronological 80/20 splits described in `split_manifest_80_20.json` and `split_manifest_80_20_unified.json`.
- Some experiments expect a CUDA-capable GPU for practical runtime.
- The public release focuses on reproducibility of the submitted manuscript, not on packaging the codebase as a polished software library.

## 7. Next documents to read

- `README.md` for the project overview
- `STRUCTURE.md` for a repository map
- `EXPERIMENTS.md` for script-by-script experiment guidance
- `DATA.md` for dataset details
