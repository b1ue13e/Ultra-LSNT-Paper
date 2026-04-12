# Ultra-LSNT

Ultra-LSNT is the public research repository accompanying the manuscript:

> "Controller-side feasibility guidance for wind power forecasting under corrupted SCADA"

The repository name retains the model identifier introduced during development, while the submitted paper emphasizes the controller-side feasibility-guidance contribution built around the Ultra-LSNT sparse forecasting backbone.

## What this repository includes

- Ultra-LSNT and Ultra-LSNT-Lite implementations
- Baseline models used in the main comparisons
- Corrupted-SCADA robustness evaluation scripts
- Batch-1 efficiency benchmarking scripts
- Ablation and safety-analysis scripts
- Secondary cross-domain diagnostic experiments
- Released data assets and split manifests for reproducibility

## Repository layout

```text
Ultra-LSNT/
|- src/                          Core models and experiment scripts
|- scripts/                      Helper scripts
|- data/
|  |- raw/                       Raw released Wind (CN) time series
|  `- processed/                 Processed datasets used by experiments
|- results/                      Figures, tables, and derived outputs
|- split_manifest_80_20.json
|- split_manifest_80_20_unified.json
|- wind_main.csv                 Legacy convenience copy for scripts
|- wind_final.csv                Legacy convenience copy for scripts
|- wind_us.csv                   Legacy convenience copy for scripts
|- air_quality_ready.csv         Legacy convenience copy for scripts
|- gefcom_ready.csv              Legacy convenience copy for scripts
`- processed_wind.csv            Legacy convenience copy for scripts
```

The canonical data location is `data/`. A small set of root-level CSV copies is intentionally retained so that legacy experiment scripts in this repository continue to run without path edits.

## Data availability

### Wind (CN)

The repository includes the following assets for the main Wind (CN) study:

- Raw SCADA-like time series: `data/raw/wind_main.csv`
- Processed forecasting tables: `data/processed/wind_final.csv` and `data/processed/processed_wind.csv`
- Public split manifests:
  - `split_manifest_80_20.json`
  - `split_manifest_80_20_unified.json`
- Preprocessing code: `src/data_preprocess.py`

### Diagnostic benchmark datasets

Processed benchmark files used for the secondary diagnostic experiments are stored at:

- `data/processed/wind_us.csv`
- `data/processed/air_quality_ready.csv`
- `data/processed/gefcom_ready.csv`

## Quick start

```bash
git clone https://github.com/b1ue13e/Ultra-LSNT.git
cd Ultra-LSNT
pip install -r requirements.txt
```

## Main reproduction commands

```bash
# Main Ultra-LSNT training on Wind (CN)
python src/experiments/train_ultra_lsnt_stable.py

# Corrupted-SCADA robustness experiments
python src/experiments/run_universal_robustness.py

# Batch-1 efficiency benchmark
python src/experiments/run_efficiency_benchmark.py

# Ablation study
python src/experiments/run_comprehensive_ablation.py

# Cross-domain diagnostics
python src/experiments/run_multi_domain_baselines.py
```

## Reproducibility notes

- The repository uses chronological 80/20 splits for the released datasets.
- The split definitions used in the paper are provided through the checked-in manifest files.
- Some scripts were developed during manuscript preparation and keep older default file names; the root-level convenience copies are present to preserve backward compatibility.
- Results in the paper should be interpreted as bounded controller-side evidence under the reported protocol, not as operator-grade deployment validation.

## Citation

If you use this repository, the corrupted-SCADA evaluation protocol, or the released data assets, please cite:

```bibtex
@article{li2026ultralsnt,
  title   = {Controller-side feasibility guidance for wind power forecasting under corrupted SCADA},
  author  = {Li, Junyu and Du, Juntao},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Under review}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
