# Experiments Guide

This file summarizes the major experiment scripts that matter most for the public manuscript release.

## Core manuscript scripts

### Main Wind (CN) training

- Script: `src/experiments/train_ultra_lsnt_stable.py`
- Purpose: train the main Ultra-LSNT forecasting model on Wind (CN)
- Notes: this is the most direct starting point for the paper's main model workflow

### Cross-domain diagnostics

- Script: `src/experiments/run_multi_domain_baselines.py`
- Purpose: compare Ultra-LSNT against baseline models across Wind (CN), Wind (US), Air Quality, and GEFCom-style data
- Notes: useful as the quickest repository-level sanity check

### Corrupted-SCADA robustness

- Script: `src/experiments/run_universal_robustness.py`
- Purpose: evaluate degradation under noise and corruption stress

### Efficiency benchmark

- Script: `src/experiments/run_efficiency_benchmark.py`
- Purpose: measure batch-1 inference and training-side runtime behavior

### Ablation study

- Script: `src/experiments/run_comprehensive_ablation.py`
- Purpose: test the contribution of major modeling choices

## Supporting script groups

### Baselines

Main baseline implementations live in `src/baselines/`, including:

- `run_dlinear.py`
- `run_latest_sota.py`
- `run_gbdt.py`
- `classical_baselines.py`
- `traditional_baselines_experiment.py`
- metaheuristic baselines such as COA and BWO variants

### Figure generation

Selected plotting and figure scripts live in `src/experiments/`, including:

- `fig_data_overview_2x2.py`
- `fig_robustness_2x2.py`
- `fig_scatter_pred_true_2x2.py`
- `fig_timeseries_compare_2x2.py`
- `plot_expert_heatmap.py`
- `plot_multi_domain.py`

### Dispatch and feasibility analysis

Relevant scripts include:

- `run_dispatch_closure_mapping_decision_4090.py`
- `run_network_constrained_dispatch_ieee24.py`
- `dispatch_mapping_utils.py`
- `wind_dispatch_model.py`

## Reproduction order

If you want a practical order rather than a complete one:

1. Check imports and paths with `python src/experiments/run_multi_domain_baselines.py --help`
2. Run `python src/experiments/train_ultra_lsnt_stable.py`
3. Run `python src/experiments/run_universal_robustness.py`
4. Run `python src/experiments/run_efficiency_benchmark.py`
5. Run `python src/experiments/run_comprehensive_ablation.py`

## Expected outputs

Most derived outputs land under:

- `results/figures/`
- `results/tables/`

Some scripts may also write checkpoints or temporary experiment artifacts outside the tracked results folders, depending on local configuration.
