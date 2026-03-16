<div align="center">

# ⚡ Ultra-LSNT

**Ultra-Lightweight Sparse Neural Transformer for Controller-Side Wind Power Forecasting**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg?logo=python)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Code-blue.svg)]()
[![Paper](https://img.shields.io/badge/Paper-Applied%20Energy-green.svg)]()

*Sparse MoE forecasting under corrupted SCADA for edge deployment*

[Overview](#-overview) • [Main Results](#-main-results) • [Reproducibility](#-reproducibility) • [Data](#-data-availability) • [Citation](#-citation)

</div>

---

<details open>
<summary><b>📋 Table of Contents</b></summary>

- [Overview](#-overview)
- [What is Included](#-what-is-included)
- [Main Results](#-main-results)
  - [1. Robustness under Corrupted SCADA](#1-robustness-under-corrupted-scada)
  - [2. Safety Effect of Feasibility Guidance](#2-safety-effect-of-feasibility-guidance)
  - [3. Batch-1 Edge Profile](#3-batch-1-edge-profile)
  - [4. Dispatch-Screening Interpretation](#4-dispatch-screening-interpretation)
  - [5. Cross-Domain Diagnostics](#5-cross-domain-diagnostics)
- [Reproducibility](#-reproducibility)
- [Data Availability](#-data-availability)
- [Repository Structure](#-repository-structure)
- [Scope Notes](#-scope-notes)
- [Citation](#-citation)
- [License](#-license)

</details>

---

## 📖 Overview

This repository contains the code, data assets, and evaluation scripts accompanying the study:

> **"Wind power forecasting under corrupted SCADA for edge deployment: sparse MoE with feasibility guidance"**

<p align="center">
  <img src="results/figures/workflow_pipeline.pdf" width="90%" alt="Workflow Pipeline">
  <br>
  <em>Figure 1: End-to-end workflow — from corrupted SCADA input to dispatch screening output</em>
</p>

The project studies a **controller-side** forecasting setting in which models must operate on **corrupted SCADA streams** under **batch-1 latency and memory constraints**, and forecast usefulness is evaluated under a **fixed forecast-to-dispatch interface**.

### Core Design

<p align="center">
  <img src="results/figures/figure2.pdf" width="85%" alt="Ultra-LSNT Architecture">
  <br>
  <em>Figure 2: Ultra-LSNT Architecture — Sparse Top-K routing with Jump Gate skipping and FG-MoE training</em>
</p>

Ultra-LSNT is a sparse mixture-of-experts forecaster combining:

| Component | Purpose |
|:----------|:--------|
| **Sparse Top-K Routing** | Conditional computation — harder windows activate more experts |
| **Jump Gate Skipping** | Reduced average inference cost via token-wise bypass |
| **FG-MoE** | Training-time feasibility guidance using deployable wind proxy |

The repository is intended as a **research codebase** for reproducibility and comparative evaluation under the protocol used in the manuscript.

---

## 📦 What is Included

```
✓ Ultra-LSNT and Ultra-LSNT-Lite implementations
✓ Baselines used in the main comparison
✓ Robustness evaluation (Gaussian noise, drift, quantization)
✓ Hardware profiling scripts for batch-1 edge inference
✓ Ablation and sensitivity experiments
✓ Secondary cross-domain diagnostic experiments
✓ Data preprocessing and evaluation assets
```

---

## 🎯 Main Results

### 1. Robustness under Corrupted SCADA

> **Key Point**: Under the controller-side protocol, Ultra-LSNT remains usable under severe corruption while weaker local baselines may collapse.

At the severe Gaussian stress slice **σₑff = 0.60** on Wind (CN):

| Model | Clean R² | Severe R² | Status |
|:-----:|:--------:|:---------:|:------:|
| **Ultra-LSNT** | 0.9346 | **0.7863** | ✅ Stable |
| iTransformer | 0.9012 | 0.8717 | ✅ Positive |
| TimeMixer | 0.8337 | 0.8886 | ✅ Positive |
| DLinear | 0.9494 | **-0.9903** | ❌ Collapsed |

<details>
<summary><b>Interpretation</b></summary>

These rows should be read as **controller-side local comparisons** under the reported protocol. Long-context foundation-model references discussed in the paper are contextual and should not be read as edge-feasible deployment peers.

</details>

---

### 2. Safety Effect of Feasibility Guidance

FG-MoE is evaluated against a matched backbone and a clip-only repair control:

| Setting | Clean R² | R²@0.60 | Exceedance (%) | Notes |
|:--------|:--------:|:-------:|:--------------:|:------|
| Ultra-LSNT (no guardrail) | 0.8727 | 0.8273 | 0.0550 | Baseline |
| Ultra-LSNT + clip-only | 0.4319 | 0.3120 | 0.0000 | Post-hoc repair |
| **FG-MoE (full-trained)** | **0.8747** | **0.8286** | **0.0092** | **Training-time guidance** |

> 💡 **Takeaway**: Training-time feasibility guidance reduces infeasible outputs **without the large accuracy loss induced by clip-only repair**.

---

### 3. Batch-1 Edge Profile

Direct batch-1 measurements for same-stack hardware comparison:

| Model | Latency (ms) | Throughput (Hz) | Memory (MiB) |
|:------|:------------:|:---------------:|:------------:|
| Ultra-LSNT (full) | 3.714 | 269.3 | 22.2830 |
| **Ultra-LSNT-Lite** | **1.171** | **854.0** | **1.3327** |
| DLinear | 0.265 | 3767.6 | 0.0178 |

> **Note**: Ultra-LSNT is not the fastest model; the point is it remains within a **controller-side screening envelope** while preserving robustness and safety properties that simpler baselines do not maintain equally well.

---

### 4. Dispatch-Screening Interpretation

Under the shared mapped-and-clipped downstream interface:

```
┌─────────────────────────────────────────────────────────────┐
│  Fixed-Interface Dispatch Screening Workflow                 │
├─────────────────────────────────────────────────────────────┤
│  Raw Forecasts → Mapping Function → Clipping → Screening    │
│       ↓              ↓               ↓           ↓          │
│   (All models    (Shared)        (Shared)    RTS-24 DC     │
│    fail closure              interface)     Admissibility   │
│    before map)                                 Check        │
└─────────────────────────────────────────────────────────────┘
```

Screening outcomes under the fixed interface:

| Model | Fallback Burden | Curtailment | Cost Rank |
|:------|:---------------:|:-----------:|:---------:|
| iTransformer | Lowest | — | 1st |
| TimeMixer | Low | — | 2nd |
| **Ultra-LSNT** | Moderate | Better than DLinear | 3rd |
| DLinear | Higher | — | 4th |

> ⚠️ The RTS-24 DC layer is a **coarse admissibility check**, not an operator-grade ranking environment.

---

### 5. Cross-Domain Diagnostics

<details>
<summary><b>🔬 Secondary Diagnostic Experiments (Click to expand)</b></summary>

Additional results on **Wind (US)**, **Air Quality**, and **GEFCom Load** are retained as **secondary diagnostic evidence**.

They are included to show boundary behavior across tasks and interfaces, **not** as an extension of the main controller-side claim.

| Dataset | Type | Purpose |
|:--------|:-----|:--------|
| Wind (US) | Time-series | Boundary behavior test |
| Air Quality | Panel data | Interface stress test |
| GEFCom Load | Multi-zone | Task transfer probe |

</details>

---

## 🚀 Reproducibility

### Installation

```bash
# Clone repository
git clone https://github.com/b1ue13e/Ultra-LSNT.git
cd Ultra-LSNT

# Install dependencies
pip install -r requirements.txt
```

### Main Scripts

```bash
# Train the main model
python src/experiments/train_ultra_lsnt_stable.py

# Run robustness experiments
python src/experiments/run_universal_robustness.py

# Run efficiency benchmarking
python src/experiments/run_efficiency_benchmark.py

# Run ablation studies
python src/experiments/run_comprehensive_ablation.py

# Run cross-domain diagnostics
python src/experiments/run_multi_domain_baselines.py
```

### Reproduction Scope

This repository is organized to reproduce:

- [x] Chronological splitting
- [x] Shared preprocessing
- [x] Corruption injection
- [x] Robustness evaluation
- [x] Hardware profiling
- [x] Safety diagnostics
- [x] Fixed-interface dispatch screening

> ⚠️ Some supplementary reference rows in the paper are archived or backfilled contextual comparisons rather than direct same-hardware, same-implementation measurements.

---

## 📊 Data Availability

### Wind (CN) — Main Dataset

The Wind (CN) SCADA dataset is released with:

- Raw time series
- Train/validation/test split indices
- Preprocessing scripts from the reported protocol

### Public Benchmarks — Diagnostic Use

| Dataset | Source | Location |
|:--------|:-------|:---------|
| Wind (US) | NREL Wind Toolkit | `data/wind_us/` |
| Air Quality | UCI Air Quality | `data/air_quality/` |
| GEFCom Load | GEFCom2012 | `data/gefcom_load/` |

Place required datasets under the `data/` directory following project-specific preprocessing scripts.

---

## 📁 Repository Structure

```
Ultra-LSNT/
├── src/
│   ├── models/                    # Ultra-LSNT & Lite variants
│   ├── baselines/                 # Baseline implementations
│   └── experiments/               # Training & evaluation scripts
├── data/                          # Dataset directory
├── results/                       # Generated figures, tables, logs
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

> Research manuscript sources and drafting files are **not included** in the public release.

---

## ⚠️ Scope Notes

This repository accompanies a research manuscript and should be read with the same scope limitations:

| Aspect | Scope |
|:-------|:------|
| Main evidence | One Wind (CN) site |
| Corruption protocol | Partly synthetic |
| Dispatch layer | Comparative screening workflow under fixed interface |
| Hardware profile | Pre-HIL, not board-level operator validation |

**The results support a bounded controller-side comparative claim, not an operator-grade deployment proof.**

---

## 📝 Citation

If you use this repository or the corrupted-SCADA evaluation protocol, please cite:

```bibtex
@article{li2026ultralsnt,
  title   = {Wind power forecasting under corrupted SCADA for edge deployment: 
             sparse MoE with feasibility guidance},
  author  = {Li, Junyu and Du, Juntao},
  journal = {Applied Energy},
  year    = {2026}
}
```

---

## 📜 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

<div align="center">

**[⬆ Back to Top](#-ultra-lsnt)**

</div>
