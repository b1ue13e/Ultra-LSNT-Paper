<div align="center">

# ⚡ Ultra-LSNT

**U**ltra-**L**ightweight **S**parse **N**eural **T**ransformer for Edge-Native Wind Power Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R² Score](https://img.shields.io/badge/R²%20Score-Positive%20@%20Severe%20Noise-brightgreen.svg)]()
[![Complexity](https://img.shields.io/badge/Complexity-O(L)-orange.svg)]()

*When SCADA Noise Meets Sparse MoE: Uncompromising Accuracy at the Edge*

[Key Features](#-key-features) • [Quick Start](#-quick-start) • [Experiments](#-experiments) • [Citation](#-citation)

</div>

---

## 🎯 Project Abstract

**Ultra-LSNT** is a sparse mixture-of-experts (MoE) forecaster engineered for **controller-side edge deployment** in wind power forecasting. While standard models often crumble under the harsh reality of corrupted SCADA sensor streams, Ultra-LSNT retains positive skill ($R^2 = 0.7863$) even at the severe Gaussian stress slice ($\sigma_{eff} = 0.60$), where linear baselines completely collapse.

Built upon two core mechanisms — **Feasibility-Guided MoE (FG-MoE)** and **Jump Gate Conditional Skipping** — Ultra-LSNT achieves linear complexity $O(L)$ while seamlessly keeping predictions physically admissible. 

> 🔥 **The Bottom Line**: FG-MoE cuts rated-power exceedance from 0.0550% to 0.0092% against a matched no-guardrail backbone, avoiding the clean/severe accuracy collapse typical of post-hoc clip-only repair strategies.

---

## ✨ Key Features

### 🛡️ Noise-Immune Architecture
- **Feasibility Guidance (FG-MoE)**: Regularizes forecasts with a deployable wind proxy, actively discouraging physically implausible outputs during training rather than just post-inference clipping.
- **Sparse Top-K Routing**: Adaptively allocates capacity so harder, noisy windows activate more experts while easier windows traverse a smaller path.

### ⚡ Edge-Native Design
```text
┌─────────────────────────────────────────────────────────────┐
│  Batch-1 Edge Profile Latency (Commodity CPU)               │
├─────────────────────────────────────────────────────────────┤
│  DLinear               ██░░░░░░░░░░░░░░░░░░░░░░░   0.10ms   │
│  Ultra-LSNT-Lite       ████████░░░░░░░░░░░░░░░░░   1.17ms   │
│  ARIMA (Ref)           ██████████████░░░░░░░░░░░   2.18ms   │
│  Ultra-LSNT (Full)     ██████████████████████░░░   3.71ms   │
└─────────────────────────────────────────────────────────────┘
```

* **Linear Complexity $O(L)$**: Replaces standard $O(L^2)$ attention with multiscale decomposition and conditional routing.
* **Jump Gate Skipping**: Token-wise conditional skipping directly bypasses redundant expert evaluations to save average compute.

### 📊 Multi-Domain Generalization

Validated across heterogeneous time-series domains:
* 🌪️ **Wind Power (CN/US)**: 15-min & 5-min SCADA data 
* 🏭 **Air Quality**: Daily city-panel records 
* ⚡ **Power Load**: GEFCom2012 multi-zone demand 

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/b1ue13e/Ultra-LSNT.git
cd Ultra-LSNT

# Install dependencies
pip install -r requirements.txt
```

### 1. Train Ultra-LSNT

```bash
python src/experiments/train_ultra_lsnt_stable.py
```

### 2. Run Baselines & Comparisons

```bash
# Multi-domain comparison
python src/experiments/run_multi_domain_baselines.py

# Comprehensive ablation
python src/experiments/run_comprehensive_ablation.py

# Universal robustness test under SCADA noise
python src/experiments/run_universal_robustness.py

# Efficiency benchmark
python src/experiments/run_efficiency_benchmark.py
```

---

## 📁 Repository Structure

```text
Ultra-LSNT/
├── src/
│   ├── models/                    # Core implementations (Ultra-LSNT, Lite variant)
│   ├── baselines/                 # Baselines (DLinear, iTransformer, TimeMixer, etc.)
│   └── experiments/               # Execution scripts for training and testing
├── data/                          # Dataset directory (Wind-CN, Wind-US, etc.)
├── results/                       # Experimental outputs (Figures and Tables)
├── requirements.txt               # Environment dependencies
└── README.md                      # Project documentation
```

*(Note: LaTeX source codes and paper drafts are explicitly excluded from this public repository.)*

---

## 📈 Performance Highlights

### Robustness Under Severe SCADA Noise (Wind-CN)

Performance degradation at the severe Gaussian stress slice ($\sigma_{eff} = 0.60$):

| Model | Clean $R^2$ | Severe Noise $R^2$ (@0.60) | Status under Noise |
| --- | --- | --- | --- |
| **Ultra-LSNT (Ours)** | 0.9346 | **0.7863** | ✅ Stable & Positive |
| iTransformer | 0.9012 | 0.8717 | ✅ Positive |
| TimeMixer | 0.8337 | 0.8886 | ✅ Positive |
| DLinear | 0.9494 | -0.9903 | ❌ Collapsed |

### Edge Inference Profile (Batch-1, CPU)

Direct batch-1 measurements for edge-device suitability:

| Model | Latency (ms) | Throughput (Hz) | Memory (MiB) |
| --- | --- | --- | --- |
| **Ultra-LSNT-Lite** | **1.171** | **854.0** | **1.3327** |
| **Ultra-LSNT (Full)** | 3.714 | 269.3 | 22.2830 |
| DLinear | 0.102 | 3767.6 | 0.0178 |

---

## 📝 Citation

If you find Ultra-LSNT or our SCADA noise robustness protocol useful in your research, please consider citing our paper:

```bibtex
@article{li2025ultralsnt,
  title={Wind power forecasting under corrupted SCADA for edge deployment: sparse MoE with feasibility guidance},
  author={Li, Junyu and Du, Juntao},
  journal={Applied Energy},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">

**[⬆ Back to Top](#-ultra-lsnt)**

*Executing dimensional strikes on engineering problems.*

</div>
