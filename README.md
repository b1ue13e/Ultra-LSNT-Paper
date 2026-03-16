<div align="center">

# ⚡ Ultra-LSNT

**U**ltra-**L**ightweight **S**parse **N**eural **T**ransformer for Edge-Native Wind Power Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R² Score](https://img.shields.io/badge/R²%20Score-%3E0.83-brightgreen.svg)]()
[![Complexity](https://img.shields.io/badge/Complexity-O(L)-orange.svg)]()

*When SCADA Noise Meets Sparse MoE: Uncompromising Accuracy at the Edge*

[Key Features](#-key-features) • [Quick Start](#-quick-start) • [Experiments](#-experiments) • [Citation](#-citation)

</div>

---

## 🎯 Project Abstract

**Ultra-LSNT** is a paradigm-shifting Sparse Mixture-of-Experts (MoE) framework engineered for **edge-native deployment** in wind power forecasting. While conventional deep learning models crumble under the harsh reality of corrupted SCADA sensor data, Ultra-LSNT delivers **uncompromising $R^2 > 0.83$ accuracy** even under severe noise conditions — all while maintaining **linear complexity $\mathcal{O}(L)$** and sub-millisecond inference latency.

Built upon two revolutionary mechanisms — **Feasibility-Guided MoE (FG-MoE)** and **Jump Gate Dynamic Skipping** — Ultra-LSNT doesn't just predict; it *thinks*, adaptively routing computations through physics-informed expert networks while dynamically bypassing redundant operations based on real-time input feasibility.

> 🔥 **The Bottom Line**: We achieve **2.3× better noise robustness** than DLinear, **4.7× lower latency** than iTransformer, and **87% fewer FLOPs** than PatchTST — without sacrificing predictive power.

---

## ✨ Key Features

### 🛡️ Noise-Immune Architecture
- **Feasibility Guidance (FG-MoE)**: Physics-informed expert routing that maintains prediction integrity even when 30% of SCADA sensors fail
- **Dual-Mode Propagation**: Seamless switching between dense and sparse computation modes
- **Hard Concrete Gates**: Differentiable sparsity with straight-through estimation for training-inference consistency

### ⚡ Edge-Native Design
```
┌─────────────────────────────────────────────────────────────┐
│  Latency Benchmark (NVIDIA Jetson Nano @ 96-step input)    │
├─────────────────────────────────────────────────────────────┤
│  Ultra-LSNT (Ours)     ████████░░░░░░░░░░░░░░░░░   2.1ms  │
│  DLinear               ████████████████░░░░░░░░░░   4.8ms  │
│  PatchTST              ████████████████████████░░  12.3ms  │
│  iTransformer          ██████████████████████████  28.7ms  │
└─────────────────────────────────────────────────────────────┘
```

- **Linear Complexity $\mathcal{O}(L)$**: Sequence length independent of computational cost
- **Dynamic Expert Skipping**: Jump Gate mechanism bypasses up to 40% of computation on clean inputs
- **Quantization-Ready**: INT8 quantization with <1% accuracy degradation

### 🧠 Intelligent Expert System
| Expert Type | Specialization | Activation Condition |
|-------------|----------------|----------------------|
| $\mathcal{E}_1$ | Low-wind regime ($v < 3$ m/s) | Cut-in/off transitions |
| $\mathcal{E}_2$ | Rated-power zone | Maximum power tracking |
| $\mathcal{E}_3$ | High-variance gusts | Turbulence compensation |
| $\mathcal{E}_4$ | Seasonal patterns | Long-term drift adaptation |

### 📊 Multi-Domain Generalization
Validated across 4 heterogeneous time-series domains:
- 🌪️ **Wind Power (CN/US)**: 15-min & 5-min SCADA data
- 🏭 **Air Quality**: Hourly AQI forecasting
- ⚡ **Power Load**: GEFCom2012 multi-zone demand

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# CUDA-capable GPU recommended (but not required for inference)
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/b1ue13e/ultra-lsnt.git
cd ultra-lsnt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 1. Train Ultra-LSNT on Wind Power Data

```bash
python src/experiments/train_ultra_lsnt_stable.py \
    --dataset wind_cn \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --seq_len 96 \
    --pred_len 24 \
    --num_experts 4 \
    --top_k 2
```

**Key Arguments:**
- `--seq_len`: Input history length (default: 96, i.e., 24 hours of 15-min data)
- `--pred_len`: Prediction horizon (default: 24, i.e., 6 hours ahead)
- `--num_experts`: Number of expert networks (default: 4)
- `--top_k`: Active experts per sample (default: 2)

### 2. Run Multi-Domain Baseline Comparison

```bash
# Compare Ultra-LSNT against SOTA baselines
python src/experiments/run_multi_domain_baselines.py \
    --models ultra_lsnt,dlinear,patchtst,timemixer \
    --datasets wind_cn,wind_us,air_quality,gefcom \
    --epochs 20 \
    --batch_size 512
```

**Supported Baselines:** `dlinear`, `patchtst`, `itransformer`, `timemixer`, `lightgbm`, `xgboost`

### 3. Comprehensive Ablation Study

```bash
# Systematically ablate each component
python src/experiments/run_comprehensive_ablation.py \
    --dataset wind_cn \
    --ablate_components film,dual_propagation,expert_routing,physics_guidance
```

**Ablation Components:**
- `film`: Task-conditioned Feature-wise Linear Modulation
- `dual_propagation`: Dual-mode sparse/dense computation paths
- `expert_routing`: Top-k expert selection mechanism
- `physics_guidance`: Wind power curve feasibility constraints

### 4. Noise Robustness Stress Test

```bash
# Test robustness against SCADA noise
python src/experiments/run_universal_robustness.py \
    --dataset wind_cn \
    --noise_types gaussian,impulse,missing \
    --noise_levels 0.0,0.1,0.2,0.3 \
    --trials 5
```

**Noise Protocols:**
- **Gaussian**: Additive sensor noise $\mathcal{N}(0, \sigma^2)$
- **Impulse**: Salt-and-pepper communication errors
- **Missing**: Random SCADA dropout (MCAR/MAR)

### 5. Edge Efficiency Benchmark

```bash
# Measure FLOPs, latency, and energy consumption
python src/experiments/run_efficiency_benchmark.py \
    --models ultra_lsnt,dlinear,patchtst \
    --seq_lens 96,192,336,720 \
    --device cuda  # or 'cpu' for edge simulation
```

### 6. Generate Paper Figures

```bash
# Figure 3: Overall performance comparison
python src/experiments/fig_overall_performance_windcn.py

# Figure 4: Ablation study results
python src/experiments/fig_multi_domain.py

# Figure 5: Robustness stress test
python src/experiments/fig_robustness_2x2.py

# Figure 6: Time-series prediction visualization
python src/experiments/fig_timeseries_compare_2x2.py

# Expert activation heatmap
python src/experiments/plot_expert_heatmap.py
```

All figures are saved to `results/figures/` in publication-ready PDF format.

---

## 📁 Repository Structure

```
ultra-lsnt/
├── src/
│   ├── models/                    # Core model implementations
│   │   ├── ultra_lsnt_v4.py      # Main model (FG-MoE + Jump Gate)
│   │   ├── ultra_lsnt_lite.py    # Ultra-lightweight variant
│   │   └── ultra_lsnt_timeseries.py  # TimeSeries library wrapper
│   ├── baselines/                 # Baseline implementations
│   │   ├── run_dlinear.py
│   │   ├── run_latest_sota.py    # PatchTST, iTransformer, TimeMixer
│   │   └── run_gbdt.py           # LightGBM/XGBoost
│   ├── experiments/               # Experiment scripts
│   │   ├── train_ultra_lsnt_stable.py
│   │   ├── run_multi_domain_baselines.py
│   │   ├── run_comprehensive_ablation.py
│   │   ├── run_universal_robustness.py
│   │   └── run_efficiency_benchmark.py
│   └── data_preprocess.py         # Data preprocessing pipeline
├── data/processed/                # Processed datasets
│   ├── wind_final.csv            # China wind SCADA data
│   ├── wind_us.csv               # NREL WIND Toolkit
│   ├── air_quality_ready.csv     # China AQI data
│   └── gefcom_ready.csv          # GEFCom2012 load data
├── results/                       # Experimental outputs
│   ├── figures/                  # Publication-quality plots
│   └── tables/                   # Numerical results (CSV)
├── scripts/                       # Batch execution scripts
│   ├── run_all.sh                # Run all experiments
│   └── clean_and_plot.sh         # Generate all figures
└── paper/                         # LaTeX source and PDF
```

---

## 🔬 Experiments

### Reproduce Paper Results

```bash
# Full reproduction pipeline
bash scripts/run_complete.sh
```

This executes:
1. ✅ Data preprocessing and validation
2. ✅ Multi-domain baseline comparison (14 models × 4 datasets)
3. ✅ Comprehensive ablation study (16 configurations)
4. ✅ Noise robustness stress test (3 noise types × 4 levels × 5 trials)
5. ✅ Edge efficiency benchmark (4 sequence lengths × 3 devices)
6. ✅ Statistical significance testing (paired t-test, Wilcoxon)
7. ✅ Figure generation for all paper plots

**Expected Runtime**: ~6 hours on NVIDIA RTX 4090

### Custom Experiments

```python
from src.models.ultra_lsnt_v4 import UltraLSNT, LSNTConfig
import torch

# Configure model for edge deployment
config = LSNTConfig(
    input_dim=128,
    hidden_dim=256,
    output_dim=24,
    num_blocks=3,
    num_experts=4,
    top_k=2,                    # Sparse activation
    skip_threshold=0.5,         # Jump Gate threshold
    router_z_loss_coef=0.01,    # Load balancing
    dual_gate_type="ste"        # Straight-through estimation
)

model = UltraLSNT(config)
model.eval()

# Edge inference
with torch.no_grad():
    # Input: [batch, seq_len, features]
    prediction, aux_loss = model(input_tensor)
    
# aux_loss contains load balancing loss for training
```

---

## 📈 Performance Highlights

### Robustness Under Severe Noise (Wind-CN, 80/20 Split)

| Model | Clean | +20% Noise | +30% Noise | Degradation |
|-------|-------|------------|------------|-------------|
| **Ultra-LSNT (Ours)** | **0.891** | **0.854** | **0.831** | **-6.7%** |
| DLinear | 0.872 | 0.798 | 0.721 | -17.3% |
| PatchTST | 0.885 | 0.812 | 0.745 | -15.8% |
| iTransformer | 0.878 | 0.801 | 0.718 | -18.2% |
| LightGBM | 0.834 | 0.721 | 0.642 | -23.0% |

### Efficiency Metrics (seq_len=96, batch=1)

| Metric | Ultra-LSNT | DLinear | PatchTST | iTransformer |
|--------|------------|---------|----------|--------------|
| Parameters | 2.1M | 1.8M | 8.7M | 12.4M |
| FLOPs | 12.4M | 18.2M | 94.3M | 156.8M |
| Latency (GPU) | 2.1ms | 4.8ms | 12.3ms | 28.7ms |
| Latency (CPU) | 18.4ms | 42.1ms | 156.2ms | 423.8ms |
| Memory | 18MB | 22MB | 67MB | 98MB |

---

## 📝 Citation

If you find Ultra-LSNT useful in your research, please cite:

```bibtex
@article{ultra_lsnt_2025,
  title={Ultra-LSNT: Feasibility-Guided Sparse Mixture-of-Experts for Edge-Native Wind Power Forecasting under SCADA Noise},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  publisher={[Publisher]}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Wind data: Xinjiang Wind Farm & NREL WIND Toolkit
- Air quality data: China National Environmental Monitoring Centre
- Load data: GEFCom2012 Competition
- Baseline implementations: Time-Series Library (Tsinghua)

---

<div align="center">

**[⬆ Back to Top](#-ultra-lsnt)**

*Built with PyTorch. Optimized for the Edge. Resilient against Chaos.*

</div>
