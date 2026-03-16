# 快速开始指南

## 环境配置

### 1. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 或使用 conda
conda create -n ultra_lsnt python=3.9
conda activate ultra_lsnt
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 快速实验

### 实验1: 单数据集训练

```bash
python src/experiments/run_ultra_lsnt_wind_cn_real.py \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --seq_len 96 \
    --pred_len 24
```

### 实验2: 多域对比

```bash
python src/experiments/run_multi_domain_baselines.py \
    --model2 timemixer \
    --epochs 20 \
    --batch_size 512
```

### 实验3: 生成论文图表

```bash
# 数据概览图
python src/experiments/fig_data_overview_2x2.py

# 多域性能图
python src/experiments/fig_multi_domain.py

# 鲁棒性测试图
python src/experiments/fig_robustness_2x2.py

# 时序对比图
python src/experiments/fig_timeseries_compare_2x2.py
```

## 模型推理

```python
import torch
from src.models.ultra_lsnt_v4 import UltraLSNT, LSNTConfig

# 加载配置
config = LSNTConfig(
    input_dim=128,
    hidden_dim=256,
    output_dim=24,
    num_blocks=3,
    num_experts=4,
    top_k=2
)

# 创建模型
model = UltraLSNT(config)
model.load_state_dict(torch.load('checkpoints/model_best.pth'))
model.eval()

# 推理
with torch.no_grad():
    prediction = model(input_data)
```

## 数据预处理

```bash
# 预处理新的风电数据
python src/data_preprocess.py \
    --data data/raw/new_wind_data.xlsx \
    --target power \
    --output data/processed/new_wind.csv
```

## 常见问题

### Q: CUDA 内存不足？

减小 batch_size:
```bash
python src/experiments/run_ultra_lsnt_wind_cn_real.py --batch_size 128
```

### Q: 如何只运行特定数据集？

编辑 `src/experiments/run_multi_domain_baselines.py` 中的 `DATASETS` 列表。

### Q: 如何修改模型结构？

编辑 `src/models/ultra_lsnt_v4.py` 中的 `LSNTConfig` 配置类。

## 更多帮助

- 完整文档: [README.md](README.md)
- 技术报告: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)
- 实验结果: [results/tables/](results/tables/)
