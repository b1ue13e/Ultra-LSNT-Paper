# 实验清单

本文档列出项目中包含的所有实验及其用途。

## 1. 主模型实验

### 1.1 单数据集训练
- **脚本**: `src/experiments/run_ultra_lsnt_wind_cn_real.py`
- **用途**: 在中国风电数据集上训练 Ultra-LSNT
- **关键参数**: 
  - `--epochs`: 训练轮数 (默认: 50)
  - `--batch_size`: 批次大小 (默认: 256)
  - `--seq_len`: 输入序列长度 (默认: 96)
  - `--pred_len`: 预测长度 (默认: 24)

### 1.2 稳定训练
- **脚本**: `src/experiments/train_ultra_lsnt_stable.py`
- **用途**: 更稳定的训练流程，适合长时间训练
- **特点**: 改进的学习率调度，早停机制

### 1.3 增强训练
- **脚本**: `src/experiments/train_ultra_lsnt_enhanced.py`
- **用途**: 增强版训练，支持更多高级功能

## 2. 多域对比实验

### 2.1 多域基线对比
- **脚本**: `src/experiments/run_multi_domain_baselines.py`
- **用途**: 在4个数据集上对比 Ultra-LSNT 与基线模型
- **支持基线**: DLinear, TimeMixer, PatchTST, iTransformer
- **输出**: `results/tables/multi_domain_baselines_r2_*.csv`

### 2.2 统一评估框架
- **脚本**: `src/experiments/evaluate_multi_domain.py`
- **用途**: 统一评估多个模型在多个数据集上的性能

## 3. 消融实验

### 3.1 综合消融实验
- **脚本**: `src/experiments/run_comprehensive_ablation.py`
- **用途**: 系统性地消融各个组件
- **消融组件**:
  - 物理信息模块
  - 混合专家机制
  - 双模传播
  - FiLM 适配
- **输出**: `results/tables/comprehensive_ablation_results_parallel.csv`

### 3.2 并行消融实验
- **脚本**: `src/experiments/run_comprehensive_ablation_parallel.py`
- **用途**: 使用多GPU并行运行消融实验

### 3.3 通用消融
- **脚本**: `src/experiments/run_universal_ablation.py`
- **用途**: 简化的消融实验流程

## 4. 鲁棒性实验

### 4.1 通用鲁棒性测试
- **脚本**: `src/experiments/run_universal_robustness.py`
- **用途**: 测试模型对不同噪声的鲁棒性
- **噪声类型**: 高斯噪声、脉冲噪声、缺失值
- **输出**: `results/tables/robustness_80_20.csv`

### 4.2 完整噪声鲁棒性测试
- **脚本**: `src/experiments/complete_noise_robustness_test.py`
- **用途**: 更全面的噪声测试
- **特点**: 多种噪声水平，多轮测试

## 5. 效率基准测试

### 5.1 效率基准
- **脚本**: `src/experiments/run_efficiency_benchmark.py`
- **用途**: 测量推理延迟和能耗
- **指标**: FLOPs, 参数量, 延迟, 能耗
- **输出**: `results/tables/efficiency_benchmark_results_enhanced.csv`

### 5.2 增强效率测试
- **脚本**: `src/experiments/run_efficiency_benchmark_enhanced.py`
- **用途**: 更详细的效率分析

## 6. 超参数搜索

### 6.1 超参数搜索
- **脚本**: `src/experiments/run_hyperparameter_search.py`
- **用途**: 随机搜索最优超参数
- **搜索空间**: learning_rate, batch_size, hidden_dim, num_experts, top_k

### 6.2 并行超参数搜索
- **脚本**: `src/experiments/run_hyperparameter_search_parallel.py`
- **用途**: 并行化超参数搜索

### 6.3 Top-K 和温度参数扫描
- **脚本**: `src/baselines/benchmark_topk_tau.py`
- **用途**: 扫描 top_k 和 tau 参数的影响
- **输出**: `results/tables/tau_sweep.csv`, `results/tables/topk_sweep.csv`

## 7. 调度实验

### 7.1 调度闭包映射
- **脚本**: `src/experiments/run_dispatch_closure_mapping_decision_4090.py`
- **用途**: 风电调度决策实验
- **输出**: `results/tables/dispatch_rolling_80_20.csv`

### 7.2 IEEE 24节点调度
- **脚本**: `src/experiments/run_network_constrained_dispatch_ieee24.py`
- **用途**: 网络约束下的调度优化

## 8. 基线模型实验

### 8.1 DLinear
- **脚本**: `src/baselines/run_dlinear.py`, `src/baselines/run_dlinear_fixed.py`
- **用途**: DLinear 基线

### 8.2 SOTA Transformer
- **脚本**: `src/baselines/run_latest_sota.py`
- **包含**: PatchTST, iTransformer, TimeMixer

### 8.3 GBDT
- **脚本**: `src/baselines/run_gbdt.py`, `src/baselines/run_gbdt_complete_fixed.py`
- **用途**: LightGBM/XGBoost 基线

### 8.4 经典机器学习
- **脚本**: `src/baselines/classical_baselines.py`
- **包含**: ARIMA, SVR, Random Forest

### 8.5 传统基线
- **脚本**: `src/baselines/traditional_baselines_experiment.py`
- **用途**: 传统时间序列方法

## 9. 元启发式算法实验

### 9.1 COA-BiLSTM
- **脚本**: `src/baselines/coa_bilstm_experiment.py`
- **用途**: 冠豪猪优化算法 + BiLSTM

### 9.2 BWO-CNN
- **脚本**: `src/baselines/bwo_cnn_experiment.py`
- **用途**: 白鲸优化算法 + CNN

### 9.3 BWO-SVR
- **脚本**: `src/baselines/bwo_svr_experiment.py`
- **用途**: 白鲸优化算法 + SVR

### 9.4 完整元启发式实验
- **脚本**: `src/experiments/run_full_coa_bwo_experiments.py`
- **用途**: 运行所有元启发式对比实验

### 9.5 扩展元启发式基线
- **脚本**: `src/experiments/run_extended_metaheuristic_baselines_4090.py`
- **用途**: 扩展的元启发式算法对比

## 10. 专家分析实验

### 10.1 专家物理分析
- **脚本**: `src/experiments/run_expert_physics.py`
- **用途**: 分析专家的物理意义

### 10.2 专家切换分析
- **脚本**: `src/experiments/analyze_expert_switching.py`
- **用途**: 分析专家切换模式

## 11. 绘图脚本

### 11.1 数据概览
- **脚本**: `src/experiments/fig_data_overview_2x2.py`
- **输出**: 四数据集概览图

### 11.2 多域性能
- **脚本**: `src/experiments/fig_multi_domain.py`, `src/experiments/plot_multi_domain.py`
- **输出**: 多域性能对比图

### 11.3 鲁棒性
- **脚本**: `src/experiments/fig_robustness_2x2.py`, `src/experiments/fig_robustness_scissor_2x2_v2.py`
- **输出**: 鲁棒性测试图

### 11.4 时序对比
- **脚本**: `src/experiments/fig_timeseries_compare_2x2.py`
- **输出**: 预测时序对比图

### 11.5 散点图
- **脚本**: `src/experiments/fig_scatter_pred_true_2x2.py`
- **输出**: 预测-真实值散点图

### 11.6 专家热力图
- **脚本**: `src/experiments/plot_expert_heatmap.py`, `src/experiments/plot_expert_heatmap_enhanced.py`
- **输出**: 专家激活热力图

### 11.7 架构图
- **脚本**: `src/experiments/plot_architecture_detailed.py`
- **输出**: 模型架构图

## 12. 其他实验

### 12.1 模型评估优化
- **脚本**: `src/experiments/evaluate_ultra_lsnt_optimized.py`
- **用途**: 优化的模型评估

### 12.2 显著性测试
- **脚本**: `src/experiments/run_significance.py`, `src/experiments/run_universal_significance.py`
- **用途**: 统计显著性检验

### 12.3 对比分析
- **脚本**: `src/experiments/compare_dispatch_metrics.py`, `src/experiments/compare_robustness_real.py`
- **用途**: 不同方法对比分析

## 运行顺序建议

对于复现论文结果，建议按以下顺序运行：

1. **数据准备**
   ```bash
   python src/data_preprocess.py
   ```

2. **基线对比**
   ```bash
   python src/experiments/run_multi_domain_baselines.py
   ```

3. **主模型训练**
   ```bash
   python src/experiments/train_ultra_lsnt_stable.py
   ```

4. **消融实验**
   ```bash
   python src/experiments/run_comprehensive_ablation.py
   ```

5. **鲁棒性测试**
   ```bash
   python src/experiments/run_universal_robustness.py
   ```

6. **效率测试**
   ```bash
   python src/experiments/run_efficiency_benchmark.py
   ```

7. **生成图表**
   ```bash
   bash scripts/clean_and_plot.sh
   ```

## 批量运行

使用提供的shell脚本批量运行：

```bash
# 运行所有主要实验
bash scripts/run_all.sh

# 完整实验流程
bash scripts/run_complete.sh

# 仅绘图
bash scripts/clean_and_plot.sh
```
