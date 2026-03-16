# 项目结构详情

```
Ultra-LSNT-Paper/
├── README.md                           # 主说明文档
├── QUICKSTART.md                       # 快速开始指南
├── EXPERIMENTS.md                      # 实验清单
├── DATA.md                             # 数据集说明
├── STRUCTURE.md                        # 本文件
├── requirements.txt                    # Python依赖
├── LICENSE                             # MIT许可证
├── .gitignore                          # Git忽略配置
│
├── src/                                # 源代码 (57 files)
│   ├── models/                         # 核心模型代码 (4 files)
│   │   ├── ultra_lsnt_v4.py           # Ultra-LSNT v4.0 主模型
│   │   ├── ultra_lsnt_timeseries.py   # 时序训练框架
│   │   ├── ultra_lsnt_lite.py         # 轻量版模型
│   │   └── ultra_lsnt_linear_branch.py # 线性分支实现
│   │
│   ├── baselines/                      # 基线模型 (14 files)
│   │   ├── run_dlinear.py             # DLinear
│   │   ├── run_dlinear_fixed.py       # DLinear (fixed)
│   │   ├── run_latest_sota.py         # PatchTST, iTransformer, TimeMixer
│   │   ├── run_gbdt.py                # LightGBM/XGBoost
│   │   ├── run_gbdt_complete_fixed.py # GBDT (完整版)
│   │   ├── classical_baselines.py     # ARIMA, SVR
│   │   ├── traditional_baselines_experiment.py
│   │   ├── quick_baselines_experiment.py
│   │   ├── run_mamba_auditable_suite.py
│   │   ├── run_ssa_elm_auditable_suite.py
│   │   ├── coa_algorithm.py           # 冠豪猪优化算法
│   │   ├── coa_bilstm_experiment.py   # COA-BiLSTM
│   │   ├── bwo_algorithm.py           # 白鲸优化算法
│   │   ├── bwo_cnn_experiment.py      # BWO-CNN
│   │   └── bwo_svr_experiment.py      # BWO-SVR
│   │
│   ├── experiments/                    # 实验脚本 (32 files)
│   │   # 核心实验
│   │   ├── run_multi_domain_baselines.py
│   │   ├── run_ultra_lsnt_wind_cn_real.py
│   │   ├── train_ultra_lsnt_stable.py
│   │   ├── train_ultra_lsnt_enhanced.py
│   │   # 消融实验
│   │   ├── run_comprehensive_ablation.py
│   │   ├── run_comprehensive_ablation_parallel.py
│   │   ├── run_universal_ablation.py
│   │   # 鲁棒性实验
│   │   ├── run_universal_robustness.py
│   │   ├── complete_noise_robustness_test.py
│   │   # 效率实验
│   │   ├── run_efficiency_benchmark.py
│   │   ├── run_efficiency_benchmark_enhanced.py
│   │   # 超参数搜索
│   │   ├── run_hyperparameter_search.py
│   │   ├── run_hyperparameter_search_parallel.py
│   │   # 调度实验
│   │   ├── run_dispatch_closure_mapping_decision_4090.py
│   │   ├── run_network_constrained_dispatch_ieee24.py
│   │   # 元启发式
│   │   ├── run_full_coa_bwo_experiments.py
│   │   ├── run_extended_metaheuristic_baselines_4090.py
│   │   # 专家分析
│   │   ├── run_expert_physics.py
│   │   # 评估
│   │   ├── evaluate_ultra_lsnt_optimized.py
│   │   ├── evaluate_multi_domain.py
│   │   ├── unified_experiment_framework.py
│   │   ├── run_all_experiments_master.py
│   │   # 绘图脚本
│   │   ├── fig_data_overview_2x2.py
│   │   ├── fig_robustness_2x2.py
│   │   ├── fig_scatter_pred_true_2x2.py
│   │   ├── fig_timeseries_compare_2x2.py
│   │   ├── fig_overall_performance_windcn.py
│   │   ├── fig_noise_protocol_schematic.py
│   │   ├── plot_results.py
│   │   ├── plot_multi_domain.py
│   │   ├── plot_expert_heatmap.py
│   │   └── plot_architecture_detailed.py
│   │
│   ├── data_preprocess.py              # 数据预处理
│   ├── unified_split_utils.py          # 数据分割工具
│   ├── noise_utils.py                  # 噪声注入工具
│   ├── dispatch_mapping_utils.py       # 调度映射工具
│   ├── wind_dispatch_model.py          # 风电调度模型
│   └── windcn_audit_common.py          # 审计工具
│
├── data/                               # 数据集 (7 files)
│   ├── raw/                            # 原始数据 (空，需自行下载)
│   └── processed/                      # 处理后的数据
│       ├── wind_final.csv              # 中国风电数据 (5.0 MB)
│       ├── wind_us.csv                 # 美国风电数据 (10.0 MB)
│       ├── air_quality_ready.csv       # 空气质量数据 (20.0 MB)
│       ├── gefcom_ready.csv            # GEFCom负荷数据 (1.0 MB)
│       ├── processed_wind.csv          # 处理后风电数据 (13.3 MB)
│       ├── wind_expert_activation.csv  # 风电专家激活日志
│       └── air_quality_expert_activation.csv
│
├── results/                            # 实验结果 (26 files)
│   ├── experiments/                    # 实验输出 (空)
│   ├── figures/                        # 论文图表 (10 PDFs)
│   │   ├── Figure_3_Final.pdf          # 主结果图
│   │   ├── Figure_4_Final.pdf          # 消融实验图
│   │   ├── Figure_5_Final.pdf          # 效率对比图
│   │   ├── fig_multi_domain.pdf        # 多域性能
│   │   ├── fig_efficiency.pdf          # 效率分析
│   │   ├── fig_ablation.pdf            # 消融结果
│   │   ├── fig_robustness_stress_test.pdf
│   │   ├── fig_dispatch_economics.pdf
│   │   ├── fig_dm_matrix_2x2.pdf
│   │   └── tau_sweep_ultra_pareto_v3.pdf
│   └── tables/                         # 结果表格 (16 CSVs)
│       ├── unified_results_final.csv
│       ├── unified_results_final_fixed.csv
│       ├── robustness_80_20.csv
│       ├── comprehensive_ablation_results_parallel.csv
│       ├── all_models_clean_80_20.csv
│       ├── metaheuristic_best_configs.csv
│       ├── metaheuristic_metrics_all_datasets.csv
│       ├── metaheuristic_latency_energy.csv
│       ├── dispatch_rolling_80_20.csv
│       ├── dispatch_rts24_80_20.csv
│       ├── efficiency_benchmark_results_enhanced.csv
│       ├── artifact_full_coverage_20260211.csv
│       ├── coa_bwo_search_trace.csv
│       ├── tau_sweep.csv
│       ├── topk_sweep.csv
│       └── latency_fixed.csv
│
├── scripts/                            # Shell脚本 (6 files)
│   ├── run_all.sh                      # 运行所有实验
│   ├── run_complete.sh                 # 完整实验流程
│   ├── run_perfect.sh                  # 完美实验流程
│   ├── run_all_fixed.sh                # 修复版运行脚本
│   ├── run_all_plots_final.sh          # 生成所有图表
│   └── clean_and_plot.sh               # 清理和绘图
│
├── paper/                              # 论文文件 (5 files)
│   ├── main_PI-MoE_final_AE_ready.tex  # 主LaTeX源文件
│   ├── supplementary_material.tex      # 补充材料
│   ├── cas-refs.bib                    # 参考文献
│   ├── main_PI-MoE_final_AE_ready_build.pdf
│   └── ultra_lsnt_architecture_tikz_fix4.pdf
│
└── docs/                               # 文档 (7 files)
    ├── TECHNICAL_REPORT.md
    ├── EXPERIMENTAL_RESULTS_COMPREHENSIVE.md
    ├── COMPLETE_EXECUTION_SUMMARY.md
    ├── FINAL_REPORT.md
    ├── ALL_EXPERIMENTS_AND_CODE_SUMMARY.md
    ├── DETAILED_EXPERIMENT_SUMMARY_ANALYSIS.md
    └── EXPERIMENT_RESULTS_MASTER_REPORT_20260303.md
```

## 文件统计

| 目录 | 文件数 | 说明 |
|------|--------|------|
| src/models | 4 | 核心模型代码 |
| src/baselines | 14 | 基线模型实现 |
| src/experiments | 32 | 实验脚本和绘图 |
| data/processed | 7 | 处理后的数据集 |
| results/figures | 10 | 论文图表(PDF) |
| results/tables | 16 | 实验结果表格(CSV) |
| scripts | 6 | Shell运行脚本 |
| paper | 5 | 论文LaTeX和PDF |
| docs | 7 | 技术报告和文档 |
| **总计** | **101** | **核心文件** |

## 存储占用

| 类别 | 大小 |
|------|------|
| 数据文件 | ~50 MB |
| 图表PDF | ~2 MB |
| 代码文件 | ~1 MB |
| 结果CSV | ~2 MB |
| 论文PDF | ~5 MB |
| 文档 | ~1 MB |
| **总计** | **~60 MB** |

## GitHub上传建议

### 推荐的仓库结构

```
github.com/yourusername/ultra-lsnt/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/           # 代码
├── data/          # 数据 (可用Git LFS)
├── results/       # 结果
├── scripts/       # 脚本
├── paper/         # 论文
└── docs/          # 文档
```

### 使用Git LFS管理大文件

数据文件较大，建议使用Git LFS：

```bash
# 安装Git LFS
git lfs install

# 追踪大文件
git lfs track "data/processed/*.csv"
git lfs track "*.pkl"
git lfs track "*.h5"

# 添加.gitattributes
git add .gitattributes
```

### 上传步骤

```bash
# 1. 创建新仓库
cd f:\Ultra-LSNT-Paper
git init
git remote add origin https://github.com/yourusername/ultra-lsnt.git

# 2. 添加所有文件
git add .

# 3. 提交
git commit -m "Initial commit: Ultra-LSNT paper code and experiments"

# 4. 推送
git push -u origin main
```
