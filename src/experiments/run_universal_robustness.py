import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import json

sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)

# 字体设置已移除，使用默认字体以避免警告

# ================= 配置区域 =================
DATASETS = [
    # (显示名称, 文件路径, 目标列, 推荐SeqLen)
    ("Wind (CN)", "wind_final.csv", "power", 96),
    ("Wind (US)", "wind_us.csv", "power (MW)", 96),
    ("GEFCom Load", "gefcom_ready.csv", "load", 96),
    ("Air Quality", "air_quality_ready.csv", "AQI", 96)
]
# ===========================================

def run_universal_robustness():
    print("启动全领域鲁棒性压力测试 (Universal Robustness Test)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备画布 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    for idx, (name, path, target, seq_len) in enumerate(DATASETS):
        print(f"\n正在测试数据集: {name} ({path})...")
        ax = axes[idx]
        
        if not os.path.exists(path):
            print(f"   文件不存在，跳过")
            continue
            
        # 1. 加载数据
        data, _ = load_csv_data(path, target)
        if data is None: continue
        input_dim = data.shape[1]
        
        # 2. 准备 Loader (只用测试集)
        ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=24, target=target)
        # 注意：这里我们现场训练一个快速模型来做基准，或者你可以尝试加载 specific 的权重
        # 为了通用性，这里我们采用"现场快速微调"策略 (Few-shot) 或者直接加载对应文件夹的权重
        # 假设你之前都跑过并保存了，我们尝试加载对应的 checkpoint
        
        # 路径映射
        ckpt_map = {
            "Wind (CN)": "checkpoints_ts/main/best_model.pth",
            "Wind (US)": "checkpoints_ts/exp_wind_us/best_model.pth",
            "GEFCom Load": "checkpoints_ts/exp_gefcom/best_model.pth", # 假设你跑过
            "Air Quality": "checkpoints_ts/exp_air_quality/best_model.pth"
        }
        
        train_config = TrainConfig(batch_size=256)
        _, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
        
        # 3. 初始化模型 - 从配置文件加载正确的配置
        # 确定配置文件路径
        ckpt_path = ckpt_map.get(name, "")
        config_path = ""
        if os.path.exists(ckpt_path):
            config_path = os.path.join(os.path.dirname(ckpt_path), 'model_config.json')
        
        if os.path.exists(config_path):
            print(f"   加载模型配置: {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # 使用配置文件中的参数，但确保input_dim与当前数据匹配
            config_dict['input_dim'] = input_dim
            model_config = LSNTConfig(**config_dict)
            print(f"   使用配置: hidden_dim={model_config.hidden_dim}, num_blocks={model_config.num_blocks}, num_experts={model_config.num_experts}, top_k={model_config.top_k}")
        else:
            print(f"   警告: 未找到配置文件 ({config_path})，使用默认配置")
            model_config = LSNTConfig(input_dim=input_dim)
        
        model = UltraLSNTForecaster(model_config, ts_config).to(device)
        
        # 4. 尝试加载权重
        if os.path.exists(ckpt_path):
            print(f"   加载权重: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            
            # 尝试加载，提供详细的错误信息
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"   权重加载成功 (strict=True)")
            except RuntimeError as e:
                print(f"   警告: 严格加载失败: {e}")
                print(f"   尝试使用 strict=False 加载...")
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print(f"   权重加载成功 (strict=False) - 注意：部分参数未加载，可能影响性能")
                except RuntimeError as e2:
                    print(f"   错误: 宽松加载也失败: {e2}")
                    print(f"   使用随机初始化进行测试")
        else:
            print(f"   警告: 没找到权重 ({ckpt_path})，使用随机初始化进行测试 (结果仅供参考趋势)")
            
        model.eval()
        
        # 4. 噪音测试循环
        r2_scores = []
        for noise in noise_levels:
            preds, trues = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    # 注入噪音
                    noise_tensor = torch.randn_like(bx) * noise
                    out, _ = model(bx + noise_tensor)
                    preds.append(out.cpu().numpy())
                    trues.append(by.cpu().numpy())
            
            # 反归一化
            target_std = scaler.std[-1]
            target_mean = scaler.mean[-1]
            p = np.concatenate(preds) * target_std + target_mean
            y = np.concatenate(trues) * target_std + target_mean
            
            metrics = compute_metrics(p, y)
            r2_scores.append(metrics['R2'])
            print(f"      噪音 {int(noise*100)}%: R2={metrics['R2']:.4f}")
            
        # 5. 绘图
        ax.plot(noise_levels, r2_scores, 'o-', color='#d62728', linewidth=2, label='Ultra-LSNT')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Noise Level ($\sigma$)')
        ax.set_ylabel('R² Score')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=0) # R2最低到0，不显示负数
        
        # 添加衰减率标注
        drop_rate = (r2_scores[0] - r2_scores[-1]) / r2_scores[0] * 100
        ax.text(0.1, 0.1, f'Decay: {drop_rate:.1f}%', transform=ax.transAxes, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('fig_universal_robustness.png')
    print("\n全领域鲁棒性图表已生成: fig_universal_robustness.png")

if __name__ == '__main__':
    run_universal_robustness()