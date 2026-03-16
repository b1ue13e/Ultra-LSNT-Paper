#!/usr/bin/env python3
# GBDT鲁棒性测试 - 彻底修复版本
# 解决 hidden_dim=128 vs 256 的权重不匹配问题
#
# 评估偏差说明：
# 1. LightGBM使用Vanilla（原始）配置，没有专门的时间序列特征工程
# 2. 输入被展平为(batch, seq_len*features)，丢失了序列的时间结构信息
# 3. 这种对比旨在展示深度学习模型对序列结构的利用能力，而非GBDT的最佳性能
# 4. 要获得GBDT的最佳性能，需要添加滞后特征、滚动统计等时间序列特征工程

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# 引入项目模块
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    compute_metrics, UltraLSNTForecaster, LSNTConfig
)
from noise_utils import inject_industrial_noise

# 使用英文标题避免中文字体问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def load_ultra_lsnt_compatible(input_dim, ts_config, device):
    """
    兼容性加载Ultra-LSNT模型
    强制使用检查点的配置 (hidden_dim=128)
    """
    print("🤖 加载 Ultra-LSNT (彻底修复版本)...")
    
    # 加载模型配置
    config_path = 'checkpoints_ts/main/model_config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 确保使用检查点的配置
    print(f"   检查点配置: hidden_dim={config_dict.get('hidden_dim', '未知')}")
    
    # 使用检查点的配置创建模型
    config = LSNTConfig(**config_dict)
    config.input_dim = input_dim
    
    model = UltraLSNTForecaster(config, ts_config).to(device)
    
    # 加载权重
    ckpt_path = 'checkpoints_ts/main/best_model.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到权重文件 {ckpt_path}！")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # 智能权重加载
    print("   🧠 智能权重匹配中...")
    model_state_dict = model.state_dict()
    
    # 首先尝试标准加载
    try:
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state_dict, strict=False)
        print(f"   ✅ 标准加载成功 (strict=False)")
        print(f"     加载 {len(ckpt_state_dict) - len(missing_keys)}/{len(model_state_dict)} 个权重")
        filtered_state_dict = None
        match_ratio = (len(ckpt_state_dict) - len(missing_keys)) / len(model_state_dict)
    except Exception as e1:
        print(f"   ⚠️ 标准加载失败: {e1}")
        print("   尝试手动适配...")
        
        # 手动过滤和适配权重
        filtered_state_dict = {}
        for k, v in ckpt_state_dict.items():
            if k in model_state_dict:
                if model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"      ⚠️ 跳过 {k}: 形状不匹配 {v.shape} vs {model_state_dict[k].shape}")
            else:
                print(f"      ⚠️ 忽略不存在键: {k}")
        
        # 检查权重匹配度
        total_params = len(model_state_dict)
        matched_params = len(filtered_state_dict)
        match_ratio = matched_params / total_params if total_params > 0 else 0
        
        if matched_params > 0:
            # 更严格的阈值：低于90%视为严重不匹配
            if match_ratio < 0.9:
                print(f"   ❌ 关键权重缺失比例过高 ({match_ratio:.1%})，模型可能处于随机状态！")
                print(f"     需要至少90%的权重匹配，但实际只有{matched_params}/{total_params}个参数匹配")
                raise RuntimeError(f"Weight mismatch is too severe to proceed. Only {matched_params}/{total_params} parameters matched (<90%).")
            
            # 加载过滤后的权重
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"   ✅ 适配加载成功: {matched_params}/{total_params} 个权重 ({match_ratio:.1%} 匹配)")
            if match_ratio < 0.95:
                print(f"   ⚠️  注意: 有{total_params - matched_params}个权重使用随机初始化")
        else:
            print("   ⚠️ 使用随机初始化权重")
            # 使用随机权重，没有缺失或多余的键
            missing_keys, unexpected_keys = [], []
            raise RuntimeError("No weights could be loaded. Model is fully random.")
    
    # 打印缺失/多余键（无论哪种加载方式）
    if missing_keys:
        print(f"   警告: 缺少 {len(missing_keys)} 个键")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"     - {key}")
    if unexpected_keys:
        print(f"   警告: 多余 {len(unexpected_keys)} 个键")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"     - {key}")
    
    print(f"   ✅ Ultra-LSNT 就位 (权重匹配度: {match_ratio:.1%})")
    return model

def run_real_battle():
    print("⚔️ [彻底修复版本] LightGBM vs Ultra-LSNT...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 准备数据
    data_path = 'wind_final.csv'
    target = 'power'
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target=target)
    train_config = TrainConfig(batch_size=512)

    data, _ = load_csv_data(data_path, target)
    if data is None:
        print("❌ 数据加载失败")
        return
        
    input_dim = data.shape[1]

    # 创建 DataLoader
    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    # 获取反归一化参数
    target_std = scaler.std[-1]
    target_mean = scaler.mean[-1]

    # 2. 加载Ultra-LSNT（彻底修复版本）
    model_ours = load_ultra_lsnt_compatible(input_dim, ts_config, device)

    # 3. 准备LightGBM
    print("🌲 正在训练 LightGBM (使用全量训练集)...")
    
    def to_numpy(loader):
        X, y = [], []
        for bx, by in loader:
            X.append(bx.reshape(bx.size(0), -1).numpy())
            y.append(by.numpy())
        return np.concatenate(X), np.concatenate(y)

    X_train, y_train = to_numpy(train_loader)
    
    lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42)
    model_lgbm = MultiOutputRegressor(lgbm)
    model_lgbm.fit(X_train, y_train)
    print("   ✅ LightGBM 训练完毕")

    # 4. 真实鲁棒性测试
    noise_type = 'gaussian'
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

    scores_ours = []
    scores_lgbm = []

    print(f"\n🌪️ 开始注入真实噪音 ({noise_type})...")

    for level in noise_levels:
        # 预分配内存以提高效率
        test_samples = len(test_loader.dataset)
        pred_len = 24  # ts_config.pred_len
        preds_o = np.empty((test_samples, pred_len), dtype=np.float32)
        preds_l = np.empty((test_samples, pred_len), dtype=np.float32)
        trues = np.empty((test_samples, pred_len), dtype=np.float32)
        
        idx = 0

        # 遍历测试集
        for bx, by in test_loader:
            bx = bx.to(device)
            batch_size = bx.size(0)

            # 注入噪音
            bx_noisy = inject_industrial_noise(bx, noise_type, level)

            # 1. Ultra-LSNT 推理
            with torch.no_grad():
                out_o, _ = model_ours(bx_noisy)
                preds_o[idx:idx+batch_size] = out_o.cpu().numpy()

            # 2. LightGBM 推理 - 使用展平输入（说明是Vanilla GBDT）
            X_numpy_noisy = bx_noisy.cpu().numpy().reshape(batch_size, -1)
            out_l = model_lgbm.predict(X_numpy_noisy)
            preds_l[idx:idx+batch_size] = out_l

            trues[idx:idx+batch_size] = by.numpy()
            
            # 显式清理内存
            del bx_noisy, out_o
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            idx += batch_size

        # 反归一化
        p_o = preds_o * target_std + target_mean
        p_l = preds_l * target_std + target_mean
        y_true = trues * target_std + target_mean

        # 计算 R2
        r2_o = compute_metrics(p_o, y_true)['R2']
        r2_l = compute_metrics(p_l, y_true)['R2']

        scores_ours.append(r2_o)
        scores_lgbm.append(r2_l)

        print(f"   Noise {level:.2f} | Ultra-LSNT: {r2_o:.4f} vs LightGBM: {r2_l:.4f}")

    # 5. 绘图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(noise_levels, scores_ours, 'r*-', linewidth=3, markersize=12, label='Ultra-LSNT (Ours)')
    plt.plot(noise_levels, scores_lgbm, 'bo--', linewidth=2, markersize=8, label='LightGBM (Baseline)')

    plt.xlabel(f'Noise Level ({noise_type})', fontsize=12, fontweight='bold')
    plt.ylabel('R-Squared ($R^2$)', fontsize=12, fontweight='bold')
    plt.title('Robustness: Ultra-LSNT vs Vanilla GBDT (Fixed Version)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('fig_gbdt_battle_complete_fixed.png', bbox_inches='tight')
    print("\n✅ 彻底修复版本图表已生成: fig_gbdt_battle_complete_fixed.png")
    
    # 保存结果
    results = {
        'noise_levels': noise_levels,
        'ultra_lsnt_scores': scores_ours,
        'lightgbm_scores': scores_lgbm,
        'data_info': {
            'dataset': 'wind_final.csv',
            'input_dim': input_dim,
            'samples': len(data)
        }
    }
    
    with open('gbdt_results_complete.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ 结果已保存: gbdt_results_complete.json")

if __name__ == '__main__':
    run_real_battle()
