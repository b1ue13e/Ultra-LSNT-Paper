"""
评估Ultra-LSNT-Lite优化版本在测试集上的R²和RMSE
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

sys.path.append(os.getcwd())

# 导入必要模块
try:
    from ultra_lsnt_timeseries import (
        load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
        compute_metrics
    )
    print("✅ 成功导入 ultra_lsnt_timeseries 工具库")
except ImportError as e:
    print(f"❌ 导入超时序列模块时出错: {e}")
    sys.exit(1)

from ultra_lsnt_lite import UltraLSNTLiteForecaster

def evaluate_optimized_model():
    print("=" * 60)
    print("🚀 Ultra-LSNT-Lite优化版本性能评估")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    print("\n📊 加载数据...")
    data_path = 'wind_final.csv'
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        sys.exit(1)
    
    data, _ = load_csv_data(data_path, 'power')
    input_dim = data.shape[1]
    print(f"数据形状: {data.shape}, 输入维度: {input_dim}")
    
    # 配置参数（与效率测试保持一致）
    seq_len = 96
    pred_len = 24
    batch_size = 256
    epochs = 3  # 为了快速评估，使用3个epoch
    
    ts_config = TimeSeriesConfig(seq_len=seq_len, pred_len=pred_len, target='power')
    train_config = TrainConfig(batch_size=batch_size, epochs=epochs)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data, ts_config, train_config
    )
    
    # 优化配置
    model_config = {
        'input_dim': input_dim,
        'hidden_dim': 96,        # 优化：减少隐藏维度
        'seq_len': seq_len,
        'pred_len': pred_len,
        'num_blocks': 1,         # 优化：减少块数
        'num_experts': 2,        # 优化：减少专家数
        'top_k': 1,              # 优化：减少top-k
        'scales': [1, 2]         # 优化：减少尺度数量
    }
    
    print("\n📋 模型配置:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # 创建模型
    model = UltraLSNTLiteForecaster(model_config).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / (1024*1024):.2f} MB")
    
    # 训练模型
    print("\n🏋️ 训练模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        epoch_loss = 0
        batch_count = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, aux_loss = model(bx)
            loss = criterion(out, by) + 0.01 * aux_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        print(f"  Epoch {epoch+1}/{epochs}: 训练时间: {epoch_time:.2f} 秒, 平均损失: {avg_loss:.4f}")
    
    # 在测试集上评估
    print("\n🧪 在测试集上评估...")
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out, _ = model(bx)
            
            all_preds.append(out.cpu().numpy())
            all_trues.append(by.cpu().numpy())
    
    # 合并所有批次
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    print(f"预测形状: {preds.shape}, 真实值形状: {trues.shape}")
    
    # 计算指标（反标准化）
    # 注意：compute_metrics期望原始尺度数据，但我们的数据已经标准化
    # 需要获取目标变量的均值和标准差进行反标准化
    tgt_mean = scaler.mean[-1] if hasattr(scaler, 'mean') else 0
    tgt_std = scaler.std[-1] if hasattr(scaler, 'std') else 1
    
    # 反标准化
    preds_original = preds * tgt_std + tgt_mean
    trues_original = trues * tgt_std + tgt_mean
    
    # 计算指标
    metrics = compute_metrics(preds_original, trues_original)
    
    print("\n📈 测试集性能指标:")
    print(f"  R² 分数: {metrics['R2']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")
    print(f"  MSE: {metrics['MSE']:.6f}")
    
    # 与原始Ultra-LSNT-Lite对比（如果有数据）
    print("\n📊 性能对比:")
    print(f"  优化版Ultra-LSNT-Lite: R² = {metrics['R2']:.6f}, RMSE = {metrics['RMSE']:.6f}")
    
    # 尝试获取原始Ultra-LSNT-Lite的基准数据
    try:
        # 从现有结果文件中读取
        if os.path.exists('gbdt_results_complete.json'):
            import json
            with open('gbdt_results_complete.json', 'r') as f:
                gbdt_data = json.load(f)
                if 'Ultra-LSNT' in gbdt_data:
                    orig_r2 = gbdt_data['Ultra-LSNT'].get('r2', 0)
                    orig_rmse = gbdt_data['Ultra-LSNT'].get('rmse', 0)
                    print(f"  原始Ultra-LSNT: R² = {orig_r2:.6f}, RMSE = {orig_rmse:.6f}")
                    
                    r2_improvement = ((metrics['R2'] - orig_r2) / abs(orig_r2)) * 100 if orig_r2 != 0 else 0
                    rmse_improvement = ((orig_rmse - metrics['RMSE']) / orig_rmse) * 100 if orig_rmse != 0 else 0
                    
                    print(f"  R² 提升: {r2_improvement:+.2f}%")
                    print(f"  RMSE 降低: {rmse_improvement:+.2f}%")
    except Exception as e:
        print(f"  无法获取对比数据: {e}")
    
    # 保存结果
    results = {
        "model": "Ultra-LSNT-Lite-Optimized",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024*1024),
        "R2": float(metrics['R2']),
        "RMSE": float(metrics['RMSE']),
        "MAE": float(metrics['MAE']),
        "MAPE": float(metrics['MAPE']),
        "MSE": float(metrics['MSE']),
        "training_epochs": epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "input_dim": input_dim,
        "hidden_dim": model_config['hidden_dim'],
        "num_blocks": model_config['num_blocks'],
        "num_experts": model_config['num_experts'],
        "top_k": model_config['top_k'],
        "scales": str(model_config['scales'])
    }
    
    # 保存为CSV
    df = pd.DataFrame([results])
    output_csv = "ultra_lsnt_lite_optimized_performance.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 性能评估结果已保存: {output_csv}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("🎉 Ultra-LSNT-Lite优化版本性能评估完成!")
    print("=" * 60)
    print(f"📊 核心指标:")
    print(f"  • R² 分数: {metrics['R2']:.6f}")
    print(f"  • RMSE: {metrics['RMSE']:.6f}")
    print(f"  • 模型大小: {total_params * 4 / (1024*1024):.2f} MB")
    print(f"  • 参数量: {total_params:,}")
    print("=" * 60)
    
    return metrics

if __name__ == '__main__':
    evaluate_optimized_model()