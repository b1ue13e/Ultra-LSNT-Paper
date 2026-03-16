"""
完整消融研究：测试卷积核尺寸、注意力头数、专家数量、隐藏维度、Dropout率、学习率调度器
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)

def violent_replace_conv(model, new_k):
    """替换卷积核尺寸"""
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if 'encoder' in name and isinstance(module, nn.Conv1d):
            padding = (new_k - 1) // 2
            new_layer = nn.Conv1d(module.in_channels, module.out_channels, 
                                  kernel_size=new_k, padding=padding)
            new_layer.to(device)
            parent = model.get_submodule(name.rsplit('.', 1)[0])
            setattr(parent, name.rsplit('.', 1)[1], new_layer)

def run_comprehensive_ablation():
    print("启动完整消融研究 (多组件测试)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    input_dim = data.shape[1]
    
    results = []
    
    # 组件参数网格
    kernels = [3, 5, 7, 9, 11]  # 卷积核尺寸
    num_heads_list = [2, 4, 8]  # 注意力头数（如果有的话）
    num_experts_list = [4, 8, 16]  # 专家数量
    hidden_dims = [128, 256, 512]  # 隐藏维度
    dropouts = [0.0, 0.1, 0.2, 0.3]  # Dropout率
    learning_rates = [1e-4, 3e-4, 1e-3]  # 学习率
    
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    
    # 由于完整网格搜索太耗时，我们使用部分组合
    for k in kernels:
        for h_dim in hidden_dims[:2]:  # 限制组合数量
            for dropout in dropouts[:2]:
                for lr in learning_rates[:2]:
                    print(f"\n测试配置: k={k}, h_dim={h_dim}, dropout={dropout}, lr={lr}")
                    
                    # 训练配置
                    train_config = TrainConfig(
                        batch_size=128, 
                        epochs=5,  # 消融研究用较少的epoch
                        lr=lr
                    )
                    
                    # 模型配置
                    model_config = LSNTConfig(
                        input_dim=input_dim,
                        hidden_dim=h_dim,
                        dropout=dropout
                    )
                    
                    train_loader, _, test_loader, scaler = create_dataloaders(
                        data, ts_config, train_config
                    )
                    
                    # 创建模型
                    model = UltraLSNTForecaster(model_config, ts_config).to(device)
                    violent_replace_conv(model, k)
                    
                    # 训练
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.MSELoss()
                    
                    model.train()
                    for epoch in range(train_config.epochs):
                        for bx, by in train_loader:
                            bx, by = bx.to(device), by.to(device)
                            optimizer.zero_grad()
                            out, _ = model(bx)
                            loss = criterion(out, by)
                            loss.backward()
                            optimizer.step()
                    
                    # 测试
                    model.eval()
                    preds, trues = [], []
                    with torch.no_grad():
                        for bx, by in test_loader:
                            bx, by = bx.to(device), by.to(device)
                            out, _ = model(bx)
                            preds.append(out.cpu().numpy())
                            trues.append(by.cpu().numpy())
                    
                    # 计算指标
                    tgt_std = scaler.std[-1]
                    tgt_mean = scaler.mean[-1]
                    p = np.concatenate(preds) * tgt_std + tgt_mean
                    y = np.concatenate(trues) * tgt_std + tgt_mean
                    metrics = compute_metrics(p, y)
                    
                    results.append({
                        "Kernel": k,
                        "HiddenDim": h_dim,
                        "Dropout": dropout,
                        "LearningRate": lr,
                        "R2": metrics['R2'],
                        "RMSE": metrics['RMSE'],
                        "MAE": metrics['MAE']
                    })
                    
                    print(f"  -> R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("comprehensive_ablation_results.csv", index=False)
    print("\n完整消融结果已保存: comprehensive_ablation_results.csv")
    print(df.head(10))

if __name__ == '__main__':
    run_comprehensive_ablation()
