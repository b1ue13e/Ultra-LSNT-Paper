"""
完整消融研究 - 并行增强版：使用多进程并行测试不同配置
增加训练轮数至10个epoch以提高精度
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

def train_and_evaluate_single_config(config):
    """训练和评估单个配置（用于并行执行）"""
    k, h_dim, dropout, lr, data, input_dim, device_id = config
    
    # 设置GPU设备（如果可用）
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
    
    print(f"进程 {os.getpid()}: 测试配置 k={k}, h_dim={h_dim}, dropout={dropout}, lr={lr} on {device}")
    
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    
    # 训练配置 - 增加训练轮数至10个epoch
    train_config = TrainConfig(
        batch_size=128, 
        epochs=10,  # 增加训练轮数以提高精度
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
        epoch_loss = 0
        batch_count = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        if epoch % 3 == 0:  # 每3个epoch打印一次进度
            print(f"  进程 {os.getpid()}: epoch {epoch+1}/{train_config.epochs}, loss: {epoch_loss/batch_count:.4f}")
    
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
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "Kernel": k,
        "HiddenDim": h_dim,
        "Dropout": dropout,
        "LearningRate": lr,
        "R2": metrics['R2'],
        "RMSE": metrics['RMSE'],
        "MAE": metrics['MAE'],
        "ProcessID": os.getpid(),
        "Device": str(device)
    }

def run_comprehensive_ablation_parallel():
    print("启动并行完整消融研究 (多组件测试)...")
    print(f"可用GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    # 加载数据（在主进程中加载一次，避免重复加载）
    data, _ = load_csv_data('wind_final.csv', 'power')
    input_dim = data.shape[1]
    
    # 组件参数网格
    kernels = [3, 5, 7, 9, 11]  # 卷积核尺寸
    hidden_dims = [128, 256, 512]  # 隐藏维度
    dropouts = [0.0, 0.1, 0.2, 0.3]  # Dropout率
    learning_rates = [1e-4, 3e-4, 1e-3]  # 学习率
    
    # 生成所有配置组合
    configs = []
    config_id = 0
    for k in kernels:
        for h_dim in hidden_dims[:2]:  # 限制组合数量
            for dropout in dropouts[:2]:
                for lr in learning_rates[:2]:
                    # 轮询分配GPU设备
                    device_id = config_id % max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 0
                    configs.append((k, h_dim, dropout, lr, data, input_dim, device_id))
                    config_id += 1
    
    print(f"总配置数: {len(configs)}")
    print(f"开始并行执行（工作进程数: {os.cpu_count()}）...")
    
    results = []
    
    # 使用ProcessPoolExecutor并行执行
    # 注意：由于PyTorch和CUDA的多进程问题，这里使用进程池但限制每个进程使用不同的GPU
    max_workers = min(len(configs), os.cpu_count(), 8)  # 限制最大工作进程数
    print(f"使用 {max_workers} 个工作进程")
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_config = {executor.submit(train_and_evaluate_single_config, config): config for config in configs}
        
        # 收集结果
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                print(f"完成配置: k={result['Kernel']}, h_dim={result['HiddenDim']}, "
                      f"dropout={result['Dropout']}, lr={result['LearningRate']:.4f}, "
                      f"R2={result['R2']:.4f}, 进程: {result['ProcessID']}")
            except Exception as e:
                print(f"配置执行失败: {config}, 错误: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"并行消融研究完成，耗时: {elapsed_time:.2f} 秒")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("comprehensive_ablation_results_parallel.csv", index=False)
    print("\n并行完整消融结果已保存: comprehensive_ablation_results_parallel.csv")
    
    # 打印最佳配置
    if len(results) > 0:
        best_idx = df['R2'].idxmax()
        best_config = df.loc[best_idx]
        print("\n=== 最佳配置 ===")
        print(f"卷积核尺寸: {best_config['Kernel']}")
        print(f"隐藏维度: {best_config['HiddenDim']}")
        print(f"Dropout率: {best_config['Dropout']}")
        print(f"学习率: {best_config['LearningRate']:.6f}")
        print(f"R2分数: {best_config['R2']:.4f}")
        print(f"RMSE: {best_config['RMSE']:.4f}")
    
    return results

if __name__ == '__main__':
    run_comprehensive_ablation_parallel()
