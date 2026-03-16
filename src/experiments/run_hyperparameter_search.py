"""
超参数优化：使用网格搜索和随机搜索优化关键参数
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random

sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)

def train_and_evaluate(params, data, ts_config):
    """训练模型并返回验证集性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 从参数创建配置
    model_config = LSNTConfig(
        input_dim=data.shape[1],
        hidden_dim=params['hidden_dim'],
        num_blocks=params['num_blocks'],
        num_experts=params['num_experts'],
        top_k=params['top_k'],
        dropout=params['dropout']
    )
    
    train_config = TrainConfig(
        batch_size=params['batch_size'],
        epochs=5,  # 超参数搜索用较少的epoch
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # 创建数据加载器
    train_loader, val_loader, _, scaler = create_dataloaders(
        data, ts_config, train_config,
        train_ratio=0.6, val_ratio=0.2  # 更小的训练集用于快速验证
    )
    
    # 创建模型
    model = UltraLSNTForecaster(model_config, ts_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    criterion = nn.MSELoss()
    
    # 训练
    model.train()
    for epoch in range(train_config.epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    
    # 验证
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in val_loader:
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
    
    return metrics['R2'], metrics['RMSE']

def run_hyperparameter_search():
    print("启动超参数优化搜索...")
    
    # 加载数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    
    # 定义超参数搜索空间
    param_grid = {
        'hidden_dim': [128, 256, 512],
        'num_blocks': [2, 3, 4, 5],
        'num_experts': [4, 8, 16],
        'top_k': [2, 4, 6],
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
        'batch_size': [64, 128, 256],
        'weight_decay': [0.0, 0.01, 0.001, 0.0001]
    }
    
    # 随机搜索参数
    random_search_results = []
    num_random_trials = 20  # 随机搜索试验次数
    
    print(f"执行随机搜索 ({num_random_trials} 次试验)...")
    
    for trial in range(num_random_trials):
        # 随机选择参数
        params = {
            'hidden_dim': random.choice(param_grid['hidden_dim']),
            'num_blocks': random.choice(param_grid['num_blocks']),
            'num_experts': random.choice(param_grid['num_experts']),
            'top_k': random.choice(param_grid['top_k']),
            'dropout': random.choice(param_grid['dropout']),
            'learning_rate': random.choice(param_grid['learning_rate']),
            'batch_size': random.choice(param_grid['batch_size']),
            'weight_decay': random.choice(param_grid['weight_decay'])
        }
        
        print(f"\n试验 {trial+1}/{num_random_trials}:")
        print(f"  参数: {params}")
        
        try:
            r2, rmse = train_and_evaluate(params, data, ts_config)
            
            result = params.copy()
            result['R2'] = r2
            result['RMSE'] = rmse
            random_search_results.append(result)
            
            print(f"  结果: R2={r2:.4f}, RMSE={rmse:.4f}")
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    # 保存随机搜索结果
    if random_search_results:
        df_random = pd.DataFrame(random_search_results)
        df_random.to_csv("hyperparameter_random_search_results.csv", index=False)
        print(f"\n随机搜索结果已保存: hyperparameter_random_search_results.csv")
        
        # 找到最佳参数
        best_idx = df_random['R2'].idxmax()
        best_params = df_random.loc[best_idx].to_dict()
        
        print("\n=== 最佳参数 ===")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # 网格搜索（在最佳参数附近）
        print("\n=== 在最佳参数附近进行网格搜索 ===")
        grid_search_results = []
        
        # 定义局部网格
        if best_params['hidden_dim'] == 128:
            hidden_grid = [128, 256]
        elif best_params['hidden_dim'] == 512:
            hidden_grid = [256, 512]
        else:
            hidden_grid = [128, 256, 512]
        
        # 执行局部网格搜索
        for hidden_dim in hidden_grid:
            for lr in [best_params['learning_rate'] * 0.5, best_params['learning_rate'], best_params['learning_rate'] * 2]:
                if lr < 1e-5 or lr > 1e-2:  # 限制范围
                    continue
                    
                params = best_params.copy()
                params['hidden_dim'] = hidden_dim
                params['learning_rate'] = lr
                
                print(f"  测试: hidden_dim={hidden_dim}, lr={lr}")
                
                try:
                    r2, rmse = train_and_evaluate(params, data, ts_config)
                    
                    result = params.copy()
                    result['R2'] = r2
                    result['RMSE'] = rmse
                    grid_search_results.append(result)
                    
                    print(f"    结果: R2={r2:.4f}, RMSE={rmse:.4f}")
                except Exception as e:
                    print(f"    错误: {e}")
                    continue
        
        # 保存网格搜索结果
        if grid_search_results:
            df_grid = pd.DataFrame(grid_search_results)
            df_grid.to_csv("hyperparameter_grid_search_results.csv", index=False)
            print(f"\n网格搜索结果已保存: hyperparameter_grid_search_results.csv")
            
            # 找到最终最佳参数
            final_best_idx = df_grid['R2'].idxmax()
            final_best_params = df_grid.loc[final_best_idx].to_dict()
            
            print("\n=== 最终最佳参数 ===")
            for key, value in final_best_params.items():
                print(f"  {key}: {value}")
            
            # 保存最佳配置
            with open("best_hyperparameters.txt", "w") as f:
                f.write("Ultra-LSNT 最佳超参数配置\n")
                f.write("=" * 40 + "\n\n")
                for key, value in final_best_params.items():
                    if key not in ['R2', 'RMSE']:
                        f.write(f"{key}: {value}\n")
                
                f.write(f"\n验证集性能:\n")
                f.write(f"  R2: {final_best_params['R2']:.4f}\n")
                f.write(f"  RMSE: {final_best_params['RMSE']:.4f}\n")
            
            print("\n最佳配置已保存到: best_hyperparameters.txt")
    
    print("\n超参数优化完成!")

if __name__ == '__main__':
    run_hyperparameter_search()
