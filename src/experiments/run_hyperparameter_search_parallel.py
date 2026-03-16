"""
超参数优化 - 并行增强版：使用并行随机搜索，增加训练轮数至10个epoch
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)

def train_and_evaluate_single_trial(params_data):
    """训练和评估单个超参数试验（用于并行执行）"""
    params, data, ts_config, trial_id, device_id = params_data
    
    # 设置GPU设备（如果可用）
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
    
    print(f"试验 {trial_id}: 在 {device} 上执行")
    print(f"  参数: {params}")
    
    try:
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
            epochs=10,  # 增加至10个epoch
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # 创建数据加载器
        train_loader, val_loader, _, scaler = create_dataloaders(
            data, ts_config, train_config,
            train_ratio=0.6, val_ratio=0.2
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
        train_losses = []
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
            
            train_losses.append(epoch_loss / batch_count)
            
            if epoch % 3 == 0:  # 每3个epoch打印一次进度
                print(f"    试验 {trial_id}: epoch {epoch+1}/{train_config.epochs}, "
                      f"loss: {train_losses[-1]:.4f}")
        
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
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = params.copy()
        result['R2'] = metrics['R2']
        result['RMSE'] = metrics['RMSE']
        result['MAE'] = metrics['MAE']
        result['FinalLoss'] = train_losses[-1] if train_losses else 0
        result['TrialID'] = trial_id
        result['ProcessID'] = os.getpid()
        result['Device'] = str(device)
        
        print(f"  试验 {trial_id}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"  试验 {trial_id} 失败: {e}")
        # 返回包含错误信息的占位结果
        error_result = params.copy()
        error_result['R2'] = -1.0
        error_result['RMSE'] = 9999.0
        error_result['MAE'] = 9999.0
        error_result['FinalLoss'] = 9999.0
        error_result['TrialID'] = trial_id
        error_result['ProcessID'] = os.getpid()
        error_result['Device'] = str(device)
        error_result['Error'] = str(e)
        return error_result

def run_hyperparameter_search_parallel():
    print("启动并行超参数优化搜索...")
    print(f"可用GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
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
    num_random_trials = 30  # 增加试验次数至30
    
    print(f"执行并行随机搜索 ({num_random_trials} 次试验)...")
    
    # 准备所有试验参数
    trials_data = []
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
        
        # 轮询分配GPU设备
        device_id = trial % max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 0
        trials_data.append((params, data, ts_config, trial + 1, device_id))
    
    # 并行执行试验
    start_time = time.time()
    
    max_workers = min(num_random_trials, os.cpu_count(), 8)  # 限制最大工作进程数
    print(f"使用 {max_workers} 个工作进程进行并行搜索")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_trial = {executor.submit(train_and_evaluate_single_trial, trial_data): trial_data 
                          for trial_data in trials_data}
        
        # 收集结果
        completed_trials = 0
        for future in as_completed(future_to_trial):
            trial_data = future_to_trial[future]
            trial_id = trial_data[3]
            
            try:
                result = future.result()
                random_search_results.append(result)
                completed_trials += 1
                
                print(f"完成试验 {trial_id}/{num_random_trials}: "
                      f"R2={result['R2']:.4f}, RMSE={result['RMSE']:.4f}")
                
            except Exception as e:
                print(f"试验 {trial_id} 执行异常: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"并行随机搜索完成，耗时: {elapsed_time:.2f} 秒")
    print(f"完成试验数: {completed_trials}/{num_random_trials}")
    
    # 保存随机搜索结果
    if random_search_results:
        df_random = pd.DataFrame(random_search_results)
        df_random.to_csv("hyperparameter_random_search_results_parallel.csv", index=False)
        print(f"\n并行随机搜索结果已保存: hyperparameter_random_search_results_parallel.csv")
        
        # 过滤掉失败的结果（R2为负值）
        df_success = df_random[df_random['R2'] >= 0]
        if len(df_success) > 0:
            # 找到最佳参数
            best_idx = df_success['R2'].idxmax()
            best_params = df_success.loc[best_idx].to_dict()
            
            print("\n=== 最佳参数 ===")
            for key, value in best_params.items():
                if key not in ['TrialID', 'ProcessID', 'Device', 'Error']:
                    print(f"  {key}: {value}")
            
            # 网格搜索（在最佳参数附近）
            print("\n=== 在最佳参数附近进行并行网格搜索 ===")
            grid_search_results = []
            
            # 定义局部网格
            if best_params['hidden_dim'] == 128:
                hidden_grid = [128, 256]
            elif best_params['hidden_dim'] == 512:
                hidden_grid = [256, 512]
            else:
                hidden_grid = [128, 256, 512]
            
            # 准备网格搜索试验
            grid_trials = []
            grid_trial_id = 1
            
            for hidden_dim in hidden_grid:
                for lr in [best_params['learning_rate'] * 0.5, best_params['learning_rate'], best_params['learning_rate'] * 2]:
                    if lr < 1e-5 or lr > 1e-2:  # 限制范围
                        continue
                        
                    for batch_size in [best_params['batch_size'] // 2, best_params['batch_size'], best_params['batch_size'] * 2]:
                        if batch_size < 32 or batch_size > 512:
                            continue
                            
                        params = best_params.copy()
                        params['hidden_dim'] = hidden_dim
                        params['learning_rate'] = lr
                        params['batch_size'] = batch_size
                        
                        device_id = grid_trial_id % max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 0
                        grid_trials.append((params, data, ts_config, f"G{grid_trial_id}", device_id))
                        grid_trial_id += 1
            
            print(f"执行 {len(grid_trials)} 个网格搜索试验...")
            
            # 并行执行网格搜索
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_grid = {executor.submit(train_and_evaluate_single_trial, grid_trial): grid_trial 
                                 for grid_trial in grid_trials}
                
                for future in as_completed(future_to_grid):
                    grid_trial = future_to_grid[future]
                    trial_id = grid_trial[3]
                    
                    try:
                        result = future.result()
                        grid_search_results.append(result)
                        print(f"  网格试验 {trial_id}: R2={result['R2']:.4f}, RMSE={result['RMSE']:.4f}")
                    except Exception as e:
                        print(f"  网格试验 {trial_id} 失败: {e}")
            
            # 保存网格搜索结果
            if grid_search_results:
                df_grid = pd.DataFrame(grid_search_results)
                df_grid.to_csv("hyperparameter_grid_search_results_parallel.csv", index=False)
                print(f"\n并行网格搜索结果已保存: hyperparameter_grid_search_results_parallel.csv")
                
                # 找到最终最佳参数
                df_grid_success = df_grid[df_grid['R2'] >= 0]
                if len(df_grid_success) > 0:
                    final_best_idx = df_grid_success['R2'].idxmax()
                    final_best_params = df_grid_success.loc[final_best_idx].to_dict()
                    
                    print("\n=== 最终最佳参数 ===")
                    for key, value in final_best_params.items():
                        if key not in ['TrialID', 'ProcessID', 'Device', 'Error']:
                            print(f"  {key}: {value}")
                    
                    # 保存最佳配置
                    with open("best_hyperparameters_parallel.txt", "w") as f:
                        f.write("Ultra-LSNT 并行搜索最佳超参数配置\n")
                        f.write("=" * 50 + "\n\n")
                        for key, value in final_best_params.items():
                            if key not in ['R2', 'RMSE', 'MAE', 'FinalLoss', 'TrialID', 'ProcessID', 'Device', 'Error']:
                                f.write(f"{key}: {value}\n")
                        
                        f.write(f"\n验证集性能:\n")
                        f.write(f"  R2: {final_best_params['R2']:.4f}\n")
                        f.write(f"  RMSE: {final_best_params['RMSE']:.4f}\n")
                        f.write(f"  MAE: {final_best_params['MAE']:.4f}\n")
                        f.write(f"  最终训练损失: {final_best_params['FinalLoss']:.4f}\n")
                    
                    print("\n并行最佳配置已保存到: best_hyperparameters_parallel.txt")
        else:
            print("警告: 没有成功的试验结果")
    
    print("\n并行超参数优化完成!")
    return random_search_results

if __name__ == '__main__':
    run_hyperparameter_search_parallel()
