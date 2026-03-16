"""
效率基准测试：系统测量训练时间、推理时间、内存占用
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import psutil
import gc

sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
    return memory_mb, gpu_memory

def run_efficiency_benchmark():
    print("启动效率基准测试...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    
    # 加载数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    input_dim = data.shape[1]
    
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=128, epochs=5)  # 减少epochs用于基准测试
    
    train_loader, _, test_loader, scaler = create_dataloaders(
        data, ts_config, train_config
    )
    
    model_config = LSNTConfig(input_dim=input_dim)
    model = UltraLSNTForecaster(model_config, ts_config).to(device)
    
    # 1. 推理时间测试
    print("\n=== 推理时间测试 ===")
    model.eval()
    
    # 预热
    with torch.no_grad():
        for bx, _ in train_loader:
            bx = bx.to(device)
            _ = model(bx)
            break
    
    # 基准测试
    inference_times = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.perf_counter()
            
            _ = model(bx)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_inference_time = np.mean(inference_times)
    print(f"平均推理时间: {avg_inference_time:.2f} ms")
    
    # 2. 训练时间测试
    print("\n=== 训练时间测试 ===")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epoch_times = []
    for epoch in range(3):  # 只训练3个epoch用于基准测试
        epoch_start = time.perf_counter()
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1} 训练时间: {epoch_time:.2f} 秒")
    
    avg_epoch_time = np.mean(epoch_times)
    print(f"平均每个epoch训练时间: {avg_epoch_time:.2f} 秒")
    
    # 3. 内存占用测试
    print("\n=== 内存占用测试 ===")
    
    # 模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    
    # 训练时内存占用
    memory_before, gpu_memory_before = get_memory_usage()
    print(f"训练前内存: {memory_before:.1f} MB, GPU内存: {gpu_memory_before:.1f} MB")
    
    # 进行一次训练步骤
    bx, by = next(iter(train_loader))
    bx, by = bx.to(device), by.to(device)
    
    optimizer.zero_grad()
    out, _ = model(bx)
    loss = criterion(out, by)
    loss.backward()
    
    memory_after, gpu_memory_after = get_memory_usage()
    print(f"训练后内存: {memory_after:.1f} MB, GPU内存: {gpu_memory_after:.1f} MB")
    
    # 4. 保存结果
    results = {
        "device": str(device),
        "avg_inference_time_ms": avg_inference_time,
        "avg_epoch_time_sec": avg_epoch_time,
        "model_parameters": num_params,
        "cpu_memory_usage_mb": memory_after - memory_before,
        "gpu_memory_usage_mb": gpu_memory_after - gpu_memory_before,
        "batch_size": train_config.batch_size,
        "sequence_length": ts_config.seq_len,
    }
    
    df = pd.DataFrame([results])
    df.to_csv("efficiency_benchmark_results.csv", index=False)
    print("\n效率基准测试结果已保存: efficiency_benchmark_results.csv")
    print(df.to_string())
    
    # 5. 生成摘要报告
    with open("efficiency_summary.txt", "w") as f:
        f.write("=== Ultra-LSNT 效率基准测试摘要 ===\n\n")
        f.write(f"测试设备: {device}\n")
        f.write(f"模型参数量: {num_params:,}\n")
        f.write(f"平均推理时间: {avg_inference_time:.2f} ms\n")
        f.write(f"平均每个epoch训练时间: {avg_epoch_time:.2f} 秒\n")
        f.write(f"CPU内存增量: {memory_after - memory_before:.1f} MB\n")
        f.write(f"GPU内存增量: {gpu_memory_after - gpu_memory_before:.1f} MB\n")

if __name__ == '__main__':
    run_efficiency_benchmark()
