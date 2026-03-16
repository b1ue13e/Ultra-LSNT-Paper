"""
效率基准测试 - 增强版：增加训练轮数至10个epoch，添加多GPU支持
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

def run_efficiency_benchmark_enhanced():
    print("启动增强版效率基准测试...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 加载数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    input_dim = data.shape[1]
    
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=128, epochs=10)  # 增加至10个epoch
    
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
    std_inference_time = np.std(inference_times)
    print(f"平均推理时间: {avg_inference_time:.2f} ms (±{std_inference_time:.2f} ms)")
    
    # 2. 训练时间测试（10个epoch）
    print("\n=== 训练时间测试 (10个epoch) ===")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epoch_times = []
    epoch_losses = []
    
    for epoch in range(train_config.epochs):
        epoch_start = time.perf_counter()
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
        
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        epoch_losses.append(epoch_loss / batch_count)
        
        print(f"  Epoch {epoch+1}/{train_config.epochs}: "
              f"训练时间: {epoch_time:.2f} 秒, "
              f"平均损失: {epoch_losses[-1]:.4f}")
    
    avg_epoch_time = np.mean(epoch_times)
    print(f"平均每个epoch训练时间: {avg_epoch_time:.2f} 秒")
    print(f"总训练时间 ({train_config.epochs}个epoch): {sum(epoch_times):.2f} 秒")
    
    # 3. 内存占用测试
    print("\n=== 内存占用测试 ===")
    
    # 模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {num_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
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
    
    # 4. 多GPU性能测试（如果有多个GPU）
    multi_gpu_results = {}
    if torch.cuda.device_count() > 1:
        print("\n=== 多GPU性能测试 ===")
        
        # 测试数据并行
        model_dp = nn.DataParallel(UltraLSNTForecaster(model_config, ts_config))
        model_dp = model_dp.to(device)
        
        # 多GPU推理时间测试
        model_dp.eval()
        dp_inference_times = []
        
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.perf_counter()
                
                _ = model_dp(bx)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.perf_counter()
                
                dp_inference_times.append((end_time - start_time) * 1000)
        
        avg_dp_inference_time = np.mean(dp_inference_times)
        speedup = avg_inference_time / avg_dp_inference_time if avg_dp_inference_time > 0 else 0
        
        multi_gpu_results = {
            "dp_avg_inference_time_ms": avg_dp_inference_time,
            "single_gpu_inference_time_ms": avg_inference_time,
            "speedup_factor": speedup,
            "gpu_count": torch.cuda.device_count()
        }
        
        print(f"多GPU数据并行平均推理时间: {avg_dp_inference_time:.2f} ms")
        print(f"加速比: {speedup:.2f}x")
    
    # 5. 保存结果
    results = {
        "device": str(device),
        "avg_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
        "avg_epoch_time_sec": avg_epoch_time,
        "total_training_time_sec": sum(epoch_times),
        "epochs": train_config.epochs,
        "model_parameters": num_params,
        "trainable_parameters": trainable_params,
        "cpu_memory_usage_mb": memory_after - memory_before,
        "gpu_memory_usage_mb": gpu_memory_after - gpu_memory_before,
        "batch_size": train_config.batch_size,
        "sequence_length": ts_config.seq_len,
        **multi_gpu_results
    }
    
    df = pd.DataFrame([results])
    df.to_csv("efficiency_benchmark_results_enhanced.csv", index=False)
    print("\n增强版效率基准测试结果已保存: efficiency_benchmark_results_enhanced.csv")
    print(df.to_string())
    
    # 6. 生成详细摘要报告
    with open("efficiency_summary_enhanced.txt", "w") as f:
        f.write("=== Ultra-LSNT 增强版效率基准测试摘要 ===\n\n")
        f.write(f"测试设备: {device}\n")
        f.write(f"训练轮数: {train_config.epochs}\n")
        f.write(f"模型总参数量: {num_params:,}\n")
        f.write(f"可训练参数量: {trainable_params:,}\n")
        f.write(f"平均推理时间: {avg_inference_time:.2f} ms (±{std_inference_time:.2f} ms)\n")
        f.write(f"平均每个epoch训练时间: {avg_epoch_time:.2f} 秒\n")
        f.write(f"总训练时间: {sum(epoch_times):.2f} 秒\n")
        f.write(f"CPU内存增量: {memory_after - memory_before:.1f} MB\n")
        f.write(f"GPU内存增量: {gpu_memory_after - gpu_memory_before:.1f} MB\n")
        
        if multi_gpu_results:
            f.write(f"\n多GPU性能:\n")
            f.write(f"  GPU数量: {multi_gpu_results['gpu_count']}\n")
            f.write(f"  多GPU推理时间: {multi_gpu_results['dp_avg_inference_time_ms']:.2f} ms\n")
            f.write(f"  加速比: {multi_gpu_results['speedup_factor']:.2f}x\n")

if __name__ == '__main__':
    run_efficiency_benchmark_enhanced()
