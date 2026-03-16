import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig,
    UltraLSNTForecaster, LSNTConfig, compute_metrics
)

# 暴力替换 Conv1d (复用你之前的逻辑) - 修正设备一致性问题
def violent_replace_conv(model, new_k):
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if 'encoder' in name and isinstance(module, nn.Conv1d):
            padding = (new_k - 1) // 2
            new_layer = nn.Conv1d(module.in_channels, module.out_channels, kernel_size=new_k, padding=padding).to(device)
            # 确保权重设备与模型一致
            parent = model.get_submodule(name.rsplit('.', 1)[0])
            setattr(parent, name.rsplit('.', 1)[1], new_layer)
            # 打印替换信息
            print(f"      🔧 已将 {name} 替换为 kernel={new_k} (设备: {device})")

DATASETS = [
    ("Wind (CN)", "wind_final.csv", "power"),
    ("Air Quality", "air_quality_ready.csv", "AQI") 
    # 为了节省时间，这里只跑这两个差异最大的代表：能源 vs 环境
]

KERNELS = [3, 7, 15] # 小、中、大三种感受野

def run_universal_ablation():
    print("启动全领域消融实验 (Kernel Sensitivity)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    for name, path, target in DATASETS:
        print(f"\nDataset: {name}")
        if not os.path.exists(path): continue
        
        data, _ = load_csv_data(path, target)
        input_dim = data.shape[1]
        ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target=target)
        train_config = TrainConfig(batch_size=512, epochs=10, num_workers=4) # RTX 4090适配：增大批量，增加数据加载线程
        
        train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
        
        for k in KERNELS:
            print(f"   Testing Kernel Size = {k} ...")
            
            # 初始化并替换
            model = UltraLSNTForecaster(LSNTConfig(input_dim=input_dim), ts_config).to(device)
            violent_replace_conv(model, k)
            model = model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
            
            # 测试
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    out, _ = model(bx)
                    preds.append(out.cpu().numpy())
                    trues.append(by.cpu().numpy())
            
            # 计算 R2
            tgt_std = scaler.std[-1]
            tgt_mean = scaler.mean[-1]
            p = np.concatenate(preds) * tgt_std + tgt_mean
            y = np.concatenate(trues) * tgt_std + tgt_mean
            r2 = compute_metrics(p, y)['R2']
            
            print(f"      -> R2: {r2:.4f}")
            results.append({"Dataset": name, "Kernel": k, "R2": r2})

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("universal_ablation_results.csv", index=False)
    print("\n消融结果已保存: universal_ablation_results.csv")
    print(df)

if __name__ == '__main__':
    run_universal_ablation()