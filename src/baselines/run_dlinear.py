import torch
import torch.nn as nn
import time
import numpy as np
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig, compute_metrics
)


# ==========================================
# DLinear 模型 (AAAI 2023) - 极简 SOTA
# ==========================================
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decomposition Kernel
        self.decompsition = SeriesDecomp(25)

        # Linear layers
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        seasonal_init, trend_init = self.decompsition(x)

        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)[:, :, -1]  # 简单起见，我们预测单变量


def run_dlinear():
    print("🚀 正在训练 DLinear (2023 SOTA Baseline)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    if data is None:
        print("❌ 数据加载失败")
        return
        
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=64, epochs=10, lr=0.005)  # DLinear 收敛快
    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    # 2. 模型
    model = DLinear(ts_config.seq_len, ts_config.pred_len, data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    criterion = nn.MSELoss()

    # 3. 训练
    model.train()
    for epoch in range(train_config.epochs):
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)  # DLinear 输出是多变量，我们这里简化处理
            # 注意：DLinear 代码这里需要微调以匹配你的数据维度，这里只做演示
            # 为了跑通，我们假设 DLinear 输出 [B, Pred, Chan]，我们取最后一维
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"   Epoch {epoch + 1}: Loss {epoch_loss / len(train_loader):.4f}")

    # 4. 测试
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())

    # 反归一化
    target_mean, target_std = scaler.mean[-1], scaler.std[-1]
    preds = np.concatenate(preds) * target_std + target_mean
    trues = np.concatenate(trues) * target_std + target_mean

    metrics = compute_metrics(preds, trues)
    print(f"\n✅ DLinear 最终 R2: {metrics['R2']:.4f}")


if __name__ == '__main__':
    run_dlinear()