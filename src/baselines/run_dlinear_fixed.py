import torch
import torch.nn as nn
import time
import numpy as np
import os
import json
from datetime import datetime
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


def run_dlinear_enhanced():
    """增强版DLinear训练，自动保存结果"""
    print("🚀 正在训练 DLinear (2023 SOTA Baseline) - 增强版...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建结果保存目录
    results_dir = 'dlinear_results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 数据
    data, _ = load_csv_data('wind_final.csv', 'power')
    if data is None:
        print("❌ 数据加载失败")
        return None
    
    print(f"✅ 数据加载成功: {data.shape}")
    
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target='power')
    train_config = TrainConfig(batch_size=64, epochs=10, lr=0.005)  # DLinear 收敛快
    train_loader, _, test_loader, scaler = create_dataloaders(data, ts_config, train_config)

    # 2. 模型
    model = DLinear(ts_config.seq_len, ts_config.pred_len, data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    criterion = nn.MSELoss()

    # 3. 训练（记录训练历史）
    model.train()
    training_history = []
    start_time = time.time()
    
    for epoch in range(train_config.epochs):
        epoch_loss = 0
        batch_count = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / max(batch_count, 1)
        elapsed = time.time() - start_time
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'time': elapsed
        })
        
        print(f"   ⏳ Epoch {epoch + 1}/{train_config.epochs}: Loss {avg_loss:.4f} | 耗时: {elapsed:.1f}s")

    # 4. 测试
    print("   🚀 开始测试 DLinear...")
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
    preds_original = np.concatenate(preds) * target_std + target_mean
    trues_original = np.concatenate(trues) * target_std + target_mean

    metrics = compute_metrics(preds_original, trues_original)
    print(f"\n✅ DLinear 最终 R2: {metrics['R2']:.4f}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE: {metrics['MAE']:.4f}")

    # 5. 保存结果
    result_file = os.path.join(results_dir, f"dlinear_results_{timestamp}.json")
    
    result_data = {
        'model': 'DLinear',
        'data_source': 'wind_final.csv',
        'target': 'power',
        'input_dim': data.shape[1],
        'seq_len': ts_config.seq_len,
        'pred_len': ts_config.pred_len,
        'batch_size': train_config.batch_size,
        'epochs': train_config.epochs,
        'learning_rate': train_config.lr,
        'metrics': metrics,
        'training_history': training_history,
        'predictions_shape': preds_original.shape,
        'timestamp': timestamp,
        'device': str(device),
        'total_training_time': time.time() - start_time
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    
    print(f"💾 详细结果已保存: {result_file}")
    
    # 保存预测值和真实值（前1000个样本）
    sample_file = os.path.join(results_dir, f"dlinear_samples_{timestamp}.npz")
    np.savez_compressed(
        sample_file,
        predictions=preds_original[:1000],
        ground_truth=trues_original[:1000]
    )
    print(f"💾 样本数据已保存: {sample_file}")
    
    # 生成性能报告
    generate_dlinear_report(metrics, training_history, timestamp)
    
    return metrics['R2'], result_file


def generate_dlinear_report(metrics, training_history, timestamp):
    """生成DLinear性能报告"""
    report_lines = []
    report_lines.append("DLinear模型性能分析报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("性能指标:")
    report_lines.append("-" * 40)
    report_lines.append(f"R² 分数: {metrics['R2']:.4f}")
    report_lines.append(f"RMSE: {metrics['RMSE']:.4f}")
    report_lines.append(f"MAE: {metrics['MAE']:.4f}")
    report_lines.append(f"MAPE: {metrics.get('MAPE', 'N/A')}")
    report_lines.append("")
    
    report_lines.append("训练过程:")
    report_lines.append("-" * 40)
    if training_history:
        first_loss = training_history[0]['loss']
        last_loss = training_history[-1]['loss']
        loss_reduction = ((first_loss - last_loss) / first_loss * 100) if first_loss > 0 else 0
        report_lines.append(f"初始损失: {first_loss:.4f}")
        report_lines.append(f"最终损失: {last_loss:.4f}")
        report_lines.append(f"损失下降: {loss_reduction:.1f}%")
        report_lines.append(f"训练轮数: {len(training_history)}")
        report_lines.append(f"总训练时间: {training_history[-1]['time']:.1f}秒")
    report_lines.append("")
    
    report_lines.append("模型配置:")
    report_lines.append("-" * 40)
    report_lines.append("模型架构: DLinear (AAAI 2023)")
    report_lines.append("分解核大小: 25")
    report_lines.append("季节性分量: Linear")
    report_lines.append("趋势分量: Linear")
    report_lines.append("")
    
    report_lines.append("与Ultra-LSNT对比:")
    report_lines.append("-" * 40)
    ultra_lsnt_r2 = 0.7513  # 参考值
    dlinear_r2 = metrics['R2']
    diff = dlinear_r2 - ultra_lsnt_r2
    
    if diff > 0:
        report_lines.append(f"DLinear 优于 Ultra-LSNT: +{diff:.4f}")
        report_lines.append("结论: DLinear在此数据集上表现更好")
    else:
        report_lines.append(f"DLinear 劣于 Ultra-LSNT: {diff:.4f}")
        report_lines.append("结论: Ultra-LSNT在此数据集上保持优势")
    
    # 保存报告
    report_file = f'dlinear_results/dlinear_report_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📝 性能报告已生成: {report_file}")
    
    # 打印报告摘要
    print("\n📋 DLinear性能摘要:")
    print("-" * 40)
    for line in report_lines[:15]:
        print(line)


def compare_with_baselines():
    """与其他基线模型比较"""
    print("\n📊 DLinear与基线模型比较分析")
    print("=" * 60)
    
    # 假设的基线模型性能（可以从实际运行中获取）
    baseline_scores = {
        'LSTM': 0.7123,  # 示例值
        'MLP': 0.6987,   # 示例值
        'Transformer': 0.7234,  # 示例值
        'Ultra-LSNT': 0.7513,
        'DLinear': None  # 将在运行时填充
    }
    
    # 运行DLinear获取实际分数
    dlinear_score, _ = run_dlinear_enhanced()
    if dlinear_score:
        baseline_scores['DLinear'] = dlinear_score
        
        # 生成比较报告
        comparison_lines = []
        comparison_lines.append("基线模型性能比较")
        comparison_lines.append("=" * 60)
        comparison_lines.append(f"比较时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        comparison_lines.append("")
        
        # 按分数排序
        sorted_scores = sorted(baseline_scores.items(), key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
        
        comparison_lines.append("性能排名:")
        comparison_lines.append("-" * 40)
        for i, (model, score) in enumerate(sorted_scores, 1):
            if score is not None:
                comparison_lines.append(f"{i}. {model:15s} : R² = {score:.4f}")
            else:
                comparison_lines.append(f"{i}. {model:15s} : 未运行")
        
        # 保存比较报告
        comparison_file = 'dlinear_results/baseline_comparison.txt'
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(comparison_lines))
        
        print(f"\n📊 基线模型比较报告已保存: {comparison_file}")


if __name__ == '__main__':
    print("🔧 DLinear模型训练系统 (增强版)")
    print("✨ 特性: 自动保存结果、生成报告、性能分析")
    print("-" * 60)
    
    # 检查必要文件
    if not os.path.exists('wind_final.csv'):
        print("❌ 找不到 wind_final.csv 文件")
        print("   请先运行 force_fix.py 或 data_preprocess.py")
        print("   或者使用其他数据文件")
        exit(1)
    
    # 运行DLinear训练
    score, result_file = run_dlinear_enhanced()
    
    if score:
        print(f"\n🎉 DLinear训练完成! R² = {score:.4f}")
        print(f"📁 结果保存在: {result_file}")
        print("📝 查看报告文件获取详细分析")
    else:
        print("\n❌ DLinear训练失败，请检查错误信息")