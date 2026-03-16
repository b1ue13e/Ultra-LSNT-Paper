"""
最新SOTA模型对比实验 - PatchTST (2023), TimeMixer (2024), iTransformer (2024)
================================================================================

本脚本实现2023-2024年最新的时序预测模型，用于风电功率预测对比实验。

包含模型：
1. PatchTST (ICLR 2023) - 基于补丁的Transformer，长时序预测SOTA
2. TimeMixer (ICLR 2024) - 多尺度混合框架，最新轻量化模型
3. iTransformer (NeurIPS 2024) - 倒置Transformer，近期性能突出的模型

使用方法：
    # 运行单个模型
    python run_latest_sota.py --model PatchTST --data wind_final.csv --target power
    
    # 运行所有最新模型
    python run_latest_sota.py --model all --data wind_final.csv --target power
    
    # 后台运行（推荐）
    nohup python run_latest_sota.py --model all --data wind_final.csv --target power > latest_sota.log 2>&1 &
    
    # 使用RTX 4090D优化配置
    python run_latest_sota.py --model all --data wind_final.csv --target power --gpu_optimized --batch_size 256

作者: Roo (AI助手)
日期: 2026-01-26
版本: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 尝试从主脚本导入工具函数
try:
    from ultra_lsnt_timeseries import (
        create_dataloaders, load_csv_data, compute_metrics,
        generate_synthetic_wind_data, generate_synthetic_load_data,
        TimeSeriesConfig, TrainConfig, get_temperature,
        StandardScaler
    )
    print("✅ 成功导入 ultra_lsnt_timeseries 工具库")
except ImportError as e:
    print(f"❌ 导入超时序列模块时出错: {e}")
    print("请确保此脚本与 ultra_lsnt_timeseries.py 在同一目录下")
    sys.exit(1)

# ============================================================
# 1. PatchTST (ICLR 2023) 实现
# ============================================================

class PatchTST(nn.Module):
    """
    PatchTST: A Transformer-based model for time series forecasting with patching mechanism.
    核心思想：将时间序列分割为不重叠的补丁，每个补丁通过线性投影到嵌入空间，然后使用标准Transformer编码器。
    
    论文: "Taming Transformers for Time Series Forecasting with PatchTST"
    年份: 2023 (ICLR)
    特点: 补丁化策略减少序列长度，降低计算复杂度，提升长序列预测能力
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.input_dim = config['input_dim']
        self.d_model = config.get('d_model', 128)
        
        # 补丁参数
        self.patch_len = config.get('patch_len', 16)  # 每个补丁长度
        self.stride = config.get('stride', 8)  # 补丁步长
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1
        
        # 补丁嵌入层
        self.patch_embedding = nn.Linear(self.patch_len * self.input_dim, self.d_model)
        
        # 位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.d_model) * 0.02)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.get('nhead', 4),
            dim_feedforward=self.d_model * 4,
            dropout=config.get('dropout', 0.1),
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_layers', 2))
        
        # 解码头
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model * self.num_patches, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.pred_len)
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(self.d_model)
        
    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入序列分割为补丁
        x: (batch, seq_len, input_dim)
        返回: (batch, num_patches, patch_len * input_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 确保序列长度足够
        if seq_len < self.patch_len:
            raise ValueError(f"序列长度{seq_len}小于补丁长度{self.patch_len}")
        
        # 创建补丁
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :]  # (batch, patch_len, input_dim)
            patch = patch.reshape(batch_size, -1)  # (batch, patch_len * input_dim)
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * input_dim)
        return patches
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: (batch, seq_len, input_dim)
        返回: (batch, pred_len), aux_loss
        """
        batch_size = x_enc.size(0)
        
        # 创建补丁
        patches = self.create_patches(x_enc)  # (batch, num_patches, patch_len * input_dim)
        
        # 补丁嵌入
        patch_emb = self.patch_embedding(patches)  # (batch, num_patches, d_model)
        
        # 位置编码
        patch_emb = patch_emb + self.position_embedding
        
        # 层归一化
        patch_emb = self.norm(patch_emb)
        
        # Transformer编码
        encoded = self.encoder(patch_emb)  # (batch, num_patches, d_model)
        
        # 展平
        encoded_flat = encoded.reshape(batch_size, -1)  # (batch, num_patches * d_model)
        
        # 解码
        output = self.decoder(encoded_flat)  # (batch, pred_len)
        
        return output.unsqueeze(-1), torch.tensor(0.0, device=x_enc.device)

# ============================================================
# 2. TimeMixer (ICLR 2024) 实现
# ============================================================

class TimeMixer(nn.Module):
    """
    TimeMixer: A Multi-Scale Framework for Time Series Forecasting
    核心思想：多尺度混合，结合局部和全局模式，使用自适应频率分解和混合机制。
    
    论文: "TimeMixer: A Multi-Scale Framework for Time Series Forecasting"
    年份: 2024 (ICLR)
    特点: 轻量化设计，多尺度特征提取，适合边缘部署
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.input_dim = config['input_dim']
        self.d_model = config.get('d_model', 128)
        
        # 多尺度参数
        self.scales = config.get('scales', [1, 2, 4])  # 多尺度因子
        
        # 输入投影
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        
        # 多尺度卷积
        self.multiscale_convs = nn.ModuleList()
        for scale in self.scales:
            kernel_size = max(3, scale * 2)
            padding = kernel_size // 2
            conv = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=kernel_size, padding=padding),
                nn.GELU(),
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            )
            self.multiscale_convs.append(conv)
        
        # 尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.d_model * len(self.scales), self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model)
        )
        
        # 自适应频率分解
        self.frequency_decomp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model * 2)
        )
        
        # 时间注意力（轻量化）
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.get('nhead', 4),
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.d_model)
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.pred_len)
        )
        
        # 跳跃连接
        self.skip = nn.Linear(self.seq_len, self.pred_len) if self.seq_len != self.pred_len else None
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: (batch, seq_len, input_dim)
        返回: (batch, pred_len), aux_loss
        """
        batch_size = x_enc.size(0)
        
        # 输入投影
        x = self.input_proj(x_enc)  # (batch, seq_len, d_model)
        
        # 多尺度特征提取 - 确保所有输出尺寸一致
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        multiscale_features = []
        for conv in self.multiscale_convs:
            scale_feat = conv(x_t)  # (batch, d_model, seq_len)
            # 确保序列长度一致（通过切片）
            if scale_feat.size(-1) != self.seq_len:
                # 如果尺寸不匹配，进行中心裁剪或填充
                diff = scale_feat.size(-1) - self.seq_len
                if diff > 0:
                    # 裁剪多余的
                    start = diff // 2
                    scale_feat = scale_feat[:, :, start:start+self.seq_len]
                else:
                    # 填充不足的
                    padding = -diff
                    scale_feat = F.pad(scale_feat, (padding//2, padding - padding//2))
            
            multiscale_features.append(scale_feat.transpose(1, 2))  # (batch, seq_len, d_model)
        
        # 尺度融合
        fused = torch.cat(multiscale_features, dim=-1)  # (batch, seq_len, d_model * len(scales))
        fused = self.scale_fusion(fused)  # (batch, seq_len, d_model)
        
        # 自适应频率分解
        freq_components = self.frequency_decomp(fused)  # (batch, seq_len, d_model * 2)
        low_freq, high_freq = torch.chunk(freq_components, 2, dim=-1)
        
        # 低频分量（趋势）
        trend = low_freq.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        
        # 高频分量（季节性）
        seasonal = high_freq
        
        # 时间注意力
        attn_out, _ = self.temporal_attn(seasonal, seasonal, seasonal)  # (batch, seq_len, d_model)
        seasonal = self.attn_norm(seasonal + attn_out)
        
        # 组合
        combined = seasonal + trend.expand_as(seasonal)
        
        # 聚合
        aggregated = combined.mean(dim=1)  # (batch, d_model)
        
        # 输出投影
        output = self.output_proj(aggregated)  # (batch, pred_len)
        
        # 跳跃连接（如果可用）
        if self.skip is not None:
            skip_out = self.skip(x_enc[:, :, -1])  # 使用最后一个特征
            output = output + skip_out * 0.1
        
        return output.unsqueeze(-1), torch.tensor(0.0, device=x_enc.device)

# ============================================================
# 3. iTransformer (NeurIPS 2024) 实现
# ============================================================

class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    核心思想：倒置Transformer架构，将时间点作为token，变量作为特征，使用自注意力跨变量建模。
    
    论文: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
    年份: 2024 (NeurIPS)
    特点: 简单高效，在多个基准测试中达到SOTA性能
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.input_dim = config['input_dim']
        self.d_model = config.get('d_model', 128)
        
        # 倒置架构：将变量维度作为序列维度
        # 输入: (batch, seq_len, input_dim) -> (batch, input_dim, d_model)
        
        # 变量嵌入（将每个变量映射到d_model维空间）
        self.var_embedding = nn.Linear(self.seq_len, self.d_model)
        
        # 位置编码（针对变量维度）
        self.pos_embedding = nn.Parameter(torch.randn(1, self.input_dim, self.d_model) * 0.02)
        
        # Transformer编码器（在变量维度上应用注意力）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.get('nhead', 4),
            dim_feedforward=self.d_model * 4,
            dropout=config.get('dropout', 0.1),
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_layers', 2))
        
        # 解码器：预测未来序列
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model * self.input_dim, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.pred_len)
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: (batch, seq_len, input_dim)
        返回: (batch, pred_len), aux_loss
        """
        batch_size = x_enc.size(0)
        
        # 倒置：转置序列和变量维度
        # (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x_inverted = x_enc.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        # 变量嵌入
        var_emb = self.var_embedding(x_inverted)  # (batch, input_dim, d_model)
        
        # 位置编码
        var_emb = var_emb + self.pos_embedding
        
        # 层归一化
        var_emb = self.norm(var_emb)
        
        # Transformer编码（在变量维度上）
        encoded = self.encoder(var_emb)  # (batch, input_dim, d_model)
        
        # 展平
        encoded_flat = encoded.reshape(batch_size, -1)  # (batch, input_dim * d_model)
        
        # 解码
        output = self.decoder(encoded_flat)  # (batch, pred_len)
        
        return output.unsqueeze(-1), torch.tensor(0.0, device=x_enc.device)

# ============================================================
# 4. 统一的训练与评估流程
# ============================================================

def train_latest_model(model_name: str, data: np.ndarray, args) -> Dict[str, float]:
    """训练最新的SOTA模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n⚡ 训练最新模型: {model_name} 在 {device} 上")
    print(f"   设备规格: RTX 4090D 24GB, 批大小: {args.batch_size}")
    
    # 转换配置
    ts_config = TimeSeriesConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        label_len=args.seq_len // 2,
        target=args.target
    )
    
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # 准备数据
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data, ts_config, train_config
    )
    
    # 初始化模型配置
    model_config = {
        'input_dim': data.shape[1],
        'output_dim': 1,
        'd_model': args.d_model,
        'seq_len': args.seq_len,
        'label_len': args.seq_len // 2,
        'pred_len': args.pred_len,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    # 特定模型配置
    if model_name == 'PatchTST':
        model_config.update({
            'patch_len': args.patch_len,
            'stride': args.stride,
        })
        model = PatchTST(model_config).to(device)
    elif model_name == 'TimeMixer':
        model_config.update({
            'scales': [1, 2, 4],
        })
        model = TimeMixer(model_config).to(device)
    elif model_name == 'iTransformer':
        model = iTransformer(model_config).to(device)
    else:
        raise ValueError(f"未知模型: {model_name}")
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 训练循环
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Forward
            pred, _ = model(batch_x)
            loss = criterion(pred.squeeze(-1), batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, _ = model(batch_x)
                loss = criterion(pred.squeeze(-1), batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'checkpoints_ts/{model_name}_best.pth')
        else:
            patience_counter += 1
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   周期 {epoch + 1}/{args.epochs} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")
        
        if patience_counter >= args.patience:
            print(f"   早停在周期 {epoch + 1}")
            break
    
    training_time = time.time() - start_time
    print(f"   训练时间: {training_time:.2f}s")
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load(f'checkpoints_ts/{model_name}_best.pth', map_location=device))
    
    # 测试评估
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred, _ = model(batch_x)
            preds.append(pred.squeeze(-1).cpu().numpy())
            trues.append(batch_y.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 反标准化
    target_std = scaler.std[-1]
    target_mean = scaler.mean[-1]
    preds = preds * target_std + target_mean
    trues = trues * target_std + target_mean
    
    # 计算指标
    metrics = compute_metrics(preds, trues)
    metrics['training_time'] = training_time
    metrics['total_params'] = total_params
    metrics['trainable_params'] = trainable_params
    
    print(f"📈 {model_name} 结果:")
    print(f"   R²: {metrics['R2']:.4f}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE: {metrics['MAE']:.4f}")
    print(f"   训练时间: {training_time:.2f}s")
    print(f"   参数量: {total_params:,}")
    
    # 保存结果
    save_path = Path(f'./checkpoints_ts/{model_name}_latest_results.json')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 保存预测结果用于后续分析
    np.save(f'checkpoints_ts/{model_name}_preds.npy', preds)
    np.save(f'checkpoints_ts/{model_name}_trues.npy', trues)
    
    return metrics

# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='运行最新SOTA模型对比实验')
    
    # 数据参数
    parser.add_argument('--data', type=str, default='wind_final.csv', help='数据文件路径')
    parser.add_argument('--target', type=str, default='power', help='目标列名')
    parser.add_argument('--synthetic', action='store_true', help='使用合成数据')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='all', 
                        choices=['PatchTST', 'TimeMixer', 'iTransformer', 'all'],
                        help='选择要运行的模型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练周期数')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小 (RTX 4090D优化)')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    
    # 模型架构参数
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=24, help='预测长度')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--nhead', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # PatchTST特定参数
    parser.add_argument('--patch_len', type=int, default=16, help='PatchTST补丁长度')
    parser.add_argument('--stride', type=int, default=8, help='PatchTST补丁步长')
    
    # 系统参数
    parser.add_argument('--gpu_optimized', action='store_true', help='使用GPU优化配置')
    parser.add_argument('--no_cuda', action='store_true', help='不使用CUDA')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # GPU优化配置
    if args.gpu_optimized and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✅ 启用GPU优化 (cuDNN基准模式)")
    
    # 加载数据
    print(f"📊 加载数据...")
    if args.synthetic:
        data, _ = generate_synthetic_wind_data(5000)
        print(f"   使用合成数据，形状: {data.shape}")
    elif args.data:
        data, _ = load_csv_data(args.data, args.target)
        if data is None:
            print("❌ 数据加载失败")
            return
        print(f"   数据形状: {data.shape}")
    else:
        print("请指定 --data 或 --synthetic")
        return
    
    # 确定要运行的模型
    if args.model == 'all':
        models_to_run = ['PatchTST', 'TimeMixer', 'iTransformer']
    else:
        models_to_run = [args.model]
    
    # 确保检查点目录存在
    Path('./checkpoints_ts').mkdir(parents=True, exist_ok=True)
    
    # 运行每个模型
    all_results = {}
    for model_name in models_to_run:
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始训练: {model_name}")
            print(f"{'='*60}")
            
            results = train_latest_model(model_name, data, args)
            all_results[model_name] = results
            
            print(f"✅ {model_name} 训练完成!")
            
        except Exception as e:
            print(f"❌ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    if all_results:
        print(f"\n{'='*60}")
        print("🏆 最终对比结果")
        print(f"{'='*60}")
        
        # 打印表格头
        print(f"{'模型':<12} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'训练时间':>10} {'参数量':>12}")
        print("-" * 68)
        
        for model_name, metrics in all_results.items():
            print(f"{model_name:<12} {metrics['R2']:>8.4f} {metrics['RMSE']:>10.2f} "
                  f"{metrics['MAE']:>10.2f} {metrics['training_time']:>10.1f}s "
                  f"{metrics['total_params']:>12,}")
        
        # 保存汇总结果
        summary_path = 'checkpoints_ts/latest_sota_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📁 结果已保存到: {summary_path}")
        
        # 与现有SOTA模型对比
        try:
            existing_metrics_path = 'my/results/metrics.json'
            if os.path.exists(existing_metrics_path):
                with open(existing_metrics_path, 'r') as f:
                    existing_metrics = json.load(f)
                
                print(f"\n{'='*60}")
                print("🔄 与现有SOTA模型对比")
                print(f"{'='*60}")
                print(f"{'模型':<15} {'R²':>8} {'年份':>8}")
                print("-" * 35)
                
                # 现有模型
                for model_name, metrics in existing_metrics.items():
                    if 'r2' in metrics:
                        print(f"{model_name:<15} {metrics['r2']:>8.4f} {'(现有)':>8}")
                
                # 新模型
                for model_name, metrics in all_results.items():
                    print(f"{model_name:<15} {metrics['R2']:>8.4f} {'2023-2024':>8}")
        
        except Exception as e:
            print(f"警告: 无法与现有模型对比: {e}")
    
    print(f"\n🎉 所有实验完成!")
    print(f"   查看详细日志: latest_sota.log")
    print(f"   结果文件: checkpoints_ts/ 目录")
    print(f"   预测数据: checkpoints_ts/*_preds.npy")

# ============================================================
# 6. 后台运行辅助函数
# ============================================================

def run_in_background():
    """
    后台运行脚本的辅助函数
    使用方法: 在终端中运行 `nohup python run_latest_sota.py --model all > latest_sota.log 2>&1 &`
    """
    print("准备在后台运行最新SOTA模型对比实验...")
    print("请使用以下命令:")
    print()
    print("  nohup python run_latest_sota.py --model all --data wind_final.csv --target power \\")
    print("      --batch_size 256 --epochs 100 --gpu_optimized > latest_sota.log 2>&1 &")
    print()
    print("这将:")
    print("  1. 在后台运行所有三个最新模型")
    print("  2. 使用RTX 4090D优化配置 (批大小256)")
    print("  3. 将日志输出到 latest_sota.log")
    print("  4. 结果保存到 checkpoints_ts/ 目录")
    print()
    print("要监控进度:")
    print("  tail -f latest_sota.log")
    print()
    print("要停止所有后台任务:")
    print("  pkill -f run_latest_sota.py")

# ============================================================
# 7. 执行
# ============================================================

if __name__ == '__main__':
    # 检查是否需要显示后台运行帮助
    if len(sys.argv) == 2 and sys.argv[1] == '--background-help':
        run_in_background()
    else:
        main()