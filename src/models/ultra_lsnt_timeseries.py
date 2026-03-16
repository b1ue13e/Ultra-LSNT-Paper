"""
Ultra-LSNT v7.0 - 风电场/电力负荷 时序预测训练脚本
====================================================

支持数据类型：
- 风电场数据：风速、风向、温度、湿度 → 发电功率预测
- 电力负荷数据：历史负荷、温度、时间特征 → 未来负荷预测

针对 RTX 4090D 24GB 优化（GPU增强版）

特性：
- 大batch_size训练 (256-512)
- 混合精度训练 (AMP)
- 多GPU支持 (单卡优化)
- 显存优化配置

使用方法：
    # 使用自己的数据
    python ultra_lsnt_timeseries.py --data your_data.csv --target power
    
    # 使用合成数据测试
    python ultra_lsnt_timeseries.py --synthetic
    
    # 快速测试
    python ultra_lsnt_timeseries.py --synthetic --quick
    
    # 使用RTX 4090D优化配置
    python ultra_lsnt_timeseries.py --data your_data.csv --target power --gpu_optimized

作者: 李俊宇
日期: 2025
更新: 2026-01-22 (RTX 4090D优化版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import time
import os
import argparse
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入误差函数 erf，兼容不同环境
try:
    from scipy.special import erf
    HAS_SCIPY_ERF = True
except ImportError:
    try:
        from numpy import erf
        HAS_SCIPY_ERF = True
    except ImportError:
        import math
        erf = np.vectorize(math.erf)  # 向量化 math.erf
        HAS_SCIPY_ERF = False


# ============================================================
# Part 1: 配置
# ============================================================

@dataclass
class LSNTConfig:
    """模型配置 - RTX 4090D优化版"""
    input_dim: int = 64           # 输入特征维度
    hidden_dim: int = 256         # 更大的隐藏维度 (128→256)
    output_dim: int = 1           # 预测目标数（单步预测=1，多步=n）
    num_blocks: int = 4           # 更多LSNT块 (3→4)
    num_experts: int = 8          # 更多专家数 (4→8)
    top_k: int = 4                # 更大的Top-K路由 (2→4)
    dropout: float = 0.1
    
    # 路由器配置
    router_z_loss_coef: float = 0.01
    router_aux_loss_coef: float = 0.01
    router_jitter_noise: float = 0.1
    
    # 跳跃门控
    skip_threshold: float = 0.5
    
    # 双模传播
    dual_mode_expansion: int = 4
    
    # GPU优化
    use_cudnn_benchmark: bool = True  # 启用cuDNN基准优化
    
    # 概率预测配置
    probabilistic_mode: str = 'none'  # 'none', 'gaussian', 'quantile'
    num_quantiles: int = 3            # 分位数回归的分位数数量
    quantiles: List[float] = None     # 分位数列表，默认为[0.1, 0.5, 0.9]
    min_std: float = 0.01             # 标准差下限（避免除零）
    heteroscedastic_moe: bool = True  # 是否使用MoE建模异方差
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        if self.probabilistic_mode == 'gaussian':
            self.output_dim = 2  # μ, σ
        elif self.probabilistic_mode == 'quantile':
            self.output_dim = self.num_quantiles
        # 其他模式保持原有output_dim
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TimeSeriesConfig:
    """时序数据配置"""
    seq_len: int = 96             # 输入序列长度（例如96个时间步=1天@15分钟间隔）
    pred_len: int = 24            # 预测长度（例如24个时间步=6小时）
    label_len: int = 48           # 标签长度（用于decoder）
    
    # 特征配置
    features: str = 'MS'          # M: 多变量预测多变量, S: 单变量, MS: 多变量预测单变量
    target: str = 'power'         # 目标列名
    
    # 数据处理
    scale: bool = True            # 是否标准化
    time_features: bool = True    # 是否添加时间特征


@dataclass
class TrainConfig:
    """训练配置 - RTX 4090D优化版"""
    batch_size: int = 256          # RTX 4090D可支持更大batch_size
    num_workers: int = 8           # 更多数据加载线程
    epochs: int = 200              # 更多epochs以获得更好收敛
    lr: float = 3e-4              # 更小的学习率适合大batch_size
    weight_decay: float = 0.01
    warmup_epochs: int = 10        # 更长的warmup
    
    # 显存优化
    accumulation_steps: int = 1    # 不需要梯度累积，24GB VRAM足够
    use_amp: bool = True           # 启用混合精度训练
    gradient_clip: float = 1.0
    
    # Loss 权重
    aux_loss_weight: float = 0.01
    
    # 温度退火
    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    
    # 保存
    save_dir: str = './checkpoints_ts_gpu'
    log_interval: int = 10         # 更频繁的日志记录
    
    # 早停
    patience: int = 30             # 更宽松的早停
    
    # GPU优化参数
    pin_memory: bool = True        # 固定内存加速数据传输
    prefetch_factor: int = 2       # 预取因子
    persistent_workers: bool = True # 持久化工作进程


# ============================================================
# Part 2: 工具函数
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_temperature(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.initial_temperature
    progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    progress = min(1.0, progress)
    return cfg.final_temperature + 0.5 * (cfg.initial_temperature - cfg.final_temperature) * (1 + np.cos(np.pi * progress))


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class StandardScaler:
    """标准化工具"""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1  # 避免除零
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# ============================================================
# Part 3: 数据集
# ============================================================

class TimeSeriesDataset(Dataset):
    """通用时序数据集"""
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, 
                 label_len: int = 0, target_idx: int = -1):
        """
        Args:
            data: shape (T, F) - T个时间步，F个特征
            seq_len: 输入序列长度
            pred_len: 预测长度
            label_len: 标签长度（用于Informer风格）
            target_idx: 目标变量的索引，-1表示最后一列
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.target_idx = target_idx
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # 输入序列
        seq_x = self.data[idx:idx + self.seq_len]
        
        # 目标序列
        seq_y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_idx]
        
        # 转换为tensor
        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        
        return seq_x, seq_y


def generate_synthetic_wind_data(n_samples: int = 10000, n_features: int = 8) -> Tuple[np.ndarray, List[str]]:
    """生成合成风电数据用于测试"""
    print("生成合成风电数据...")
    
    t = np.arange(n_samples)
    
    # 风速 (主要特征) - 包含日周期和随机波动
    wind_speed = (
        8 + 
        3 * np.sin(2 * np.pi * t / 96) +      # 日周期 (96 = 1天@15分钟)
        1.5 * np.sin(2 * np.pi * t / 672) +   # 周周期
        np.random.normal(0, 1.5, n_samples)    # 随机波动
    )
    wind_speed = np.clip(wind_speed, 0, 25)
    
    # 风向 (0-360度)
    wind_direction = (
        180 + 
        45 * np.sin(2 * np.pi * t / 96 + 1) +
        np.random.normal(0, 20, n_samples)
    ) % 360
    
    # 温度
    temperature = (
        15 + 
        10 * np.sin(2 * np.pi * t / 96 - np.pi/2) +  # 日周期
        5 * np.sin(2 * np.pi * t / 2688) +            # 月周期
        np.random.normal(0, 2, n_samples)
    )
    
    # 湿度
    humidity = (
        60 + 
        20 * np.sin(2 * np.pi * t / 96 + np.pi) +
        np.random.normal(0, 5, n_samples)
    )
    humidity = np.clip(humidity, 20, 100)
    
    # 气压
    pressure = 1013 + np.random.normal(0, 5, n_samples)
    
    # 时间特征
    hour_sin = np.sin(2 * np.pi * (t % 96) / 96)
    hour_cos = np.cos(2 * np.pi * (t % 96) / 96)
    
    # 发电功率 (目标) - 与风速的立方成正比（简化的风机功率曲线）
    cut_in = 3  # 切入风速
    cut_out = 25  # 切出风速
    rated = 12  # 额定风速
    
    power = np.zeros(n_samples)
    mask1 = (wind_speed >= cut_in) & (wind_speed < rated)
    mask2 = (wind_speed >= rated) & (wind_speed < cut_out)
    
    power[mask1] = (wind_speed[mask1] - cut_in) ** 3 / (rated - cut_in) ** 3 * 100
    power[mask2] = 100
    power += np.random.normal(0, 3, n_samples)
    power = np.clip(power, 0, 100)
    
    # 组合数据
    data = np.column_stack([
        wind_speed, wind_direction, temperature, humidity,
        pressure, hour_sin, hour_cos, power
    ])
    
    feature_names = [
        'wind_speed', 'wind_direction', 'temperature', 'humidity',
        'pressure', 'hour_sin', 'hour_cos', 'power'
    ]
    
    print(f"   生成数据形状: {data.shape}")
    print(f"   特征: {feature_names}")
    
    return data, feature_names


def generate_synthetic_load_data(n_samples: int = 10000, n_features: int = 6) -> Tuple[np.ndarray, List[str]]:
    """生成合成电力负荷数据"""
    print("生成合成电力负荷数据...")
    
    t = np.arange(n_samples)
    
    # 基础负荷 - 包含多周期
    base_load = (
        500 +                                        # 基础
        150 * np.sin(2 * np.pi * t / 96) +          # 日周期
        50 * np.sin(2 * np.pi * t / 672) +          # 周周期  
        30 * np.sin(2 * np.pi * t / 2688)           # 月周期
    )
    
    # 温度影响（U型曲线 - 过冷过热都增加负荷）
    temperature = (
        20 + 
        8 * np.sin(2 * np.pi * t / 96 - np.pi/2) +
        np.random.normal(0, 2, n_samples)
    )
    temp_effect = 0.5 * (temperature - 22) ** 2
    
    # 工作日/周末效应
    day_of_week = (t // 96) % 7
    weekday_effect = np.where(day_of_week < 5, 50, -30)
    
    # 最终负荷
    load = base_load + temp_effect + weekday_effect + np.random.normal(0, 20, n_samples)
    load = np.clip(load, 100, 1000)
    
    # 时间特征
    hour_sin = np.sin(2 * np.pi * (t % 96) / 96)
    hour_cos = np.cos(2 * np.pi * (t % 96) / 96)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    data = np.column_stack([
        temperature, hour_sin, hour_cos, day_sin, day_cos, load
    ])
    
    feature_names = ['temperature', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'load']
    
    print(f"   生成数据形状: {data.shape}")
    
    return data, feature_names


def load_csv_data(filepath: str, target: str = 'power') -> Tuple[np.ndarray, List[str]]:
    """从CSV加载数据（改进版）
    
    改进点：
    - 更智能的缺失值处理（使用插值而非简单前向填充）
    - 删除重复列和无用列
    - 检测目标列是否存在
    - 验证数据质量
    """
    print(f"加载数据: {filepath}")
    
    df = pd.read_csv(filepath)
    original_shape = df.shape
    
    # 检查目标列是否存在
    if target not in df.columns:
        raise ValueError(f"目标列 '{target}' 不在数据中。可用列: {list(df.columns)}")
    
    print(f"   原始形状: {original_shape}")
    print(f"   列数: {len(df.columns)}")
    
    # 1. 删除完全重复的列
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    if df.shape[1] < original_shape[1]:
        print(f"   移除了 {original_shape[1] - df.shape[1]} 个重复列")
    
    # 2. 删除时间戳列（如果有）
    time_cols = ['date', 'time', 'datetime', 'timestamp', '时间', '日期']
    cols_to_drop = [col for col in time_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"   删除了时间戳列: {cols_to_drop}")
    
    # 3. 删除序列号列（如果有）
    seq_cols = ['序列', 'id', 'index', 'seq', '序号']
    seq_to_drop = [col for col in seq_cols if col in df.columns]
    if seq_to_drop:
        df = df.drop(columns=seq_to_drop)
        print(f"   删除了序列号列: {seq_to_drop}")
    
    # 4. 删除非数值列（如场站名）
    non_numeric_cols = []
    for col in df.columns:
        if col == target:
            continue
        try:
            pd.to_numeric(df[col], errors='coerce')
        except:
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        df = df.drop(columns=non_numeric_cols)
        print(f"   删除了非数值列: {non_numeric_cols}")
    
    # 5. 检查缺失值
    missing_ratio = df.isnull().sum() / len(df)
    high_missing = missing_ratio[missing_ratio > 0.3]
    if len(high_missing) > 0:
        print(f"   警告: 以下列的缺失率 > 30%:")
        for col, ratio in high_missing.items():
            print(f"      - {col}: {ratio:.1%}")
        df = df.drop(columns=high_missing.index)
    
    # 6. 改进的缺失值处理
    # 使用线性插值处理缺失值（保留风电数据的物理意义）
    df = df.interpolate(method='linear', limit_direction='both', axis=0)
    # 剩余的缺失值（如序列开头）用前向填充
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')
    
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"   警告: 仍有 {remaining_missing} 个缺失值，使用0填充")
        df = df.fillna(0)
    else:
        print(f"   缺失值已全部处理")
    
    # 7. 移动目标列到最后
    if target in df.columns:
        cols = [c for c in df.columns if c != target] + [target]
        df = df[cols]
    else:
        raise ValueError(f"目标列 '{target}' 在处理过程中被删除")
    
    # 8. 转换为numpy数组
    data = df.values.astype(np.float32)
    feature_names = list(df.columns)
    
    # 9. 验证数据质量
    invalid_count = np.isnan(data).sum() + np.isinf(data).sum()
    if invalid_count > 0:
        print(f"   错误: 存在 {invalid_count} 个无效值 (NaN/Inf)")
        return None, None
    
    print(f"   最终形状: {data.shape}")
    print(f"   特征: {feature_names}")
    print(f"   数据范围: [{data.min():.4f}, {data.max():.4f}]")
    
    return data, feature_names


def create_dataloaders(data: np.ndarray, ts_config: TimeSeriesConfig, 
                       train_config: TrainConfig, train_ratio: float = 0.7,
                       val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """创建数据加载器"""
    
    # 划分数据
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # 标准化（只用训练集fit）
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(
        train_data_scaled, ts_config.seq_len, ts_config.pred_len,
        ts_config.label_len, target_idx=-1
    )
    val_dataset = TimeSeriesDataset(
        val_data_scaled, ts_config.seq_len, ts_config.pred_len,
        ts_config.label_len, target_idx=-1
    )
    test_dataset = TimeSeriesDataset(
        test_data_scaled, ts_config.seq_len, ts_config.pred_len,
        ts_config.label_len, target_idx=-1
    )
    
    # 创建加载器
    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size,
        shuffle=True, num_workers=train_config.num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size * 2,
        shuffle=False, num_workers=train_config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.batch_size * 2,
        shuffle=False, num_workers=train_config.num_workers,
        pin_memory=True
    )
    
    print(f"   训练样本: {len(train_dataset)}")
    print(f"   验证样本: {len(val_dataset)}")
    print(f"   测试样本: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, scaler


# ============================================================
# Part 4: Ultra-LSNT 核心模型（时序版）
# ============================================================

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft, hard):
        return hard
    
    @staticmethod
    def backward(ctx, grad):
        return grad, None


def ste_discretize(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    hard = (x > threshold).float()
    return StraightThroughEstimator.apply(x, hard)


class SparseMoERouter(nn.Module):
    """稀疏 MoE 路由器"""
    
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2,
                 jitter_noise: float = 0.0, z_loss_coef: float = 0.01,
                 aux_loss_coef: float = 0.01):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.jitter_noise = jitter_noise
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef
        
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        batch_size, dim = x.shape
        device = x.device
        
        router_logits = self.router(x)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise
        
        router_probs = F.softmax(router_logits / temperature, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 稀疏计算
        output = torch.zeros(batch_size, dim, device=device, dtype=x.dtype)
        flat_indices = top_k_indices.view(-1)
        flat_weights = top_k_weights.view(-1)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        for expert_idx in range(self.num_experts):
            mask = (flat_indices == expert_idx)
            if not mask.any():
                continue
            token_indices = batch_indices[mask]
            token_weights = flat_weights[mask]
            expert_output = self.experts[expert_idx](x[token_indices])
            output.index_add_(0, token_indices, expert_output * token_weights.unsqueeze(-1))
        
        # 辅助损失
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        expert_usage = router_probs.mean(dim=0)
        expert_selection = expert_mask.sum(dim=1).mean(dim=0) / self.top_k
        load_balance_loss = self.num_experts * (expert_usage * expert_selection).sum()
        
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        aux_loss = self.aux_loss_coef * load_balance_loss + self.z_loss_coef * z_loss

        stats = {
            "load_balance_loss": load_balance_loss.item(),
            "z_loss": z_loss.item(),
            "expert_usage": expert_usage.detach().cpu().tolist(),
            "usage_std": expert_usage.std().item(),
            "compute_ratio": self.top_k / self.num_experts,
            "raw_probs": router_probs.detach().cpu()
        }
        
        return output, aux_loss, stats


class HeteroscedasticMoERouter(nn.Module):
    """异方差 MoE 路由器 - 基于不确定性感知的路由"""
    
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2,
                 jitter_noise: float = 0.0, z_loss_coef: float = 0.01,
                 aux_loss_coef: float = 0.01, uncertainty_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.jitter_noise = jitter_noise
        self.z_loss_coef = z_loss_coef
        self.aux_loss_coef = aux_loss_coef
        
        # 不确定性估计网络
        self.uncertainty_net = nn.Sequential(
            nn.Linear(dim, uncertainty_dim),
            nn.GELU(),
            nn.Linear(uncertainty_dim, 1),
            nn.Softplus()  # 输出正值不确定性分数
        )
        
        # 路由器：输入为原始特征 + 不确定性分数（拼接）
        self.router_input_dim = dim + 1
        self.router = nn.Linear(self.router_input_dim, num_experts, bias=False)
        
        # 专家网络（与原始相同）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        batch_size, dim = x.shape
        device = x.device
        
        # 估计每个样本的不确定性
        uncertainty = self.uncertainty_net(x)  # (batch, 1)
        uncertainty_norm = uncertainty / (uncertainty.mean() + 1e-8)  # 归一化
        
        # 拼接特征和不确定性作为路由器输入
        router_input = torch.cat([x, uncertainty_norm], dim=-1)  # (batch, dim+1)
        
        router_logits = self.router(router_input)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise
        
        router_probs = F.softmax(router_logits / temperature, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 稀疏计算
        output = torch.zeros(batch_size, dim, device=device, dtype=x.dtype)
        flat_indices = top_k_indices.view(-1)
        flat_weights = top_k_weights.view(-1)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        for expert_idx in range(self.num_experts):
            mask = (flat_indices == expert_idx)
            if not mask.any():
                continue
            token_indices = batch_indices[mask]
            token_weights = flat_weights[mask]
            expert_output = self.experts[expert_idx](x[token_indices])
            output.index_add_(0, token_indices, expert_output * token_weights.unsqueeze(-1))
        
        # 辅助损失
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        expert_usage = router_probs.mean(dim=0)
        expert_selection = expert_mask.sum(dim=1).mean(dim=0) / self.top_k
        load_balance_loss = self.num_experts * (expert_usage * expert_selection).sum()
        
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        aux_loss = self.aux_loss_coef * load_balance_loss + self.z_loss_coef * z_loss

        stats = {
            "load_balance_loss": load_balance_loss.item(),
            "z_loss": z_loss.item(),
            "expert_usage": expert_usage.detach().cpu().tolist(),
            "usage_std": expert_usage.std().item(),
            "compute_ratio": self.top_k / self.num_experts,
            "uncertainty_mean": uncertainty.mean().item(),
            "uncertainty_std": uncertainty.std().item(),
            "raw_probs": router_probs.detach().cpu()
        }
        
        return output, aux_loss, stats


class ConsistentSkipGate(nn.Module):
    """跳跃门控"""
    
    def __init__(self, dim: int, init_threshold: float = 0.5):
        super().__init__()
        self.importance_net = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1), nn.Sigmoid()
        )
        self.change_detector = nn.Sequential(
            nn.Linear(dim * 2, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1), nn.Sigmoid()
        )
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
    
    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> Tuple[torch.Tensor, float]:
        importance = self.importance_net(x)
        combined = torch.cat([x, layer_output], dim=-1)
        change_score = self.change_detector(combined)
        skip_score = (1 - importance) * (1 - change_score)
        
        threshold = torch.sigmoid(self.threshold)
        
        if self.training:
            hard = (skip_score > threshold).float()
            skip_decision = hard.detach() - skip_score.detach() + skip_score
        else:
            skip_decision = (skip_score > threshold).float()
        
        output = skip_decision * x + (1 - skip_decision) * layer_output
        return output, skip_decision.mean().item()


class DualModePropagation(nn.Module):
    """双模传播"""
    
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        
        self.mode_selector = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1), nn.Sigmoid()
        )
        self.fast_path = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.slow_path = nn.Sequential(
            nn.Linear(dim, dim * expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim), nn.LayerNorm(dim)
        )
        self.skip_gate = ConsistentSkipGate(dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        mode_score = self.mode_selector(x)
        
        if self.training:
            hard = (mode_score > 0.5).float()
            use_slow = hard.detach() - mode_score.detach() + mode_score
        else:
            use_slow = (mode_score > 0.5).float()
        
        fast_out = self.fast_path(x)
        slow_mask = use_slow.squeeze(-1) > 0.5
        slow_out = fast_out.clone()
        
        if slow_mask.any():
            slow_out[slow_mask] = self.slow_path(x[slow_mask])
        
        output = (1 - use_slow) * fast_out + use_slow * slow_out
        output, skip_rate = self.skip_gate(x, output)
        
        return output, {"slow_mode_ratio": use_slow.mean().item(), "skip_rate": skip_rate}


class UltraLSNTBlock(nn.Module):
    """Ultra-LSNT 块"""
    
    def __init__(self, config: LSNTConfig):
        super().__init__()
        dim = config.hidden_dim
        
        self.input_norm = nn.LayerNorm(dim)
        # 根据配置选择异方差路由器或普通路由器
        if config.heteroscedastic_moe:
            self.moe_router = HeteroscedasticMoERouter(
                dim=dim, num_experts=config.num_experts, top_k=config.top_k,
                jitter_noise=config.router_jitter_noise,
                z_loss_coef=config.router_z_loss_coef,
                aux_loss_coef=config.router_aux_loss_coef,
                uncertainty_dim=16  # 默认值，可考虑添加到config
            )
        else:
            self.moe_router = SparseMoERouter(
                dim=dim, num_experts=config.num_experts, top_k=config.top_k,
                jitter_noise=config.router_jitter_noise,
                z_loss_coef=config.router_z_loss_coef,
                aux_loss_coef=config.router_aux_loss_coef
            )
        self.dual_prop = DualModePropagation(
            dim=dim, expansion=config.dual_mode_expansion, dropout=config.dropout
        )
        self.output_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        residual = x
        x = self.input_norm(x)
        x, router_aux_loss, router_stats = self.moe_router(x, temperature)
        x, dual_stats = self.dual_prop(x)
        x = self.output_norm(x)
        x = self.dropout(x)
        x = x + residual
        
        return x, {"router": router_stats, "dual_mode": dual_stats, "aux_loss": router_aux_loss}


class TemporalEncoder(nn.Module):
    """时序编码器 - 将序列编码为固定维度"""
    
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        # 特征投影
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # 时序卷积（捕获局部模式）
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        
        # 时序注意力（简化版）
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # 序列聚合
        self.aggregate = nn.Sequential(
            nn.Linear(seq_len * hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        
        # 特征投影
        x = self.feature_proj(x)  # (batch, seq_len, hidden_dim)
        
        # 位置编码
        x = x + self.pos_embedding
        
        # 时序卷积
        x_conv = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        x_conv = self.temporal_conv(x_conv)
        x = x + x_conv.transpose(1, 2)
        
        # 时序注意力
        x_attn, _ = self.temporal_attn(x, x, x)
        x = self.attn_norm(x + x_attn)
        
        # 聚合为固定维度
        x = x.reshape(batch_size, -1)  # (batch, seq_len * hidden_dim)
        x = self.aggregate(x)  # (batch, hidden_dim)
        
        return x


class TemporalDecoder(nn.Module):
    """时序解码器 - 将固定维度解码为预测序列"""
    
    def __init__(self, hidden_dim: int, output_dim: int, pred_len: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # 展开到预测长度
        self.expand = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * pred_len),
            nn.GELU(),
        )
        
        # 预测头（基础版本）
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim)
        batch_size = x.size(0)
        
        # 展开
        x = self.expand(x)  # (batch, hidden_dim * pred_len)
        x = x.reshape(batch_size, self.pred_len, self.hidden_dim)  # (batch, pred_len, hidden_dim)
        
        # 预测
        output = self.pred_head(x)  # (batch, pred_len, output_dim)
        
        # 如果output_dim=1，保持形状 (batch, pred_len)
        if self.output_dim == 1:
            output = output.squeeze(-1)
        
        return output

class ProbabilisticDecoder(nn.Module):
    """概率解码器 - 输出概率分布参数（高斯或分位数）"""
    
    def __init__(self, hidden_dim: int, output_dim: int, pred_len: int,
                 mode: str = 'gaussian', quantiles: List[float] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.mode = mode
        
        # 展开到预测长度
        self.expand = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * pred_len),
            nn.GELU(),
        )
        
        # 基础特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
        )
        
        # 根据模式设置输出头
        if mode == 'gaussian':
            # 输出 μ 和 log(σ)
            self.mu_head = nn.Linear(hidden_dim // 4, 1)
            self.logvar_head = nn.Linear(hidden_dim // 4, 1)
            self.output_dim = 2
        elif mode == 'quantile':
            self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
            self.num_quantiles = len(self.quantiles)
            self.quantile_heads = nn.ModuleList([
                nn.Linear(hidden_dim // 4, 1) for _ in range(self.num_quantiles)
            ])
            self.output_dim = self.num_quantiles
        else:
            raise ValueError(f"未知模式: {mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim)
        batch_size = x.size(0)
        
        # 展开
        x = self.expand(x)  # (batch, hidden_dim * pred_len)
        x = x.reshape(batch_size, self.pred_len, self.hidden_dim)  # (batch, pred_len, hidden_dim)
        
        # 特征提取
        features = self.feature_net(x)  # (batch, pred_len, hidden_dim//4)
        
        if self.mode == 'gaussian':
            mu = self.mu_head(features)  # (batch, pred_len, 1)
            logvar = self.logvar_head(features)
            # 确保标准差为正数
            logvar = torch.clamp(logvar, -10, 10)
            sigma = torch.exp(0.5 * logvar) + 1e-6
            # 拼接 μ 和 σ
            output = torch.cat([mu, sigma], dim=-1)  # (batch, pred_len, 2)
        elif self.mode == 'quantile':
            quantile_outputs = []
            for head in self.quantile_heads:
                q = head(features)  # (batch, pred_len, 1)
                quantile_outputs.append(q)
            output = torch.cat(quantile_outputs, dim=-1)  # (batch, pred_len, num_quantiles)
        else:
            raise RuntimeError(f"无效模式: {self.mode}")
        
        return output


class UltraLSNTForecaster(nn.Module):
    """Ultra-LSNT 时序预测模型（支持概率输出）"""
    
    def __init__(self, model_config: LSNTConfig, ts_config: TimeSeriesConfig):
        super().__init__()
        self.model_config = model_config
        self.ts_config = ts_config
        
        # 编码器
        self.encoder = TemporalEncoder(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            seq_len=ts_config.seq_len
        )
        
        # LSNT 块
        self.blocks = nn.ModuleList([
            UltraLSNTBlock(model_config) for _ in range(model_config.num_blocks)
        ])
        
        # 根据概率模式选择解码器
        if model_config.probabilistic_mode == 'none':
            self.decoder = TemporalDecoder(
                hidden_dim=model_config.hidden_dim,
                output_dim=model_config.output_dim,
                pred_len=ts_config.pred_len
            )
            self.is_probabilistic = False
        else:
            self.decoder = ProbabilisticDecoder(
                hidden_dim=model_config.hidden_dim,
                output_dim=model_config.output_dim,
                pred_len=ts_config.pred_len,
                mode=model_config.probabilistic_mode,
                quantiles=model_config.quantiles
            )
            self.is_probabilistic = True
        
        # 确保解码器输出维度与配置一致
        self.output_dim = model_config.output_dim
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0,
                return_stats: bool = False):
        # 编码
        x = self.encoder(x)  # (batch, hidden_dim)
        
        # LSNT 处理
        all_stats = []
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        for block in self.blocks:
            x, stats = block(x, temperature)
            all_stats.append(stats)
            total_aux_loss = total_aux_loss + stats["aux_loss"]
        
        # 解码
        output = self.decoder(x)  # (batch, pred_len, output_dim) 或 (batch, pred_len)
        
        if return_stats:
            efficiency = {
                "avg_compute_ratio": np.mean([s["router"]["compute_ratio"] for s in all_stats]),
                "avg_skip_rate": np.mean([s["dual_mode"]["skip_rate"] for s in all_stats]),
                "avg_slow_ratio": np.mean([s["dual_mode"]["slow_mode_ratio"] for s in all_stats]),
            }
            return output, total_aux_loss, {"block_stats": all_stats, "efficiency": efficiency}
        
        return output, total_aux_loss


# ============================================================
# Part 5: 基线模型
# ============================================================

class LSTMBaseline(nn.Module):
    """LSTM 基线"""
    def __init__(self, input_dim, hidden_dim, pred_len, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, pred_len)
    
    def forward(self, x, temperature=1.0, return_stats=False):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        aux_loss = torch.tensor(0.0, device=x.device)
        if return_stats:
            return out, aux_loss, {"efficiency": {"avg_compute_ratio": 1.0, "avg_skip_rate": 0.0, "avg_slow_ratio": 1.0}}
        return out, aux_loss


class MLPBaseline(nn.Module):
    """MLP 基线"""
    def __init__(self, input_dim, seq_len, hidden_dim, pred_len):
        super().__init__()
        self.flatten_dim = input_dim * seq_len
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len)
        )
    
    def forward(self, x, temperature=1.0, return_stats=False):
        x = x.reshape(x.size(0), -1)
        out = self.net(x)
        aux_loss = torch.tensor(0.0, device=x.device)
        if return_stats:
            return out, aux_loss, {"efficiency": {"avg_compute_ratio": 1.0, "avg_skip_rate": 0.0, "avg_slow_ratio": 1.0}}
        return out, aux_loss


# ============================================================
# Part 6: 评估指标
# ============================================================

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    pred = pred.flatten()
    true = true.flatten()
    
    # MSE
    mse = np.mean((pred - true) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(pred - true))
    
    # MAPE (避免除零和极小值问题)
    epsilon = 1e-2  # 阈值，忽略绝对值小于此值的真实值
    mask = np.abs(true) > epsilon
    if mask.any():
        mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100
    else:
        mape = 0
    
    # R²
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2)
    }


# ============================================================
# Part 6.1: 概率损失函数
# ============================================================

def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor,
                      min_std: float = 0.01) -> torch.Tensor:
    """
    高斯负对数似然损失 (NLL)
    mu: 预测均值 (batch, pred_len)
    sigma: 预测标准差 (batch, pred_len)，必须为正数
    target: 真实值 (batch, pred_len)
    """
    sigma = torch.clamp(sigma, min=min_std)
    nll = 0.5 * torch.log(2 * torch.pi * sigma**2) + 0.5 * ((target - mu) / sigma)**2
    return nll.mean()


def pinball_loss(pred_quantiles: torch.Tensor, target: torch.Tensor,
                 quantiles: List[float]) -> torch.Tensor:
    """
    分位数回归的 Pinball 损失
    pred_quantiles: (batch, pred_len, num_quantiles) 或 (batch, pred_len) 如果 num_quantiles=1
    target: (batch, pred_len)
    quantiles: 分位数列表，例如 [0.1, 0.5, 0.9]
    """
    if pred_quantiles.dim() == 2:
        # 假设只有一个分位数
        pred_quantiles = pred_quantiles.unsqueeze(-1)
    batch_size, pred_len, num_q = pred_quantiles.shape
    target_expanded = target.unsqueeze(-1).expand(-1, -1, num_q)
    quantiles_tensor = torch.tensor(quantiles, device=pred_quantiles.device).view(1, 1, -1)
    
    error = target_expanded - pred_quantiles
    loss = torch.max(quantiles_tensor * error, (quantiles_tensor - 1) * error)
    return loss.mean()


def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    连续分级概率评分 (CRPS) - 高斯分布版本
    近似公式: CRPS = sigma * (1/√π - 2*φ(z) - z*(2*Φ(z) - 1))
    其中 z = (target - mu) / sigma
    """
    eps = 1e-8
    sigma = torch.clamp(sigma, min=eps)
    z = (target - mu) / sigma
    # 标准正态 PDF 和 CDF
    phi = torch.exp(-0.5 * z**2) / torch.sqrt(2 * torch.pi)
    Phi = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))
    
    crps = sigma * (1 / torch.sqrt(torch.pi) - 2 * phi - z * (2 * Phi - 1))
    return crps.mean()


def compute_probabilistic_metrics(pred_params: np.ndarray, true: np.ndarray,
                                  mode: str = 'gaussian',
                                  quantiles: List[float] = None) -> Dict[str, float]:
    """
    计算概率预测评估指标
    pred_params: 预测参数，形状取决于模式:
        - 'gaussian': (n_samples, 2) 每行 [mu, sigma]
        - 'quantile': (n_samples, num_quantiles) 每行分位数预测
    true: 真实值 (n_samples,)
    """
    pred_params = np.asarray(pred_params)
    true = np.asarray(true).flatten()
    
    metrics = {}
    
    if mode == 'gaussian':
        # 分离 mu 和 sigma
        if pred_params.shape[1] != 2:
            raise ValueError("高斯模式需要2列参数")
        mu = pred_params[:, 0]
        sigma = pred_params[:, 1]
        
        # 计算 CRPS（使用近似）
        z = (true - mu) / np.clip(sigma, 1e-8, None)
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
        crps = sigma * (1 / np.sqrt(np.pi) - 2 * phi - z * (2 * Phi - 1))
        metrics['CRPS'] = float(np.mean(crps))
        
        # 区间覆盖率 (PICP) 和宽度 (MPIW)
        lower = mu - 1.96 * sigma  # 95% 置信区间
        upper = mu + 1.96 * sigma
        coverage = np.logical_and(true >= lower, true <= upper).mean()
        avg_width = np.mean(upper - lower)
        metrics['PICP_95'] = float(coverage)
        metrics['MPIW_95'] = float(avg_width)
        
        # 可靠性图统计（简化）
        # 计算标准化残差的分位数
        std_residual = (true - mu) / np.clip(sigma, 1e-8, None)
        metrics['std_residual_mean'] = float(std_residual.mean())
        metrics['std_residual_std'] = float(std_residual.std())
        
    elif mode == 'quantile':
        num_q = pred_params.shape[1]
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        if len(quantiles) != num_q:
            raise ValueError("分位数数量不匹配")
        
        # 计算每个分位数的 Pinball 损失
        total_pinball = 0
        for i, q in enumerate(quantiles):
            pred_q = pred_params[:, i]
            error = true - pred_q
            loss = np.maximum(q * error, (q - 1) * error)
            total_pinball += np.mean(loss)
            metrics[f'pinball_q{q}'] = float(np.mean(loss))
        metrics['pinball_avg'] = float(total_pinball / num_q)
        
        # 区间覆盖率（使用外部分位数）
        if num_q >= 2:
            lower = pred_params[:, 0]   # 例如 q=0.1
            upper = pred_params[:, -1]  # 例如 q=0.9
            coverage = np.logical_and(true >= lower, true <= upper).mean()
            avg_width = np.mean(upper - lower)
            metrics['PICP_interval'] = float(coverage)
            metrics['MPIW_interval'] = float(avg_width)
        
        # 中位数作为点预测
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else num_q // 2
        median_pred = pred_params[:, median_idx]
        metrics['median_MAE'] = float(np.mean(np.abs(true - median_pred)))
        metrics['median_RMSE'] = float(np.sqrt(np.mean((true - median_pred)**2)))
    
    return metrics


# ============================================================
# Part 7: 训练和评估
# ============================================================

def train_epoch(model, loader, optimizer, scaler, device, temperature, cfg: TrainConfig):
    """训练一个 epoch（支持概率输出）"""
    model.train()
    
    loss_meter = AverageMeter()
    main_loss_meter = AverageMeter()  # 主损失（NLL/pinball/MSE）
    aux_meter = AverageMeter()
    
    optimizer.zero_grad()
    
    for batch_idx, (seq_x, seq_y) in enumerate(loader):
        seq_x = seq_x.to(device, non_blocking=True)
        seq_y = seq_y.to(device, non_blocking=True)
        
        with autocast(enabled=cfg.use_amp):
            pred, aux_loss = model(seq_x, temperature=temperature)
            
            # 根据模型概率模式选择损失函数
            if hasattr(model, 'is_probabilistic') and model.is_probabilistic:
                mode = model.model_config.probabilistic_mode
                if mode == 'gaussian':
                    # 输出形状 (batch, pred_len, 2): [mu, sigma]
                    mu = pred[:, :, 0]
                    sigma = pred[:, :, 1]
                    main_loss = gaussian_nll_loss(mu, sigma, seq_y, min_std=model.model_config.min_std)
                elif mode == 'quantile':
                    # 输出形状 (batch, pred_len, num_quantiles)
                    quantiles = model.model_config.quantiles
                    main_loss = pinball_loss(pred, seq_y, quantiles)
                else:
                    main_loss = F.mse_loss(pred, seq_y)
            else:
                # 传统点预测
                main_loss = F.mse_loss(pred, seq_y)
            
            loss = (main_loss + cfg.aux_loss_weight * aux_loss) / cfg.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % cfg.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        loss_meter.update(loss.item() * cfg.accumulation_steps, seq_x.size(0))
        main_loss_meter.update(main_loss.item(), seq_x.size(0))
        aux_meter.update(aux_loss.item(), seq_x.size(0))
        
        if batch_idx % cfg.log_interval == 0:
            print(f'  [{batch_idx:4d}/{len(loader)}] Loss: {loss_meter.avg:.4f} | Main: {main_loss_meter.avg:.4f} | Aux: {aux_meter.avg:.4f}')
    
    return {'loss': loss_meter.avg, 'main_loss': main_loss_meter.avg, 'aux': aux_meter.avg}


@torch.no_grad()
def evaluate(model, loader, device, temperature, scaler_obj: StandardScaler = None):
    """评估"""
    model.eval()
    
    all_preds = []
    all_trues = []
    all_stats = []
    
    for seq_x, seq_y in loader:
        seq_x = seq_x.to(device, non_blocking=True)
        seq_y = seq_y.to(device, non_blocking=True)
        
        with autocast():
            pred, _, stats = model(seq_x, temperature=temperature, return_stats=True)
        
        all_preds.append(pred.cpu().numpy())
        all_trues.append(seq_y.cpu().numpy())
        all_stats.append(stats['efficiency'])
    
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    # 反标准化（如果需要）
    if scaler_obj is not None:
        # 只对目标变量反标准化
        target_mean = scaler_obj.mean[-1]
        target_std = scaler_obj.std[-1]
        preds = preds * target_std + target_mean
        trues = trues * target_std + target_mean
    
    metrics = compute_metrics(preds, trues)
    
    efficiency = {
        'avg_compute_ratio': np.mean([s['avg_compute_ratio'] for s in all_stats]),
        'avg_skip_rate': np.mean([s['avg_skip_rate'] for s in all_stats]),
        'avg_slow_ratio': np.mean([s['avg_slow_ratio'] for s in all_stats]),
    }
    
    return metrics, efficiency, preds, trues


def train(model_config: LSNTConfig, ts_config: TimeSeriesConfig, 
          train_config: TrainConfig, data: np.ndarray, 
          experiment_name: str = 'default'):
    """完整训练流程"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print(f"Ultra-LSNT Time Series Training - {experiment_name}")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {model_config.num_blocks} blocks, {model_config.num_experts} experts, top-{model_config.top_k}")
    print(f"Sequence: {ts_config.seq_len} → {ts_config.pred_len}")
    print("=" * 70)
    
    # 数据加载器
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data, ts_config, train_config
    )
    
    # 更新 input_dim
    model_config.input_dim = data.shape[1]
    
    # 模型
    model = UltraLSNTForecaster(model_config, ts_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs)
    grad_scaler = GradScaler(enabled=train_config.use_amp)
    
    # 保存目录
    save_dir = Path(train_config.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model_config.save(save_dir / 'model_config.json')
    
    # 训练
    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(train_config.epochs):
        epoch_start = time.time()
        temperature = get_temperature(epoch, train_config)
        
        print(f"\nEpoch {epoch+1}/{train_config.epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | Temp: {temperature:.3f}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, grad_scaler, device, temperature, train_config)
        
        # 验证
        val_metrics, val_efficiency, _, _ = evaluate(model, val_loader, device, temperature)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        record = {
            'epoch': epoch + 1,
            'temperature': temperature,
            'train': train_metrics,
            'val': val_metrics,
            'efficiency': val_efficiency,
            'time': epoch_time,
        }
        history.append(record)
        
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | MSE: {train_metrics['mse']:.4f}")
        print(f"  Val   | MSE: {val_metrics['MSE']:.4f} | RMSE: {val_metrics['RMSE']:.4f} | MAE: {val_metrics['MAE']:.4f}")
        print(f"  Efficiency | Compute: {val_efficiency['avg_compute_ratio']:.2%} | Skip: {val_efficiency['avg_skip_rate']:.2%}")
        
        # 早停
        if val_metrics['MSE'] < best_val_loss:
            best_val_loss = val_metrics['MSE']
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, save_dir / 'best_model.pth')
            print(f"  New best: MSE={best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # 保存历史
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # 测试
    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)
    
    # 加载最佳模型
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_efficiency, preds, trues = evaluate(model, test_loader, device, temperature, scaler)
    
    print(f"\nTest Metrics:")
    print(f"  MSE:  {test_metrics['MSE']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE:  {test_metrics['MAE']:.4f}")
    print(f"  MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"  R2:   {test_metrics['R2']:.4f}")
    print(f"\nEfficiency:")
    print(f"  Compute Ratio: {test_efficiency['avg_compute_ratio']:.2%}")
    print(f"  Skip Rate: {test_efficiency['avg_skip_rate']:.2%}")
    
    # 保存测试结果
    final_results = {
        'test_metrics': test_metrics,
        'efficiency': test_efficiency,
        'model_config': model_config.to_dict(),
    }
    with open(save_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 保存预测结果（用于可视化）
    np.savez(save_dir / 'predictions.npz', predictions=preds, ground_truth=trues)
    
    return history, test_metrics


# ============================================================
# Part 8: 可视化
# ============================================================

def plot_predictions(save_dir: str):
    """绘制预测结果"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed")
        return
    
    save_dir = Path(save_dir)
    
    # 加载预测
    data = np.load(save_dir / 'predictions.npz')
    preds = data['predictions']
    trues = data['ground_truth']
    
    # 只显示前500个点
    n = min(500, len(preds))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 预测 vs 真实值
    axes[0, 0].plot(trues[:n, 0], 'b-', label='Ground Truth', alpha=0.7)
    axes[0, 0].plot(preds[:n, 0], 'r-', label='Prediction', alpha=0.7)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('(a) Prediction vs Ground Truth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 误差分布
    errors = (preds - trues).flatten()
    axes[0, 1].hist(errors, bins=50, color='steelblue', edgecolor='white')
    axes[0, 1].axvline(x=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Error')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('(b) Error Distribution')
    
    # 3. 散点图
    axes[1, 0].scatter(trues.flatten()[:1000], preds.flatten()[:1000], alpha=0.3, s=10)
    min_val = min(trues.min(), preds.min())
    max_val = max(trues.max(), preds.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    axes[1, 0].set_xlabel('Ground Truth')
    axes[1, 0].set_ylabel('Prediction')
    axes[1, 0].set_title('(c) Scatter Plot')
    axes[1, 0].legend()
    
    # 4. 训练曲线
    history_path = save_dir / 'history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train']['mse'] for h in history]
        val_mse = [h['val']['MSE'] for h in history]
        
        axes[1, 1].plot(epochs, train_loss, 'b-', label='Train MSE')
        axes[1, 1].plot(epochs, val_mse, 'r-', label='Val MSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('(d) Training Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_dir / 'results.png'}")


# ============================================================
# Part 9: 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Ultra-LSNT Time Series Forecasting')
    
    # 数据
    parser.add_argument('--data', type=str, default=None, help='CSV data file path')
    parser.add_argument('--target', type=str, default='power', help='Target column name')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--data_type', type=str, default='wind', choices=['wind', 'load'], help='Synthetic data type')
    
    # GPU优化配置
    parser.add_argument('--gpu_optimized', action='store_true', help='Use RTX 4090D optimized configuration')
    
    # 模型
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension (128 for CPU, 256 for GPU)')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of LSNT blocks (3 for CPU, 4 for GPU)')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts (4 for CPU, 8 for GPU)')
    parser.add_argument('--top_k', type=int, default=4, help='Top-K routing (2 for CPU, 4 for GPU)')
    
    # 时序配置
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction length')
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (64 for CPU, 256 for GPU)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick test (20 epochs)')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparison')
    parser.add_argument('--visualize', type=str, default=None, help='Visualize results from checkpoint dir')
    parser.add_argument('--experiment_name', type=str, default='main', help='Name of experiment')

    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 可视化模式
    if args.visualize:
        plot_predictions(args.visualize)
        return
    
    # 加载数据
    if args.data:
        data, feature_names = load_csv_data(args.data, args.target)
        if data is None:
            print("数据加载失败")
            return
    elif args.synthetic:
        if args.data_type == 'wind':
            data, feature_names = generate_synthetic_wind_data(n_samples=15000)
        else:
            data, feature_names = generate_synthetic_load_data(n_samples=15000)
    else:
        print("请指定 --data 或 --synthetic")
        print("   示例: python ultra_lsnt_timeseries.py --synthetic")
        print("   示例: python ultra_lsnt_timeseries.py --data wind_data.csv --target power")
        return
    
    # 验证数据
    print("\n" + "="*70)
    print("数据验证")
    print("="*70)
    if len(data) < args.seq_len + args.pred_len:
        print(f"错误: 数据太少 ({len(data)} 行) < 所需最小数据 ({args.seq_len + args.pred_len} 行)")
        return
    
    # 检查数据异常
    data_mean = np.mean(data)
    data_std = np.std(data)
    print(f"数据均值: {data_mean:.4f}")
    print(f"数据标差: {data_std:.4f}")
    print(f"数据类型: {data.dtype}")
    print(f"样本数: {len(data)}")
    print(f"特征数: {data.shape[1]}")
    
    # GPU优化配置检测
    if args.gpu_optimized:
        print("\n[GPU优化模式] RTX 4090D 24GB 配置启用")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"检测到GPU: {gpu_name}, 显存: {gpu_memory:.1f} GB")
            # 自动调整为GPU优化参数
            if args.hidden_dim == 256:  # 默认值
                print("使用GPU优化隐藏维度: 256")
            if args.num_blocks == 4:    # 默认值
                print("使用GPU优化块数: 4")
            if args.num_experts == 8:   # 默认值
                print("使用GPU优化专家数: 8")
            if args.top_k == 4:         # 默认值
                print("使用GPU优化Top-K: 4")
            if args.batch_size == 256:  # 默认值
                print("使用GPU优化批量大小: 256")
        else:
            print("警告: GPU优化模式启用但未检测到CUDA设备，将使用CPU配置")
    
    # 配置
    model_config = LSNTConfig(
        input_dim=data.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_blocks=args.num_blocks,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    
    ts_config = TimeSeriesConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target=args.target,
    )
    
    # 根据GPU优化调整训练配置
    if args.gpu_optimized and torch.cuda.is_available():
        train_config = TrainConfig(
            batch_size=args.batch_size,
            epochs=20 if args.quick else args.epochs,  # 快速模式20个epoch
            lr=args.lr,
            use_amp=True,          # 启用混合精度
            num_workers=8,         # 更多数据加载线程
            pin_memory=True,       # 固定内存
            prefetch_factor=2,     # 预取因子
            persistent_workers=True, # 持久化工作进程
        )
    else:
        train_config = TrainConfig(
            batch_size=args.batch_size,
            epochs=10 if args.quick else args.epochs,
            lr=args.lr,
            use_amp=False,         # CPU模式禁用AMP
            num_workers=4,         # 较少线程
            pin_memory=False,      # 禁用固定内存
        )
    
    # 训练
    history, test_metrics = train(
        model_config, ts_config, train_config, data,
        experiment_name=args.experiment_name
    )
    
    # 生成可视化
    plot_predictions(f'{train_config.save_dir}/main')
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
