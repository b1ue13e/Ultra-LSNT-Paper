"""
Ultra-LSNT with Linear Branch (DLinear-inspired)
在 TemporalDecoder 之前加入线性分流，专门处理线性部分，MoE模块专门学习残差（非线性波动）
同时捕获路由器输出用于专家切换分析
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import json

# 导入原始模型组件
from ultra_lsnt_timeseries import (
    LSNTConfig, TimeSeriesConfig, TrainConfig,
    TemporalEncoder, UltraLSNTBlock, TemporalDecoder, ProbabilisticDecoder,
    SparseMoERouter, HeteroscedasticMoERouter, DualModePropagation
)

class LinearBranch(nn.Module):
    """线性分流 - 专门处理线性部分 (DLinear思想)"""
    
    def __init__(self, input_dim: int, hidden_dim: int, pred_len: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        
        # 简单的线性层序列
        self.linear_seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len)  # 直接输出预测序列
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim) 来自编码器的输出
        linear_pred = self.linear_seq(x)  # (batch, pred_len)
        return linear_pred

class UltraLSNTLinearBranchForecaster(nn.Module):
    """带线性分流的 Ultra-LSNT 时序预测模型"""
    
    def __init__(self, model_config: LSNTConfig, ts_config: TimeSeriesConfig,
                 linear_weight: float = 0.5, return_weights: bool = False):
        super().__init__()
        self.model_config = model_config
        self.ts_config = ts_config
        self.linear_weight = linear_weight  # 线性分支的权重（可学习或固定）
        self.return_weights = return_weights  # 是否返回路由器权重
        
        # 编码器
        self.encoder = TemporalEncoder(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            seq_len=ts_config.seq_len
        )
        
        # LSNT 块（MoE模块，专门学习非线性残差）
        self.blocks = nn.ModuleList([
            UltraLSNTBlock(model_config) for _ in range(model_config.num_blocks)
        ])
        
        # 线性分流
        self.linear_branch = LinearBranch(
            input_dim=model_config.hidden_dim,
            hidden_dim=model_config.hidden_dim,
            pred_len=ts_config.pred_len
        )
        
        # 非线性MoE解码器（处理残差）
        if model_config.probabilistic_mode == 'none':
            self.nonlinear_decoder = TemporalDecoder(
                hidden_dim=model_config.hidden_dim,
                output_dim=model_config.output_dim,
                pred_len=ts_config.pred_len
            )
            self.is_probabilistic = False
        else:
            self.nonlinear_decoder = ProbabilisticDecoder(
                hidden_dim=model_config.hidden_dim,
                output_dim=model_config.output_dim,
                pred_len=ts_config.pred_len,
                mode=model_config.probabilistic_mode,
                quantiles=model_config.quantiles
            )
            self.is_probabilistic = True
        
        # 自适应权重学习（可选）
        self.weight_learner = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 2, model_config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 注册钩子以捕获路由器输出
        self.gate_weights = []  # 存储路由器权重
        self.router_hooks = []
        
    def _register_router_hooks(self):
        """注册钩子以捕获路由器输出"""
        self.gate_weights.clear()
        self.router_hooks.clear()
        
        def hook_fn(module, input, output):
            # output: (tensor, aux_loss, stats)
            if len(output) >= 3:
                stats = output[2]
                if 'raw_probs' in stats:
                    self.gate_weights.append(stats['raw_probs'].detach().cpu())
                elif 'router_probs' in stats:
                    self.gate_weights.append(stats['router_probs'].detach().cpu())
        
        for block in self.blocks:
            router = block.moe_router
            hook = router.register_forward_hook(hook_fn)
            self.router_hooks.append(hook)
    
    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.router_hooks:
            hook.remove()
        self.router_hooks.clear()
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0,
                return_weights: bool = False):
        """
        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            temperature: 路由器温度
            return_weights: 是否返回路由器权重
        
        Returns:
            如果 return_weights=True: (prediction, linear_pred, nonlinear_pred, gate_weights)
            否则: (prediction, linear_pred, nonlinear_pred)
        """
        # 注册钩子以捕获路由器权重
        if return_weights:
            self._register_router_hooks()
        
        # 编码
        encoded = self.encoder(x)  # (batch, hidden_dim)
        
        # LSNT 处理（MoE学习非线性残差）
        all_stats = []
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        for block in self.blocks:
            encoded, stats = block(encoded, temperature)
            all_stats.append(stats)
            total_aux_loss = total_aux_loss + stats["aux_loss"]
        
        # 线性分支预测
        linear_pred = self.linear_branch(encoded)  # (batch, pred_len)
        
        # 非线性分支预测（基于MoE处理后的特征）
        nonlinear_pred = self.nonlinear_decoder(encoded)  # (batch, pred_len) 或 (batch, pred_len, output_dim)
        
        # 如果是概率输出，需要特殊处理
        if self.is_probabilistic:
            if self.model_config.probabilistic_mode == 'gaussian':
                # nonlinear_pred: (batch, pred_len, 2) [mu, sigma]
                # 我们只取mu作为非线性预测，sigma保持不变
                nonlinear_mu = nonlinear_pred[:, :, 0]
                nonlinear_sigma = nonlinear_pred[:, :, 1]
                
                # 线性预测与非线性预测的mu结合
                combined_mu = self.linear_weight * linear_pred + (1 - self.linear_weight) * nonlinear_mu
                combined_pred = torch.stack([combined_mu, nonlinear_sigma], dim=-1)
                
                # 分离的预测
                linear_full = torch.stack([linear_pred, torch.zeros_like(linear_pred)], dim=-1)
                nonlinear_full = nonlinear_pred
            elif self.model_config.probabilistic_mode == 'quantile':
                # nonlinear_pred: (batch, pred_len, num_quantiles)
                # 线性预测需要扩展到相同维度
                num_quantiles = nonlinear_pred.shape[-1]
                linear_expanded = linear_pred.unsqueeze(-1).expand(-1, -1, num_quantiles)
                combined_pred = self.linear_weight * linear_expanded + (1 - self.linear_weight) * nonlinear_pred
                
                linear_full = linear_expanded
                nonlinear_full = nonlinear_pred
            else:
                combined_pred = nonlinear_pred
                linear_full = linear_pred.unsqueeze(-1) if linear_pred.dim() == 2 else linear_pred
                nonlinear_full = nonlinear_pred
        else:
            # 点预测：加权结合
            combined_pred = self.linear_weight * linear_pred + (1 - self.linear_weight) * nonlinear_pred
            linear_full = linear_pred
            nonlinear_full = nonlinear_pred
        
        # 自适应权重学习（可选）
        # 可以基于编码特征动态调整linear_weight
        # weight_features = torch.cat([encoded, encoded.mean(dim=1, keepdim=True).expand_as(encoded)], dim=-1)
        # adaptive_weight = self.weight_learner(weight_features.mean(dim=1)).squeeze()
        
        # 如果需要返回权重
        if return_weights:
            gate_weights = self.gate_weights.copy() if self.gate_weights else []
            self._remove_hooks()
            return combined_pred, linear_full, nonlinear_full, gate_weights, total_aux_loss
        
        return combined_pred, linear_full, nonlinear_full, total_aux_loss

def train_linear_branch_model(model_config: LSNTConfig, ts_config: TimeSeriesConfig,
                              train_config: TrainConfig, data: np.ndarray,
                              experiment_name: str = 'linear_branch',
                              linear_weight: float = 0.5):
    """训练带线性分支的模型"""
    import time
    from pathlib import Path
    from torch.utils.data import DataLoader
    from ultra_lsnt_timeseries import (
        TimeSeriesDataset, StandardScaler, train_epoch, evaluate
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print(f"Ultra-LSNT with Linear Branch Training - {experiment_name}")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Linear weight: {linear_weight}")
    print(f"Config: {model_config.num_blocks} blocks, {model_config.num_experts} experts")
    print("=" * 70)
    
    # 数据划分
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # 标准化
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
    
    # 更新 input_dim
    model_config.input_dim = data.shape[1]
    
    # 创建模型
    model = UltraLSNTLinearBranchForecaster(
        model_config, ts_config, linear_weight=linear_weight
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr,
                                  weight_decay=train_config.weight_decay)
    
    # 训练循环（简化版，基于原始训练函数）
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(train_config.epochs):
        model.train()
        train_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            pred, linear_pred, nonlinear_pred, aux_loss = model(bx)
            loss = F.mse_loss(pred, by) + train_config.aux_loss_weight * aux_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred, _, _, _ = model(bx)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(by.cpu().numpy())
        
        # 计算验证损失
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_mse = np.mean((val_preds - val_trues) ** 2)
        
        print(f"Epoch {epoch+1}/{train_config.epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Val MSE: {val_mse:.4f}")
        
        # 早停
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            # 保存模型
            save_dir = Path(train_config.save_dir) / experiment_name
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_mse': val_mse,
                'linear_weight': linear_weight,
                'model_config': model_config.to_dict(),
            }, save_dir / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("Training completed!")
    return model

if __name__ == '__main__':
    # 示例用法
    import argparse
    from ultra_lsnt_timeseries import load_csv_data, main as original_main
    
    parser = argparse.ArgumentParser(description='Train Ultra-LSNT with Linear Branch')
    parser.add_argument('--data', type=str, default='wind_final.csv', help='Data file')
    parser.add_argument('--target', type=str, default='power', help='Target column')
    parser.add_argument('--linear_weight', type=float, default=0.5, help='Weight for linear branch')
    parser.add_argument('--experiment_name', type=str, default='linear_branch', help='Experiment name')
    
    args = parser.parse_args()
    
    # 加载数据
    data, feature_names = load_csv_data(args.data, args.target)
    
    # 配置
    model_config = LSNTConfig(
        input_dim=data.shape[1],
        hidden_dim=256,
        num_blocks=4,
        num_experts=8,
        top_k=4
    )
    
    ts_config = TimeSeriesConfig(
        seq_len=96,
        pred_len=24,
        target=args.target
    )
    
    train_config = TrainConfig(
        batch_size=256,
        epochs=200,
        lr=3e-4
    )
    
    # 训练模型
    train_linear_branch_model(
        model_config, ts_config, train_config, data,
        experiment_name=args.experiment_name,
        linear_weight=args.linear_weight
    )