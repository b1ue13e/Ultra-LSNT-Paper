"""
train_ultra_lsnt_stable.py - 稳定版 Ultra-LSNT 训练脚本
解决数值不稳定问题，提供可靠的模型性能
"""
import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from pathlib import Path
from datetime import datetime

from ultra_lsnt_timeseries import (
    load_csv_data, TimeSeriesConfig, TrainConfig, LSNTConfig,
    UltraLSNTForecaster, create_dataloaders, compute_metrics,
    set_seed, get_temperature, StandardScaler
)

def create_stable_config(data_dim: int = 8) -> LSNTConfig:
    """创建稳定版模型配置"""
    return LSNTConfig(
        input_dim=data_dim,
        hidden_dim=192,           # 适中隐藏维度 (128→192)
        output_dim=1,
        num_blocks=3,             # 保持3块
        num_experts=6,            # 适中专家数
        top_k=2,                  # 保守的Top-K
        dropout=0.15,             # 适中Dropout
        router_z_loss_coef=0.01,
        router_aux_loss_coef=0.01,
        router_jitter_noise=0.1,
        skip_threshold=0.5,
        dual_mode_expansion=4,
        probabilistic_mode='none',
        heteroscedastic_moe=False  # 禁用异方差MoE以简化
    )

def create_stable_train_config() -> TrainConfig:
    """创建稳定版训练配置"""
    return TrainConfig(
        batch_size=128,
        num_workers=4,
        epochs=150,               # 减少epochs
        lr=1e-3,                  # 更低学习率
        weight_decay=0.01,
        warmup_epochs=5,          # 更短的warmup
        accumulation_steps=1,
        use_amp=True,
        gradient_clip=0.5,        # 更严格的梯度裁剪
        aux_loss_weight=0.01,
        initial_temperature=2.0,
        final_temperature=0.1,
        save_dir='./checkpoints_ts_stable',
        log_interval=20,          # 减少日志频率
        patience=20,              # 减少早停耐心
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

def safe_compute_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    """安全的指标计算，处理NaN/Inf"""
    preds = preds.flatten().astype(np.float64)
    trues = trues.flatten().astype(np.float64)
    
    # 检查NaN/Inf
    mask = np.isfinite(preds) & np.isfinite(trues)
    if np.sum(mask) == 0:
        return {'MSE': float('inf'), 'RMSE': float('inf'), 'MAE': float('inf'), 
                'MAPE': float('inf'), 'R2': float('-inf')}
    
    preds = preds[mask]
    trues = trues[mask]
    
    try:
        metrics = compute_metrics(preds, trues)
        # 确保所有指标都是有限值
        for k, v in metrics.items():
            if not np.isfinite(v):
                metrics[k] = float('inf') if k != 'R2' else float('-inf')
    except Exception as e:
        print(f"  警告: 指标计算失败: {e}")
        metrics = {'MSE': float('inf'), 'RMSE': float('inf'), 'MAE': float('inf'), 
                  'MAPE': float('inf'), 'R2': float('-inf')}
    
    return metrics

def train_epoch_stable(model, loader, optimizer, scaler, device, temperature, cfg: TrainConfig):
    """稳定版训练循环"""
    model.train()
    
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    total_samples = 0
    grad_norms = []
    
    optimizer.zero_grad()
    
    for batch_idx, (seq_x, seq_y) in enumerate(loader):
        seq_x = seq_x.to(device, non_blocking=True)
        seq_y = seq_y.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            pred, aux_loss = model(seq_x, temperature=temperature)
            
            # 检查NaN
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"  警告: batch {batch_idx} 预测包含NaN/Inf")
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            
            main_loss = nn.MSELoss()(pred, seq_y)
            
            # 检查损失NaN
            if torch.isnan(main_loss) or torch.isinf(main_loss):
                print(f"  警告: batch {batch_idx} 主损失为NaN/Inf，使用替代值")
                main_loss = torch.tensor(1.0, device=device)
            
            loss = (main_loss + cfg.aux_loss_weight * aux_loss) / cfg.accumulation_steps
        
        scaler.scale(loss).backward()
        
        # 计算梯度范数
        if (batch_idx + 1) % cfg.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.gradient_clip, error_if_nonfinite=False
            )
            grad_norms.append(total_norm.item() if torch.isfinite(total_norm) else 0.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        batch_size = seq_x.size(0)
        total_loss += loss.item() * cfg.accumulation_steps * batch_size
        total_main_loss += main_loss.item() * batch_size
        total_aux_loss += aux_loss.item() * batch_size
        total_samples += batch_size
        
        if batch_idx % cfg.log_interval == 0 and total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_main = total_main_loss / total_samples
            avg_aux = total_aux_loss / total_samples
            grad_avg = np.mean(grad_norms[-10:]) if grad_norms else 0.0
            print(f'  [{batch_idx:4d}/{len(loader)}] Loss: {avg_loss:.4f} | Main: {avg_main:.4f} | Aux: {avg_aux:.4f} | Grad: {grad_avg:.2f}')
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_main = total_main_loss / total_samples if total_samples > 0 else 0.0
    avg_aux = total_aux_loss / total_samples if total_samples > 0 else 0.0
    avg_grad = np.mean(grad_norms) if grad_norms else 0.0
    
    return {'loss': avg_loss, 'main_loss': avg_main, 'aux': avg_aux, 'grad_norm': avg_grad}

@torch.no_grad()
def evaluate_stable(model, loader, device, temperature, scaler_obj=None):
    """稳定版评估"""
    model.eval()
    
    all_preds = []
    all_trues = []
    
    for seq_x, seq_y in loader:
        seq_x = seq_x.to(device, non_blocking=True)
        seq_y = seq_y.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            pred, _ = model(seq_x, temperature=temperature)
        
        # 检查NaN
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print(f"   评估警告: 预测包含NaN/Inf，进行清理")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        all_preds.append(pred.cpu().numpy())
        all_trues.append(seq_y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    # 反标准化
    if scaler_obj is not None:
        target_mean = scaler_obj.mean[-1]
        target_std = scaler_obj.std[-1]
        preds = preds * target_std + target_mean
        trues = trues * target_std + target_mean
    
    metrics = safe_compute_metrics(preds, trues)
    
    return metrics, preds, trues

def train_stable_model():
    """主训练函数"""
    set_seed(42)
    
    print("=" * 80)
    print("Ultra-LSNT 稳定版训练 - 解决数值不稳定问题")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 加载数据
    data_path = 'wind_final.csv'
    target_col = 'power'
    
    print(f"加载数据: {data_path}")
    data, feature_names = load_csv_data(data_path, target_col)
    if data is None:
        print("❌ 数据加载失败")
        return
    
    print(f"数据形状: {data.shape}")
    print(f"特征: {feature_names}")
    
    # 创建配置
    ts_config = TimeSeriesConfig(
        seq_len=96,
        pred_len=24,
        target='power'
    )
    
    model_config = create_stable_config(data.shape[1])
    train_config = create_stable_train_config()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 数据加载器
    print("\n准备数据加载器...")
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        data, ts_config, train_config
    )
    
    # 模型
    print("\n初始化模型...")
    model = UltraLSNTForecaster(model_config, ts_config).to(device)
    
    # 自定义初始化（更稳定）
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {num_params:,}")
    print(f"配置: {model_config.num_blocks} blocks, {model_config.num_experts} experts, top-{model_config.top_k}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.lr, 
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器 - 线性预热 + 余弦退火
    def lr_lambda(epoch):
        # 线性预热
        if epoch < train_config.warmup_epochs:
            return float(epoch) / float(max(1, train_config.warmup_epochs))
        # 余弦退火
        progress = (epoch - train_config.warmup_epochs) / (train_config.epochs - train_config.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 混合精度
    grad_scaler = torch.cuda.amp.GradScaler(enabled=train_config.use_amp)
    
    # 保存目录
    experiment_name = 'stable_v1'
    save_dir = Path(train_config.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    model_config.save(save_dir / 'model_config.json')
    with open(save_dir / 'train_config.json', 'w') as f:
        json.dump(train_config.__dict__, f, indent=2)
    
    print(f"\n保存目录: {save_dir}")
    
    # 训练历史
    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n开始训练...")
    print("-" * 80)
    
    for epoch in range(train_config.epochs):
        epoch_start = time.time()
        temperature = get_temperature(epoch, train_config)
        
        print(f"\nEpoch {epoch+1}/{train_config.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Temp: {temperature:.3f}")
        
        # 训练
        train_metrics = train_epoch_stable(
            model, train_loader, optimizer, grad_scaler, device, temperature, train_config
        )
        
        # 验证
        val_metrics, _, _ = evaluate_stable(model, val_loader, device, temperature, scaler)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 记录
        record = {
            'epoch': epoch + 1,
            'temperature': temperature,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'train': train_metrics,
            'val': val_metrics,
            'time': epoch_time,
        }
        history.append(record)
        
        # 打印结果，处理Infinity显示
        val_mse = val_metrics['MSE']
        val_mse_str = f"{val_mse:.4f}" if np.isfinite(val_mse) else "Inf"
        val_r2 = val_metrics['R2']
        val_r2_str = f"{val_r2:.4f}" if np.isfinite(val_r2) else "-Inf"
        val_rmse = val_metrics['RMSE']
        val_rmse_str = f"{val_rmse:.4f}" if np.isfinite(val_rmse) else "Inf"
        
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | Main: {train_metrics['main_loss']:.4f} | Grad: {train_metrics.get('grad_norm', 0):.2f}")
        print(f"  Val   | MSE: {val_mse_str} | RMSE: {val_rmse_str} | R²: {val_r2_str}")
        print(f"  时间: {epoch_time:.1f}s")
        
        # 早停 - 只使用有效的验证损失
        if np.isfinite(val_mse) and val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
            }, save_dir / 'best_model.pth')
            
            print(f"  ✅ 新最佳: MSE={best_val_loss:.4f} (已保存)")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"\n⚠️  早停触发于 epoch {epoch+1}")
                break
        
        # 定期保存历史
        if (epoch + 1) % 5 == 0:
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2, default=str)
    
    # 训练结束
    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    
    # 加载最佳模型进行测试（如果存在）
    best_model_path = save_dir / 'best_model.pth'
    if best_model_path.exists():
        print("加载最佳模型进行测试...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 测试
        test_metrics, test_preds, test_trues = evaluate_stable(
            model, test_loader, device, temperature, scaler
        )
        
        print(f"\n测试集性能:")
        print(f"  MSE:  {test_metrics['MSE']:.2f}" if np.isfinite(test_metrics['MSE']) else "  MSE:  Inf")
        print(f"  RMSE: {test_metrics['RMSE']:.2f}" if np.isfinite(test_metrics['RMSE']) else "  RMSE: Inf")
        print(f"  MAE:  {test_metrics['MAE']:.2f}" if np.isfinite(test_metrics['MAE']) else "  MAE:  Inf")
        print(f"  MAPE: {test_metrics['MAPE']:.2f}%" if np.isfinite(test_metrics['MAPE']) else "  MAPE: Inf%")
        print(f"  R²:   {test_metrics['R2']:.4f}" if np.isfinite(test_metrics['R2']) else "  R²:   -Inf")
        
        # 保存测试结果
        final_results = {
            'test_metrics': test_metrics,
            'model_config': model_config.to_dict(),
            'best_epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['best_val_loss'],
            'training_time': sum([h['time'] for h in history]),
            'num_epochs_trained': len(history),
        }
        
        with open(save_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # 保存预测结果
        np.savez(save_dir / 'test_predictions.npz', 
                 predictions=test_preds, 
                 ground_truth=test_trues)
        
        print(f"\n✅ 所有结果已保存至: {save_dir}")
        
        # 对比现有基线
        try:
            from checkpoints_ts.main.final_results import test_metrics as baseline_metrics
            print("\n📊 与基线模型对比:")
            print(f"  基线 R²: {baseline_metrics.get('R2', 'N/A'):.4f}")
            print(f"  当前 R²: {test_metrics['R2']:.4f}" if np.isfinite(test_metrics['R2']) else "  当前 R²: -Inf")
        except:
            pass
    
    else:
        print("❌ 未找到最佳模型，可能训练过程中未保存")
    
    return save_dir

if __name__ == '__main__':
    train_stable_model()