"""
train_ultra_lsnt_enhanced.py - 增强版 Ultra-LSNT 训练脚本
针对风功率预测优化，旨在超越 LightGBM 基线
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
    set_seed, get_temperature
)

def create_enhanced_config(data_dim: int = 8) -> LSNTConfig:
    """创建增强版模型配置"""
    return LSNTConfig(
        input_dim=data_dim,
        hidden_dim=512,           # 增加隐藏维度 (原128→512)
        output_dim=1,
        num_blocks=4,             # 增加块数 (原3→4)
        num_experts=8,            # 增加专家数 (原4→8)
        top_k=4,                  # 增加Top-K (原2→4)
        dropout=0.2,              # 增加Dropout (原0.1→0.2)
        router_z_loss_coef=0.01,
        router_aux_loss_coef=0.01,
        router_jitter_noise=0.15, # 增加噪声 (原0.1→0.15)
        skip_threshold=0.5,
        dual_mode_expansion=4,
        probabilistic_mode='none',  # 点预测
        heteroscedastic_moe=True   # 启用异方差MoE
    )

def create_enhanced_train_config() -> TrainConfig:
    """创建增强版训练配置"""
    return TrainConfig(
        batch_size=128,           # 平衡训练速度与梯度稳定性
        num_workers=8,
        epochs=300,               # 增加训练轮数 (原200→300)
        lr=2e-4,                  # 降低学习率 (原3e-4→2e-4)
        weight_decay=0.02,        # 增加权重衰减 (原0.01→0.02)
        warmup_epochs=15,         # 增加warmup (原10→15)
        accumulation_steps=1,
        use_amp=True,             # 启用混合精度
        gradient_clip=1.0,
        aux_loss_weight=0.01,
        initial_temperature=2.0,
        final_temperature=0.05,   # 降低最终温度 (原0.1→0.05)
        save_dir='./checkpoints_ts_enhanced',
        log_interval=10,
        patience=40,              # 增加早停耐心 (原30→40)
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

def train_epoch_enhanced(model, loader, optimizer, scaler, device, temperature, cfg: TrainConfig):
    """增强版训练循环"""
    model.train()
    
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (seq_x, seq_y) in enumerate(loader):
        seq_x = seq_x.to(device, non_blocking=True)
        seq_y = seq_y.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            pred, aux_loss = model(seq_x, temperature=temperature)
            
            # 主损失 (MSE)
            main_loss = nn.MSELoss()(pred, seq_y)
            
            # 总损失
            loss = (main_loss + cfg.aux_loss_weight * aux_loss) / cfg.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % cfg.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        batch_size = seq_x.size(0)
        total_loss += loss.item() * cfg.accumulation_steps * batch_size
        total_main_loss += main_loss.item() * batch_size
        total_aux_loss += aux_loss.item() * batch_size
        total_samples += batch_size
        
        if batch_idx % cfg.log_interval == 0:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            avg_main = total_main_loss / total_samples if total_samples > 0 else 0
            avg_aux = total_aux_loss / total_samples if total_samples > 0 else 0
            print(f'  [{batch_idx:4d}/{len(loader)}] Loss: {avg_loss:.4f} | Main: {avg_main:.4f} | Aux: {avg_aux:.4f}')
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_main = total_main_loss / total_samples if total_samples > 0 else 0
    avg_aux = total_aux_loss / total_samples if total_samples > 0 else 0
    
    return {'loss': avg_loss, 'main_loss': avg_main, 'aux': avg_aux}

@torch.no_grad()
def evaluate_enhanced(model, loader, device, temperature, scaler_obj=None):
    """增强版评估"""
    model.eval()
    
    all_preds = []
    all_trues = []
    
    for seq_x, seq_y in loader:
        seq_x = seq_x.to(device, non_blocking=True)
        seq_y = seq_y.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            pred, _ = model(seq_x, temperature=temperature)
        
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
    
    metrics = compute_metrics(preds, trues)
    
    return metrics, preds, trues

def train_enhanced_model():
    """主训练函数"""
    set_seed(42)
    
    print("=" * 80)
    print("Ultra-LSNT 增强版训练 - 针对风功率预测优化")
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
    
    model_config = create_enhanced_config(data.shape[1])
    train_config = create_enhanced_train_config()
    
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
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {num_params:,}")
    print(f"配置: {model_config.num_blocks} blocks, {model_config.num_experts} experts, top-{model_config.top_k}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.lr, 
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=30,  # 周期长度
        T_mult=2,  # 周期倍增因子
        eta_min=1e-6  # 最小学习率
    )
    
    # 混合精度
    grad_scaler = torch.cuda.amp.GradScaler(enabled=train_config.use_amp)
    
    # 保存目录
    experiment_name = 'enhanced_v1'
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
        train_metrics = train_epoch_enhanced(
            model, train_loader, optimizer, grad_scaler, device, temperature, train_config
        )
        
        # 验证
        val_metrics, _, _ = evaluate_enhanced(model, val_loader, device, temperature, scaler)
        
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
        
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | Main: {train_metrics['main_loss']:.4f}")
        print(f"  Val   | MSE: {val_metrics['MSE']:.4f} | RMSE: {val_metrics['RMSE']:.4f} | MAE: {val_metrics['MAE']:.4f}")
        print(f"  R²: {val_metrics['R2']:.4f} | 时间: {epoch_time:.1f}s")
        
        # 早停
        if val_metrics['MSE'] < best_val_loss:
            best_val_loss = val_metrics['MSE']
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
        if (epoch + 1) % 10 == 0:
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
    
    # 训练结束
    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    
    # 加载最佳模型进行测试
    print("加载最佳模型进行测试...")
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试
    test_metrics, test_preds, test_trues = evaluate_enhanced(
        model, test_loader, device, temperature, scaler
    )
    
    print(f"\n测试集性能:")
    print(f"  MSE:  {test_metrics['MSE']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE:  {test_metrics['MAE']:.4f}")
    print(f"  MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"  R²:   {test_metrics['R2']:.4f}")
    
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
        json.dump(final_results, f, indent=2)
    
    # 保存预测结果
    np.savez(save_dir / 'test_predictions.npz', 
             predictions=test_preds, 
             ground_truth=test_trues)
    
    # 对比LightGBM基线
    try:
        import lightgbm as lgb
        print("\n对比LightGBM基线...")
        # 这里可以添加与LightGBM的对比逻辑
    except ImportError:
        print("LightGBM未安装，跳过基线对比")
    
    print(f"\n✅ 所有结果已保存至: {save_dir}")
    print(f"📊 最终R²分数: {test_metrics['R2']:.4f}")
    print(f"🕒 总训练时间: {sum([h['time'] for h in history]):.1f}s")
    
    return test_metrics, save_dir

if __name__ == '__main__':
    train_enhanced_model()