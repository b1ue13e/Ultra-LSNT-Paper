"""
run_gbdt.py - LightGBM 基线模型 (能源领域必比)
================================================
使用 MultiOutputRegressor 策略进行多步预测。
"""
import numpy as np
import time
import argparse
import joblib
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from ultra_lsnt_timeseries import (
    load_csv_data, create_dataloaders, TimeSeriesConfig, TrainConfig, compute_metrics
)


def flatten_data(loader):
    """将时序数据展平供 GBDT 使用"""
    X_list, y_list = [], []
    for bx, by in loader:
        # bx: [Batch, Seq, Feat] -> [Batch, Seq*Feat]
        batch_size = bx.shape[0]
        X_flat = bx.reshape(batch_size, -1).numpy()
        y_list.append(by.numpy())
        X_list.append(X_flat)
    
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def run_gbdt():
    print("🌲 启动 LightGBM 基线训练 (能源界守门员)...")
    
    # 1. 准备数据
    data_path = 'wind_final.csv'
    target = 'power'
    # 注意：GBDT 不需要 seq_len 那么长，通常短一点效果更好，但为了公平我们保持一致
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target=target)
    # batch_size 设大点是为了加快数据加载，GBDT 本身是全量训练
    train_config = TrainConfig(batch_size=1024) 
    
    data, _ = load_csv_data(data_path, target)
    if data is None:
        print("❌ 数据加载失败")
        return
        
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts_config, train_config)
    
    print("   🔄 正在展平数据...")
    X_train, y_train = flatten_data(train_loader)
    X_test, y_test = flatten_data(test_loader)
    
    print(f"   训练集形状: {X_train.shape}")
    
    # 2. 定义模型
    # LightGBM 默认支持并行，速度极快
    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    # 多步预测包装器
    model = MultiOutputRegressor(lgbm)
    
    # 3. 训练
    print("   ⏳ 开始训练 (这通常很快)...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"   ✅ 训练完成，耗时: {time.time() - start_time:.2f}s")
    
    # 4. 预测
    print("   🚀 开始预测...")
    preds = model.predict(X_test)
    
    # 5. 反归一化 & 评估
    target_mean = scaler.mean[-1]
    target_std = scaler.std[-1]
    
    preds_orig = preds * target_std + target_mean
    y_test_orig = y_test * target_std + target_mean
    
    metrics = compute_metrics(preds_orig, y_test_orig)
    
    print("\n📊 LightGBM 最终成绩:")
    print(f"   R²  : {metrics['R2']:.4f}")
    print(f"   MSE : {metrics['MSE']:.4f}")
    print(f"   MAE : {metrics['MAE']:.4f}")
    
    # 保存结果供 DM Test 使用
    np.save('preds_lgbm.npy', preds_orig)
    np.save('trues_lgbm.npy', y_test_orig)
    print("   💾 预测结果已保存至 preds_lgbm.npy")


if __name__ == '__main__':
    run_gbdt()
