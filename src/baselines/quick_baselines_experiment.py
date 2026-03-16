"""
Quick Traditional Baseline Experiments - Reduced epochs for faster execution
Generates results for all 4 datasets, 3 seeds, 4 models
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
import os
from typing import Dict, Tuple, Any
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check ARIMA availability
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class QuickBaselinesExperiment:
    """Quick baseline experiments with reduced epochs."""
    
    def __init__(self, protocol_path: str = "experiment_protocol_v2.json"):
        with open(protocol_path, 'r', encoding='utf-8') as f:
            self.protocol = json.load(f)
        
        self.datasets = self.protocol['datasets']['list']
        self.seeds = self.protocol['random_seeds']['values']
        self.results = []
        
        manifest_path = self.protocol['split_policy']['manifest_file']
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.split_manifest = json.load(f)
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset with unified 80/20 split."""
        dataset_info = self.protocol['datasets'][dataset_name]
        filepath = dataset_info['file']
        
        manifest_entry = None
        for ds in self.split_manifest['datasets']:
            if ds['name'] == dataset_name:
                manifest_entry = ds
                break
        
        if 'Load_history' in filepath:
            df = pd.read_csv(filepath, thousands=',')
            numeric_cols = [col for col in df.columns if col not in ['zone_id', 'year', 'month', 'day']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
        else:
            df = pd.read_csv(filepath)
        
        total_samples = len(df)
        
        train_indices = manifest_entry['training_set']['indices']
        train_start, train_end = map(int, train_indices.split(':'))
        
        if total_samples != manifest_entry['total_samples']:
            actual_train_end = int(total_samples * 0.8)
            train_df = df.iloc[:actual_train_end].copy()
            test_df = df.iloc[actual_train_end:].copy()
        else:
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()
        
        return train_df, test_df
    
    def get_target_data(self, dataset_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
        target_col = self.protocol['datasets'][dataset_name]['target_column']
        if target_col is None or target_col not in train_df.columns:
            target_col = train_df.columns[-1]
        return train_df[target_col].values, test_df[target_col].values
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        
        if len(y_true) == 0:
            return {'R2': -999, 'RMSE': 999, 'MAE': 999, 'MAPE': 999}
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    
    def run_arima(self, dataset_name: str, seed: int) -> Dict[str, Any]:
        if not STATSMODELS_AVAILABLE:
            return None
        
        np.random.seed(seed)
        
        try:
            train_df, test_df = self.load_dataset(dataset_name)
            y_train, y_test = self.get_target_data(dataset_name, train_df, test_df)
            
            print(f"  ARIMA on {dataset_name} (seed={seed})")
            
            start_time = time.time()
            
            # Use small subset for ARIMA to speed up
            max_train = min(5000, len(y_train))
            y_train_subset = y_train[-max_train:]
            
            model = ARIMA(y_train_subset, order=(1, 1, 1))
            model_fit = model.fit(method='statespace')
            y_pred = model_fit.forecast(steps=min(len(y_test), 1000))
            train_time = time.time() - start_time
            
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test_adj = y_test[:min_len]
            
            metrics = self.compute_metrics(y_test_adj, y_pred)
            
            return {
                'dataset': dataset_name, 'model': 'ARIMA', 'seed': seed,
                'split_id': '80_20_unified', 'train_time_s': train_time,
                'params': 3, **metrics
            }
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def run_svr(self, dataset_name: str, seed: int) -> Dict[str, Any]:
        np.random.seed(seed)
        
        try:
            train_df, test_df = self.load_dataset(dataset_name)
            y_train, y_test = self.get_target_data(dataset_name, train_df, test_df)
            
            print(f"  SVR on {dataset_name} (seed={seed})")
            
            seq_len = 24
            
            # Sample data for speed
            max_samples = min(20000, len(y_train) - seq_len)
            y_train_sample = y_train[-max_samples-seq_len:]
            
            X_train, y_train_seq = self._create_sequences(y_train_sample, seq_len, 1)
            X_test, y_test_seq = self._create_sequences(
                np.concatenate([y_train[-seq_len:], y_test]), seq_len, 1
            )
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
            
            start_time = time.time()
            model = SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
            model.fit(X_train_scaled, y_train_scaled)
            train_time = time.time() - start_time
            
            X_test_scaled = scaler_X.transform(X_test)
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test_adj = y_test[:min_len]
            
            metrics = self.compute_metrics(y_test_adj, y_pred)
            
            return {
                'dataset': dataset_name, 'model': 'SVR', 'seed': seed,
                'split_id': '80_20_unified', 'train_time_s': train_time,
                'params': len(model.support_), **metrics
            }
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def run_lstm(self, dataset_name: str, seed: int) -> Dict[str, Any]:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        try:
            train_df, test_df = self.load_dataset(dataset_name)
            y_train, y_test = self.get_target_data(dataset_name, train_df, test_df)
            
            print(f"  LSTM on {dataset_name} (seed={seed})")
            
            seq_len, pred_len = 96, 24
            
            # Sample for speed
            max_samples = min(10000, len(y_train) - seq_len - pred_len)
            y_train_sample = y_train[-max_samples-seq_len-pred_len:]
            
            X_train, y_train_seq = self._create_sequences(y_train_sample, seq_len, pred_len)
            X_test, y_test_seq = self._create_sequences(
                np.concatenate([y_train[-seq_len:], y_test]), seq_len, pred_len
            )
            
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
            y_train_scaled = scaler_y.fit_transform(y_train_seq)
            
            model = Sequential([
                LSTM(64, input_shape=(seq_len, 1)),
                Dense(pred_len)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            start_time = time.time()
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
            
            val_split = 0.1
            val_size = int(len(X_train_scaled) * val_split)
            
            model.fit(
                X_train_scaled[:-val_size], y_train_scaled[:-val_size],
                validation_data=(X_train_scaled[-val_size:], y_train_scaled[-val_size:]),
                epochs=20, batch_size=32, callbacks=[early_stop], verbose=0
            )
            train_time = time.time() - start_time
            
            X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            y_test_flat = y_test_seq.flatten()[:len(y_pred.flatten())]
            y_pred_flat = y_pred.flatten()[:len(y_test_flat)]
            
            metrics = self.compute_metrics(y_test_flat, y_pred_flat)
            
            return {
                'dataset': dataset_name, 'model': 'LSTM', 'seed': seed,
                'split_id': '80_20_unified', 'train_time_s': train_time,
                'params': model.count_params(), **metrics
            }
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def run_cnn_lstm(self, dataset_name: str, seed: int) -> Dict[str, Any]:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        try:
            train_df, test_df = self.load_dataset(dataset_name)
            y_train, y_test = self.get_target_data(dataset_name, train_df, test_df)
            
            print(f"  CNN-LSTM on {dataset_name} (seed={seed})")
            
            seq_len, pred_len = 96, 24
            
            max_samples = min(10000, len(y_train) - seq_len - pred_len)
            y_train_sample = y_train[-max_samples-seq_len-pred_len:]
            
            X_train, y_train_seq = self._create_sequences(y_train_sample, seq_len, pred_len)
            X_test, y_test_seq = self._create_sequences(
                np.concatenate([y_train[-seq_len:], y_test]), seq_len, pred_len
            )
            
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
            y_train_scaled = scaler_y.fit_transform(y_train_seq)
            
            model = Sequential([
                Conv1D(32, 3, activation='relu', input_shape=(seq_len, 1)),
                MaxPooling1D(2),
                LSTM(32),
                Dense(pred_len)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            start_time = time.time()
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
            
            val_split = 0.1
            val_size = int(len(X_train_scaled) * val_split)
            
            model.fit(
                X_train_scaled[:-val_size], y_train_scaled[:-val_size],
                validation_data=(X_train_scaled[-val_size:], y_train_scaled[-val_size:]),
                epochs=20, batch_size=32, callbacks=[early_stop], verbose=0
            )
            train_time = time.time() - start_time
            
            X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            y_test_flat = y_test_seq.flatten()[:len(y_pred.flatten())]
            y_pred_flat = y_pred.flatten()[:len(y_test_flat)]
            
            metrics = self.compute_metrics(y_test_flat, y_pred_flat)
            
            return {
                'dataset': dataset_name, 'model': 'CNN-LSTM', 'seed': seed,
                'split_id': '80_20_unified', 'train_time_s': train_time,
                'params': model.count_params(), **metrics
            }
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def _create_sequences(self, data: np.ndarray, seq_len: int, pred_len: int):
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+pred_len])
        return np.array(X), np.array(y)
    
    def run_all(self, models: list = None):
        if models is None:
            models = ['ARIMA', 'SVR', 'LSTM', 'CNN-LSTM']
        
        print("="*70)
        print("Quick Traditional Baselines - 80/20 Unified")
        print("="*70)
        
        for dataset in self.datasets:
            print(f"\nDataset: {dataset}")
            print("-"*50)
            
            for seed in self.seeds:
                print(f"\n  Seed: {seed}")
                
                for model_name in models:
                    if model_name == 'ARIMA':
                        result = self.run_arima(dataset, seed)
                    elif model_name == 'SVR':
                        result = self.run_svr(dataset, seed)
                    elif model_name == 'LSTM':
                        result = self.run_lstm(dataset, seed)
                    elif model_name == 'CNN-LSTM':
                        result = self.run_cnn_lstm(dataset, seed)
                    else:
                        continue
                    
                    if result:
                        self.results.append(result)
                        print(f"    R2={result['R2']:.4f}, RMSE={result['RMSE']:.4f}")
        
        # Save results
        df = pd.DataFrame(self.results)
        output_file = "traditional_baselines_80_20.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to {output_file}")
        print(f"Total experiments: {len(df)}")
        print("="*70)
        
        return df


if __name__ == "__main__":
    exp = QuickBaselinesExperiment()
    results = exp.run_all()
    print("\nSummary:")
    print(results.groupby(['dataset', 'model'])[['R2', 'RMSE']].mean())
