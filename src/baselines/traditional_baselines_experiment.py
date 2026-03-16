"""
Traditional Baseline Experiments - 80/20 Unified Protocol
ARIMA, SVR, LSTM, CNN-LSTM on four datasets with 3 seeds
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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available, ARIMA experiments will be skipped")


class TraditionalBaselinesExperiment:
    """Traditional baseline experiments following unified protocol."""
    
    def __init__(self, protocol_path: str = "experiment_protocol_v2.json"):
        with open(protocol_path, 'r', encoding='utf-8') as f:
            self.protocol = json.load(f)
        
        self.datasets = self.protocol['datasets']['list']
        self.seeds = self.protocol['random_seeds']['values']
        self.results = []
        
        # Load split manifest
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
        
        if manifest_entry is None:
            raise ValueError(f"Dataset {dataset_name} not found in split manifest")
        
        # Load data
        if 'Load_history' in filepath:
            df = pd.read_csv(filepath, thousands=',')
            numeric_cols = [col for col in df.columns if col not in ['zone_id', 'year', 'month', 'day']]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
        else:
            df = pd.read_csv(filepath)
        
        total_samples = len(df)
        
        # Apply split
        train_indices = manifest_entry['training_set']['indices']
        test_indices = manifest_entry['test_set']['indices']
        train_start, train_end = map(int, train_indices.split(':'))
        
        if total_samples != manifest_entry['total_samples']:
            actual_train_end = int(total_samples * 0.8)
            train_df = df.iloc[:actual_train_end].copy()
            test_df = df.iloc[actual_train_end:].copy()
        else:
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[train_end:].copy()
        
        return train_df, test_df
    
    def get_target_data(self, dataset_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract target variable from dataframes."""
        target_col = self.protocol['datasets'][dataset_name]['target_column']
        
        if target_col is None or target_col not in train_df.columns:
            # Use last column
            target_col = train_df.columns[-1]
        
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        
        return y_train, y_test
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standardized metrics."""
        # Filter NaN
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {'R2': -999, 'RMSE': 999, 'MAE': 999, 'MAPE': 999}
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    
    def run_arima(self, dataset_name: str, seed: int) -> Dict[str, Any]:
        """Run ARIMA experiment."""
        if not STATSMODELS_AVAILABLE:
            return None
        
        np.random.seed(seed)
        
        try:
            train_df, test_df = self.load_dataset(dataset_name)
            y_train, y_test = self.get_target_data(dataset_name, train_df, test_df)
            
            print(f"  ARIMA on {dataset_name} (seed={seed}): train={len(y_train)}, test={len(y_test)}")
            
            # Use simple ARIMA(1,1,1) for speed
            start_time = time.time()
            
            model = ARIMA(y_train, order=(1, 1, 1))
            model_fit = model.fit(method='statespace')
            
            # Forecast
            y_pred = model_fit.forecast(steps=len(y_test))
            train_time = time.time() - start_time
            
            # Handle shape mismatch
            if len(y_pred) != len(y_test):
                min_len = min(len(y_pred), len(y_test))
                y_pred = y_pred[:min_len]
                y_test_adj = y_test[:min_len]
            else:
                y_test_adj = y_test
            
            metrics = self.compute_metrics(y_test_adj, y_pred)
            
            result = {
                'dataset': dataset_name,
                'model': 'ARIMA',
                'seed': seed,
                'split_id': '80_20_unified',
                'train_time_s': train_time,
                'params': 3,  # p, d, q
                **metrics
            }
            
            print(f"    R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")
            return result
            
        except Exception as e:
            print(f"  ARIMA error on {dataset_name}: {e}")
            return None
    
    def run_svr(self, dataset_name: str, seed: int) -> Dict[str, Any]:
        """Run SVR experiment."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        try:
            train_df, test_df = self.load_dataset(dataset_name)
            y_train, y_test = self.get_target_data(dataset_name, train_df, test_df)
            
            print(f"  SVR on {dataset_name} (seed={seed}): train={len(y_train)}, test={len(y_test)}")
            
            # Create sequences for SVR (single-step)
            seq_len = 24
            
            X_train, y_train_seq = self._create_sequences(y_train, seq_len, 1)
            X_test, y_test_seq = self._create_sequences(
                np.concatenate([y_train[-seq_len:], y_test]), seq_len, 1
            )
            
            # Scale features
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
            
            # Train SVR
            start_time = time.time()
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
            model.fit(X_train_scaled, y_train_scaled)
            train_time = time.time() - start_time
            
            # Predict
            X_test_scaled = scaler_X.transform(X_test)
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Align lengths
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test_adj = y_test[:min_len]
            
            metrics = self.compute_metrics(y_test_adj, y_pred)
            
            # Measure latency
            sample_input = X_test_scaled[:1]
            latencies = []
            for _ in range(110):
                t0 = time.perf_counter()
                _ = model.predict(sample_input)
                latencies.append((time.perf_counter() - t0) * 1000)
            latency_ms = np.median(latencies[10:])  # Exclude warm-up
            
            result = {
                'dataset': dataset_name,
                'model': 'SVR',
                'seed': seed,
                'split_id': '80_20_unified',
                'train_time_s': train_time,
                'latency_ms': latency_ms,
                'infer_ms': latency_ms,
                'params': len(model.support_),
                **metrics
            }
            
            print(f"    R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, latency={latency_ms:.2f}ms")
            return result
            
        except Exception as e:
            print(f"  SVR error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_lstm(self, dataset_name: str, seed: int) -> Dict[str, Any]:
        """Run LSTM experiment."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        try:
            train_df, test_df = self.load_dataset(dataset_name)
            y_train, y_test = self.get_target_data(dataset_name, train_df, test_df)
            
            print(f"  LSTM on {dataset_name} (seed={seed}): train={len(y_train)}, test={len(y_test)}")
            
            seq_len = 96
            pred_len = 24
            
            # Create sequences
            X_train, y_train_seq = self._create_sequences(y_train, seq_len, pred_len)
            X_test, y_test_seq = self._create_sequences(
                np.concatenate([y_train[-seq_len:], y_test]), seq_len, pred_len
            )
            
            # Reshape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Scale
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
            y_train_scaled = scaler_y.fit_transform(y_train_seq)
            
            # Build model
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
                LSTM(32),
                Dense(pred_len)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            # Train
            start_time = time.time()
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            
            # Simple validation split
            val_split = 0.1
            val_size = int(len(X_train_scaled) * val_split)
            
            history = model.fit(
                X_train_scaled[:-val_size], y_train_scaled[:-val_size],
                validation_data=(X_train_scaled[-val_size:], y_train_scaled[-val_size:]),
                epochs=50, batch_size=32, callbacks=[early_stop], verbose=0
            )
            train_time = time.time() - start_time
            
            # Predict
            X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            # Flatten for metrics
            y_test_flat = y_test_seq.flatten()[:len(y_pred.flatten())]
            y_pred_flat = y_pred.flatten()[:len(y_test_flat)]
            
            metrics = self.compute_metrics(y_test_flat, y_pred_flat)
            
            # Measure latency
            sample_input = X_test_scaled[:1]
            latencies = []
            for _ in range(110):
                t0 = time.perf_counter()
                _ = model.predict(sample_input, verbose=0)
                latencies.append((time.perf_counter() - t0) * 1000)
            latency_ms = np.median(latencies[10:])
            
            result = {
                'dataset': dataset_name,
                'model': 'LSTM',
                'seed': seed,
                'split_id': '80_20_unified',
                'train_time_s': train_time,
                'latency_ms': latency_ms,
                'infer_ms': latency_ms,
                'params': model.count_params(),
                **metrics
            }
            
            print(f"    R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, latency={latency_ms:.2f}ms")
            return result
            
        except Exception as e:
            print(f"  LSTM error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_cnn_lstm(self, dataset_name: str, seed: int) -> Dict[str, Any]:
        """Run CNN-LSTM experiment."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        try:
            train_df, test_df = self.load_dataset(dataset_name)
            y_train, y_test = self.get_target_data(dataset_name, train_df, test_df)
            
            print(f"  CNN-LSTM on {dataset_name} (seed={seed}): train={len(y_train)}, test={len(y_test)}")
            
            seq_len = 96
            pred_len = 24
            
            # Create sequences
            X_train, y_train_seq = self._create_sequences(y_train, seq_len, pred_len)
            X_test, y_test_seq = self._create_sequences(
                np.concatenate([y_train[-seq_len:], y_test]), seq_len, pred_len
            )
            
            # Reshape
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Scale
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
            y_train_scaled = scaler_y.fit_transform(y_train_seq)
            
            # Build CNN-LSTM
            model = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=(seq_len, 1)),
                MaxPooling1D(2),
                LSTM(64, return_sequences=True),
                LSTM(32),
                Dense(pred_len)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            # Train
            start_time = time.time()
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            
            val_split = 0.1
            val_size = int(len(X_train_scaled) * val_split)
            
            history = model.fit(
                X_train_scaled[:-val_size], y_train_scaled[:-val_size],
                validation_data=(X_train_scaled[-val_size:], y_train_scaled[-val_size:]),
                epochs=50, batch_size=32, callbacks=[early_stop], verbose=0
            )
            train_time = time.time() - start_time
            
            # Predict
            X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            # Flatten
            y_test_flat = y_test_seq.flatten()[:len(y_pred.flatten())]
            y_pred_flat = y_pred.flatten()[:len(y_test_flat)]
            
            metrics = self.compute_metrics(y_test_flat, y_pred_flat)
            
            # Measure latency
            sample_input = X_test_scaled[:1]
            latencies = []
            for _ in range(110):
                t0 = time.perf_counter()
                _ = model.predict(sample_input, verbose=0)
                latencies.append((time.perf_counter() - t0) * 1000)
            latency_ms = np.median(latencies[10:])
            
            result = {
                'dataset': dataset_name,
                'model': 'CNN-LSTM',
                'seed': seed,
                'split_id': '80_20_unified',
                'train_time_s': train_time,
                'latency_ms': latency_ms,
                'infer_ms': latency_ms,
                'params': model.count_params(),
                **metrics
            }
            
            print(f"    R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, latency={latency_ms:.2f}ms")
            return result
            
        except Exception as e:
            print(f"  CNN-LSTM error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_sequences(self, data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences."""
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+pred_len])
        return np.array(X), np.array(y)
    
    def run_all_experiments(self, models: list = None) -> pd.DataFrame:
        """Run all traditional baseline experiments."""
        if models is None:
            models = ['ARIMA', 'SVR', 'LSTM', 'CNN-LSTM']
        
        print("=" * 70)
        print("Traditional Baselines - 80/20 Unified Protocol")
        print("=" * 70)
        
        for dataset in self.datasets:
            print(f"\nDataset: {dataset}")
            print("-" * 50)
            
            for seed in self.seeds:
                print(f"\n  Seed: {seed}")
                
                for model_name in models:
                    if model_name == 'ARIMA' and dataset in ['wind_cn', 'electricity', 'air_quality', 'gefcom']:
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
        
        # Save results
        df = pd.DataFrame(self.results)
        output_file = "traditional_baselines_80_20.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{'=' * 70}")
        print(f"Results saved to {output_file}")
        print(f"Total experiments: {len(df)}")
        print("=" * 70)
        
        return df


if __name__ == "__main__":
    experiment = TraditionalBaselinesExperiment()
    results_df = experiment.run_all_experiments()
    print("\nSummary:")
    print(results_df.groupby(['dataset', 'model'])[['R2', 'RMSE']].mean())
