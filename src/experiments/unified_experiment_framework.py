"""
Unified Experiment Framework for 80/20 Time Series Forecasting

This module provides a standardized framework for running all experiments
according to the experiment_protocol_v2.json specifications.
"""

import json
import numpy as np
import pandas as pd
import time
import os
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

warnings.filterwarnings('ignore')


@dataclass
class ExperimentResult:
    """Standardized experiment result container."""
    dataset: str
    model: str
    seed: int
    split_id: str
    R2: float
    RMSE: float
    MAE: float
    MAPE: float = np.nan
    latency_ms_batch1: float = np.nan
    model_size_mib: float = np.nan
    active_params_m: float = np.nan
    energy_mj_sample: float = np.nan
    train_time_s: float = np.nan
    infer_ms: float = np.nan
    params: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnifiedExperimentFramework:
    """Unified framework for all experiments."""
    
    def __init__(self, protocol_path: str = "experiment_protocol_v2.json"):
        """Initialize framework with protocol file."""
        with open(protocol_path, 'r', encoding='utf-8') as f:
            self.protocol = json.load(f)
        
        self.datasets = self.protocol['datasets']['list']
        self.seeds = self.protocol['random_seeds']['values']
        self.metrics_fields = self.protocol['metrics']['output_fields']
        
        # Load split manifest
        manifest_path = self.protocol['split_policy']['manifest_file']
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.split_manifest = json.load(f)
        
        self.results = []
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset according to unified 80/20 split."""
        dataset_info = self.protocol['datasets'][dataset_name]
        filepath = dataset_info['file']
        
        # Find in manifest
        manifest_entry = None
        for ds in self.split_manifest['datasets']:
            if ds['name'] == dataset_name:
                manifest_entry = ds
                break
        
        if manifest_entry is None:
            raise ValueError(f"Dataset {dataset_name} not found in split manifest")
        
        # Load data with special handling
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
        test_start, test_end = map(int, test_indices.split(':'))
        
        if total_samples != manifest_entry['total_samples']:
            actual_train_end = int(total_samples * 0.8)
            train_df = df.iloc[:actual_train_end].copy()
            test_df = df.iloc[actual_train_end:].copy()
        else:
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
        
        return train_df, test_df
    
    def get_target_column(self, dataset_name: str) -> str:
        """Get target column for dataset."""
        return self.protocol['datasets'][dataset_name]['target_column']
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standardized metrics."""
        # R2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def measure_latency(self, model, sample_input: np.ndarray, 
                       num_warmup: int = 10, num_runs: int = 100) -> float:
        """Measure CPU batch-1 inference latency."""
        import time
        
        # Warm-up
        for _ in range(num_warmup):
            try:
                _ = model.predict(sample_input[:1]) if hasattr(model, 'predict') else model(sample_input[:1])
            except:
                pass
        
        # Measurement
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            try:
                _ = model.predict(sample_input[:1]) if hasattr(model, 'predict') else model(sample_input[:1])
            except:
                pass
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return np.median(latencies)
    
    def add_result(self, result: ExperimentResult):
        """Add experiment result."""
        self.results.append(result.to_dict())
    
    def save_results(self, filepath: str = "all_models_clean_80_20.csv"):
        """Save results to CSV."""
        if not self.results:
            print("Warning: No results to save")
            return
        
        df = pd.DataFrame(self.results)
        
        # Ensure all required fields are present
        for field in self.metrics_fields:
            if field not in df.columns:
                df[field] = np.nan
        
        # Reorder columns
        df = df[[c for c in self.metrics_fields if c in df.columns]]
        
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return df
    
    def create_sequences(self, data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series forecasting."""
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+pred_len])
        return np.array(X), np.array(y)


def load_protocol(protocol_path: str = "experiment_protocol_v2.json") -> Dict[str, Any]:
    """Load experiment protocol."""
    with open(protocol_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test framework
    framework = UnifiedExperimentFramework()
    print("Unified Experiment Framework initialized")
    print(f"Datasets: {framework.datasets}")
    print(f"Seeds: {framework.seeds}")
    
    # Test loading
    for dataset in framework.datasets[:1]:
        train, test = framework.load_dataset(dataset)
        print(f"\nDataset: {dataset}")
        print(f"  Train: {len(train)}, Test: {len(test)}")
        print(f"  Target: {framework.get_target_column(dataset)}")
