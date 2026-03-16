"""
COA-BiLSTM Experiment Script

This script implements BiLSTM with Cuckoo Optimization Algorithm (COA)
for hyperparameter optimization across five datasets with unified 80/20 splits.

Features:
1. BiLSTM model with configurable architecture
2. COA for hyperparameter optimization
3. Five datasets: wind_cn, wind_us, electricity, air_quality, gefcom
4. Unified 80/20 chronological splits
5. 3 random seeds for robust evaluation
6. Intensive search with high parameter space coverage
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import json
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Add project path
sys.path.append('论文1')

# Import unified split utilities
from unified_split_utils import load_and_split_dataset
from coa_algorithm import CuckooOptimizationAlgorithm

warnings.filterwarnings('ignore')


class BiLSTMRegressor(nn.Module):
    """Bidirectional LSTM for time series forecasting."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # Use the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


def prepare_sequences(data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for time series forecasting."""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, -1])  # Last column is target
    return np.array(X), np.array(y)


def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                lr: float, weight_decay: float, epochs: int, device: torch.device) -> Dict[str, Any]:
    """Train the BiLSTM model and return metrics."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }


def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                   device: torch.device) -> Dict[str, float]:
    """Evaluate model on test set and return metrics."""
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
    
    y_true = y_test
    
    # Calculate metrics
    mse = np.mean((predictions - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_true))
    
    # R² score
    ss_res = np.sum((y_true - predictions) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'R2': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MSE': float(mse)
    }


def create_objective_function(dataset_name: str, seq_len: int, pred_len: int,
                              device: torch.device, seed: int = 42) -> callable:
    """Create objective function for COA optimization."""
    
    def objective(params: np.ndarray) -> float:
        """Objective function to minimize (validation loss)."""
        # Decode parameters
        hidden_dim = int(params[0])  # 32 to 256
        num_layers = int(params[1])  # 1 to 4
        dropout = params[2]          # 0.1 to 0.5
        lr = params[3]               # 1e-4 to 1e-2
        weight_decay = params[4]     # 1e-6 to 1e-3
        batch_size = int(params[5])  # 16 to 128
        
        # Load and prepare dataset
        try:
            if dataset_name == "wind_cn":
                data_path = "论文1/processed_wind.csv"
                target_col = "power"
            elif dataset_name == "wind_us":
                data_path = "论文1/wind_us.csv"
                target_col = "power (MW)"
            elif dataset_name == "electricity":
                data_path = "论文1/Load_history.csv"
                target_col = "load"
            elif dataset_name == "air_quality":
                data_path = "论文1/air_quality_ready.csv"
                target_col = "PM2.5"
            elif dataset_name == "gefcom":
                data_path = "论文1/gefcom_ready.csv"
                target_col = "load"
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Load data with unified split
            train_df, test_df = load_and_split_dataset(dataset_name, data_path)
            
            # Use last column as target, all others as features
            data = train_df.values
            if dataset_name in ["wind_cn", "wind_us"]:
                # For wind datasets, use appropriate target column name
                target_col = "power" if dataset_name == "wind_cn" else "power (MW)"
                if target_col in train_df.columns:
                    # Reorder columns to put target last
                    cols = [c for c in train_df.columns if c != target_col] + [target_col]
                    data = train_df[cols].values
            
            # Prepare sequences
            X, y = prepare_sequences(data, seq_len, pred_len)
            
            # Split into train/val (80% of training for actual training, 20% for validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create data loaders
            train_loader, val_loader = create_dataloaders(
                X_train, y_train, X_val, y_val, batch_size
            )
            
            # Create model
            input_dim = X.shape[2]  # Number of features
            model = BiLSTMRegressor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=pred_len,
                dropout=dropout
            )
            
            # Train model
            training_results = train_model(
                model, train_loader, val_loader,
                lr=lr, weight_decay=weight_decay,
                epochs=20,  # Fixed epochs for quick evaluation
                device=device
            )
            
            return training_results['best_val_loss']
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return float('inf')  # Return worst possible value
    
    return objective


def run_coa_bilstm_experiment(dataset_name: str, seed: int = 42, 
                              intensive_search: bool = True) -> Dict[str, Any]:
    """Run COA-BiLSTM experiment for a single dataset and seed."""
    print(f"\n{'='*60}")
    print(f"COA-BiLSTM Experiment: {dataset_name} (seed={seed})")
    print(f"{'='*60}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Fixed parameters
    seq_len = 96  # Lookback window
    pred_len = 24  # Prediction horizon
    
    # Define parameter bounds for COA
    # [hidden_dim, num_layers, dropout, lr, weight_decay, batch_size]
    bounds = [
        (32, 256),      # hidden_dim
        (1, 4),         # num_layers
        (0.1, 0.5),     # dropout
        (1e-4, 1e-2),   # learning rate
        (1e-6, 1e-3),   # weight decay
        (16, 128)       # batch_size
    ]
    
    # Create objective function
    objective_func = create_objective_function(dataset_name, seq_len, pred_len, device, seed)
    
    # COA configuration
    population_size = 30 if intensive_search else 20
    max_iterations = 50 if intensive_search else 30
    
    # Initialize COA
    coa = CuckooOptimizationAlgorithm(
        objective_func=objective_func,
        bounds=bounds,
        population_size=population_size,
        nest_abandonment_prob=0.25,
        step_size_alpha=0.01,
        levy_beta=1.5,
        max_iterations=max_iterations,
        random_seed=seed
    )
    
    # Run optimization
    print(f"\nStarting COA optimization ({max_iterations} iterations)...")
    start_time = time.time()
    result = coa.optimize(verbose=True)
    optimization_time = time.time() - start_time
    
    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    print(f"Best fitness (validation loss): {result['best_fitness']:.6f}")
    print(f"Best parameters:")
    param_names = ["hidden_dim", "num_layers", "dropout", "lr", "weight_decay", "batch_size"]
    for name, value in zip(param_names, result['best_solution']):
        print(f"  {name}: {value}")
    
    # Final evaluation with best parameters
    print(f"\nFinal evaluation on test set...")
    
    # Decode best parameters
    best_params = result['best_solution']
    hidden_dim = int(best_params[0])
    num_layers = int(best_params[1])
    dropout = best_params[2]
    lr = best_params[3]
    weight_decay = best_params[4]
    batch_size = int(best_params[5])
    
    # Load data for final training
    if dataset_name == "wind_cn":
        data_path = "论文1/processed_wind.csv"
        target_col = "power"
    elif dataset_name == "wind_us":
        data_path = "论文1/wind_us.csv"
        target_col = "power (MW)"
    elif dataset_name == "electricity":
        data_path = "论文1/Load_history.csv"
        target_col = "load"
    elif dataset_name == "air_quality":
        data_path = "论文1/air_quality_ready.csv"
        target_col = "PM2.5"
    elif dataset_name == "gefcom":
        data_path = "论文1/gefcom_ready.csv"
        target_col = "load"
    
    train_df, test_df = load_and_split_dataset(dataset_name, data_path)
    
    # Prepare full training data
    train_data = train_df.values
    if dataset_name in ["wind_cn", "wind_us"]:
        # Use appropriate target column name
        target_col = "power" if dataset_name == "wind_cn" else "power (MW)"
        if target_col in train_df.columns:
            cols = [c for c in train_df.columns if c != target_col] + [target_col]
            train_data = train_df[cols].values
    
    # Prepare test data
    test_data = test_df.values
    if dataset_name in ["wind_cn", "wind_us"]:
        target_col = "power" if dataset_name == "wind_cn" else "power (MW)"
        if target_col in test_df.columns:
            cols = [c for c in test_df.columns if c != target_col] + [target_col]
            test_data = test_df[cols].values
    
    # Prepare sequences
    X_train, y_train = prepare_sequences(train_data, seq_len, pred_len)
    X_test, y_test = prepare_sequences(test_data, seq_len, pred_len)
    
    # Split training into train/val for final model
    split_idx = int(len(X_train) * 0.8)
    X_train_final, X_val_final = X_train[:split_idx], X_train[split_idx:]
    y_train_final, y_val_final = y_train[:split_idx], y_train[split_idx:]
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        X_train_final, y_train_final, X_val_final, y_val_final, batch_size
    )
    
    # Create and train final model
    input_dim = X_train.shape[2]
    model = BiLSTMRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=pred_len,
        dropout=dropout
    ).to(device)
    
    # Train with more epochs for final model
    final_training_results = train_model(
        model, train_loader, val_loader,
        lr=lr, weight_decay=weight_decay,
        epochs=50,  # More epochs for final model
        device=device
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, device)
    
    # Compile results
    experiment_results = {
        'dataset': dataset_name,
        'seed': seed,
        'best_parameters': {
            param_names[i]: float(best_params[i]) for i in range(len(param_names))
        },
        'optimization_history': {
            'best_fitness_history': result['history']['best_fitness'],
            'avg_fitness_history': result['history']['avg_fitness'],
            'iterations': result['history']['iterations']
        },
        'training_metrics': final_training_results,
        'test_metrics': test_metrics,
        'optimization_time': optimization_time,
        'model_architecture': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'output_dim': pred_len
        },
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\nTest Metrics:")
    print(f"  R²: {test_metrics['R2']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE: {test_metrics['MAE']:.4f}")
    
    return experiment_results


def run_all_coa_bilstm_experiments():
    """Run COA-BiLSTM experiments for all datasets and seeds."""
    datasets = ["wind_cn", "wind_us", "electricity", "air_quality", "gefcom"]
    seeds = [42, 123, 999]  # 3 seeds as requested (consistent with other experiments)
    
    all_results = []
    
    for dataset in datasets:
        dataset_results = []
        for seed in seeds:
            try:
                result = run_coa_bilstm_experiment(
                    dataset_name=dataset,
                    seed=seed,
                    intensive_search=True
                )
                dataset_results.append(result)
                
                # Save intermediate result
                with open(f'coa_bilstm_{dataset}_seed{seed}.json', 'w') as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                print(f"Error running experiment for {dataset} (seed={seed}): {e}")
                continue
        
        # Aggregate results for this dataset
        if dataset_results:
            all_results.extend(dataset_results)
    
    # Save all results
    if all_results:
        with open('coa_bilstm_all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary CSV
        summary_rows = []
        for result in all_results:
            summary_rows.append({
                'dataset': result['dataset'],
                'seed': result['seed'],
                'R2': result['test_metrics']['R2'],
                'RMSE': result['test_metrics']['RMSE'],
                'MAE': result['test_metrics']['MAE'],
                'hidden_dim': result['best_parameters']['hidden_dim'],
                'num_layers': result['best_parameters']['num_layers'],
                'dropout': result['best_parameters']['dropout'],
                'lr': result['best_parameters']['lr'],
                'weight_decay': result['best_parameters']['weight_decay'],
                'batch_size': result['best_parameters']['batch_size'],
                'optimization_time': result['optimization_time']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('coa_bilstm_summary.csv', index=False)
        
        print(f"\n{'='*60}")
        print("All COA-BiLSTM Experiments Completed!")
        print(f"{'='*60}")
        print(f"Results saved to:")
        print(f"  - coa_bilstm_all_results.json (detailed results)")
        print(f"  - coa_bilstm_summary.csv (summary table)")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        for dataset in datasets:
            dataset_results = [r for r in all_results if r['dataset'] == dataset]
            if dataset_results:
                r2_scores = [r['test_metrics']['R2'] for r in dataset_results]
                rmse_scores = [r['test_metrics']['RMSE'] for r in dataset_results]
                print(f"\n{dataset}:")
                print(f"  R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
                print(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    
    return all_results


if __name__ == "__main__":
    print("COA-BiLSTM Experimental Suite")
    print("=============================")
    print("This script runs COA-optimized BiLSTM on four datasets")
    print("with unified 80/20 splits and 3 random seeds.")
    print("\nDatasets: wind, electricity, air_quality, gefcom")
    print("Seeds: 42, 123, 456")
    print("Intensive search: Yes")
    
    # Check if required files exist
    required_files = [
        "论文1/processed_wind.csv",
        "论文1/Load_history.csv", 
        "论文1/air_quality_ready.csv",
        "论文1/gefcom_ready.csv",
        "split_manifest_80_20_unified.json",
        "unified_split_utils.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"\nWarning: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run coa_bwo_unified_split.py first to generate split manifest.")
    else:
        print("\nAll required files found. Starting experiments...")
        run_all_coa_bilstm_experiments()