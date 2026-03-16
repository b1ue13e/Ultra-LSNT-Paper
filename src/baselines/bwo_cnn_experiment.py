"""
BWO-CNN Experiment Script

This script implements Convolutional Neural Network (CNN) with
Black Widow Optimization Algorithm (BWO) for hyperparameter optimization
across five datasets with unified 80/20 splits.

Features:
1. 1D CNN for time series forecasting
2. BWO for hyperparameter optimization (filters, kernel sizes, dropout, etc.)
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
sys.path.append('.')

# Import unified split utilities and BWO algorithm
from unified_split_utils import load_and_split_dataset
from bwo_algorithm import BlackWidowOptimization

warnings.filterwarnings('ignore')


class CNNRegressor(nn.Module):
    """1D CNN for time series forecasting."""
    
    def __init__(self, input_dim: int, seq_len: int, pred_len: int,
                 filters: List[int], kernel_sizes: List[int], 
                 dropout: float = 0.2, dense_units: int = 128):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.pred_len = pred_len
        
        # CNN layers
        cnn_layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            padding = kernel_size // 2  # Same padding
            cnn_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            
            # Max pooling for first two layers
            if i < 2:
                cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate output size after CNN layers
        with torch.no_grad():
            test_input = torch.zeros(1, input_dim, seq_len)
            test_output = self.cnn(test_input)
            cnn_output_size = test_output.numel() // 1  # Flattened size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units // 2, pred_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        # Convert to (batch, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # CNN layers
        x = self.cnn(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        
        return x


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
    """Train the CNN model and return metrics."""
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


def decode_cnn_params(params: np.ndarray) -> Dict[str, Any]:
    """Decode BWO parameters to CNN architecture."""
    # Parameter mapping:
    # [filter1, filter2, filter3, kernel1, kernel2, kernel3, dropout, lr, weight_decay, batch_size, dense_units]
    
    filters = [
        int(16 * (2 ** params[0])),  # 16, 32, 64, 128
        int(16 * (2 ** params[1])),  # 16, 32, 64, 128  
        int(16 * (2 ** params[2]))   # 16, 32, 64, 128
    ]
    
    kernel_sizes = [
        int(3 + 2 * params[3]),  # 3, 5, 7, 9
        int(3 + 2 * params[4]),  # 3, 5, 7, 9
        int(3 + 2 * params[5])   # 3, 5, 7, 9
    ]
    
    dropout = 0.1 + 0.4 * params[6]  # 0.1 to 0.5
    lr = 10 ** (-4 + 2 * params[7])  # 10^-4 to 10^-2
    weight_decay = 10 ** (-6 + 3 * params[8])  # 10^-6 to 10^-3
    batch_size = int(16 * (2 ** params[9]))  # 16, 32, 64, 128
    dense_units = int(64 * (2 ** params[10]))  # 64, 128, 256
    
    return {
        'filters': filters,
        'kernel_sizes': kernel_sizes,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'dense_units': dense_units
    }


def create_cnn_objective_function(dataset_name: str, seq_len: int, pred_len: int,
                                  device: torch.device, seed: int = 42) -> callable:
    """Create objective function for BWO optimization of CNN."""
    
    def objective(params: np.ndarray) -> float:
        """Objective function to minimize (validation loss)."""
        try:
            # Decode parameters
            decoded = decode_cnn_params(params)
            
            # Load dataset
            if dataset_name == "wind_cn":
                data_path = "论文1/processed_wind.csv"
                target_col = "power"
            elif dataset_name == "wind_us":
                data_path = "论文1/wind_us.csv"
                target_col = "power (MW)"
            elif dataset_name == "electricity":
                data_path = "论文1/Load_history.csv"
            elif dataset_name == "air_quality":
                data_path = "论文1/air_quality_ready.csv"
            elif dataset_name == "gefcom":
                data_path = "论文1/gefcom_ready.csv"
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Load data with unified split
            train_df, _ = load_and_split_dataset(dataset_name, data_path)
            
            # Use last column as target, all others as features
            data = train_df.values
            if dataset_name in ["wind_cn", "wind_us"] and target_col in train_df.columns:
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
                X_train, y_train, X_val, y_val, decoded['batch_size']
            )
            
            # Create model
            input_dim = X.shape[2]  # Number of features
            model = CNNRegressor(
                input_dim=input_dim,
                seq_len=seq_len,
                pred_len=pred_len,
                filters=decoded['filters'],
                kernel_sizes=decoded['kernel_sizes'],
                dropout=decoded['dropout'],
                dense_units=decoded['dense_units']
            )
            
            # Train model
            training_results = train_model(
                model, train_loader, val_loader,
                lr=decoded['lr'],
                weight_decay=decoded['weight_decay'],
                epochs=15,  # Reduced epochs for faster evaluation
                device=device
            )
            
            return training_results['best_val_loss']
            
        except Exception as e:
            print(f"Error in CNN objective function: {e}")
            return float('inf')  # Return worst possible value
    
    return objective


def run_bwo_cnn_experiment(dataset_name: str, seed: int = 42,
                           intensive_search: bool = True) -> Dict[str, Any]:
    """Run BWO-CNN experiment for a single dataset and seed."""
    print(f"\n{'='*60}")
    print(f"BWO-CNN Experiment: {dataset_name} (seed={seed})")
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
    
    # Define parameter bounds for BWO
    # [filter1_pow, filter2_pow, filter3_pow, kernel1_idx, kernel2_idx, kernel3_idx, 
    #  dropout, log10_lr, log10_weight_decay, batch_size_pow, dense_units_pow]
    # All parameters normalized to [0, 1] range
    bounds = [
        (0, 3),  # filter1: 2^0=1 to 2^3=8 -> 16*2^param (16 to 128)
        (0, 3),  # filter2
        (0, 3),  # filter3
        (0, 3),  # kernel1: 0->3, 1->5, 2->7, 3->9
        (0, 3),  # kernel2
        (0, 3),  # kernel3
        (0, 1),  # dropout: 0.1 to 0.5
        (0, 1),  # log10_lr: -4 to -2
        (0, 1),  # log10_weight_decay: -6 to -3
        (0, 3),  # batch_size_pow: 16 to 128
        (0, 2),  # dense_units_pow: 64 to 256
    ]
    
    # Create objective function
    objective_func = create_cnn_objective_function(dataset_name, seq_len, pred_len, device, seed)
    
    # BWO configuration
    population_size = 35 if intensive_search else 25
    max_iterations = 50 if intensive_search else 30
    
    # Initialize BWO
    bwo = BlackWidowOptimization(
        objective_func=objective_func,
        bounds=bounds,
        population_size=population_size,
        procreation_rate=0.8,
        cannibalism_rate=0.3,
        mutation_rate=0.1,
        max_iterations=max_iterations,
        random_seed=seed
    )
    
    # Run optimization
    print(f"\nStarting BWO optimization ({max_iterations} iterations)...")
    start_time = time.time()
    result = bwo.optimize(verbose=True)
    optimization_time = time.time() - start_time
    
    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    print(f"Best fitness (validation loss): {result['best_fitness']:.6f}")
    
    # Decode best parameters
    best_params = result['best_solution']
    decoded_params = decode_cnn_params(best_params)
    
    print(f"Best parameters:")
    print(f"  Filters: {decoded_params['filters']}")
    print(f"  Kernel sizes: {decoded_params['kernel_sizes']}")
    print(f"  Dropout: {decoded_params['dropout']:.3f}")
    print(f"  Learning rate: {decoded_params['lr']:.6f}")
    print(f"  Weight decay: {decoded_params['weight_decay']:.6f}")
    print(f"  Batch size: {decoded_params['batch_size']}")
    print(f"  Dense units: {decoded_params['dense_units']}")
    
    # Final evaluation with best parameters
    print(f"\nFinal evaluation on test set...")
    
    # Load full dataset
    if dataset_name in ["wind_cn", "wind_us"]:
        data_path = "论文1/processed_wind.csv"
    elif dataset_name == "electricity":
        data_path = "论文1/Load_history.csv"
    elif dataset_name == "air_quality":
        data_path = "论文1/air_quality_ready.csv"
    elif dataset_name == "gefcom":
        data_path = "论文1/gefcom_ready.csv"
    
    train_df, test_df = load_and_split_dataset(dataset_name, data_path)
    
    # Prepare training data
    train_data = train_df.values
    if dataset_name in ["wind_cn", "wind_us"] and "power" in train_df.columns:
        cols = [c for c in train_df.columns if c != "power"] + ["power"]
        train_data = train_df[cols].values
    
    # Prepare test data
    test_data = test_df.values
    if dataset_name in ["wind_cn", "wind_us"] and "power" in test_df.columns:
        cols = [c for c in test_df.columns if c != "power"] + ["power"]
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
        X_train_final, y_train_final, X_val_final, y_val_final, decoded_params['batch_size']
    )
    
    # Create and train final model
    input_dim = X_train.shape[2]
    model = CNNRegressor(
        input_dim=input_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        filters=decoded_params['filters'],
        kernel_sizes=decoded_params['kernel_sizes'],
        dropout=decoded_params['dropout'],
        dense_units=decoded_params['dense_units']
    ).to(device)
    
    # Train with more epochs for final model
    final_training_results = train_model(
        model, train_loader, val_loader,
        lr=decoded_params['lr'],
        weight_decay=decoded_params['weight_decay'],
        epochs=30,  # More epochs for final model
        device=device
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, device)
    
    # Compile results
    experiment_results = {
        'dataset': dataset_name,
        'seed': seed,
        'best_parameters': decoded_params,
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
            'seq_len': seq_len,
            'pred_len': pred_len,
            'filters': decoded_params['filters'],
            'kernel_sizes': decoded_params['kernel_sizes'],
            'dropout': decoded_params['dropout'],
            'dense_units': decoded_params['dense_units']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\nTest Metrics:")
    print(f"  R²: {test_metrics['R2']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE: {test_metrics['MAE']:.4f}")
    
    return experiment_results


def run_all_bwo_cnn_experiments():
    """Run BWO-CNN experiments for all datasets and seeds."""
    datasets = ["wind_cn", "wind_us", "electricity", "air_quality", "gefcom"]
    seeds = [42, 123, 999]  # 3 seeds as requested (consistent with other experiments)
    
    all_results = []
    
    for dataset in datasets:
        dataset_results = []
        for seed in seeds:
            try:
                result = run_bwo_cnn_experiment(
                    dataset_name=dataset,
                    seed=seed,
                    intensive_search=True
                )
                dataset_results.append(result)
                
                # Save intermediate result
                with open(f'bwo_cnn_{dataset}_seed{seed}.json', 'w') as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                print(f"Error running BWO-CNN experiment for {dataset} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Aggregate results for this dataset
        if dataset_results:
            all_results.extend(dataset_results)
    
    # Save all results
    if all_results:
        with open('bwo_cnn_all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary CSV
        summary_rows = []
        for result in all_results:
            params = result['best_parameters']
            summary_rows.append({
                'dataset': result['dataset'],
                'seed': result['seed'],
                'R2': result['test_metrics']['R2'],
                'RMSE': result['test_metrics']['RMSE'],
                'MAE': result['test_metrics']['MAE'],
                'filter1': params['filters'][0],
                'filter2': params['filters'][1],
                'filter3': params['filters'][2],
                'kernel1': params['kernel_sizes'][0],
                'kernel2': params['kernel_sizes'][1],
                'kernel3': params['kernel_sizes'][2],
                'dropout': params['dropout'],
                'lr': params['lr'],
                'weight_decay': params['weight_decay'],
                'batch_size': params['batch_size'],
                'dense_units': params['dense_units'],
                'optimization_time': result['optimization_time']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('bwo_cnn_summary.csv', index=False)
        
        print(f"\n{'='*60}")
        print("All BWO-CNN Experiments Completed!")
        print(f"{'='*60}")
        print(f"Results saved to:")
        print(f"  - bwo_cnn_all_results.json (detailed results)")
        print(f"  - bwo_cnn_summary.csv (summary table)")
        
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


def quick_test():
    """Quick test to verify the script works."""
    print("Running quick test of BWO-CNN experiment...")
    
    # Test with wind dataset, seed 42, reduced search
    try:
        result = run_bwo_cnn_experiment(
            dataset_name="wind",
            seed=42,
            intensive_search=False  # Quick test mode
        )
        print("\nQuick test completed successfully!")
        return True
    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("BWO-CNN Experimental Suite")
    print("==========================")
    print("This script runs BWO-optimized CNN on four datasets")
    print("with unified 80/20 splits and 3 random seeds.")
    print("\nDatasets: wind, electricity, air_quality, gefcom")
    print("Seeds: 42, 123, 456")
    print("Intensive search: Yes")
    print("CNN architecture: 3-layer Conv1D with pooling")
    
    # Check if required files exist
    required_files = [
        "论文1/processed_wind.csv",
        "论文1/Load_history.csv", 
        "论文1/air_quality_ready.csv",
        "论文1/gefcom_ready.csv",
        "split_manifest_80_20_unified.json",
        "unified_split_utils.py",
        "bwo_algorithm.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"\nWarning: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run coa_bwo_unified_split.py first to generate split manifest.")
        print("Also ensure bwo_algorithm.py is in the current directory.")
    else:
        print("\nAll required files found.")
        
        # Ask user for mode
        print("\nSelect mode:")
        print("1. Run quick test (wind dataset only, reduced search)")
        print("2. Run all experiments (all 4 datasets, 3 seeds, intensive search)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            print("\nRunning quick test...")
            quick_test()
        elif choice == "2":
            print("\nStarting all BWO-CNN experiments...")
            run_all_bwo_cnn_experiments()
        else:
            print("\nExiting.")