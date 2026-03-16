"""
BWO-SVR Experiment Script

This script implements Support Vector Regression (SVR) with
Black Widow Optimization Algorithm (BWO) for hyperparameter optimization
across five datasets with unified 80/20 splits.

Features:
1. SVR with RBF kernel (configurable)
2. BWO for hyperparameter optimization (C, epsilon, gamma)
3. Five datasets: wind_cn, wind_us, electricity, air_quality, gefcom
4. Unified 80/20 chronological splits
5. 3 random seeds for robust evaluation
6. Intensive search with high parameter space coverage
"""

import numpy as np
import pandas as pd
import sys
import os
import json
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Scikit-learn imports
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project path
sys.path.append('.')

# Import unified split utilities and BWO algorithm
from unified_split_utils import load_and_split_dataset
from bwo_algorithm import BlackWidowOptimization

warnings.filterwarnings('ignore')


def prepare_sequences_sklearn(data: np.ndarray, seq_len: int, pred_len: int, 
                             target_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for time series forecasting for sklearn models.
    
    Args:
        data: 2D array with features and target
        seq_len: Lookback window length
        pred_len: Prediction horizon
        target_idx: Index of target column (default: last column)
        
    Returns:
        X: 2D array of features (samples, seq_len * features)
        y: 1D array of target values
    """
    X, y = [], []
    n_features = data.shape[1]
    
    for i in range(len(data) - seq_len - pred_len + 1):
        # Flatten the sequence into features
        sequence = data[i:i+seq_len]
        X.append(sequence.flatten())
        
        # Target is the next pred_len values of the target variable
        target_values = data[i+seq_len:i+seq_len+pred_len, target_idx]
        y.append(np.mean(target_values))  # Use mean for single-step prediction
    
    return np.array(X), np.array(y)


def prepare_features_targets(df: pd.DataFrame, target_col: Optional[str] = None) -> np.ndarray:
    """
    Prepare features and targets from dataframe.
    
    Args:
        df: Input dataframe
        target_col: Name of target column (if None, use last column)
        
    Returns:
        data: 2D numpy array with features and target
    """
    if target_col is not None and target_col in df.columns:
        # Reorder columns to put target last
        feature_cols = [c for c in df.columns if c != target_col]
        data = df[feature_cols + [target_col]].values
    else:
        # Use all columns, last column as target
        data = df.values
    
    return data


def train_evaluate_svr(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       C: float, epsilon: float, gamma: float,
                       kernel: str = 'rbf') -> Dict[str, Any]:
    """
    Train SVR model and evaluate on validation set.
    
    Returns:
        Dictionary with model and metrics
    """
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # Scale target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    
    # Create and train SVR
    svr = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
    
    start_time = time.time()
    svr.fit(X_train_scaled, y_train_scaled)
    training_time = time.time() - start_time
    
    # Predict on validation set
    y_pred_scaled = svr.predict(X_val_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    return {
        'model': svr,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'val_mse': mse,
        'val_rmse': rmse,
        'val_mae': mae,
        'val_r2': r2,
        'training_time': training_time,
        'n_support_vectors': len(svr.support_vectors_) if hasattr(svr, 'support_vectors_') else 0
    }


def create_svr_objective_function(dataset_name: str, seq_len: int, pred_len: int,
                                  seed: int = 42) -> callable:
    """Create objective function for BWO optimization of SVR."""
    
    def objective(params: np.ndarray) -> float:
        """Objective function to minimize (validation MSE)."""
        try:
            # Decode parameters
            C = 10 ** params[0]  # Log scale: 10^-2 to 10^3
            epsilon = params[1]  # 0.01 to 0.5
            gamma = 10 ** params[2]  # Log scale: 10^-4 to 10^1
            
            # Load dataset
            if dataset_name == "wind_cn":
                data_path = "论文1/processed_wind.csv"
                target_col = "power"
            elif dataset_name == "wind_us":
                data_path = "论文1/wind_us.csv"
                target_col = "power (MW)"
            elif dataset_name == "electricity":
                data_path = "论文1/Load_history.csv"
                target_col = None  # Use last column
            elif dataset_name == "air_quality":
                data_path = "论文1/air_quality_ready.csv"
                target_col = "PM2.5"
            elif dataset_name == "gefcom":
                data_path = "论文1/gefcom_ready.csv"
                target_col = "load"
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Load data with unified split
            train_df, _ = load_and_split_dataset(dataset_name, data_path)
            
            # Prepare data
            data = prepare_features_targets(train_df, target_col)
            
            # Prepare sequences
            X, y = prepare_sequences_sklearn(data, seq_len, pred_len)
            
            # Split into train/val (80% of training for actual training, 20% for validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train and evaluate SVR
            result = train_evaluate_svr(
                X_train, y_train, X_val, y_val,
                C=C, epsilon=epsilon, gamma=gamma,
                kernel='rbf'
            )
            
            return result['val_mse']  # Minimize MSE
            
        except Exception as e:
            print(f"Error in SVR objective function: {e}")
            return float('inf')  # Return worst possible value
    
    return objective


def run_bwo_svr_experiment(dataset_name: str, seed: int = 42,
                           intensive_search: bool = True) -> Dict[str, Any]:
    """Run BWO-SVR experiment for a single dataset and seed."""
    print(f"\n{'='*60}")
    print(f"BWO-SVR Experiment: {dataset_name} (seed={seed})")
    print(f"{'='*60}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Fixed parameters
    seq_len = 24  # Smaller lookback for SVR (faster training)
    pred_len = 1   # Single-step prediction for SVR
    
    # Define parameter bounds for BWO (log scale for C and gamma)
    # [log10(C), epsilon, log10(gamma)]
    bounds = [
        (-2, 3),     # log10(C): 10^-2 to 10^3
        (0.01, 0.5), # epsilon: 0.01 to 0.5
        (-4, 1)      # log10(gamma): 10^-4 to 10^1
    ]
    
    # Create objective function
    objective_func = create_svr_objective_function(dataset_name, seq_len, pred_len, seed)
    
    # BWO configuration
    population_size = 40 if intensive_search else 25
    max_iterations = 60 if intensive_search else 40
    
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
    print(f"Best fitness (validation MSE): {result['best_fitness']:.6f}")
    
    # Decode best parameters
    best_params = result['best_solution']
    best_C = 10 ** best_params[0]
    best_epsilon = best_params[1]
    best_gamma = 10 ** best_params[2]
    
    print(f"Best parameters:")
    print(f"  C: {best_C:.4f}")
    print(f"  epsilon: {best_epsilon:.4f}")
    print(f"  gamma: {best_gamma:.6f}")
    
    # Final evaluation with best parameters
    print(f"\nFinal evaluation on test set...")
    
    # Load full dataset
    if dataset_name in ["wind_cn", "wind_us"]:
        data_path = "论文1/processed_wind.csv"
        target_col = "power"
    elif dataset_name == "electricity":
        data_path = "论文1/Load_history.csv"
        target_col = None
    elif dataset_name == "air_quality":
        data_path = "论文1/air_quality_ready.csv"
        target_col = "PM2.5"
    elif dataset_name == "gefcom":
        data_path = "论文1/gefcom_ready.csv"
        target_col = "load"
    
    train_df, test_df = load_and_split_dataset(dataset_name, data_path)
    
    # Prepare training data
    train_data = prepare_features_targets(train_df, target_col)
    test_data = prepare_features_targets(test_df, target_col)
    
    # Prepare sequences
    X_train, y_train = prepare_sequences_sklearn(train_data, seq_len, pred_len)
    X_test, y_test = prepare_sequences_sklearn(test_data, seq_len, pred_len)
    
    # Split training into train/val for final model
    split_idx = int(len(X_train) * 0.8)
    X_train_final, X_val_final = X_train[:split_idx], X_train[split_idx:]
    y_train_final, y_val_final = y_train[:split_idx], y_train[split_idx:]
    
    # Train final model with best parameters
    final_result = train_evaluate_svr(
        X_train_final, y_train_final, X_val_final, y_val_final,
        C=best_C, epsilon=best_epsilon, gamma=best_gamma,
        kernel='rbf'
    )
    
    # Evaluate on test set
    svr_model = final_result['model']
    scaler_X = final_result['scaler_X']
    scaler_y = final_result['scaler_y']
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    y_pred_scaled = svr_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate test metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Training metrics
    y_val_pred_scaled = svr_model.predict(scaler_X.transform(X_val_final))
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
    val_r2 = r2_score(y_val_final, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_final, y_val_pred))
    
    # Compile results
    experiment_results = {
        'dataset': dataset_name,
        'seed': seed,
        'best_parameters': {
            'C': float(best_C),
            'epsilon': float(best_epsilon),
            'gamma': float(best_gamma),
            'kernel': 'rbf'
        },
        'optimization_history': {
            'best_fitness_history': result['history']['best_fitness'],
            'avg_fitness_history': result['history']['avg_fitness'],
            'iterations': result['history']['iterations']
        },
        'training_metrics': {
            'val_r2': float(val_r2),
            'val_rmse': float(val_rmse),
            'val_mae': float(mean_absolute_error(y_val_final, y_val_pred)),
            'training_time': final_result['training_time'],
            'n_support_vectors': final_result['n_support_vectors']
        },
        'test_metrics': {
            'R2': float(test_r2),
            'RMSE': float(test_rmse),
            'MAE': float(test_mae),
            'MSE': float(test_mse)
        },
        'optimization_time': optimization_time,
        'model_info': {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'n_features_train': X_train.shape[1],
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\nTest Metrics:")
    print(f"  R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  Training time: {final_result['training_time']:.2f}s")
    print(f"  Support vectors: {final_result['n_support_vectors']}")
    
    return experiment_results


def run_all_bwo_svr_experiments():
    """Run BWO-SVR experiments for all datasets and seeds."""
    datasets = ["wind_cn", "wind_us", "electricity", "air_quality", "gefcom"]
    seeds = [42, 123, 999]  # 3 seeds as requested (consistent with other experiments)
    
    all_results = []
    
    for dataset in datasets:
        dataset_results = []
        for seed in seeds:
            try:
                result = run_bwo_svr_experiment(
                    dataset_name=dataset,
                    seed=seed,
                    intensive_search=True
                )
                dataset_results.append(result)
                
                # Save intermediate result
                with open(f'bwo_svr_{dataset}_seed{seed}.json', 'w') as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                print(f"Error running BWO-SVR experiment for {dataset} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Aggregate results for this dataset
        if dataset_results:
            all_results.extend(dataset_results)
    
    # Save all results
    if all_results:
        with open('bwo_svr_all_results.json', 'w') as f:
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
                'C': result['best_parameters']['C'],
                'epsilon': result['best_parameters']['epsilon'],
                'gamma': result['best_parameters']['gamma'],
                'training_time': result['training_metrics']['training_time'],
                'n_support_vectors': result['training_metrics']['n_support_vectors'],
                'optimization_time': result['optimization_time']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('bwo_svr_summary.csv', index=False)
        
        print(f"\n{'='*60}")
        print("All BWO-SVR Experiments Completed!")
        print(f"{'='*60}")
        print(f"Results saved to:")
        print(f"  - bwo_svr_all_results.json (detailed results)")
        print(f"  - bwo_svr_summary.csv (summary table)")
        
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
    print("Running quick test of BWO-SVR experiment...")
    
    # Test with wind dataset, seed 42, reduced search
    try:
        result = run_bwo_svr_experiment(
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
    print("BWO-SVR Experimental Suite")
    print("==========================")
    print("This script runs BWO-optimized SVR on four datasets")
    print("with unified 80/20 splits and 3 random seeds.")
    print("\nDatasets: wind, electricity, air_quality, gefcom")
    print("Seeds: 42, 123, 456")
    print("Intensive search: Yes")
    print("SVR kernel: RBF")
    
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
            print("\nStarting all BWO-SVR experiments...")
            run_all_bwo_svr_experiments()
        else:
            print("\nExiting.")