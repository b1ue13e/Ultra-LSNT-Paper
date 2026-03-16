"""
Utility functions for applying unified 80/20 splits to datasets.

This module provides functions to load and split datasets according to the
unified 80/20 chronological split scheme for COA/BWO experiments.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import json


def load_and_split_dataset(dataset_name: str, data_path: str, 
                          split_manifest_path: str = "split_manifest_80_20_unified.json") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a dataset and split it according to the unified 80/20 scheme.
    
    Args:
        dataset_name: One of ['wind', 'electricity', 'air_quality', 'gefcom']
        data_path: Path to the data file
        split_manifest_path: Path to the split manifest JSON file
        
    Returns:
        train_df: Training data (first 80% chronologically)
        test_df: Testing data (last 20% chronologically)
    """
    # Load split manifest
    with open(split_manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # Find dataset configuration
    dataset_config = None
    for ds in manifest['datasets']:
        if ds['name'] == dataset_name:
            dataset_config = ds
            break
    
    if dataset_config is None:
        raise ValueError(f"Dataset {dataset_name} not found in split manifest")
    
    # Load data with special handling for different formats
    if 'Load_history' in data_path:
        # Electricity load data has comma as thousands separator
        df = pd.read_csv(data_path, thousands=',')
        # Also ensure numeric columns are properly typed
        # Identify numeric columns (excluding zone_id, year, month, day)
        numeric_cols = [col for col in df.columns if col not in ['zone_id', 'year', 'month', 'day']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
    else:
        df = pd.read_csv(data_path)
    
    total_samples = len(df)
    
    # Get split indices
    train_indices = dataset_config['training_set']['indices']
    test_indices = dataset_config['test_set']['indices']
    
    # Parse indices (format: "start:end")
    train_start, train_end = map(int, train_indices.split(':'))
    test_start, test_end = map(int, test_indices.split(':'))
    
    # Adjust indices if actual data size differs from expected
    if total_samples != dataset_config['total_samples']:
        print(f"Warning: Actual data size ({total_samples}) differs from expected "
              f"({dataset_config['total_samples']}). Adjusting split proportionally.")
        
        # Maintain 80/20 ratio on actual data
        actual_train_end = int(total_samples * 0.8)
        train_df = df.iloc[:actual_train_end].copy()
        test_df = df.iloc[actual_train_end:].copy()
    else:
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
    
    print(f"Dataset: {dataset_name}")
    print(f"  Total samples: {total_samples}")
    print(f"  Training samples: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
    print(f"  Testing samples: {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")
    
    return train_df, test_df


def get_split_indices(dataset_name: str, 
                     split_manifest_path: str = "split_manifest_80_20_unified.json") -> Dict[str, Any]:
    """
    Get split indices for a dataset without loading the data.
    
    Returns:
        Dictionary with split information
    """
    with open(split_manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    for ds in manifest['datasets']:
        if ds['name'] == dataset_name:
            return {
                'training_indices': ds['training_set']['indices'],
                'test_indices': ds['test_set']['indices'],
                'training_samples': ds['training_set']['samples'],
                'test_samples': ds['test_set']['samples'],
                'total_samples': ds['total_samples']
            }
    
    raise ValueError(f"Dataset {dataset_name} not found in split manifest")


def validate_split_consistency(dataset_name: str, train_df: pd.DataFrame, 
                              test_df: pd.DataFrame) -> bool:
    """
    Validate that the split follows the unified 80/20 scheme.
    
    Returns:
        True if split is valid, False otherwise
    """
    total_samples = len(train_df) + len(test_df)
    train_ratio = len(train_df) / total_samples
    test_ratio = len(test_df) / total_samples
    
    # Check ratios (allow small tolerance for rounding)
    ratio_valid = (abs(train_ratio - 0.8) < 0.01 and abs(test_ratio - 0.2) < 0.01)
    
    # Check no overlap (assuming data is sorted chronologically)
    # This is a simple check - in practice you might need to check timestamps
    overlap_valid = True  # Placeholder
    
    return ratio_valid and overlap_valid


# Example usage
if __name__ == "__main__":
    # Example: Split wind dataset
    try:
        train, test = load_and_split_dataset(
            dataset_name="wind",
            data_path="论文1/processed_wind.csv"
        )
        print(f"Wind dataset split successfully")
        print(f"  Train shape: {train.shape}")
        print(f"  Test shape: {test.shape}")
        
        # Validate the split
        is_valid = validate_split_consistency("wind", train, test)
        print(f"  Split valid: {is_valid}")
        
    except Exception as e:
        print(f"Error: {e}")
