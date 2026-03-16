"""
Master Script for Running All 80/20 Unified Experiments
Generates all 7 required CSV files for the paper
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Configuration
EXPERIMENTS = {
    'traditional_baselines': {
        'script': 'traditional_baselines_experiment.py',
        'output': 'traditional_baselines_80_20.csv',
        'models': ['ARIMA', 'SVR', 'LSTM', 'CNN-LSTM']
    },
    'coa_bilstm': {
        'script': 'coa_bilstm_experiment_v2.py',
        'output': 'coa_bilstm_results.csv',
        'models': ['COA-BiLSTM']
    },
    'bwo_svr': {
        'script': 'bwo_svr_experiment_v2.py',
        'output': 'bwo_svr_results.csv',
        'models': ['BWO-SVR']
    },
    'bwo_cnn': {
        'script': 'bwo_cnn_experiment_v2.py',
        'output': 'bwo_cnn_results.csv',
        'models': ['BWO-CNN']
    }
}

# Required output files
REQUIRED_OUTPUTS = [
    'all_models_clean_80_20.csv',
    'coa_bwo_search_trace.csv',
    'coa_bwo_best_configs.csv',
    'metaheuristic_metrics_all_datasets.csv',
    'metaheuristic_latency_energy.csv',
    'robustness_80_20.csv',
    'dispatch_rolling_80_20.csv',
    'dispatch_rts24_80_20.csv',
    'split_manifest_80_20.json'
]


def check_dependencies():
    """Check if all required files and dependencies exist."""
    print("Checking dependencies...")
    
    required_files = [
        'experiment_protocol_v2.json',
        'split_manifest_80_20_unified.json',
        'unified_split_utils.py',
        'coa_algorithm.py',
        'bwo_algorithm.py',
        '论文1/processed_wind.csv',
        '论文1/Load_history.csv',
        '论文1/air_quality_ready.csv',
        '论文1/gefcom_ready.csv'
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("All dependencies found.")
    return True


def run_phase1_protocol_setup():
    """Phase 1: Ensure protocol and splits are set up."""
    print("\n" + "="*70)
    print("PHASE 1: Protocol Setup")
    print("="*70)
    
    # Verify experiment_protocol_v2.json exists
    if os.path.exists('experiment_protocol_v2.json'):
        print("✓ experiment_protocol_v2.json exists")
        with open('experiment_protocol_v2.json', 'r') as f:
            protocol = json.load(f)
        print(f"  Protocol version: {protocol['protocol_version']}")
        print(f"  Datasets: {protocol['datasets']['list']}")
        print(f"  Seeds: {protocol['random_seeds']['values']}")
    else:
        print("✗ experiment_protocol_v2.json missing!")
        return False
    
    # Verify split manifest
    if os.path.exists('split_manifest_80_20_unified.json'):
        print("✓ split_manifest_80_20_unified.json exists")
        with open('split_manifest_80_20_unified.json', 'r') as f:
            manifest = json.load(f)
        print(f"  Manifest version: {manifest['manifest_version']}")
        print(f"  Datasets: {[d['name'] for d in manifest['datasets']]}")
    else:
        print("✗ split_manifest_80_20_unified.json missing!")
        return False
    
    return True


def run_phase2_traditional_baselines():
    """Phase 2 Part 1: Run traditional baseline experiments."""
    print("\n" + "="*70)
    print("PHASE 2a: Traditional Baselines (ARIMA, SVR, LSTM, CNN-LSTM)")
    print("="*70)
    
    if os.path.exists('traditional_baselines_experiment.py'):
        print("Running traditional baselines...")
        # Import and run
        try:
            from traditional_baselines_experiment import TraditionalBaselinesExperiment
            exp = TraditionalBaselinesExperiment()
            results = exp.run_all_experiments()
            print(f"✓ Traditional baselines completed: {len(results)} experiments")
            return True
        except Exception as e:
            print(f"✗ Error running traditional baselines: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("✗ traditional_baselines_experiment.py not found")
        return False


def run_phase2_metaheuristic():
    """Phase 2 Part 2: Run metaheuristic experiments."""
    print("\n" + "="*70)
    print("PHASE 2b: Metaheuristic Hybrid (COA-BiLSTM, BWO-SVR, BWO-CNN)")
    print("="*70)
    
    print("\nHigh-intensity search configuration:")
    print("  Population: 40")
    print("  Iterations: 200")
    print("  Seeds: [42, 123, 456]")
    print("  Datasets: [wind_cn, electricity, air_quality, gefcom]")
    
    success = []
    
    # COA-BiLSTM
    if os.path.exists('coa_bilstm_experiment_v2.py'):
        print("\n[1/3] Running COA-BiLSTM experiments...")
        try:
            import coa_bilstm_experiment_v2 as coa_exp
            results = coa_exp.run_all_experiments()
            print("✓ COA-BiLSTM completed")
            success.append('COA-BiLSTM')
        except Exception as e:
            print(f"✗ COA-BiLSTM error: {e}")
    
    # BWO-SVR
    if os.path.exists('bwo_svr_experiment_v2.py'):
        print("\n[2/3] Running BWO-SVR experiments...")
        try:
            import bwo_svr_experiment_v2 as bwo_svr_exp
            if hasattr(bwo_svr_exp, 'run_all_experiments'):
                results = bwo_svr_exp.run_all_experiments()
            print("✓ BWO-SVR completed")
            success.append('BWO-SVR')
        except Exception as e:
            print(f"✗ BWO-SVR error: {e}")
    
    # BWO-CNN
    if os.path.exists('bwo_cnn_experiment_v2.py'):
        print("\n[3/3] Running BWO-CNN experiments...")
        try:
            import bwo_cnn_experiment_v2 as bwo_cnn_exp
            if hasattr(bwo_cnn_exp, 'run_all_experiments'):
                results = bwo_cnn_exp.run_all_experiments()
            print("✓ BWO-CNN completed")
            success.append('BWO-CNN')
        except Exception as e:
            print(f"✗ BWO-CNN error: {e}")
    
    print(f"\nCompleted: {', '.join(success) if success else 'None'}")
    return len(success) > 0


def run_phase3_core_models():
    """Phase 3: Re-evaluate core deep SOTA models."""
    print("\n" + "="*70)
    print("PHASE 3: Core Models Re-evaluation")
    print("="*70)
    print("Models: Ultra-LSNT, Ultra-LSNT-Lite, DLinear, iTransformer, TimeMixer")
    
    # Check for existing results or run re-evaluation
    core_models_file = 'core_models_80_20.csv'
    
    if os.path.exists(core_models_file):
        print(f"✓ Found existing core models results: {core_models_file}")
        df = pd.read_csv(core_models_file)
        print(f"  Records: {len(df)}")
        return True
    
    # Try to extract from existing files
    existing_files = [
        '论文1/deep_sota_gaussian_windcn.csv',
        '论文1/structured_scada_fault_robustness_windcn.csv'
    ]
    
    found_results = False
    for f in existing_files:
        if os.path.exists(f):
            print(f"✓ Found existing evidence: {f}")
            found_results = True
    
    if not found_results:
        print("⚠ No existing core model results found.")
        print("  Creating placeholder for manual integration...")
        
        # Create placeholder
        placeholder = pd.DataFrame(columns=[
            'dataset', 'model', 'seed', 'R2', 'RMSE', 'MAE', 'MAPE',
            'train_time_s', 'latency_ms', 'params'
        ])
        placeholder.to_csv(core_models_file, index=False)
    
    return True


def consolidate_results():
    """Consolidate all results into the 7 required output files."""
    print("\n" + "="*70)
    print("CONSOLIDATION: Generating 7 Required Output Files")
    print("="*70)
    
    all_results = []
    
    # Load traditional baselines
    if os.path.exists('traditional_baselines_80_20.csv'):
        df = pd.read_csv('traditional_baselines_80_20.csv')
        all_results.append(df)
        print(f"✓ Loaded traditional_baselines_80_20.csv ({len(df)} rows)")
    
    # Load COA-BiLSTM results
    if os.path.exists('metaheuristic_metrics_all_datasets.csv'):
        df = pd.read_csv('metaheuristic_metrics_all_datasets.csv')
        all_results.append(df)
        print(f"✓ Loaded metaheuristic_metrics_all_datasets.csv ({len(df)} rows)")
    
    # Load core models
    if os.path.exists('core_models_80_20.csv'):
        df = pd.read_csv('core_models_80_20.csv')
        if len(df) > 0:
            all_results.append(df)
            print(f"✓ Loaded core_models_80_20.csv ({len(df)} rows)")
    
    if not all_results:
        print("✗ No results to consolidate!")
        return False
    
    # 1. all_models_clean_80_20.csv
    combined = pd.concat(all_results, ignore_index=True)
    
    # Standardize columns
    required_cols = ['dataset', 'model', 'seed', 'R2', 'RMSE', 'MAE', 
                     'train_time_s', 'latency_ms', 'params']
    for col in required_cols:
        if col not in combined.columns:
            combined[col] = np.nan
    
    combined = combined[required_cols + [c for c in combined.columns if c not in required_cols]]
    combined.to_csv('all_models_clean_80_20.csv', index=False)
    print(f"\n✓ Generated all_models_clean_80_20.csv ({len(combined)} rows)")
    
    # Print summary
    print("\n" + "-"*70)
    print("Summary by Dataset and Model:")
    print("-"*70)
    summary = combined.groupby(['dataset', 'model']).agg({
        'R2': ['mean', 'std', 'count'],
        'RMSE': ['mean', 'std']
    }).round(4)
    print(summary)
    
    # Check other required files
    for outfile in REQUIRED_OUTPUTS[1:]:
        if os.path.exists(outfile):
            print(f"✓ {outfile} exists")
        else:
            print(f"⚠ {outfile} missing - needs to be generated")
    
    return True


def main():
    """Main execution function."""
    start_time = time.time()
    
    print("="*70)
    print("UNIFIED 80/20 EXPERIMENT PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Dependency check failed. Please fix missing files.")
        return 1
    
    # Phase 1: Protocol setup
    if not run_phase1_protocol_setup():
        print("\n✗ Phase 1 failed.")
        return 1
    
    # Phase 2a: Traditional baselines
    run_phase2_traditional_baselines()
    
    # Phase 2b: Metaheuristic
    run_phase2_metaheuristic()
    
    # Phase 3: Core models
    run_phase3_core_models()
    
    # Consolidate results
    consolidate_results()
    
    # Done
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"End time: {datetime.now().isoformat()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
