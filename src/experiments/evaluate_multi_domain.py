import json
import pandas as pd
import os
from pathlib import Path

# 定义你的实验路径 (根据你实际跑的名字)
EXPERIMENTS = {
    "Wind (CN)": "checkpoints_ts/main",
    "Wind (US)": "checkpoints_ts/exp_wind_us",
    "Air Quality": "checkpoints_ts/exp_air_quality",
    "GEFCom Load": "checkpoints_ts/exp_gefcom"
}

def load_metrics(exp_name, path):
    """从 final_results.json 中提取真实指标"""
    json_path = Path(path) / "final_results.json"
    
    if not json_path.exists():
        print(f"Warning: Results file not found for {exp_name} ({json_path})")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    metrics = data['test_metrics']
    efficiency = data['efficiency']
    
    return {
        "Dataset": exp_name,
        "R2 Score": f"{metrics['R2']:.4f}",
        "RMSE": f"{metrics['RMSE']:.4f}",
        "MAE": f"{metrics['MAE']:.4f}",
        "Skip Rate": f"{efficiency['avg_skip_rate']*100:.1f}%",
        "Status": "Success"
    }

def main():
    print("Summarizing multi-domain experimental results...")
    print("="*60)
    
    results = []
    for name, path in EXPERIMENTS.items():
        res = load_metrics(name, path)
        if res:
            results.append(res)
    
    if not results:
        print("Error: No experimental results found! Please check paths.")
        return

    # Convert to DataFrame and print
    df = pd.DataFrame(results)
    
    # Adjust column order
    cols = ["Dataset", "R2 Score", "RMSE", "MAE", "Skip Rate", "Status"]
    df = df[cols]
    
    print("\nUltra-LSNT Multi-Domain Generalization Evaluation")
    print("-" * 75)
    print(df.to_string(index=False))
    print("-" * 75)
    
    # Save as CSV for paper tables
    df.to_csv("table_multi_domain.csv", index=False)
    print("\nTable saved as: table_multi_domain.csv")
    print("Note: Use this data for the 'Results' section in your paper.")

if __name__ == "__main__":
    main()