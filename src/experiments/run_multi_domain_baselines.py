#!/usr/bin/env python3
"""
Run multi-domain baseline experiments and export per-dataset R2 for at least 2 baselines.

Default baselines:
  - DLinear
  - TimeMixer

Optional second baseline:
  - PatchTST
  - iTransformer
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from run_dlinear import DLinear
from run_latest_sota import PatchTST, TimeMixer, iTransformer
from ultra_lsnt_timeseries import (
    TimeSeriesConfig,
    TrainConfig,
    compute_metrics,
    create_dataloaders,
    load_csv_data,
    set_seed,
)


ROOT = Path(__file__).resolve().parent


@dataclass
class RunConfig:
    model2: str = "timemixer"
    seq_len: int = 96
    pred_len: int = 24
    epochs: int = 10
    batch_size: int = 512  # 增加以适应RTX 4090大显存
    lr: float = 1e-3
    patience: int = 5
    # 模型容量增强参数
    d_model: int = 512  # 增加模型维度以匹配4090能力
    nhead: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    seed: int = 42


DATASETS: List[Tuple[str, str, str]] = [
    ("Wind (CN)", "wind_final.csv", "power"),
    ("Wind (US)", "wind_us.csv", "power (MW)"),
    ("Air Quality", "air_quality_ready.csv", "AQI"),
    ("GEFCom Load", "gefcom_ready.csv", "load"),
]

DATASET_ALIAS: Dict[str, str] = {
    "wind_cn": "Wind (CN)",
    "wind_us": "Wind (US)",
    "air_quality": "Air Quality",
    "gefcom": "GEFCom Load",
}


def _eval_model(
    model: nn.Module,
    test_loader,
    scaler,
    device: torch.device,
    model_type: str,
) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            if model_type == "dlinear":
                pred = model(bx)
            else:
                pred, _ = model(bx)
                pred = pred.squeeze(-1)
            preds.append(pred.cpu().numpy())
            trues.append(by.numpy())

    p = np.concatenate(preds, axis=0)
    y = np.concatenate(trues, axis=0)
    p = p * scaler.std[-1] + scaler.mean[-1]
    y = y * scaler.std[-1] + scaler.mean[-1]
    return compute_metrics(p, y)


def _train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    model_type: str,
    epochs: int,
    lr: float,
    patience: int,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    bad_epochs = 0
    best_state = None

    start = time.time()
    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            if model_type == "dlinear":
                pred = model(bx)
                loss = criterion(pred, by)
            else:
                pred, _ = model(bx)
                loss = criterion(pred.squeeze(-1), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                if model_type == "dlinear":
                    pred = model(bx)
                    loss = criterion(pred, by)
                else:
                    pred, _ = model(bx)
                    loss = criterion(pred.squeeze(-1), by)
                val_loss += float(loss.item())
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return time.time() - start


def run_for_dataset(cfg: RunConfig, dataset_name: str, csv_name: str, target: str, device: torch.device) -> Dict[str, float]:
    path = ROOT / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data, _ = load_csv_data(str(path), target)
    if data is None:
        raise RuntimeError(f"Failed to load data from: {path}")

    ts = TimeSeriesConfig(seq_len=cfg.seq_len, pred_len=cfg.pred_len, target=target)
    tc = TrainConfig(
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        num_workers=0,
        patience=cfg.patience,
    )
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, ts, tc)

    input_dim = data.shape[1]

    dlinear = DLinear(cfg.seq_len, cfg.pred_len, input_dim).to(device)
    d_t = _train_model(
        dlinear, train_loader, val_loader, device, "dlinear", cfg.epochs, cfg.lr, cfg.patience
    )
    d_metrics = _eval_model(dlinear, test_loader, scaler, device, "dlinear")

    model2_name = cfg.model2.lower()
    if model2_name == "timemixer":
        model2 = TimeMixer(
            {
                "input_dim": input_dim,
                "seq_len": cfg.seq_len,
                "pred_len": cfg.pred_len,
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "scales": [1, 2, 4],
            }
        ).to(device)
    elif model2_name == "patchtst":
        model2 = PatchTST(
            {
                "input_dim": input_dim,
                "seq_len": cfg.seq_len,
                "pred_len": cfg.pred_len,
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "patch_len": cfg.patch_len,
                "stride": cfg.stride,
            }
        ).to(device)
    elif model2_name == "itransformer":
        model2 = iTransformer(
            {
                "input_dim": input_dim,
                "seq_len": cfg.seq_len,
                "pred_len": cfg.pred_len,
                "d_model": cfg.d_model,
                "nhead": cfg.nhead,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
            }
        ).to(device)
    else:
        raise ValueError(f"Unsupported --model2: {cfg.model2}")

    m2_t = _train_model(
        model2, train_loader, val_loader, device, "latest", cfg.epochs, cfg.lr, cfg.patience
    )
    m2_metrics = _eval_model(model2, test_loader, scaler, device, "latest")

    return {
        "dataset": dataset_name,
        "target": target,
        "DLinear_R2": float(d_metrics["R2"]),
        f"{cfg.model2}_R2": float(m2_metrics["R2"]),
        "DLinear_RMSE": float(d_metrics["RMSE"]),
        "DLinear_MAE": float(d_metrics["MAE"]),
        f"{cfg.model2}_RMSE": float(m2_metrics["RMSE"]),
        f"{cfg.model2}_MAE": float(m2_metrics["MAE"]),
        "DLinear_train_sec": float(d_t),
        f"{cfg.model2}_train_sec": float(m2_t),
    }


def parse_dataset_filter(raw: str) -> List[Tuple[str, str, str]]:
    raw = raw.strip().lower()
    if raw in {"", "all"}:
        return DATASETS

    aliases = [s.strip() for s in raw.split(",") if s.strip()]
    wanted = {DATASET_ALIAS[a] for a in aliases if a in DATASET_ALIAS}
    if not wanted:
        raise ValueError(
            f"Invalid --datasets value: {raw}. "
            f"Use comma-separated aliases from: {', '.join(DATASET_ALIAS.keys())}"
        )
    return [d for d in DATASETS if d[0] in wanted]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-domain baseline R2 experiments.")
    p.add_argument("--model2", type=str, default="timemixer", choices=["timemixer", "patchtst", "itransformer"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset aliases: wind_cn,wind_us,air_quality,gefcom (default: all).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Force execution device (default: auto).",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default="multi_domain_baselines_r2",
        help="Output prefix used for csv/json files.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(
        model2=args.model2,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {cfg}")

    selected_datasets = parse_dataset_filter(args.datasets)

    rows: List[Dict[str, float]] = []
    for ds_name, csv_name, target in selected_datasets:
        print(f"\n=== {ds_name} | target={target} ===")
        try:
            row = run_for_dataset(cfg, ds_name, csv_name, target, device)
            rows.append(row)
            print(
                f"R2: DLinear={row['DLinear_R2']:.4f}, {cfg.model2}={row[f'{cfg.model2}_R2']:.4f}"
            )
        except Exception as exc:
            print(f"[WARN] {ds_name} failed: {exc}")
            rows.append(
                {
                    "dataset": ds_name,
                    "target": target,
                    "DLinear_R2": np.nan,
                    f"{cfg.model2}_R2": np.nan,
                    "error": str(exc),
                }
            )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix.strip() or "multi_domain_baselines_r2"
    csv_path = out_dir / f"{prefix}_{ts}.csv"
    latest_csv = out_dir / f"{prefix}_latest.csv"
    json_path = out_dir / f"{prefix}_{ts}.json"

    import csv as csv_lib

    all_keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_lib.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(cfg),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {latest_csv}")
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
