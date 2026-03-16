#!/usr/bin/env python3
"""
Extended metaheuristic baseline suite for Wind(CN), optimized for RTX 4090.

This script adds three additional hybrid baselines:
1) PSO-BPNN
2) GWO-SVR
3) HPO-CNN-LSTM

Outputs are written to applied_energy_final_results/supplementary_evidence by default:
- extended_metaheuristic_clean_windcn.csv
- extended_metaheuristic_robustness_windcn.csv
- extended_metaheuristic_latency_windcn.csv
- extended_metaheuristic_search_trace.csv
- extended_metaheuristic_best_configs.csv
- extended_metaheuristic_meta.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RunConfig:
    seed: int = 42
    seq_len: int = 96
    pred_len: int = 24
    train_ratio: float = 0.8
    target: str = "power"

    pso_pop: int = 10
    pso_iters: int = 15
    gwo_pop: int = 10
    gwo_iters: int = 15
    hpo_pop: int = 8
    hpo_iters: int = 12

    objective_samples: int = 5000
    final_svr_samples: int = 4000
    final_torch_epochs: int = 10
    objective_torch_epochs: int = 2

    batch_size: int = 256
    num_workers: int = 4
    gaussian_sigma_eff: float = 0.6
    spike_severity: float = 0.6
    drift_severity: float = 0.6
    latency_runs: int = 200


class CNNLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        channels: int = 32,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)  # [B, F, L]
        z = self.conv(z)       # [B, C, L]
        z = z.transpose(1, 2)  # [B, L, C]
        z, _ = self.lstm(z)
        return self.head(z[:, -1, :])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_numeric_dataset(path: str, target: str) -> Tuple[np.ndarray, List[str], int]:
    df = pd.read_csv(path)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if target not in num_df.columns:
        raise ValueError(f"Target '{target}' not found in numeric columns of {path}")
    feature_cols = [c for c in num_df.columns if c != target]
    ordered = feature_cols + [target]
    arr = num_df[ordered].to_numpy(dtype=np.float32)
    target_idx = len(ordered) - 1
    return arr, ordered, target_idx


def create_windows(features: np.ndarray, target: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    n = len(target)
    for i in range(n - seq_len - pred_len + 1):
        xs.append(features[i : i + seq_len])
        ys.append(target[i + seq_len : i + seq_len + pred_len])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_scale_windows(data: np.ndarray, target_idx: int, cfg: RunConfig):
    n = len(data)
    n_train = int(n * cfg.train_ratio)
    train = data[:n_train]
    test = data[n_train:]

    sx = StandardScaler()
    sy = StandardScaler()

    x_train_raw = sx.fit_transform(train[:, :target_idx])
    y_train_raw = sy.fit_transform(train[:, target_idx : target_idx + 1]).reshape(-1)
    x_test_raw = sx.transform(test[:, :target_idx])
    y_test_raw = sy.transform(test[:, target_idx : target_idx + 1]).reshape(-1)

    x_train, y_train = create_windows(x_train_raw, y_train_raw, cfg.seq_len, cfg.pred_len)
    x_test, y_test = create_windows(x_test_raw, y_test_raw, cfg.seq_len, cfg.pred_len)

    n_win = len(x_train)
    n_val = max(256, int(0.1 * n_win))
    n_val = min(n_val, max(1, n_win - 512))
    n_tr = n_win - n_val
    if n_tr <= 0:
        raise RuntimeError("Insufficient train windows after validation split.")

    x_tr, y_tr = x_train[:n_tr], y_train[:n_tr]
    x_va, y_va = x_train[n_tr:], y_train[n_tr:]

    return x_tr, y_tr, x_va, y_va, x_test, y_test, sx, sy


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def inverse_scale_y(y_scaled: np.ndarray, sy: StandardScaler) -> np.ndarray:
    return sy.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def inject_gaussian_np(x: np.ndarray, sigma_eff: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = np.std(x, axis=1, keepdims=True) + 1e-6
    return x + rng.normal(0.0, sigma_eff, size=x.shape).astype(np.float32) * std


def inject_spike_np(x: np.ndarray, severity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = x.copy()
    b, l, c = z.shape
    p = min(0.15, 0.02 + 0.10 * severity)
    mask = rng.random((b, l, c)) < p
    feat_std = np.std(z, axis=(0, 1), keepdims=True) + 1e-6
    amp = (3.0 + 6.0 * severity) * feat_std
    spike = rng.laplace(0.0, 1.0, size=z.shape).astype(np.float32) * amp
    z[mask] = z[mask] + spike[mask]
    return z


def inject_drift_np(x: np.ndarray, severity: float) -> np.ndarray:
    z = x.copy()
    _, l, _ = z.shape
    ramp = np.linspace(0.0, 1.0, l, dtype=np.float32).reshape(1, l, 1)
    feat_std = np.std(z, axis=(0, 1), keepdims=True) + 1e-6
    z = z + ramp * (0.2 + severity) * feat_std
    return z


def sklearn_model_size_mib(model) -> float:
    return len(pickle.dumps(model)) / (1024.0 * 1024.0)


def torch_model_size_mib(model: nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024.0 * 1024.0)


def measure_sklearn_latency_ms(model, sample_flat: np.ndarray, runs: int = 200) -> Dict[str, float]:
    for _ in range(20):
        _ = model.predict(sample_flat)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(sample_flat)
        ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ts, dtype=np.float64)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_std_ms": float(arr.std()),
        "latency_p95_ms": float(np.percentile(arr, 95)),
    }


def measure_torch_latency_ms(model: nn.Module, sample: torch.Tensor, device: torch.device, runs: int = 200) -> Dict[str, float]:
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(30):
            _ = model(sample)
    ts = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ts, dtype=np.float64)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_std_ms": float(arr.std()),
        "latency_p95_ms": float(np.percentile(arr, 95)),
    }


def pso_optimize(objective, bounds: List[Tuple[float, float]], pop_size: int, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    span = hi - lo

    x = lo + rng.random((pop_size, dim)) * span
    v = np.zeros_like(x)

    pbest = x.copy()
    pbest_score = np.full(pop_size, np.inf, dtype=np.float64)

    gbest = x[0].copy()
    gbest_score = float("inf")
    trace = []

    for it in range(iters):
        for i in range(pop_size):
            score = float(objective(x[i]))
            if score < pbest_score[i]:
                pbest_score[i] = score
                pbest[i] = x[i].copy()
            if score < gbest_score:
                gbest_score = score
                gbest = x[i].copy()

        trace.append({"iter": it + 1, "best_score": float(gbest_score), "best_params": gbest.tolist()})

        w = 0.72
        c1 = 1.49
        c2 = 1.49
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest[None, :] - x)
        x = x + v
        x = np.clip(x, lo, hi)

    return gbest, gbest_score, trace


def gwo_optimize(objective, bounds: List[Tuple[float, float]], pop_size: int, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    span = hi - lo

    wolves = lo + rng.random((pop_size, dim)) * span
    trace = []
    alpha = beta = delta = None
    alpha_score = beta_score = delta_score = float("inf")

    for it in range(iters):
        for i in range(pop_size):
            score = float(objective(wolves[i]))
            if score < alpha_score:
                delta_score, delta = beta_score, None if beta is None else beta.copy()
                beta_score, beta = alpha_score, None if alpha is None else alpha.copy()
                alpha_score, alpha = score, wolves[i].copy()
            elif score < beta_score:
                delta_score, delta = beta_score, None if beta is None else beta.copy()
                beta_score, beta = score, wolves[i].copy()
            elif score < delta_score:
                delta_score, delta = score, wolves[i].copy()

        trace.append({"iter": it + 1, "best_score": float(alpha_score), "best_params": alpha.tolist()})

        a = 2.0 - 2.0 * (it / max(iters - 1, 1))
        for i in range(pop_size):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            ref_beta = alpha if beta is None else beta
            D_beta = np.abs(C2 * ref_beta - wolves[i])
            X2 = ref_beta - A2 * D_beta

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            ref_delta = ref_beta if delta is None else delta
            D_delta = np.abs(C3 * ref_delta - wolves[i])
            X3 = ref_delta - A3 * D_delta

            wolves[i] = (X1 + X2 + X3) / 3.0
        wolves = np.clip(wolves, lo, hi)

    return alpha, alpha_score, trace


def hpo_optimize(objective, bounds: List[Tuple[float, float]], pop_size: int, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    span = hi - lo

    pop = lo + rng.random((pop_size, dim)) * span
    best = pop[0].copy()
    best_score = float("inf")
    trace = []

    for it in range(iters):
        for i in range(pop_size):
            score = float(objective(pop[i]))
            if score < best_score:
                best_score = score
                best = pop[i].copy()

        trace.append({"iter": it + 1, "best_score": float(best_score), "best_params": best.tolist()})

        exploit = 0.65 + 0.30 * (it / max(iters - 1, 1))
        for i in range(pop_size):
            direction = best - pop[i]
            noise = rng.normal(0.0, 1.0, size=dim)
            step = exploit * direction + (1.0 - exploit) * 0.25 * span * noise
            candidate = pop[i] + step
            if rng.random() < 0.25:
                candidate = candidate + 0.35 * span * rng.normal(0.0, 1.0, size=dim)
            pop[i] = np.clip(candidate, lo, hi)

    return best, best_score, trace


def train_cnn_lstm(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    patience: int = 3,
) -> Tuple[nn.Module, float]:
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if num_workers > 0 else 0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=pin,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    best_loss = float("inf")
    best_state = None
    stall = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp and device.type == "cuda"):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast(enabled=use_amp and device.type == "cuda"):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                val_sum += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_sum / max(val_n, 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def infer_cnn_lstm(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb)
            out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extended metaheuristic baselines for Wind(CN) on RTX 4090.")
    ap.add_argument("--data", type=str, default="wind_final.csv")
    ap.add_argument("--target", type=str, default="power")
    ap.add_argument("--out_dir", type=str, default=os.path.join("applied_energy_final_results", "supplementary_evidence"))
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--pso_pop", type=int, default=10)
    ap.add_argument("--pso_iters", type=int, default=15)
    ap.add_argument("--gwo_pop", type=int, default=10)
    ap.add_argument("--gwo_iters", type=int, default=15)
    ap.add_argument("--hpo_pop", type=int, default=8)
    ap.add_argument("--hpo_iters", type=int, default=12)
    ap.add_argument("--objective_samples", type=int, default=5000)
    ap.add_argument("--final_svr_samples", type=int, default=4000)
    ap.add_argument("--final_torch_epochs", type=int, default=10)
    ap.add_argument("--objective_torch_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gaussian_sigma_eff", type=float, default=0.6)
    ap.add_argument("--spike_severity", type=float, default=0.6)
    ap.add_argument("--drift_severity", type=float, default=0.6)
    ap.add_argument("--latency_runs", type=int, default=200)
    args = ap.parse_args()

    cfg = RunConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target=args.target,
        pso_pop=args.pso_pop,
        pso_iters=args.pso_iters,
        gwo_pop=args.gwo_pop,
        gwo_iters=args.gwo_iters,
        hpo_pop=args.hpo_pop,
        hpo_iters=args.hpo_iters,
        objective_samples=args.objective_samples,
        final_svr_samples=args.final_svr_samples,
        final_torch_epochs=args.final_torch_epochs,
        objective_torch_epochs=args.objective_torch_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gaussian_sigma_eff=args.gaussian_sigma_eff,
        spike_severity=args.spike_severity,
        drift_severity=args.drift_severity,
        latency_runs=args.latency_runs,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data, _, target_idx = load_numeric_dataset(args.data, cfg.target)
    x_tr, y_tr, x_va, y_va, x_te, y_te, _, sy = split_scale_windows(data, target_idx, cfg)
    x_tr_flat = flatten_windows(x_tr)
    x_va_flat = flatten_windows(x_va)
    x_te_flat = flatten_windows(x_te)

    clean_rows: List[Dict] = []
    robust_rows: List[Dict] = []
    latency_rows: List[Dict] = []
    trace_rows: List[Dict] = []
    best_rows: List[Dict] = []

    # ------------------------------
    # 1) PSO-BPNN
    # ------------------------------
    obj_n = min(cfg.objective_samples, x_tr_flat.shape[0])
    xtr_obj = x_tr_flat[:obj_n]
    ytr_obj = y_tr[:obj_n]
    xva_obj = x_va_flat
    yva_obj = y_va

    def decode_bpnn(v: np.ndarray) -> Dict[str, float]:
        hidden = int(np.clip(round(v[0]), 32, 384))
        alpha = float(10 ** np.clip(v[1], -6.0, -2.0))
        lr = float(10 ** np.clip(v[2], -4.5, -2.0))
        return {"hidden": hidden, "alpha": alpha, "lr": lr}

    def bpnn_objective(v: np.ndarray) -> float:
        hp = decode_bpnn(v)
        mdl = MLPRegressor(
            hidden_layer_sizes=(hp["hidden"], max(16, hp["hidden"] // 2)),
            activation="relu",
            solver="adam",
            alpha=hp["alpha"],
            learning_rate_init=hp["lr"],
            max_iter=120,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=cfg.seed,
        )
        mdl.fit(xtr_obj, ytr_obj)
        pred = mdl.predict(xva_obj)
        return float(np.sqrt(mean_squared_error(yva_obj.reshape(-1), pred.reshape(-1))))

    bp_bounds = [(32.0, 384.0), (-6.0, -2.0), (-4.5, -2.0)]
    best_bp, best_bp_score, bp_trace = pso_optimize(bpnn_objective, bp_bounds, cfg.pso_pop, cfg.pso_iters, cfg.seed)
    for t in bp_trace:
        trace_rows.append(
            {
                "model": "PSO-BPNN",
                "algorithm": "PSO",
                "iter": int(t["iter"]),
                "best_score": float(t["best_score"]),
                "best_params": json.dumps(t["best_params"]),
            }
        )
    hp_bp = decode_bpnn(best_bp)
    best_rows.append({"model": "PSO-BPNN", **hp_bp, "best_objective": float(best_bp_score)})

    t0 = time.time()
    model_bp = MLPRegressor(
        hidden_layer_sizes=(hp_bp["hidden"], max(16, hp_bp["hidden"] // 2)),
        activation="relu",
        solver="adam",
        alpha=hp_bp["alpha"],
        learning_rate_init=hp_bp["lr"],
        max_iter=220,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=cfg.seed,
    )
    model_bp.fit(x_tr_flat, y_tr)
    train_time_bp = time.time() - t0

    pred_bp = model_bp.predict(x_te_flat)
    pred_bp_phys = inverse_scale_y(pred_bp, sy)
    true_phys = inverse_scale_y(y_te, sy)
    m_bp_clean = metric_dict(true_phys, pred_bp_phys)
    clean_rows.append(
        {
            "dataset": "wind_cn",
            "model": "PSO-BPNN",
            "R2": m_bp_clean["R2"],
            "RMSE": m_bp_clean["RMSE"],
            "MAE": m_bp_clean["MAE"],
            "train_time_s": train_time_bp,
        }
    )

    # ------------------------------
    # 2) GWO-SVR
    # ------------------------------
    ytr1 = ytr_obj[:, 0]
    yva1 = yva_obj[:, 0]

    def decode_svr(v: np.ndarray) -> Dict[str, float]:
        c = float(10 ** np.clip(v[0], -2.0, 3.0))
        eps = float(np.clip(v[1], 0.001, 0.4))
        gamma = float(10 ** np.clip(v[2], -4.0, 0.5))
        return {"C": c, "epsilon": eps, "gamma": gamma}

    def svr_objective(v: np.ndarray) -> float:
        hp = decode_svr(v)
        mdl = SVR(kernel="rbf", C=hp["C"], epsilon=hp["epsilon"], gamma=hp["gamma"])
        mdl.fit(xtr_obj, ytr1)
        pred = mdl.predict(xva_obj)
        return float(np.sqrt(mean_squared_error(yva1, pred)))

    svr_bounds = [(-2.0, 3.0), (0.001, 0.4), (-4.0, 0.5)]
    best_svr, best_svr_score, svr_trace = gwo_optimize(svr_objective, svr_bounds, cfg.gwo_pop, cfg.gwo_iters, cfg.seed + 7)
    for t in svr_trace:
        trace_rows.append(
            {
                "model": "GWO-SVR",
                "algorithm": "GWO",
                "iter": int(t["iter"]),
                "best_score": float(t["best_score"]),
                "best_params": json.dumps(t["best_params"]),
            }
        )
    hp_svr = decode_svr(best_svr)
    best_rows.append({"model": "GWO-SVR", **hp_svr, "best_objective": float(best_svr_score)})

    n_svr = min(cfg.final_svr_samples, x_tr_flat.shape[0])
    t0 = time.time()
    base_svr = SVR(kernel="rbf", C=hp_svr["C"], epsilon=hp_svr["epsilon"], gamma=hp_svr["gamma"])
    model_svr = MultiOutputRegressor(base_svr, n_jobs=1)
    model_svr.fit(x_tr_flat[:n_svr], y_tr[:n_svr])
    train_time_svr = time.time() - t0

    pred_svr = model_svr.predict(x_te_flat)
    pred_svr_phys = inverse_scale_y(pred_svr, sy)
    m_svr_clean = metric_dict(true_phys, pred_svr_phys)
    clean_rows.append(
        {
            "dataset": "wind_cn",
            "model": "GWO-SVR",
            "R2": m_svr_clean["R2"],
            "RMSE": m_svr_clean["RMSE"],
            "MAE": m_svr_clean["MAE"],
            "train_time_s": train_time_svr,
        }
    )

    # ------------------------------
    # 3) HPO-CNN-LSTM
    # ------------------------------
    n_obj_torch = min(cfg.objective_samples, x_tr.shape[0])
    xtr_t_obj = x_tr[:n_obj_torch]
    ytr_t_obj = y_tr[:n_obj_torch]

    def decode_cnn(v: np.ndarray) -> Dict[str, float]:
        channels = int(np.clip(round(v[0]), 16, 128))
        hidden = int(np.clip(round(v[1]), 32, 256))
        kernel = int(np.clip(round(v[2]), 3, 9))
        if kernel % 2 == 0:
            kernel += 1
        lr = float(10 ** np.clip(v[3], -4.2, -2.0))
        dropout = float(np.clip(v[4], 0.05, 0.35))
        return {
            "channels": channels,
            "hidden_dim": hidden,
            "kernel_size": kernel,
            "lr": lr,
            "dropout": dropout,
        }

    def cnn_objective(v: np.ndarray) -> float:
        hp = decode_cnn(v)
        mdl = CNNLSTM(
            input_dim=x_tr.shape[-1],
            pred_len=cfg.pred_len,
            channels=hp["channels"],
            hidden_dim=hp["hidden_dim"],
            kernel_size=hp["kernel_size"],
            dropout=hp["dropout"],
        )
        mdl, val_loss = train_cnn_lstm(
            mdl,
            xtr_t_obj,
            ytr_t_obj,
            x_va,
            y_va,
            device=device,
            epochs=cfg.objective_torch_epochs,
            lr=hp["lr"],
            batch_size=min(cfg.batch_size, 256),
            num_workers=max(0, cfg.num_workers // 2),
            use_amp=True,
            patience=2,
        )
        del mdl
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float(np.sqrt(max(val_loss, 1e-12)))

    cnn_bounds = [(16.0, 128.0), (32.0, 256.0), (3.0, 9.0), (-4.2, -2.0), (0.05, 0.35)]
    best_cnn, best_cnn_score, cnn_trace = hpo_optimize(cnn_objective, cnn_bounds, cfg.hpo_pop, cfg.hpo_iters, cfg.seed + 19)
    for t in cnn_trace:
        trace_rows.append(
            {
                "model": "HPO-CNN-LSTM",
                "algorithm": "HPO",
                "iter": int(t["iter"]),
                "best_score": float(t["best_score"]),
                "best_params": json.dumps(t["best_params"]),
            }
        )
    hp_cnn = decode_cnn(best_cnn)
    best_rows.append({"model": "HPO-CNN-LSTM", **hp_cnn, "best_objective": float(best_cnn_score)})

    t0 = time.time()
    model_cnn = CNNLSTM(
        input_dim=x_tr.shape[-1],
        pred_len=cfg.pred_len,
        channels=hp_cnn["channels"],
        hidden_dim=hp_cnn["hidden_dim"],
        kernel_size=hp_cnn["kernel_size"],
        dropout=hp_cnn["dropout"],
    )
    model_cnn, _ = train_cnn_lstm(
        model_cnn,
        x_tr,
        y_tr,
        x_va,
        y_va,
        device=device,
        epochs=cfg.final_torch_epochs,
        lr=hp_cnn["lr"],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_amp=True,
        patience=3,
    )
    train_time_cnn = time.time() - t0

    pred_cnn = infer_cnn_lstm(model_cnn, x_te, device=device)
    pred_cnn_phys = inverse_scale_y(pred_cnn, sy)
    m_cnn_clean = metric_dict(true_phys, pred_cnn_phys)
    clean_rows.append(
        {
            "dataset": "wind_cn",
            "model": "HPO-CNN-LSTM",
            "R2": m_cnn_clean["R2"],
            "RMSE": m_cnn_clean["RMSE"],
            "MAE": m_cnn_clean["MAE"],
            "train_time_s": train_time_cnn,
        }
    )

    clean_r2_map = {r["model"]: r["R2"] for r in clean_rows}

    # ------------------------------
    # Robustness scenarios
    # ------------------------------
    scenarios = [
        ("gaussian", cfg.gaussian_sigma_eff),
        ("spike", cfg.spike_severity),
        ("drift", cfg.drift_severity),
    ]

    for fault, sev in scenarios:
        if fault == "gaussian":
            x_mod = inject_gaussian_np(x_te, sigma_eff=sev, seed=cfg.seed + 100)
            sigma_eff = sev
        elif fault == "spike":
            x_mod = inject_spike_np(x_te, severity=sev, seed=cfg.seed + 200)
            sigma_eff = np.nan
        elif fault == "drift":
            x_mod = inject_drift_np(x_te, severity=sev)
            sigma_eff = np.nan
        else:
            raise ValueError(fault)

        x_mod_flat = flatten_windows(x_mod)

        pred_bp_mod = model_bp.predict(x_mod_flat)
        pred_svr_mod = model_svr.predict(x_mod_flat)
        pred_cnn_mod = infer_cnn_lstm(model_cnn, x_mod, device=device)

        preds = {
            "PSO-BPNN": inverse_scale_y(pred_bp_mod, sy),
            "GWO-SVR": inverse_scale_y(pred_svr_mod, sy),
            "HPO-CNN-LSTM": inverse_scale_y(pred_cnn_mod, sy),
        }
        for model_name, pred_phys in preds.items():
            m = metric_dict(true_phys, pred_phys)
            robust_rows.append(
                {
                    "dataset": "wind_cn",
                    "model": model_name,
                    "fault_type": fault,
                    "severity": float(sev),
                    "sigma_eff": float(sigma_eff) if not np.isnan(sigma_eff) else np.nan,
                    "R2": m["R2"],
                    "RMSE": m["RMSE"],
                    "MAE": m["MAE"],
                    "R2_clean": clean_r2_map[model_name],
                    "relative_drop_pct": 100.0 * (clean_r2_map[model_name] - m["R2"]) / (abs(clean_r2_map[model_name]) + 1e-12),
                }
            )

    # ------------------------------
    # Latency and model size
    # ------------------------------
    sample_flat = x_te_flat[:1]
    lat_bp = measure_sklearn_latency_ms(model_bp, sample_flat, runs=cfg.latency_runs)
    lat_svr = measure_sklearn_latency_ms(model_svr, sample_flat, runs=cfg.latency_runs)

    model_cnn_cpu = copy.deepcopy(model_cnn).to(torch.device("cpu")).eval()
    sample_seq = torch.from_numpy(x_te[:1])
    lat_cnn = measure_torch_latency_ms(model_cnn_cpu, sample_seq, torch.device("cpu"), runs=cfg.latency_runs)

    latency_rows.extend(
        [
            {
                "dataset": "wind_cn",
                "model": "PSO-BPNN",
                **lat_bp,
                "model_size_mib": sklearn_model_size_mib(model_bp),
                "active_params_m": np.nan,
            },
            {
                "dataset": "wind_cn",
                "model": "GWO-SVR",
                **lat_svr,
                "model_size_mib": sklearn_model_size_mib(model_svr),
                "active_params_m": np.nan,
            },
            {
                "dataset": "wind_cn",
                "model": "HPO-CNN-LSTM",
                **lat_cnn,
                "model_size_mib": torch_model_size_mib(model_cnn_cpu),
                "active_params_m": sum(p.numel() for p in model_cnn_cpu.parameters()) / 1e6,
            },
        ]
    )

    # ------------------------------
    # Persist outputs
    # ------------------------------
    clean_csv = os.path.join(args.out_dir, "extended_metaheuristic_clean_windcn.csv")
    robust_csv = os.path.join(args.out_dir, "extended_metaheuristic_robustness_windcn.csv")
    latency_csv = os.path.join(args.out_dir, "extended_metaheuristic_latency_windcn.csv")
    trace_csv = os.path.join(args.out_dir, "extended_metaheuristic_search_trace.csv")
    best_csv = os.path.join(args.out_dir, "extended_metaheuristic_best_configs.csv")
    meta_json = os.path.join(args.out_dir, "extended_metaheuristic_meta.json")

    pd.DataFrame(clean_rows).to_csv(clean_csv, index=False)
    pd.DataFrame(robust_rows).to_csv(robust_csv, index=False)
    pd.DataFrame(latency_rows).to_csv(latency_csv, index=False)
    pd.DataFrame(trace_rows).to_csv(trace_csv, index=False)
    pd.DataFrame(best_rows).to_csv(best_csv, index=False)

    meta = {
        "script": "run_extended_metaheuristic_baselines_4090.py",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": args.data,
        "target": cfg.target,
        "split": "80_20_chronological",
        "window": {"seq_len": cfg.seq_len, "pred_len": cfg.pred_len},
        "device": str(device),
        "config": asdict(cfg),
        "outputs": {
            "clean_csv": clean_csv,
            "robust_csv": robust_csv,
            "latency_csv": latency_csv,
            "trace_csv": trace_csv,
            "best_csv": best_csv,
        },
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[done] outputs:")
    print(f"- {clean_csv}")
    print(f"- {robust_csv}")
    print(f"- {latency_csv}")
    print(f"- {trace_csv}")
    print(f"- {best_csv}")
    print(f"- {meta_json}")


if __name__ == "__main__":
    main()
