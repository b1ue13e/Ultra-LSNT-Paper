import argparse
import json
import os
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RunConfig:
    seed: int = 42
    seq_len: int = 96
    pred_len: int = 24
    train_ratio: float = 0.8
    sigma_eff_list: Tuple[float, ...] = (0.15, 0.30, 0.60)
    spike_severity: float = 0.6
    batch_size: int = 256
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-2
    hidden_dim: int = 64
    n_layers: int = 2
    d_state: int = 16
    dropout: float = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wind_data(path: str = "wind_final.csv", target: str = "power") -> Tuple[np.ndarray, List[str], int]:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != target]
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in {path}")
    cols = feature_cols + [target]
    arr = df[cols].to_numpy(dtype=np.float32)
    return arr, cols, len(feature_cols)


def create_windows(features: np.ndarray, target: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    n = len(target)
    for i in range(n - seq_len - pred_len + 1):
        xs.append(features[i : i + seq_len])
        ys.append(target[i + seq_len : i + seq_len + pred_len])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def split_and_scale(data: np.ndarray, target_idx: int, cfg: RunConfig):
    n = len(data)
    n_train = int(n * cfg.train_ratio)
    train = data[:n_train]
    test = data[n_train:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    x_train, y_train = create_windows(train_scaled[:, :target_idx], train_scaled[:, target_idx], cfg.seq_len, cfg.pred_len)
    x_test, y_test = create_windows(test_scaled[:, :target_idx], test_scaled[:, target_idx], cfg.seq_len, cfg.pred_len)

    return x_train, y_train, x_test, y_test, scaler


class SelectiveSSMBlock(nn.Module):
    """Auditable lightweight SSM-style block for Mamba-like sequence modeling."""

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.state_proj = nn.Linear(d_model, d_state)
        self.state_back = nn.Linear(d_state, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_main, x_gate = self.in_proj(x).chunk(2, dim=-1)
        gate = torch.sigmoid(x_gate)

        # causal depthwise conv surrogate via shifted cumulative state mixing
        s = torch.tanh(self.state_proj(x_main))
        s = torch.cumsum(s, dim=1) / torch.arange(1, s.size(1) + 1, device=s.device).view(1, -1, 1)
        s = self.state_back(s)

        y = x_main + s
        y = y * gate
        y = self.dropout(self.out_proj(y))
        return residual + y


class AuditableMambaForecaster(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, pred_len: int, hidden_dim: int, n_layers: int, d_state: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            SelectiveSSMBlock(hidden_dim, d_state=d_state, dropout=dropout) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h_last = h[:, -1, :]
        return self.head(h_last)


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def inject_gaussian(x: torch.Tensor, sigma_eff: float) -> torch.Tensor:
    if sigma_eff <= 0:
        return x
    std = x.std(dim=1, keepdim=True)
    return x + torch.randn_like(x) * sigma_eff * std


def inject_spike(x: torch.Tensor, severity: float) -> torch.Tensor:
    if severity <= 0:
        return x
    out = x.clone()
    b, l, c = out.shape
    n_spikes = max(1, int(l * severity * 0.1))
    for bi in range(b):
        pos = torch.randperm(l, device=out.device)[:n_spikes]
        amp = severity * (out[bi].std(dim=0, keepdim=True) + 1e-6)
        sign = torch.where(torch.rand((n_spikes, c), device=out.device) > 0.5, 1.0, -1.0)
        out[bi, pos, :] += sign * amp
    return out


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    target_idx: int,
    device: torch.device,
    perturb: str = "none",
    severity: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if perturb == "gaussian":
                xb = inject_gaussian(xb, severity)
            elif perturb == "spike":
                xb = inject_spike(xb, severity)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())

    p = np.concatenate(preds, axis=0)
    y = np.concatenate(trues, axis=0)
    target_mean = float(scaler.mean_[target_idx])
    target_std = float(scaler.scale_[target_idx])
    p = p * target_std + target_mean
    y = y * target_std + target_mean

    rmse = float(np.sqrt(mean_squared_error(y.flatten(), p.flatten())))
    mae = float(mean_absolute_error(y.flatten(), p.flatten()))
    r2 = float(r2_score(y.flatten(), p.flatten()))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def measure_latency(model: nn.Module, sample: torch.Tensor, device: torch.device, warmup: int = 40, runs: int = 180):
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)

        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_std_ms": float(arr.std()),
        "latency_p95_ms": float(np.percentile(arr, 95)),
    }


def run(cfg: RunConfig, out_dir: str, device_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data, cols, target_idx = load_wind_data("wind_final.csv", target="power")

    x_train, y_train, x_test, y_test, scaler = split_and_scale(data, target_idx, cfg)
    train_loader = to_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    test_loader = to_loader(x_test, y_test, cfg.batch_size, shuffle=False)

    model = AuditableMambaForecaster(
        input_dim=x_train.shape[-1],
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    history = []
    for ep in range(cfg.epochs):
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))

    ckpt_path = os.path.join(out_dir, "mamba_auditable_checkpoint.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)

    clean = evaluate(model, test_loader, scaler, target_idx, device, perturb="none", severity=0.0)
    pd.DataFrame([
        {
            "dataset": "wind_cn",
            "model": "Mamba (auditable)",
            "R2": clean["R2"],
            "RMSE": clean["RMSE"],
            "MAE": clean["MAE"],
        }
    ]).to_csv(os.path.join(out_dir, "mamba_clean_windcn.csv"), index=False)

    robust_rows = []
    for s in cfg.sigma_eff_list:
        m = evaluate(model, test_loader, scaler, target_idx, device, perturb="gaussian", severity=float(s))
        robust_rows.append(
            {
                "fault_type": "gaussian",
                "severity": float(s),
                "sigma_eff": float(s),
                "model": "Mamba (auditable)",
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "R2_clean": clean["R2"],
                "relative_drop_pct": 100.0 * (clean["R2"] - m["R2"]) / (abs(clean["R2"]) + 1e-12),
            }
        )

    m_spike = evaluate(model, test_loader, scaler, target_idx, device, perturb="spike", severity=cfg.spike_severity)
    robust_rows.append(
        {
            "fault_type": "spike",
            "severity": float(cfg.spike_severity),
            "sigma_eff": np.nan,
            "model": "Mamba (auditable)",
            "R2": m_spike["R2"],
            "RMSE": m_spike["RMSE"],
            "MAE": m_spike["MAE"],
            "R2_clean": clean["R2"],
            "relative_drop_pct": 100.0 * (clean["R2"] - m_spike["R2"]) / (abs(clean["R2"]) + 1e-12),
        }
    )

    pd.DataFrame(robust_rows).to_csv(os.path.join(out_dir, "mamba_robustness_windcn.csv"), index=False)

    sample = torch.from_numpy(x_test[:1]).to(torch.float32)
    lat = measure_latency(model, sample, device)
    n_params = sum(p.numel() for p in model.parameters())
    size_mib = os.path.getsize(ckpt_path) / (1024.0 * 1024.0)
    throughput = 1000.0 / max(lat["latency_mean_ms"], 1e-9)

    pd.DataFrame(
        [
            {
                "dataset": "wind_cn",
                "model": "Mamba (auditable)",
                "latency_mean_ms": lat["latency_mean_ms"],
                "latency_std_ms": lat["latency_std_ms"],
                "latency_p95_ms": lat["latency_p95_ms"],
                "throughput_hz": throughput,
                "active_params_m": n_params / 1e6,
                "model_size_mib": size_mib,
            }
        ]
    ).to_csv(os.path.join(out_dir, "mamba_latency_windcn.csv"), index=False)

    meta = {
        "script": "run_mamba_auditable_suite.py",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "wind_final.csv",
        "target": "power",
        "split": "80_20_chronological",
        "window": {"seq_len": cfg.seq_len, "pred_len": cfg.pred_len},
        "device": str(device),
        "seed": cfg.seed,
        "model": {
            "hidden_dim": cfg.hidden_dim,
            "n_layers": cfg.n_layers,
            "d_state": cfg.d_state,
            "dropout": cfg.dropout,
        },
        "train": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "loss_history": history,
        },
        "robust_protocol": {
            "gaussian_sigma_eff": list(cfg.sigma_eff_list),
            "spike_severity": cfg.spike_severity,
        },
        "outputs": {
            "checkpoint": os.path.join(out_dir, "mamba_auditable_checkpoint.pth"),
            "clean_csv": os.path.join(out_dir, "mamba_clean_windcn.csv"),
            "robust_csv": os.path.join(out_dir, "mamba_robustness_windcn.csv"),
            "latency_csv": os.path.join(out_dir, "mamba_latency_windcn.csv"),
        },
    }

    with open(os.path.join(out_dir, "mamba_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditable Mamba suite for Wind(CN)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_manifest", type=str, default="split_manifest_80_20_unified.json")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--out_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--audit_tag", type=str, default="mamba_auditable_suite")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    cfg = RunConfig(seed=args.seed, seq_len=args.seq_len, pred_len=args.pred_len, epochs=args.epochs)

    # split manifest snapshot (auditable run context)
    manifest = {
        "audit_tag": args.audit_tag,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "split": "80_20_chronological",
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "dataset": "wind_final.csv",
    }
    with open(args.split_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    run(cfg, args.out_dir, args.device)


if __name__ == "__main__":
    main()
