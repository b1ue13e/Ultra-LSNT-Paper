from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from noise_utils import inject_industrial_noise
from ultra_lsnt_timeseries import compute_metrics, load_csv_data


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "split_manifest_80_20_unified.json"


@dataclass
class ZScoreStats:
    mean: np.ndarray
    std: np.ndarray


def utc_run_id(suffix: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + suffix


def sigma_eff_from_cfg(sigma_cfg: float, alpha: float = 1.5) -> float:
    return float(alpha * sigma_cfg)


def scenario_name(sigma_cfg: float, alpha: float = 1.5) -> str:
    sigma_eff = sigma_eff_from_cfg(sigma_cfg, alpha=alpha)
    if np.isclose(sigma_cfg, 0.0):
        return "clean"
    return f"gaussian_sigma_eff_{int(round(sigma_eff * 100)):03d}"


def rel_rmse_increase(clean_rmse: float, stress_rmse: float) -> float:
    if not np.isfinite(clean_rmse) or abs(clean_rmse) < 1e-12:
        return np.nan
    return float(100.0 * (stress_rmse - clean_rmse) / abs(clean_rmse))


def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def read_split_from_manifest(
    total_len: int,
    dataset_name: str = "wind_cn",
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> Tuple[int, int]:
    path = _resolve_local_path(manifest_path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for item in manifest.get("datasets", []):
            if str(item.get("name", "")).strip() == dataset_name:
                train_end = int(item["training_set"]["samples"])
                test_end = int(item["test_set"]["samples"]) + train_end
                return train_end, min(test_end, total_len)
    train_end = int(total_len * 0.8)
    return train_end, total_len


def load_windcn_matrix(
    data_path: str | Path = "wind_final.csv",
    target: str = "power",
) -> Tuple[np.ndarray, List[str], int]:
    data, feature_names = load_csv_data(str(_resolve_local_path(data_path)), target)
    if data is None or feature_names is None:
        raise RuntimeError(f"failed to load data from {data_path}")
    target_idx = feature_names.index(target)
    return data, feature_names, target_idx


def fit_zscore(train_data: np.ndarray) -> ZScoreStats:
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return ZScoreStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_zscore(data: np.ndarray, stats: ZScoreStats) -> np.ndarray:
    return ((data - stats.mean) / stats.std).astype(np.float32)


def inverse_target_scale(values: np.ndarray, stats: ZScoreStats, target_idx: int = -1) -> np.ndarray:
    return values * stats.std[target_idx] + stats.mean[target_idx]


def make_multivariate_windows(
    data_scaled: np.ndarray,
    seq_len: int,
    pred_len: int,
    target_idx: int = -1,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    max_start = len(data_scaled) - seq_len - pred_len + 1
    for start in range(0, max_start, step):
        stop = start + seq_len
        pred_stop = stop + pred_len
        xs.append(data_scaled[start:stop])
        ys.append(data_scaled[stop:pred_stop, target_idx])
    if not xs:
        raise ValueError("no windows created; check split, seq_len, and pred_len")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def make_univariate_windows(
    series_scaled: np.ndarray,
    seq_len: int,
    pred_len: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    series_2d = np.asarray(series_scaled, dtype=np.float32).reshape(-1, 1)
    xs, ys = make_multivariate_windows(series_2d, seq_len=seq_len, pred_len=pred_len, target_idx=0, step=step)
    return xs[..., 0], ys


def maybe_limit_windows(
    x: np.ndarray,
    y: np.ndarray,
    max_windows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_windows is None or max_windows <= 0 or len(x) <= max_windows:
        return x, y
    return x[:max_windows], y[:max_windows]


def inject_gaussian_windows(
    x: np.ndarray,
    sigma_cfg: float,
    noise_scale: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if np.isclose(sigma_cfg, 0.0):
        return arr.copy()
    tensor = torch.from_numpy(arr)
    # `inject_industrial_noise(..., "gaussian", level)` applies N(0, 1.5 * level).
    # `noise_scale` lets callers preserve the same sigma semantics after moving to raw units.
    scaled_level = float(sigma_cfg) * float(noise_scale)
    noisy = inject_industrial_noise(tensor, "gaussian", scaled_level).cpu().numpy()
    return noisy.astype(np.float32)


def audit_row(
    model: str,
    scenario: str,
    sigma_cfg: float,
    pred_denorm: np.ndarray,
    true_denorm: np.ndarray,
    run_id: str,
) -> dict:
    metrics = compute_metrics(pred_denorm, true_denorm)
    return {
        "model": model,
        "scenario": scenario,
        "sigma_cfg": float(sigma_cfg),
        "sigma_eff": sigma_eff_from_cfg(sigma_cfg),
        "R2": float(metrics["R2"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "run_id": run_id,
    }


def write_csv(rows: Sequence[dict], out_csv: str | Path, columns: Iterable[str] | None = None) -> Path:
    out_path = _resolve_local_path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(columns)]
    df.to_csv(out_path, index=False)
    return out_path
