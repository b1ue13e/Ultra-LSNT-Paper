from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from paper_figure_style import ensure_output_dir, save_figure, set_paper_style
from ultra_lsnt_timeseries import (
    LSNTConfig,
    TimeSeriesConfig,
    TrainConfig,
    UltraLSNTForecaster,
    create_dataloaders,
    load_csv_data,
)


set_paper_style()
OUT_DIR = ensure_output_dir("figure")


def _load_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _safe_idx(feature_names: list[str], name: str) -> int | None:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def analyze_physics(
    *,
    data_path: str | Path = "wind_final.csv",
    ckpt_path: str | Path = "checkpoints_ts/main/best_model.pth",
    config_path: str | Path = "checkpoints_ts/main/model_config.json",
    max_batches: int = 220,
) -> None:
    data_path = Path(data_path)
    ckpt_path = Path(ckpt_path)
    config_path = Path(config_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, feature_names = load_csv_data(str(data_path), "power")
    if data is None:
        raise RuntimeError("load_csv_data returned None")

    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    train_config = TrainConfig(batch_size=128, num_workers=0)
    _, _, test_loader, _ = create_dataloaders(data, ts_config, train_config)
    input_dim = int(data.shape[1])

    # Load checkpoint first so we can infer whether the router expects dim or dim+1.
    state = _load_state_dict(ckpt_path, device)
    router_w = state.get("blocks.0.moe_router.router.weight", None)
    router_in_dim = int(router_w.shape[1]) if hasattr(router_w, "shape") else None

    model_config = LSNTConfig(input_dim=input_dim, heteroscedastic_moe=False)
    if config_path.exists():
        import json

        raw = json.loads(config_path.read_text(encoding="utf-8", errors="ignore"))
        allowed = set(LSNTConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in raw.items() if k in allowed}
        filtered["input_dim"] = input_dim
        filtered.setdefault("heteroscedastic_moe", False)
        if router_in_dim is not None:
            hidden_dim = int(filtered.get("hidden_dim", router_in_dim))
            if router_in_dim == hidden_dim + 1:
                filtered["heteroscedastic_moe"] = True
            elif router_in_dim == hidden_dim:
                filtered["heteroscedastic_moe"] = False
            else:
                filtered["hidden_dim"] = router_in_dim
                filtered["heteroscedastic_moe"] = False
        model_config = LSNTConfig(**filtered)

    model = UltraLSNTForecaster(model_config, ts_config).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    idx_ws = _safe_idx(feature_names, "windspeed") or 0
    idx_temp = _safe_idx(feature_names, "temperature")
    idx_hum = _safe_idx(feature_names, "humidity")

    expert_probs: list[np.ndarray] = []
    wind_speed: list[np.ndarray] = []
    turbulence: list[np.ndarray] = []
    temperature: list[np.ndarray] = []
    humidity: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (bx, _) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            bx = bx.to(device)
            _, _, info = model(bx, return_stats=True)
            raw_probs = info["block_stats"][-1]["router"]["raw_probs"].numpy()  # (B, E)
            expert_probs.append(raw_probs)

            # Physical proxies from normalized input (Spearman is rank-based).
            ws = bx[:, :, idx_ws].mean(dim=1).detach().cpu().numpy()
            turb = bx[:, :, idx_ws].std(dim=1).detach().cpu().numpy()
            wind_speed.append(ws)
            turbulence.append(turb)

            if idx_temp is not None:
                temperature.append(bx[:, :, idx_temp].mean(dim=1).detach().cpu().numpy())
            if idx_hum is not None:
                humidity.append(bx[:, :, idx_hum].mean(dim=1).detach().cpu().numpy())

    probs = np.concatenate(expert_probs, axis=0)  # (N, E)
    ws_all = np.concatenate(wind_speed, axis=0)
    turb_all = np.concatenate(turbulence, axis=0)
    temp_all = np.concatenate(temperature, axis=0) if temperature else None
    hum_all = np.concatenate(humidity, axis=0) if humidity else None

    variables: list[tuple[str, np.ndarray]] = [
        ("Wind Speed", ws_all),
        ("Turbulence", turb_all),
    ]
    if temp_all is not None:
        variables.append(("Temperature", temp_all))
    if hum_all is not None:
        variables.append(("Humidity", hum_all))

    corr = np.zeros((probs.shape[1], len(variables)), dtype=np.float64)
    for e in range(probs.shape[1]):
        for j, (_, v) in enumerate(variables):
            r, _ = spearmanr(probs[:, e], v)
            corr[e, j] = float(r) if np.isfinite(r) else 0.0

    df = pd.DataFrame(corr, columns=[name for name, _ in variables])
    df.index = [f"Expert {i+1}" for i in range(probs.shape[1])]

    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=300)
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Spearman Correlation"},
    )
    ax.set_title("Expert–Physics Correlation Matrix")
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "fig_physics_corr")
    plt.close(fig)


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    analyze_physics()
