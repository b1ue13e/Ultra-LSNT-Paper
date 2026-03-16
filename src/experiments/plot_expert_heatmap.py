from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

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


def _heatmap_from_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df.sort_values("Hour")
    cols = [c for c in df.columns if c.lower().startswith("expert_")]
    if not cols:
        cols = [c for c in df.columns if c.lower().startswith("expert")]
    if not cols:
        raise RuntimeError(f"No expert columns found in {csv_path}")
    mat = df[cols].to_numpy().T  # (experts, 24)
    if mat.shape[1] != 24:
        raise RuntimeError(f"Expected 24 hours, got {mat.shape}")
    return mat


def _compute_heatmap_from_model(
    *,
    data_path: Path,
    ckpt_path: Path,
    config_path: Path,
    max_samples: int = 96 * 60,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, _ = load_csv_data(str(data_path), "power")
    ts_config = TimeSeriesConfig(seq_len=96, pred_len=24, target="power")
    train_config = TrainConfig(batch_size=32, num_workers=0)
    _, _, test_loader, _ = create_dataloaders(data, ts_config, train_config)
    input_dim = int(data.shape[1])

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

    # Router: use the first block for interpretability
    router = model.blocks[0].moe_router.router
    input_norm = model.blocks[0].input_norm

    num_experts = int(model_config.num_experts)
    hourly_sum = {h: np.zeros(num_experts, dtype=np.float64) for h in range(24)}
    hourly_cnt = {h: 0 for h in range(24)}

    seen = 0
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            enc = model.encoder(bx)
            norm = input_norm(enc)
            logits = router(norm)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # (B, E)

            for i in range(probs.shape[0]):
                time_step = seen % 96
                hour = int(time_step / 4)
                hourly_sum[hour] += probs[i]
                hourly_cnt[hour] += 1
                seen += 1
                if seen >= max_samples:
                    break
            if seen >= max_samples:
                break

    mat = np.zeros((num_experts, 24), dtype=np.float64)
    for h in range(24):
        if hourly_cnt[h] > 0:
            mat[:, h] = hourly_sum[h] / hourly_cnt[h]
    return mat


def plot_real_expert_heatmap() -> None:
    csv_path = Path("wind_expert_activation.csv")
    data_path = Path("wind_final.csv")
    ckpt_path = Path("checkpoints_ts/main/best_model.pth")
    config_path = Path("checkpoints_ts/main/model_config.json")

    if csv_path.exists():
        mat = _heatmap_from_csv(csv_path)
    else:
        if not (data_path.exists() and ckpt_path.exists()):
            raise FileNotFoundError("Need wind_expert_activation.csv or (wind_final.csv + checkpoints_ts/main/best_model.pth).")
        mat = _compute_heatmap_from_model(data_path=data_path, ckpt_path=ckpt_path, config_path=config_path)

    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=300)
    sns.heatmap(
        mat,
        cmap="Reds",
        ax=ax,
        xticklabels=list(range(24)),
        yticklabels=[f"Expert {i + 1}" for i in range(mat.shape[0])],
        cbar_kws={"label": "Activation Probability"},
        vmin=0.0,
        vmax=float(np.max(mat)) if float(np.max(mat)) > 0 else 1.0,
    )
    ax.set_xlabel("Hour of Day (0–23)", fontweight="bold")
    ax.set_ylabel("Experts", fontweight="bold")
    ax.set_title("Expert Activation vs. Hour of Day")

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "fig_real_expert_heatmap")
    plt.close(fig)


if __name__ == "__main__":
    plot_real_expert_heatmap()
