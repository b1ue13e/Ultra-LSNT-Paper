from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from ae_plot_style import save_ae, set_ae_style


def panel_label(ax: plt.Axes, s: str) -> None:
    ax.text(
        0.02,
        0.95,
        s,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        fontweight="bold",
    )


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _load_from_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load real multi-horizon prediction results from NPZ.
    Expected keys in NPZ: predictions, ground_truth.
    """
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"Prediction file not found: {p}")

    data = np.load(p, allow_pickle=False)
    if "predictions" not in data or "ground_truth" not in data:
        raise KeyError(f"{p} missing keys. Found: {list(data.keys())}")

    y_pred = _ensure_2d(np.asarray(data["predictions"], dtype=float))
    y_true = _ensure_2d(np.asarray(data["ground_truth"], dtype=float))
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return y_true, y_pred


def _load_from_npy_pair(trues_path: str, preds_path: str) -> tuple[np.ndarray, np.ndarray]:
    pt = Path(trues_path)
    pp = Path(preds_path)
    if not pt.exists() or not pp.exists():
        raise FileNotFoundError(f"Missing file(s): {pt} or {pp}")

    y_true = _ensure_2d(np.asarray(np.load(pt, allow_pickle=False), dtype=float))
    y_pred = _ensure_2d(np.asarray(np.load(pp, allow_pickle=False), dtype=float))
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return y_true, y_pred


def load_real_pred_true(
    dataset: str = "wind_cn_main",
    model: str = "TimeMixer",
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    dataset:
      - wind_cn_main: use checkpoints_ts/*_trues.npy + *_preds.npy
      - gefcom: use checkpoints_ts/exp_gefcom/predictions.npz
      - air_quality: use checkpoints_ts/exp_air_quality/predictions.npz
    model (for wind_cn_main): TimeMixer / iTransformer / PatchTST
    """
    ds = dataset.strip().lower()
    md = model.strip()

    if ds == "wind_cn_main":
        valid = {"TimeMixer", "iTransformer", "PatchTST"}
        if md not in valid:
            raise ValueError(f"Invalid model for wind_cn_main: {md}. Choose one of {sorted(valid)}")
        y_true, y_pred = _load_from_npy_pair(
            trues_path=f"checkpoints_ts/{md}_trues.npy",
            preds_path=f"checkpoints_ts/{md}_preds.npy",
        )
        source = f"wind_cn_main ({md})"
        return y_true, y_pred, source

    if ds == "gefcom":
        y_true, y_pred = _load_from_npz("checkpoints_ts/exp_gefcom/predictions.npz")
        return y_true, y_pred, "gefcom"

    if ds == "air_quality":
        y_true, y_pred = _load_from_npz("checkpoints_ts/exp_air_quality/predictions.npz")
        return y_true, y_pred, "air_quality"

    raise ValueError("Unsupported dataset. Use one of: wind_cn_main, gefcom, air_quality")


def scatter_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: tuple[int, int, int, int] = (1, 3, 5, 7),
    max_points: int = 4000,
) -> plt.Figure:
    set_ae_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2), constrained_layout=True)

    for i, (ax, h) in enumerate(zip(axes.ravel(), horizons, strict=True)):
        idx = h - 1
        if idx < 0 or idx >= y_true.shape[1]:
            raise ValueError(f"Horizon h={h} out of range for shape {y_true.shape}")

        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]

        if len(yt) > max_points:
            sel = np.linspace(0, len(yt) - 1, max_points, dtype=int)
            yt = yt[sel]
            yp = yp[sel]

        ax.scatter(yp, yt, s=6, alpha=0.4)
        mn = float(min(yt.min(), yp.min()))
        mx = float(max(yt.max(), yp.max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)

        r2 = r2_score(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        a, b = np.polyfit(yp, yt, deg=1)
        ax.text(
            0.98,
            0.02,
            f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}\n$y$={a:.2f}x+{b:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"h={h}")
        panel_label(ax, f"({chr(ord('a') + i)})")

    return fig


if __name__ == "__main__":
    DATASET = "wind_cn_main"  # wind_cn_main / gefcom / air_quality
    MODEL = "TimeMixer"       # only used when DATASET=wind_cn_main

    y_true, y_pred, source = load_real_pred_true(dataset=DATASET, model=MODEL)

    print(f"Loaded dataset: {source}")
    print(f"Shape: {y_true.shape}")
    for h in (1, 3, 5, 7):
        idx = h - 1
        r2 = float(r2_score(y_true[:, idx], y_pred[:, idx]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx])))
        a, b = np.polyfit(y_pred[:, idx], y_true[:, idx], deg=1)
        print(f"h={h}: R2={r2:.4f}, RMSE={rmse:.4f}, y={a:.4f}x+{b:.4f}")

    fig = scatter_panel(y_true, y_pred, horizons=(1, 3, 5, 7), max_points=4000)
    out_prefix = f"pred_true_scatter_2x2_{DATASET}"
    if DATASET == "wind_cn_main":
        out_prefix = f"{out_prefix}_{MODEL}"
    save_ae(fig, out_prefix=out_prefix, out_dir="figures_out")
    print(f"Saved: figures_out/{out_prefix}.pdf and .tif")
    plt.show()
