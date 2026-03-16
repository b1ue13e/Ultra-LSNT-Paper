from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_ae_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, s: str) -> None:
    ax.text(0.02, 0.95, s, transform=ax.transAxes, va="top", ha="left", fontweight="bold")


def plot_pair(ax: plt.Axes, noise: np.ndarray, y_ultra: np.ndarray, y_lgbm: np.ndarray, title: str) -> None:
    ax.axvspan(0.20, 0.40, color="0.93", zorder=0)
    ax.plot(noise, y_ultra, marker="o", linewidth=1.0, label="Ultra-LSNT")
    ax.plot(noise, y_lgbm, marker="s", linewidth=1.0, label="LightGBM")
    ax.set_title(title, pad=2)
    ax.set_xticks(noise)
    ax.set_xlim(-0.01, 0.41)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _read_gaussian_from_json(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    noise = np.asarray(obj["noise_levels"], dtype=float)
    ultra = np.asarray(obj["ultra_lsnt_scores"], dtype=float)
    lgbm = np.asarray(obj["lightgbm_scores"], dtype=float)
    return noise, ultra, lgbm


def _find_latest_log(log_dir: Path, prefix: str) -> Path:
    candidates = sorted(log_dir.glob(f"{prefix}_*.log"))
    if not candidates:
        raise FileNotFoundError(f"No log found for pattern: {prefix}_*.log")
    return candidates[-1]


def _read_scores_from_log(path: Path, noise_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    line_pat = re.compile(
        r"Noise\s+([0-9]*\.?[0-9]+)\s*\|\s*Ultra-LSNT:\s*([0-9]*\.?[0-9]+)\s*vs\s*LightGBM:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    )
    score_map: dict[float, tuple[float, float]] = {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = line_pat.search(line)
        if m:
            n = round(float(m.group(1)), 4)
            score_map[n] = (float(m.group(2)), float(m.group(3)))

    if not score_map:
        raise ValueError(f"No valid score lines parsed from {path}")

    ultra = []
    lgbm = []
    for n in noise_ref:
        key = round(float(n), 4)
        if key not in score_map:
            raise ValueError(f"Noise level {n:.2f} not found in {path}")
        u, g = score_map[key]
        ultra.append(u)
        lgbm.append(g)
    return np.asarray(ultra, dtype=float), np.asarray(lgbm, dtype=float)


def load_real_robustness_data() -> dict[str, np.ndarray]:
    json_path = Path("gbdt_results_complete.json")
    log_dir = Path("logs_complete")

    noise, r2_g_ultra, r2_g_lgbm = _read_gaussian_from_json(json_path)

    drift_log = _find_latest_log(log_dir, "gbdt_drift")
    quant_log = _find_latest_log(log_dir, "gbdt_quantization")
    r2_d_ultra, r2_d_lgbm = _read_scores_from_log(drift_log, noise)
    r2_q_ultra, r2_q_lgbm = _read_scores_from_log(quant_log, noise)

    return {
        "noise": noise,
        "r2_g_ultra": r2_g_ultra,
        "r2_g_lgbm": r2_g_lgbm,
        "r2_d_ultra": r2_d_ultra,
        "r2_d_lgbm": r2_d_lgbm,
        "r2_q_ultra": r2_q_ultra,
        "r2_q_lgbm": r2_q_lgbm,
        "drift_log": np.array([str(drift_log)]),
        "quant_log": np.array([str(quant_log)]),
    }


def make_robustness_2x2(out_prefix: str = "robustness_2x2_real", out_dir: str = "figures_out") -> tuple[Path, Path]:
    set_ae_style()
    data = load_real_robustness_data()

    noise = data["noise"]
    r2_g_ultra = data["r2_g_ultra"]
    r2_g_lgbm = data["r2_g_lgbm"]
    r2_d_ultra = data["r2_d_ultra"]
    r2_d_lgbm = data["r2_d_lgbm"]
    r2_q_ultra = data["r2_q_ultra"]
    r2_q_lgbm = data["r2_q_lgbm"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    plot_pair(ax1, noise, r2_g_ultra, r2_g_lgbm, "Gaussian noise")
    panel_label(ax1, "(a)")
    ax1.set_ylabel("$R^2$")

    plot_pair(ax2, noise, r2_d_ultra, r2_d_lgbm, "Drift noise")
    panel_label(ax2, "(b)")

    plot_pair(ax3, noise, r2_q_ultra, r2_q_lgbm, "Quantization noise")
    panel_label(ax3, "(c)")
    ax3.set_xlabel("Noise level $\\epsilon$")
    ax3.set_ylabel("$R^2$")

    ax4.axvspan(0.20, 0.40, color="0.93", zorder=0)
    ax4.axhline(0, color="0.35", linewidth=0.8)
    ax4.plot(noise, r2_g_ultra - r2_g_lgbm, marker="o", linewidth=1.0, label="Gaussian")
    ax4.plot(noise, r2_d_ultra - r2_d_lgbm, marker="s", linewidth=1.0, label="Drift")
    ax4.plot(noise, r2_q_ultra - r2_q_lgbm, marker="^", linewidth=1.0, label="Quant.")
    ax4.set_title("$\\Delta R^2$ (Ultra-LSNT $-$ LightGBM)", pad=2)
    ax4.set_xticks(noise)
    ax4.set_xlim(-0.01, 0.41)
    ax4.set_xlabel("Noise level $\\epsilon$")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.legend(frameon=False, fontsize=7, loc="lower left")
    panel_label(ax4, "(d)")

    all_r2 = np.concatenate([r2_g_ultra, r2_g_lgbm, r2_d_ultra, r2_d_lgbm, r2_q_ultra, r2_q_lgbm])
    y_min = np.floor((all_r2.min() - 0.02) * 10) / 10
    y_max = np.ceil((all_r2.max() + 0.02) * 10) / 10
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(y_min, y_max)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path / f"{out_prefix}.pdf"
    png_path = out_path / f"{out_prefix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


if __name__ == "__main__":
    pdf, png = make_robustness_2x2()
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")
