from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ae_plot_style import save_ae, set_ae_style as set_base_style


def mm_to_in(mm: float) -> float:
    return mm / 25.4


PALETTE = {
    "ink": "#1F2933",
    "muted": "#52606D",
    "grid": "#D9E2EC",
    "ultra": "#0072B2",
    "lgbm": "#D55E00",
    "gaussian": "#009E73",
    "drift": "#CC79A7",
    "quant": "#E69F00",
    "input_tint": "#EAF2FB",
    "generator_tint": "#F4F6F8",
    "perturbed_tint": "#FFF1E8",
    "model_tint": "#EEF2F5",
    "gaussian_tint": "#E7F4EE",
    "drift_tint": "#F8EAF2",
    "quant_tint": "#FDF3E0",
    "high_noise_tint": "#FDECC8",
}


def set_ae_style() -> None:
    # Use a guaranteed local font to avoid platform-dependent missing-font warnings.
    set_base_style(font="DejaVu Sans", base_fontsize=8)


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fontsize: float = 8,
    *,
    facecolor: str = "white",
    edgecolor: str = PALETTE["muted"],
    text_color: str = PALETTE["ink"],
    linewidth: float = 0.9,
) -> FancyBboxPatch:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color)
    return p


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str = PALETTE["muted"]) -> None:
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=color)
    ax.add_patch(a)


def _read_gaussian_json(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return (
        np.asarray(obj["noise_levels"], dtype=float),
        np.asarray(obj["ultra_lsnt_scores"], dtype=float),
        np.asarray(obj["lightgbm_scores"], dtype=float),
    )


def _read_pair_curve_from_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pattern = re.compile(
        r"Noise\s+([0-9]*\.?[0-9]+)\s*\|\s*Ultra-LSNT:\s*([0-9]*\.?[0-9]+)\s*vs\s*LightGBM:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    )
    rows: list[tuple[float, float, float]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not rows:
        raise ValueError(f"No score rows found in {path}")
    rows.sort(key=lambda t: t[0])
    return (
        np.asarray([x[0] for x in rows], dtype=float),
        np.asarray([x[1] for x in rows], dtype=float),
        np.asarray([x[2] for x in rows], dtype=float),
    )


def _latest_log(prefix: str) -> Path:
    logs = sorted(Path("logs_complete").glob(f"{prefix}_*.log"))
    if not logs:
        raise FileNotFoundError(f"Missing logs_complete/{prefix}_*.log")
    return logs[-1]


def _levels_text(levels: np.ndarray) -> str:
    return "{" + ",".join(f"{x:.1f}" for x in levels) + "}"


def _summary_text(ultra: np.ndarray, lgbm: np.ndarray) -> str:
    return f"U/L R2: {ultra[0]:.3f}/{lgbm[0]:.3f} -> {ultra[-1]:.3f}/{lgbm[-1]:.3f}"


def load_curves() -> dict[str, np.ndarray]:
    g_noise, g_ultra, g_lgbm = _read_gaussian_json(Path("gbdt_results_complete.json"))
    d_noise, d_ultra, d_lgbm = _read_pair_curve_from_log(_latest_log("gbdt_drift"))
    q_noise, q_ultra, q_lgbm = _read_pair_curve_from_log(_latest_log("gbdt_quantization"))
    return {
        "g_noise": g_noise,
        "g_ultra": g_ultra,
        "g_lgbm": g_lgbm,
        "d_noise": d_noise,
        "d_ultra": d_ultra,
        "d_lgbm": d_lgbm,
        "q_noise": q_noise,
        "q_ultra": q_ultra,
        "q_lgbm": q_lgbm,
    }


def draw_schematic(ax: plt.Axes, curves: dict[str, np.ndarray]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    input_box = box(
        ax,
        0.02,
        0.39,
        0.15,
        0.22,
        "Input features\n$\\mathbf{X}$",
        fontsize=7.8,
        facecolor=PALETTE["input_tint"],
        edgecolor=PALETTE["ultra"],
    )
    gen_box = box(ax, 0.21, 0.18, 0.46, 0.64, "", facecolor=PALETTE["generator_tint"], edgecolor=PALETTE["muted"])
    perturbed_box = box(
        ax,
        0.70,
        0.39,
        0.15,
        0.22,
        "Perturbed input\n$\\tilde{\\mathbf{X}}$",
        fontsize=7.8,
        facecolor=PALETTE["perturbed_tint"],
        edgecolor=PALETTE["lgbm"],
    )
    model_box = box(
        ax,
        0.865,
        0.31,
        0.115,
        0.38,
        "Models\nUltra-LSNT\nLightGBM",
        fontsize=7.0,
        facecolor=PALETTE["model_tint"],
        edgecolor=PALETTE["muted"],
    )

    section_x = 0.225
    section_w = 0.43
    ax.add_patch(Rectangle((section_x, 0.57), section_w, 0.17, facecolor=PALETTE["gaussian_tint"], edgecolor="none"))
    ax.add_patch(Rectangle((section_x, 0.38), section_w, 0.17, facecolor=PALETTE["drift_tint"], edgecolor="none"))
    ax.add_patch(Rectangle((section_x, 0.20), section_w, 0.16, facecolor=PALETTE["quant_tint"], edgecolor="none"))

    arrow(ax, 0.17, 0.50, 0.21, 0.50, color=PALETTE["muted"])
    arrow(ax, 0.67, 0.50, 0.70, 0.50, color=PALETTE["muted"])
    arrow(ax, 0.85, 0.50, 0.865, 0.50, color=PALETTE["muted"])

    # Clip all generator texts to the generator container to avoid cross-box overlap.
    clip_text = {"clip_on": True, "clip_path": gen_box}

    ax.text(
        0.44,
        0.765,
        "Perturbation generator\n(shared across models)",
        ha="center",
        va="center",
        fontsize=7.6,
        color=PALETTE["ink"],
        **clip_text,
    )

    ax.text(
        0.24,
        0.715,
        "Gaussian noise:",
        ha="left",
        va="center",
        fontsize=7.1,
        color=PALETTE["gaussian"],
        fontweight="bold",
        **clip_text,
    )
    ax.text(
        0.24,
        0.685,
        r"$\tilde{\mathbf{x}}=\mathbf{x}+\sigma\varepsilon,\ \varepsilon\sim\mathcal{N}(0,\mathrm{std}(\mathbf{x})^2)$",
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["ink"],
        **clip_text,
    )
    ax.text(
        0.24,
        0.64,
        rf"$\sigma\in{_levels_text(curves['g_noise'])}$",
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["muted"],
        **clip_text,
    )
    ax.text(
        0.24,
        0.608,
        f"R2 U/L: {curves['g_ultra'][0]:.3f}/{curves['g_lgbm'][0]:.3f} -> "
        f"{curves['g_ultra'][-1]:.3f}/{curves['g_lgbm'][-1]:.3f}",
        ha="left",
        va="top",
        fontsize=5.9,
        color=PALETTE["muted"],
        **clip_text,
    )

    ax.text(
        0.24,
        0.525,
        "Drift (sensor bias):",
        ha="left",
        va="center",
        fontsize=7.1,
        color=PALETTE["drift"],
        fontweight="bold",
        **clip_text,
    )
    ax.text(
        0.24,
        0.495,
        r"$\tilde{\mathbf{x}}_t=\mathbf{x}_t+\delta(t),\ \delta(t)=\rho\cdot(t/T)\cdot\mathrm{std}(\mathbf{x})$",
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["ink"],
        **clip_text,
    )
    ax.text(
        0.24,
        0.45,
        rf"$\rho\in{_levels_text(curves['d_noise'])}$",
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["muted"],
        **clip_text,
    )
    ax.text(
        0.24,
        0.418,
        f"R2 U/L: {curves['d_ultra'][0]:.3f}/{curves['d_lgbm'][0]:.3f} -> "
        f"{curves['d_ultra'][-1]:.3f}/{curves['d_lgbm'][-1]:.3f}",
        ha="left",
        va="top",
        fontsize=5.9,
        color=PALETTE["muted"],
        **clip_text,
    )

    ax.text(
        0.24,
        0.34,
        "Quantization:",
        ha="left",
        va="center",
        fontsize=7.1,
        color=PALETTE["quant"],
        fontweight="bold",
        **clip_text,
    )
    ax.text(
        0.24,
        0.31,
        r"$\tilde{\mathbf{x}}=\Delta\cdot\mathrm{round}(\mathbf{x}/\Delta)$ (fixed/per-channel step)",
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["ink"],
        **clip_text,
    )
    ax.text(
        0.24,
        0.265,
        rf"$\epsilon\in{_levels_text(curves['q_noise'])}$",
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["muted"],
        **clip_text,
    )
    ax.text(
        0.24,
        0.235,
        f"R2 U/L: {curves['q_ultra'][0]:.3f}/{curves['q_lgbm'][0]:.3f} -> "
        f"{curves['q_ultra'][-1]:.3f}/{curves['q_lgbm'][-1]:.3f}",
        ha="left",
        va="top",
        fontsize=5.9,
        color=PALETTE["muted"],
        **clip_text,
    )

    ax.text(
        0.02,
        0.07,
        "Key point: same generator + same levels for all models (fair comparison).",
        fontsize=7.0,
        color=PALETTE["ink"],
    )


def _style_curve_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["muted"])
    ax.spines["bottom"].set_color(PALETTE["muted"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.tick_params(axis="both", colors=PALETTE["muted"])
    ax.set_facecolor("#FBFCFE")


def _plot_pair(ax: plt.Axes, x: np.ndarray, y_u: np.ndarray, y_l: np.ndarray, title: str, tag: str) -> None:
    ax.axvspan(float(x[-2]), float(x[-1]), color=PALETTE["high_noise_tint"], alpha=0.7, zorder=0)
    ax.plot(
        x,
        y_u,
        marker="o",
        markersize=3.7,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linewidth=1.25,
        color=PALETTE["ultra"],
        label="Ultra-LSNT",
    )
    ax.plot(
        x,
        y_l,
        marker="s",
        markersize=3.3,
        markerfacecolor="white",
        markeredgecolor=PALETTE["lgbm"],
        markeredgewidth=0.8,
        linewidth=1.2,
        linestyle="--",
        color=PALETTE["lgbm"],
        label="LightGBM",
    )
    ax.set_title(f"{tag} {title}", loc="left", pad=2, color=PALETTE["ink"])
    ax.set_xticks(x)
    ax.set_xlim(float(x[0]) - 0.01, float(x[-1]) + 0.01)
    _style_curve_ax(ax)


def draw_curves(ax_top: plt.Axes, ax_mid: plt.Axes, ax_bot: plt.Axes, curves: dict[str, np.ndarray]) -> None:
    _plot_pair(ax_top, curves["g_noise"], curves["g_ultra"], curves["g_lgbm"], "Gaussian", "(a)")
    _plot_pair(ax_mid, curves["d_noise"], curves["d_ultra"], curves["d_lgbm"], "Drift", "(b)")
    _plot_pair(ax_bot, curves["q_noise"], curves["q_ultra"], curves["q_lgbm"], "Quantization", "(c)")

    all_y = np.concatenate(
        [
            curves["g_ultra"],
            curves["g_lgbm"],
            curves["d_ultra"],
            curves["d_lgbm"],
            curves["q_ultra"],
            curves["q_lgbm"],
        ]
    )
    y_min = float(np.floor((all_y.min() - 0.02) * 20.0) / 20.0)
    y_max = float(np.ceil((all_y.max() + 0.02) * 20.0) / 20.0)
    for ax in [ax_top, ax_mid, ax_bot]:
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("$R^2$", color=PALETTE["ink"])
    ax_top.set_xticklabels([])
    ax_mid.set_xticklabels([])
    ax_bot.set_xlabel("Noise level $\\epsilon$", color=PALETTE["ink"])

    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(handles, labels, loc="lower left", frameon=False)


def main() -> None:
    set_ae_style()
    curves = load_curves()

    fig = plt.figure(figsize=(mm_to_in(190), mm_to_in(98)))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.06)
    ax_left = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(3, 1, hspace=0.30)
    ax_r1 = fig.add_subplot(right[0, 0])
    ax_r2 = fig.add_subplot(right[1, 0])
    ax_r3 = fig.add_subplot(right[2, 0])

    draw_schematic(ax_left, curves)
    draw_curves(ax_r1, ax_r2, ax_r3, curves)

    out_dir = Path("figures_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_noise_protocol_schematic_final"
    try:
        save_ae(fig, out_prefix=stem, out_dir=str(out_dir))
        out_png = out_dir / f"{stem}.png"
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
    except PermissionError:
        stem = "fig_noise_protocol_schematic_final_v2"
        save_ae(fig, out_prefix=stem, out_dir=str(out_dir))
        out_png = out_dir / f"{stem}.png"
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_dir / f'{stem}.pdf'}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
