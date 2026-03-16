from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figure_style import ensure_output_dir, save_figure, set_paper_style


set_paper_style()
OUT_DIR = ensure_output_dir("figure")


def plot_domain_comparison(csv_path: str | Path = "table_multi_domain.csv") -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run evaluate_multi_domain.py first.")

    df = pd.read_csv(csv_path)
    datasets = df["Dataset"].astype(str).tolist()
    r2_scores = df["R2 Score"].astype(float).to_numpy()
    skip_rates = df["Skip Rate"].astype(str).str.rstrip("%").astype(float).to_numpy()

    x = np.arange(len(datasets))
    bar_w = 0.52

    fig, ax1 = plt.subplots(figsize=(7.4, 4.6), dpi=300)

    # Reference palette from user sample:
    # blues: #D5E1EF, #9DC2D5, #5A94B9, #2A6398; green: #7EB87B
    domain_colors = {
        "Wind (CN)": "#2A6398",
        "Wind (US)": "#5A94B9",
        "Air Quality": "#D5E1EF",
        "GEFCom Load": "#9DC2D5",
    }
    color_r2 = "#2A6398"
    bars = ax1.bar(
        x,
        r2_scores,
        width=bar_w,
        color=[domain_colors.get(d, "#9DC2D5") for d in datasets],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.0,
        label=r"Accuracy ($R^2$)",
        zorder=2,
    )
    ax1.set_ylabel(r"R-Squared ($R^2$)", fontweight="bold", color=color_r2)
    ax1.tick_params(axis="y", labelcolor=color_r2)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontweight="bold")
    ax1.grid(axis="x", visible=False)

    for bar in bars:
        height = float(bar.get_height())
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    color_skip = "#111111"
    ax2.plot(
        x,
        skip_rates,
        label="Computational Saving (Skip%)",
        color=color_skip,
        marker="D",
        linewidth=2.2,
        linestyle="--",
        zorder=3,
    )
    ax2.set_ylabel("Skip Rate (%)", fontweight="bold", color=color_skip)
    ax2.tick_params(axis="y", labelcolor=color_skip)
    upper = max(5.0, float(skip_rates.max()) * 1.4 + 1.0)
    ax2.set_ylim(0.0, upper)
    ax2.grid(False)

    for i, v in enumerate(skip_rates.tolist()):
        ax2.annotate(
            f"{v:.1f}%",
            (x[i], v),
            textcoords="offset points",
            xytext=(6, 8),
            ha="left",
            color=color_skip,
            fontweight="bold",
            fontsize=10,
        )

    ax1.set_title("Cross-Domain Generalization & Efficiency Analysis")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "fig_multi_domain")
    plt.close(fig)


if __name__ == "__main__":
    plot_domain_comparison()
