from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ae_plot_style import set_ae_style, save_ae
from fig_from_tables import load_metrics_json


def metric_panel(ax, df, metric, better="high"):
    """
    Draw a horizontal scatter panel for one metric.
    df: DataFrame with index = model names, columns include the metric.
    better: "high" if higher values are better (e.g., R^2), "low" otherwise.
    """
    d = df.copy()
    # Sort so that the best model appears at the top
    if better == "high":
        d = d.sort_values(metric, ascending=True)  # for horizontal plot bottom->top
    else:
        d = d.sort_values(metric, ascending=False)

    y = np.arange(len(d))
    ax.scatter(d[metric].values, y, s=40, alpha=0.8)
    ax.set_yticks(y, d.index.tolist())
    ax.set_xlabel(metric)
    ax.margins(x=0.08)

    # highlight Ultra-LSNT
    for tick in ax.get_yticklabels():
        if "Ultra-LSNT" in tick.get_text():
            tick.set_fontweight("bold")


def main():
    set_ae_style()

    # Load metrics from the real experiment JSON
    df = load_metrics_json("results/metrics.json")
    # load_metrics_json returns columns "R^2", "RMSE", "MAE"
    # Rename R^2 to R2 for compatibility with the original code
    df = df.rename(columns={"R^2": "R2"})

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    metric_panel(axes[0], df, "R2", better="high")
    metric_panel(axes[1], df, "RMSE", better="low")
    metric_panel(axes[2], df, "MAE", better="low")

    axes[0].set_title("$R^2$ (higher is better)")
    axes[1].set_title("RMSE (lower is better)")
    axes[2].set_title("MAE (lower is better)")

    save_ae(fig, "FIG_windcn_overall", out_dir="figures_out", dpi_mixed=600)
    plt.close(fig)


if __name__ == "__main__":
    main()