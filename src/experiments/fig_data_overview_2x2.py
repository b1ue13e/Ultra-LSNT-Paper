from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _min_max_norm_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if np.isclose(xmax, xmin):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def _downsample_1d(y: np.ndarray, max_points: int = 600) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.arange(n, dtype=int)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return idx, y[idx]


def _smooth_1d(y: np.ndarray, window: int = 9) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y
    w = max(5, int(window))
    if w % 2 == 0:
        w += 1
    return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _load_cn_wind(cn_csv: Path) -> np.ndarray:
    df = pd.read_csv(cn_csv, usecols=["DATATIME", "PREPOWER"])
    df["DATATIME"] = pd.to_datetime(df["DATATIME"], errors="coerce")
    df["PREPOWER"] = pd.to_numeric(df["PREPOWER"], errors="coerce")
    df = df.dropna(subset=["DATATIME", "PREPOWER"]).sort_values("DATATIME")
    daily = df.set_index("DATATIME")["PREPOWER"].resample("D").mean().dropna()
    return daily.to_numpy(dtype=float)


def _load_us_wind(us_csv: Path) -> np.ndarray:
    df = pd.read_csv(us_csv)
    minute = df["Minute"] if "Minute" in df.columns else 0
    df["timestamp"] = pd.to_datetime(
        {
            "year": df["Year"],
            "month": df["Month"],
            "day": df["Day"],
            "hour": df["Hour"],
            "minute": minute,
        },
        errors="coerce",
    )
    df["power (MW)"] = pd.to_numeric(df["power (MW)"], errors="coerce")
    df = df.dropna(subset=["timestamp", "power (MW)"]).sort_values("timestamp")
    daily = df.set_index("timestamp")["power (MW)"].resample("D").mean().dropna()
    return daily.to_numpy(dtype=float)


def _load_air_quality(air_csv: Path, air_ready_csv: Path) -> tuple[np.ndarray, str]:
    # Prefer raw air_quality.csv to keep real date aggregation.
    if air_csv.exists():
        try:
            df = pd.read_csv(air_csv, usecols=["日期", "aqi指数"])
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df["aqi指数"] = pd.to_numeric(df["aqi指数"], errors="coerce")
            df = df.dropna(subset=["日期", "aqi指数"]).sort_values("日期")
            daily = df.groupby("日期", as_index=True)["aqi指数"].mean().sort_index()
            if len(daily) > 0:
                return daily.to_numpy(dtype=float), "air_quality.csv (daily mean AQI)"
        except Exception:
            pass

    df = pd.read_csv(air_ready_csv, usecols=["AQI"])
    seq = pd.to_numeric(df["AQI"], errors="coerce").dropna()
    return seq.to_numpy(dtype=float), "air_quality_ready.csv (sequential AQI)"


def _load_gefcom(gefcom_csv: Path) -> np.ndarray:
    df = pd.read_csv(gefcom_csv, usecols=["date", "load"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df = df.dropna(subset=["date", "load"]).sort_values("date")
    daily = df.set_index("date")["load"].resample("D").mean().dropna()
    return daily.to_numpy(dtype=float)


def load_real_data(
    cn_csv: str = "wind_main.csv",
    us_csv: str = "wind_us.csv",
    air_csv: str = "air_quality.csv",
    air_ready_csv: str = "air_quality_ready.csv",
    gefcom_csv: str = "gefcom_ready.csv",
) -> tuple[list[tuple[str, np.ndarray]], dict[str, str]]:
    dataset_series = [
        ("Wind (CN)", _load_cn_wind(Path(cn_csv))),
        ("Wind (US)", _load_us_wind(Path(us_csv))),
    ]
    air_series, air_source = _load_air_quality(Path(air_csv), Path(air_ready_csv))
    dataset_series.append(("Air Quality", air_series))
    dataset_series.append(("GEFCom Load", _load_gefcom(Path(gefcom_csv))))

    sources = {
        "Wind (CN)": cn_csv,
        "Wind (US)": us_csv,
        "Air Quality": air_source,
        "GEFCom Load": gefcom_csv,
    }
    return dataset_series, sources


def plot_overview(dataset_series: list[tuple[str, np.ndarray]]) -> plt.Figure:
    """
    2x2 overview for four real datasets:
    Wind (CN), Wind (US), Air Quality, GEFCom Load.
    """
    if len(dataset_series) != 4:
        raise ValueError(f"Expected 4 datasets for 2x2 plotting, got {len(dataset_series)}")

    set_ae_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#EDEDED")

    descriptor = {
        "Wind (CN)": "volatile + clustered ramps",
        "Wind (US)": "high-frequency fluctuations",
        "Air Quality": "seasonal decay + spikes",
        "GEFCom Load": "periodic load cycles",
    }

    for i, (title, ts) in enumerate(dataset_series):
        ax = axes.ravel()[i]
        ts_norm = _min_max_norm_1d(ts)
        x, y = _downsample_1d(ts_norm, max_points=600)
        # Use a thin, smooth line style consistent with the requested reference.
        y_s = _smooth_1d(y, window=max(7, len(y) // 90))
        ax.set_facecolor("#EDEDED")
        ax.plot(x, y_s, linewidth=0.95, color="#1f77b4")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Target (normalized)")
        ax.set_title(title)
        ax.grid(True, color="#D0D0D0", alpha=0.75, linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, f"({chr(ord('a') + i)})")
        tag = descriptor.get(title, "")
        if tag:
            ax.text(
                0.98,
                0.95,
                tag,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.7,
                color="#334155",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1", "alpha": 0.95},
            )

    return fig


def sync_manuscript_aliases(out_prefix: str, out_dir: str = "figures_out") -> None:
    generated_dir = Path(out_dir)
    aliases = {
        generated_dir / f"{out_prefix}.pdf": Path("data_statistics_2x2_real.pdf"),
        generated_dir / f"{out_prefix}.tif": Path("data_statistics_2x2_real.tif"),
    }
    for src, dst in aliases.items():
        if src.exists():
            copyfile(src, dst)


if __name__ == "__main__":
    print("Loading real datasets: wind(CN), wind(US), air quality, GEFCom")
    dataset_series, sources = load_real_data()

    for name, ts in dataset_series:
        print(f"{name:12s} points={len(ts):5d}, range=[{np.nanmin(ts):.3f}, {np.nanmax(ts):.3f}]")
        print(f"  source: {sources[name]}")

    fig = plot_overview(dataset_series)
    out_prefix = "data_statistics_2x2_real"
    try:
        save_ae(fig, out_prefix=out_prefix, out_dir="figures_out")
    except PermissionError:
        out_prefix = "data_statistics_2x2_real_v2"
        save_ae(fig, out_prefix=out_prefix, out_dir="figures_out")
        print("Primary file is locked; used fallback prefix data_statistics_2x2_real_v2")
    sync_manuscript_aliases(out_prefix=out_prefix, out_dir="figures_out")
    print(f"Saved: figures_out/{out_prefix}.pdf and .tif")
    if plt.get_backend().lower() != "agg":
        plt.show()
