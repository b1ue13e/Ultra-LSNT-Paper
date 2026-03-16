#!/usr/bin/env python3
"""
Network-constrained rolling dispatch evaluation (IEEE RTS-24, DC-OPF).

This script upgrades the single-node dispatch proxy by:
1) Mapping wind forecasts to an IEEE RTS-24 injection bus.
2) Solving day-ahead DC-OPF with explicit line thermal limits.
3) Replaying real-time DC-OPF with actual wind and fixed day-ahead thermal
   schedules, then recomputing curtailment under network constraints.

Outputs are written to `results/supplementary_evidence/`.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pypower.case24_ieee_rts import case24_ieee_rts
from pypower.idx_brch import PF, RATE_A
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    MBASE,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)
from pypower.ppoption import ppoption
from pypower.rundcopf import rundcopf


MODEL_PATHS = {
    "Ultra-LSNT": Path("results/supplementary_evidence/ultra_predictions_main_fulltest.npz"),
    "DLinear": Path("results/supplementary_evidence/dlinear_predictions_full.npz"),
    "TimeMixer": Path("checkpoints_ts/TimeMixer_preds.npy"),
    "iTransformer": Path("checkpoints_ts/iTransformer_preds.npy"),
}

DEFAULT_OUTPUT_DIR = Path("results/supplementary_evidence")

# Keep the same source range used in prior supplementary scaling metadata.
DEFAULT_SRC_WIND_RANGE_KW = (14250.0, 131839.0)

# Hourly profile to avoid a fully flat demand trace.
DEFAULT_LOAD_PROFILE = np.array(
    [
        0.82, 0.80, 0.78, 0.77, 0.78, 0.82,
        0.90, 0.98, 1.04, 1.08, 1.10, 1.12,
        1.14, 1.13, 1.10, 1.08, 1.09, 1.15,
        1.20, 1.22, 1.18, 1.08, 0.96, 0.88,
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IEEE RTS-24 network-constrained rolling DC-OPF")
    parser.add_argument("--days", type=int, default=120, help="Number of rolling days (default: 120)")
    parser.add_argument("--wind-bus", type=int, default=7, help="Wind injection bus ID (RTS-24 bus number)")
    parser.add_argument("--slack-bus", type=int, default=13, help="Emergency balancing generator bus ID")
    parser.add_argument("--load-scale", type=float, default=0.75, help="Global scaling factor on RTS load")
    parser.add_argument("--wind-min-mw", type=float, default=0.0, help="Mapped wind lower bound (MW)")
    parser.add_argument("--wind-max-mw", type=float, default=300.0, help="Mapped wind upper bound (MW)")
    parser.add_argument(
        "--slack-cost-linear",
        type=float,
        default=2000.0,
        help="Linear cost coefficient for emergency slack generation",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dispatch_network_ieee24_dcopf",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output files",
    )
    return parser.parse_args()


def load_prediction_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        payload = np.load(path)
        if "predictions" not in payload:
            raise KeyError(f"`predictions` key missing in {path}")
        arr = payload["predictions"]
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if arr.ndim != 2 or arr.shape[1] < 24:
        raise ValueError(f"Expected [N, >=24] forecasts in {path}, got {arr.shape}")
    return arr[:, :24].astype(np.float64)


def load_forecasts_and_truth(days: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    model_forecasts: Dict[str, np.ndarray] = {}
    for model, path in MODEL_PATHS.items():
        model_forecasts[model] = load_prediction_array(path)

    # Ground truth from the shared supplementary NPZ.
    truth_path = MODEL_PATHS["Ultra-LSNT"]
    payload = np.load(truth_path)
    if "ground_truth" not in payload:
        raise KeyError(f"`ground_truth` key missing in {truth_path}")
    truth = payload["ground_truth"][:, :24].astype(np.float64)

    max_days = min(min(v.shape[0] for v in model_forecasts.values()), truth.shape[0], days)
    for k in list(model_forecasts.keys()):
        model_forecasts[k] = model_forecasts[k][:max_days]
    truth = truth[:max_days]
    return model_forecasts, truth


def affine_map_to_mw(
    x_kw: np.ndarray,
    src_low_kw: float,
    src_high_kw: float,
    dst_low_mw: float,
    dst_high_mw: float,
) -> np.ndarray:
    denom = max(src_high_kw - src_low_kw, 1e-9)
    z = (x_kw - src_low_kw) / denom
    z = np.clip(z, 0.0, 1.0)
    return dst_low_mw + z * (dst_high_mw - dst_low_mw)


def build_hour_case(
    base_case: dict,
    wind_avail_mw: float,
    load_scale: float,
    hour_factor: float,
    wind_bus: int,
    slack_bus: int,
    slack_cost_linear: float,
    thermal_fix_pg: np.ndarray | None,
) -> dict:
    ppc = copy.deepcopy(base_case)
    ppc["bus"] = ppc["bus"].copy()
    ppc["gen"] = ppc["gen"].copy()
    ppc["gencost"] = ppc["gencost"].copy()

    # Apply scaled load for the hour.
    ppc["bus"][:, 2] *= load_scale * hour_factor  # Pd
    ppc["bus"][:, 3] *= load_scale * hour_factor  # Qd

    n_gen_cols = ppc["gen"].shape[1]

    # Wind generator (dispatchable up to available wind).
    wind_gen = np.zeros((1, n_gen_cols), dtype=float)
    wind_gen[0, GEN_BUS] = wind_bus
    wind_gen[0, PG] = 0.0
    wind_gen[0, QG] = 0.0
    wind_gen[0, QMAX] = 0.0
    wind_gen[0, QMIN] = 0.0
    wind_gen[0, VG] = 1.0
    wind_gen[0, MBASE] = ppc["baseMVA"]
    wind_gen[0, GEN_STATUS] = 1.0
    wind_gen[0, PMAX] = max(float(wind_avail_mw), 0.0)
    wind_gen[0, PMIN] = 0.0

    # Emergency balancing generator (very expensive).
    slack_gen = np.zeros((1, n_gen_cols), dtype=float)
    slack_gen[0, GEN_BUS] = slack_bus
    slack_gen[0, PG] = 0.0
    slack_gen[0, QG] = 0.0
    slack_gen[0, QMAX] = 0.0
    slack_gen[0, QMIN] = 0.0
    slack_gen[0, VG] = 1.0
    slack_gen[0, MBASE] = ppc["baseMVA"]
    slack_gen[0, GEN_STATUS] = 1.0
    slack_gen[0, PMAX] = 2500.0
    slack_gen[0, PMIN] = 0.0

    ppc["gen"] = np.vstack([ppc["gen"], wind_gen, slack_gen])

    # Quadratic format: [2, startup, shutdown, n, c2, c1, c0]
    wind_cost = np.array([[2, 0, 0, 3, 0.0, 0.0, 0.0]], dtype=float)
    slack_cost = np.array([[2, 0, 0, 3, 0.0, slack_cost_linear, 0.0]], dtype=float)
    ppc["gencost"] = np.vstack([ppc["gencost"], wind_cost, slack_cost])

    if thermal_fix_pg is not None:
        n_base_gen = base_case["gen"].shape[0]
        ppc["gen"][:n_base_gen, PMIN] = thermal_fix_pg
        ppc["gen"][:n_base_gen, PMAX] = thermal_fix_pg

    return ppc


def run_dcopf_safe(ppc: dict, ppopt: dict) -> dict | None:
    try:
        result = rundcopf(ppc, ppopt)
    except Exception:
        return None
    if not bool(result.get("success", False)):
        return None
    return result


def evaluate_model_rolling(
    model_name: str,
    forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    base_case: dict,
    ppopt: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    n_days = forecast_mw.shape[0]
    n_base_gen = base_case["gen"].shape[0]
    rows = []

    for day in range(n_days):
        day_rt_cost = 0.0
        day_curtail_mwh = 0.0
        day_slack_mwh = 0.0
        day_congestion_hours = 0
        day_success_hours = 0

        for h in range(24):
            f_wind = float(forecast_mw[day, h])
            a_wind = float(actual_mw[day, h])
            h_factor = float(DEFAULT_LOAD_PROFILE[h])

            # Day-ahead DC-OPF (uses forecast wind).
            da_case = build_hour_case(
                base_case=base_case,
                wind_avail_mw=f_wind,
                load_scale=args.load_scale,
                hour_factor=h_factor,
                wind_bus=args.wind_bus,
                slack_bus=args.slack_bus,
                slack_cost_linear=args.slack_cost_linear,
                thermal_fix_pg=None,
            )
            da_result = run_dcopf_safe(da_case, ppopt)
            if da_result is None:
                continue

            thermal_schedule_pg = da_result["gen"][:n_base_gen, PG].copy()

            # Real-time DC-OPF (actual wind + fixed DA thermal schedule).
            rt_case = build_hour_case(
                base_case=base_case,
                wind_avail_mw=a_wind,
                load_scale=args.load_scale,
                hour_factor=h_factor,
                wind_bus=args.wind_bus,
                slack_bus=args.slack_bus,
                slack_cost_linear=args.slack_cost_linear,
                thermal_fix_pg=thermal_schedule_pg,
            )
            rt_result = run_dcopf_safe(rt_case, ppopt)
            if rt_result is None:
                continue

            day_success_hours += 1
            day_rt_cost += float(rt_result["f"])

            wind_pg = float(rt_result["gen"][n_base_gen, PG])
            slack_pg = float(rt_result["gen"][n_base_gen + 1, PG])
            day_slack_mwh += max(slack_pg, 0.0)
            day_curtail_mwh += max(a_wind - wind_pg, 0.0)

            branch = rt_result["branch"]
            rate_a = branch[:, RATE_A]
            flow = np.abs(branch[:, PF])
            valid = rate_a > 0
            congested = bool(np.any(flow[valid] >= 0.99 * rate_a[valid])) if np.any(valid) else False
            if congested:
                day_congestion_hours += 1

        rows.append(
            {
                "model": model_name,
                "day_idx": int(day),
                "rt_cost_total": float(day_rt_cost),
                "curtailment_mwh_total": float(day_curtail_mwh),
                "slack_mwh_total": float(day_slack_mwh),
                "congestion_hours": int(day_congestion_hours),
                "opf_success_hours": int(day_success_hours),
            }
        )

        if (day + 1) % 20 == 0 or day + 1 == n_days:
            print(f"[{model_name}] processed {day + 1}/{n_days} days")

    return pd.DataFrame(rows)


def build_aggregate(daily: pd.DataFrame) -> pd.DataFrame:
    agg = (
        daily.groupby("model", as_index=False)
        .agg(
            days=("day_idx", "count"),
            total_rt_cost=("rt_cost_total", "sum"),
            mean_rt_cost_per_day=("rt_cost_total", "mean"),
            total_curtailment_mwh=("curtailment_mwh_total", "sum"),
            mean_curtailment_mwh_per_day=("curtailment_mwh_total", "mean"),
            total_slack_mwh=("slack_mwh_total", "sum"),
            mean_slack_mwh_per_day=("slack_mwh_total", "mean"),
            total_congestion_hours=("congestion_hours", "sum"),
            total_opf_success_hours=("opf_success_hours", "sum"),
        )
    )

    agg["congestion_hour_ratio"] = agg["total_congestion_hours"] / (agg["days"] * 24.0)
    agg["opf_success_ratio"] = agg["total_opf_success_hours"] / (agg["days"] * 24.0)

    if "DLinear" in set(agg["model"]):
        base = agg.loc[agg["model"] == "DLinear"].iloc[0]
        base_cost = float(base["total_rt_cost"])
        base_curt = float(base["total_curtailment_mwh"])
        agg["cost_reduction_vs_dlinear_pct"] = (
            100.0 * (base_cost - agg["total_rt_cost"]) / max(base_cost, 1e-9)
        )
        agg["curtailment_reduction_vs_dlinear_pct"] = (
            100.0 * (base_curt - agg["total_curtailment_mwh"]) / max(base_curt, 1e-9)
        )
    else:
        agg["cost_reduction_vs_dlinear_pct"] = np.nan
        agg["curtailment_reduction_vs_dlinear_pct"] = np.nan

    agg = agg.sort_values("total_rt_cost", ascending=True).reset_index(drop=True)
    return agg


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_forecasts, ground_truth = load_forecasts_and_truth(args.days)
    actual_days = ground_truth.shape[0]
    print(f"Loaded rolling horizon: {actual_days} days x 24 hours")

    src_low, src_high = DEFAULT_SRC_WIND_RANGE_KW
    mapped_truth_mw = affine_map_to_mw(
        ground_truth,
        src_low_kw=src_low,
        src_high_kw=src_high,
        dst_low_mw=args.wind_min_mw,
        dst_high_mw=args.wind_max_mw,
    )

    mapped_forecasts_mw = {
        model: affine_map_to_mw(
            arr,
            src_low_kw=src_low,
            src_high_kw=src_high,
            dst_low_mw=args.wind_min_mw,
            dst_high_mw=args.wind_max_mw,
        )
        for model, arr in model_forecasts.items()
    }

    base_case = case24_ieee_rts()
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)

    daily_frames = []
    for model, forecast_mw in mapped_forecasts_mw.items():
        print(f"Running model: {model}")
        daily_model = evaluate_model_rolling(
            model_name=model,
            forecast_mw=forecast_mw,
            actual_mw=mapped_truth_mw,
            base_case=base_case,
            ppopt=ppopt,
            args=args,
        )
        daily_frames.append(daily_model)

    daily = pd.concat(daily_frames, ignore_index=True)
    agg = build_aggregate(daily)

    prefix = args.output_prefix
    daily_path = out_dir / f"{prefix}_daily.csv"
    agg_path = out_dir / f"{prefix}_aggregate.csv"
    meta_path = out_dir / f"{prefix}_meta.json"

    daily.to_csv(daily_path, index=False)
    agg.to_csv(agg_path, index=False)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "two-stage rolling DC-OPF on IEEE RTS-24 (PYPOWER rundcopf)",
        "days": int(actual_days),
        "network": "IEEE RTS-24",
        "wind_bus": int(args.wind_bus),
        "slack_bus": int(args.slack_bus),
        "load_scale": float(args.load_scale),
        "hourly_load_profile": DEFAULT_LOAD_PROFILE.tolist(),
        "wind_mapping": {
            "source_kw_range": [src_low, src_high],
            "target_mw_range": [float(args.wind_min_mw), float(args.wind_max_mw)],
            "rule": "global affine + clipping",
        },
        "slack_cost_linear": float(args.slack_cost_linear),
        "model_sources": {k: str(v) for k, v in MODEL_PATHS.items()},
        "outputs": {
            "daily_csv": str(daily_path),
            "aggregate_csv": str(agg_path),
            "meta_json": str(meta_path),
        },
        "notes": [
            "Day-ahead stage uses model forecast as wind availability.",
            "Real-time stage uses actual wind and fixes thermal Pg to day-ahead schedules.",
            "Wind curtailment is recomputed under line-constrained network dispatch.",
            "Emergency slack generator captures shortfall balancing cost.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {daily_path}")
    print(f"Saved: {agg_path}")
    print(f"Saved: {meta_path}")
    print("\nAggregate summary:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
