#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pypower.case24_ieee_rts import case24_ieee_rts
from pypower.idx_brch import PF, RATE_A
from pypower.idx_gen import PG
from pypower.ppoption import ppoption

from dispatch_mapping_utils import fit_mapping
from run_network_constrained_dispatch_ieee24 import (
    DEFAULT_LOAD_PROFILE,
    DEFAULT_SRC_WIND_RANGE_KW,
    MODEL_PATHS,
    affine_map_to_mw,
    build_hour_case,
    load_forecasts_and_truth,
    run_dcopf_safe,
)


OUT_DIR = Path("results") / "supplementary_evidence"
OUT_HOURLY = OUT_DIR / "dispatch_closure_mapping_decision_hourly.csv"
OUT_DAILY = OUT_DIR / "dispatch_closure_mapping_decision_daily.csv"
OUT_AGG = OUT_DIR / "dispatch_closure_mapping_decision_aggregate.csv"
OUT_RANK = OUT_DIR / "dispatch_closure_mapping_decision_ranking_stability.csv"
OUT_META = OUT_DIR / "dispatch_closure_mapping_decision.meta.json"


def parse_methods(raw: str) -> List[str]:
    methods = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not methods:
        raise ValueError("No mapping method was provided.")
    allowed = {"affine", "isotonic", "quantile"}
    unknown = [m for m in methods if m not in allowed]
    if unknown:
        raise ValueError(f"Unsupported mapping method(s): {unknown}")
    return methods


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(y_t) & np.isfinite(y_p)
    if int(np.sum(m)) < 2:
        return float("nan")
    t = y_t[m]
    p = y_p[m]
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _global_hour_idx(day_idx: int, hour: int) -> int:
    return day_idx * 24 + hour


def evaluate_hourly(
    *,
    mapping_method: str,
    model_name: str,
    forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    base_case: dict,
    ppopt: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict] = []
    n_days = int(forecast_mw.shape[0])
    n_base_gen = int(base_case["gen"].shape[0])
    for day in range(n_days):
        for hour in range(24):
            f_wind = float(forecast_mw[day, hour])
            a_wind = float(actual_mw[day, hour])
            load_factor = float(DEFAULT_LOAD_PROFILE[hour])
            gidx = _global_hour_idx(day, hour)

            if gidx == 0:
                prev_f = f_wind
                prev_a = a_wind
            else:
                prev_day = (gidx - 1) // 24
                prev_h = (gidx - 1) % 24
                prev_f = float(forecast_mw[prev_day, prev_h])
                prev_a = float(actual_mw[prev_day, prev_h])

            ramp_actual = abs(a_wind - prev_a)
            ramp_forecast = abs(f_wind - prev_f)
            ramp_error = abs(ramp_forecast - ramp_actual)
            forecast_error = abs(f_wind - a_wind)

            da_case = build_hour_case(
                base_case=base_case,
                wind_avail_mw=f_wind,
                load_scale=args.load_scale,
                hour_factor=load_factor,
                wind_bus=args.wind_bus,
                slack_bus=args.slack_bus,
                slack_cost_linear=args.slack_cost_linear,
                thermal_fix_pg=None,
            )
            da_result = run_dcopf_safe(da_case, ppopt)
            da_success = da_result is not None
            rt_success = False
            curtail = np.nan
            rt_cost = np.nan
            slack = np.nan
            congestion = np.nan

            if da_success:
                thermal_schedule_pg = da_result["gen"][:n_base_gen, PG].copy()
                rt_case = build_hour_case(
                    base_case=base_case,
                    wind_avail_mw=a_wind,
                    load_scale=args.load_scale,
                    hour_factor=load_factor,
                    wind_bus=args.wind_bus,
                    slack_bus=args.slack_bus,
                    slack_cost_linear=args.slack_cost_linear,
                    thermal_fix_pg=thermal_schedule_pg,
                )
                rt_result = run_dcopf_safe(rt_case, ppopt)
                if rt_result is not None:
                    rt_success = True
                    rt_cost = float(rt_result["f"])
                    wind_pg = float(rt_result["gen"][n_base_gen, PG])
                    slack = float(rt_result["gen"][n_base_gen + 1, PG])
                    curtail = float(max(a_wind - wind_pg, 0.0))
                    branch = rt_result["branch"]
                    flow = np.abs(branch[:, PF])
                    rate_a = branch[:, RATE_A]
                    valid = rate_a > 0
                    if np.any(valid):
                        congestion = float(np.any(flow[valid] >= 0.99 * rate_a[valid]))
                    else:
                        congestion = 0.0

            success = da_success and rt_success
            rows.append(
                {
                    "mapping_method": mapping_method,
                    "model": model_name,
                    "day_idx": int(day),
                    "hour": int(hour),
                    "global_hour_idx": int(gidx),
                    "forecast_mw": float(f_wind),
                    "actual_mw": float(a_wind),
                    "forecast_error_mw": float(forecast_error),
                    "ramp_actual_mw": float(ramp_actual),
                    "ramp_forecast_mw": float(ramp_forecast),
                    "ramp_error_mw": float(ramp_error),
                    "load_factor": float(load_factor),
                    "da_success": int(da_success),
                    "rt_success": int(rt_success),
                    "success": int(success),
                    "feasibility_fail": int(1 - int(success)),
                    "rt_cost": float(rt_cost) if np.isfinite(rt_cost) else np.nan,
                    "curtailment_mwh": float(curtail) if np.isfinite(curtail) else np.nan,
                    "slack_mwh": float(max(slack, 0.0)) if np.isfinite(slack) else np.nan,
                    "congestion_flag": float(congestion) if np.isfinite(congestion) else np.nan,
                }
            )

        if (day + 1) % 20 == 0 or day + 1 == n_days:
            print(f"[{mapping_method}/{model_name}] processed {day + 1}/{n_days} days")

    return pd.DataFrame(rows)


def add_risk_and_backfire_flags(df_hourly: pd.DataFrame, risk_q: float, ramp_q: float) -> pd.DataFrame:
    out = df_hourly.copy()
    out["is_extreme_ramp"] = 0
    out["is_high_risk"] = 0
    out["is_backfire_vs_dlinear"] = 0

    for mapping in sorted(out["mapping_method"].unique()):
        dm = out[out["mapping_method"] == mapping].copy()
        base_grid = dm[dm["model"] == "DLinear"]
        if base_grid.empty:
            base_grid = dm.groupby(["day_idx", "hour"], as_index=False).first()

        load_thr = float(np.nanquantile(base_grid["load_factor"], risk_q))
        ramp_thr = float(np.nanquantile(base_grid["ramp_actual_mw"], ramp_q))

        for model in sorted(dm["model"].unique()):
            idx = (out["mapping_method"] == mapping) & (out["model"] == model)
            err_thr = float(np.nanquantile(out.loc[idx, "forecast_error_mw"], risk_q))
            extreme = out.loc[idx, "ramp_actual_mw"] >= ramp_thr
            high_risk = (
                (out.loc[idx, "load_factor"] >= load_thr)
                | (out.loc[idx, "ramp_actual_mw"] >= ramp_thr)
                | (out.loc[idx, "forecast_error_mw"] >= err_thr)
            )
            out.loc[idx, "is_extreme_ramp"] = extreme.fillna(False).astype(int)
            out.loc[idx, "is_high_risk"] = high_risk.fillna(False).astype(int)

        base = out[(out["mapping_method"] == mapping) & (out["model"] == "DLinear")][
            ["day_idx", "hour", "success", "rt_cost", "curtailment_mwh"]
        ].rename(
            columns={
                "success": "base_success",
                "rt_cost": "base_cost",
                "curtailment_mwh": "base_curt",
            }
        )
        if base.empty:
            continue

        for model in sorted(dm["model"].unique()):
            if model == "DLinear":
                continue
            sub = out[(out["mapping_method"] == mapping) & (out["model"] == model)].merge(
                base,
                on=["day_idx", "hour"],
                how="left",
            )
            cond_fail = (sub["base_success"] == 1) & (sub["success"] == 0)
            cond_worse = (
                (sub["base_success"] == 1)
                & (sub["success"] == 1)
                & (sub["rt_cost"] > sub["base_cost"])
                & (sub["curtailment_mwh"] > sub["base_curt"])
            )
            backfire = (cond_fail | cond_worse).astype(int).to_numpy()
            out.loc[(out["mapping_method"] == mapping) & (out["model"] == model), "is_backfire_vs_dlinear"] = backfire

    return out


def build_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df_hourly.groupby(["mapping_method", "model", "day_idx"], as_index=False)
        .agg(
            hours=("hour", "count"),
            success_hours=("success", "sum"),
            feasibility_fail_hours=("feasibility_fail", "sum"),
            rt_cost_total=("rt_cost", "sum"),
            curtailment_total_mwh=("curtailment_mwh", "sum"),
            slack_total_mwh=("slack_mwh", "sum"),
            congestion_hours=("congestion_flag", "sum"),
            backfire_hours=("is_backfire_vs_dlinear", "sum"),
            extreme_ramp_hours=("is_extreme_ramp", "sum"),
            high_risk_hours=("is_high_risk", "sum"),
        )
    )
    return daily


def _aggregate_subset(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    rows = []
    for (mapping, model), g in df.groupby(["mapping_method", "model"]):
        hours = int(len(g))
        success_rate = float(np.mean(g["success"])) if hours else np.nan
        rows.append(
            {
                "mapping_method": mapping,
                "subset": subset_name,
                "model": model,
                "hours": hours,
                "success_rate": success_rate,
                "feasibility_fail_rate": float(1.0 - success_rate) if np.isfinite(success_rate) else np.nan,
                "total_rt_cost": float(np.nansum(g["rt_cost"])),
                "mean_rt_cost": float(np.nanmean(g["rt_cost"])) if np.any(np.isfinite(g["rt_cost"])) else np.nan,
                "total_curtailment_mwh": float(np.nansum(g["curtailment_mwh"])),
                "mean_curtailment_mwh": float(np.nanmean(g["curtailment_mwh"])) if np.any(np.isfinite(g["curtailment_mwh"])) else np.nan,
                "total_slack_mwh": float(np.nansum(g["slack_mwh"])),
                "mean_slack_mwh": float(np.nanmean(g["slack_mwh"])) if np.any(np.isfinite(g["slack_mwh"])) else np.nan,
                "congestion_rate": float(np.nanmean(g["congestion_flag"])) if np.any(np.isfinite(g["congestion_flag"])) else np.nan,
                "backfire_rate_vs_dlinear": float(np.mean(g["is_backfire_vs_dlinear"])) if hours else np.nan,
                "mean_ramp_error_mw": float(np.nanmean(g["ramp_error_mw"])) if np.any(np.isfinite(g["ramp_error_mw"])) else np.nan,
                "p95_ramp_error_mw": float(np.nanquantile(g["ramp_error_mw"], 0.95)) if np.any(np.isfinite(g["ramp_error_mw"])) else np.nan,
                "mean_forecast_error_mw": float(np.nanmean(g["forecast_error_mw"])) if np.any(np.isfinite(g["forecast_error_mw"])) else np.nan,
                "forecast_r2": _safe_r2(g["actual_mw"].to_numpy(), g["forecast_mw"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def build_aggregate(df_hourly: pd.DataFrame) -> pd.DataFrame:
    all_df = _aggregate_subset(df_hourly, "all")
    extreme_df = _aggregate_subset(df_hourly[df_hourly["is_extreme_ramp"] == 1], "extreme_ramp")
    risk_df = _aggregate_subset(df_hourly[df_hourly["is_high_risk"] == 1], "high_risk")
    out = pd.concat([all_df, extreme_df, risk_df], axis=0, ignore_index=True)
    return out


def _spearman_from_rank_vectors(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return float("nan")
    am = float(np.mean(a))
    bm = float(np.mean(b))
    num = float(np.sum((a - am) * (b - bm)))
    den = float(np.sqrt(np.sum((a - am) ** 2) * np.sum((b - bm) ** 2)))
    if den <= 1e-12:
        return float("nan")
    return num / den


def build_ranking_stability(df_agg: pd.DataFrame) -> pd.DataFrame:
    all_df = df_agg[df_agg["subset"] == "all"].copy()
    rank_rows = []
    rank_vectors: Dict[str, Dict[str, float]] = {}
    for mapping, g in all_df.groupby("mapping_method"):
        gg = g.sort_values("total_rt_cost", ascending=True).reset_index(drop=True)
        rank_vectors[mapping] = {}
        for i, r in gg.iterrows():
            model = str(r["model"])
            rank = int(i + 1)
            rank_vectors[mapping][model] = rank
            rank_rows.append(
                {
                    "record_type": "rank",
                    "mapping_method": mapping,
                    "model": model,
                    "rank_by_total_rt_cost": rank,
                    "top1_model": str(gg.iloc[0]["model"]),
                    "spearman_rank_corr": np.nan,
                    "top1_consistency": np.nan,
                }
            )

    map_methods = sorted(rank_vectors.keys())
    pair_rows = []
    for i in range(len(map_methods)):
        for j in range(i + 1, len(map_methods)):
            m1 = map_methods[i]
            m2 = map_methods[j]
            models = sorted(set(rank_vectors[m1].keys()).intersection(rank_vectors[m2].keys()))
            r1 = np.asarray([rank_vectors[m1][m] for m in models], dtype=np.float64)
            r2 = np.asarray([rank_vectors[m2][m] for m in models], dtype=np.float64)
            pair_rows.append(
                {
                    "record_type": "pairwise",
                    "mapping_method": f"{m1}__vs__{m2}",
                    "model": "ALL",
                    "rank_by_total_rt_cost": np.nan,
                    "top1_model": np.nan,
                    "spearman_rank_corr": _spearman_from_rank_vectors(r1, r2),
                    "top1_consistency": float(
                        int(
                            min(rank_vectors[m1], key=rank_vectors[m1].get)
                            == min(rank_vectors[m2], key=rank_vectors[m2].get)
                        )
                    ),
                }
            )

    top1_models = [min(v, key=v.get) for v in rank_vectors.values()] if rank_vectors else []
    summary = {
        "record_type": "summary",
        "mapping_method": "ALL",
        "model": "ALL",
        "rank_by_total_rt_cost": np.nan,
        "top1_model": ",".join(top1_models),
        "spearman_rank_corr": float(np.nanmean([r["spearman_rank_corr"] for r in pair_rows])) if pair_rows else np.nan,
        "top1_consistency": float(len(set(top1_models)) == 1) if top1_models else np.nan,
    }
    return pd.DataFrame(rank_rows + pair_rows + [summary])


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch closure with mapping robustness and decision-aware consistency.")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--mapping-methods", type=str, default="affine,isotonic,quantile")
    parser.add_argument("--risk-q", type=float, default=0.9)
    parser.add_argument("--ramp-q", type=float, default=0.9)
    parser.add_argument("--calibration-ratio", type=float, default=0.5)
    parser.add_argument("--wind-min-mw", type=float, default=0.0)
    parser.add_argument("--wind-max-mw", type=float, default=300.0)
    parser.add_argument("--wind-bus", type=int, default=7)
    parser.add_argument("--slack-bus", type=int, default=13)
    parser.add_argument("--load-scale", type=float, default=0.75)
    parser.add_argument("--slack-cost-linear", type=float, default=2000.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    methods = parse_methods(args.mapping_methods)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_dispatch_closure_mapping_decision"

    model_forecasts_raw, truth_raw = load_forecasts_and_truth(args.days)
    n_days = int(truth_raw.shape[0])
    src_low, src_high = DEFAULT_SRC_WIND_RANGE_KW
    truth_mw = affine_map_to_mw(
        truth_raw,
        src_low_kw=src_low,
        src_high_kw=src_high,
        dst_low_mw=args.wind_min_mw,
        dst_high_mw=args.wind_max_mw,
    )

    calib_days = max(1, int(n_days * float(args.calibration_ratio)))
    mapping_params: Dict[str, Dict[str, Dict]] = {}
    mapped_predictions: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
    for method in methods:
        mapping_params[method] = {}
        for model, pred_raw in model_forecasts_raw.items():
            if method == "affine":
                pred_mw = affine_map_to_mw(
                    pred_raw,
                    src_low_kw=src_low,
                    src_high_kw=src_high,
                    dst_low_mw=args.wind_min_mw,
                    dst_high_mw=args.wind_max_mw,
                )
                mapping_params[method][model] = {
                    "rule": "fixed_affine_clipping",
                    "src_low_kw": float(src_low),
                    "src_high_kw": float(src_high),
                    "dst_low_mw": float(args.wind_min_mw),
                    "dst_high_mw": float(args.wind_max_mw),
                }
            else:
                bundle = fit_mapping(
                    method=method,
                    pred_cal=pred_raw[:calib_days],
                    true_cal=truth_mw[:calib_days],
                    dst_low=args.wind_min_mw,
                    dst_high=args.wind_max_mw,
                )
                pred_mw = bundle.fn(pred_raw)
                mapping_params[method][model] = bundle.params
            mapped_predictions[method][model] = pred_mw

    base_case = case24_ieee_rts()
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    hourly_frames = []
    for method in methods:
        for model, pred in mapped_predictions[method].items():
            print(f"[run] method={method}, model={model}")
            dfh = evaluate_hourly(
                mapping_method=method,
                model_name=model,
                forecast_mw=pred,
                actual_mw=truth_mw,
                base_case=base_case,
                ppopt=ppopt,
                args=args,
            )
            hourly_frames.append(dfh)

    hourly = pd.concat(hourly_frames, axis=0, ignore_index=True)
    hourly = add_risk_and_backfire_flags(hourly, risk_q=float(args.risk_q), ramp_q=float(args.ramp_q))
    hourly["run_id"] = run_id
    hourly.to_csv(OUT_HOURLY, index=False)

    daily = build_daily(hourly)
    daily["run_id"] = run_id
    daily.to_csv(OUT_DAILY, index=False)

    agg = build_aggregate(hourly)
    agg["run_id"] = run_id
    agg.to_csv(OUT_AGG, index=False)

    rank = build_ranking_stability(agg)
    rank["run_id"] = run_id
    rank.to_csv(OUT_RANK, index=False)

    meta = {
        "script": "run_dispatch_closure_mapping_decision_4090.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "days": n_days,
        "mapping_methods": methods,
        "calibration_days": calib_days,
        "risk_threshold_quantile": float(args.risk_q),
        "ramp_threshold_quantile": float(args.ramp_q),
        "high_risk_definition": "peak_load(q>=risk_q) OR extreme_ramp(q>=ramp_q) OR severe_pollution_proxy(forecast_error>=q_risk)",
        "severe_pollution_proxy": "absolute forecast error in MW",
        "network": "IEEE RTS-24 DC-OPF",
        "wind_mapping": {
            "src_range_kw": [float(src_low), float(src_high)],
            "dst_range_mw": [float(args.wind_min_mw), float(args.wind_max_mw)],
        },
        "model_sources": {k: str(v) for k, v in MODEL_PATHS.items()},
        "mapping_params": mapping_params,
        "outputs": {
            "hourly_csv": str(OUT_HOURLY),
            "daily_csv": str(OUT_DAILY),
            "aggregate_csv": str(OUT_AGG),
            "ranking_stability_csv": str(OUT_RANK),
            "meta_json": str(OUT_META),
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] wrote {OUT_HOURLY}")
    print(f"[ok] wrote {OUT_DAILY}")
    print(f"[ok] wrote {OUT_AGG}")
    print(f"[ok] wrote {OUT_RANK}")
    print(f"[ok] wrote {OUT_META}")


if __name__ == "__main__":
    main()

