from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


MapperFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MapperBundle:
    method: str
    fn: MapperFn
    params: Dict[str, float | int | str]


def _clean_pairs(pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(t)
    if not np.any(m):
        raise ValueError("No valid finite calibration pairs were found.")
    return p[m], t[m]


def _clip(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), low, high)


def fit_affine_mapper(
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
) -> MapperBundle:
    p, t = _clean_pairs(pred_cal, true_cal)
    p_var = float(np.var(p))
    if p_var <= 1e-12:
        slope = 0.0
        intercept = float(np.mean(t))
    else:
        slope = float(np.cov(p, t, ddof=0)[0, 1] / p_var)
        intercept = float(np.mean(t) - slope * np.mean(p))

    def mapper(x: np.ndarray) -> np.ndarray:
        y = slope * np.asarray(x, dtype=np.float64) + intercept
        return _clip(y, dst_low, dst_high)

    return MapperBundle(
        method="affine",
        fn=mapper,
        params={
            "slope": slope,
            "intercept": intercept,
            "dst_low": float(dst_low),
            "dst_high": float(dst_high),
        },
    )


def fit_isotonic_mapper(
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
) -> MapperBundle:
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception as exc:
        raise RuntimeError(
            "isotonic mapping requires scikit-learn. Install it before running dispatch closure."
        ) from exc

    p, t = _clean_pairs(pred_cal, true_cal)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, t)

    def mapper(x: np.ndarray) -> np.ndarray:
        y = ir.predict(np.asarray(x, dtype=np.float64).reshape(-1))
        y = y.reshape(np.asarray(x).shape)
        return _clip(y, dst_low, dst_high)

    return MapperBundle(
        method="isotonic",
        fn=mapper,
        params={
            "n_calibration": int(p.size),
            "x_min": float(np.min(p)),
            "x_max": float(np.max(p)),
            "dst_low": float(dst_low),
            "dst_high": float(dst_high),
        },
    )


def fit_quantile_mapper(
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
    n_quantiles: int = 1001,
) -> MapperBundle:
    p, t = _clean_pairs(pred_cal, true_cal)
    q = np.linspace(0.0, 1.0, int(n_quantiles), dtype=np.float64)
    p_q = np.quantile(p, q)
    t_q = np.quantile(t, q)

    # Guard against repeated bins: enforce weak monotonicity.
    p_q = np.maximum.accumulate(p_q)
    t_q = np.maximum.accumulate(t_q)

    def mapper(x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        x_rank = np.interp(x_arr.reshape(-1), p_q, q, left=0.0, right=1.0)
        y = np.interp(x_rank, q, t_q, left=float(t_q[0]), right=float(t_q[-1]))
        y = y.reshape(x_arr.shape)
        return _clip(y, dst_low, dst_high)

    return MapperBundle(
        method="quantile",
        fn=mapper,
        params={
            "n_quantiles": int(n_quantiles),
            "dst_low": float(dst_low),
            "dst_high": float(dst_high),
        },
    )


def fit_mapping(
    method: str,
    pred_cal: np.ndarray,
    true_cal: np.ndarray,
    *,
    dst_low: float,
    dst_high: float,
) -> MapperBundle:
    m = method.strip().lower()
    if m == "affine":
        return fit_affine_mapper(pred_cal, true_cal, dst_low=dst_low, dst_high=dst_high)
    if m == "isotonic":
        return fit_isotonic_mapper(pred_cal, true_cal, dst_low=dst_low, dst_high=dst_high)
    if m == "quantile":
        return fit_quantile_mapper(pred_cal, true_cal, dst_low=dst_low, dst_high=dst_high)
    raise ValueError(f"Unsupported mapping method: {method}")

