import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def create_windows(
    features: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    n = len(features)
    for i in range(seq_len, n - pred_len + 1):
        x.append(features[i - seq_len : i])
        y.append(target[i : i + pred_len])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def activation_fn(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class ELMConfig:
    hidden_units: int
    ridge_alpha: float
    activation: str
    input_scale: float


class ELMRegressor:
    def __init__(self, cfg: ELMConfig, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.w = None
        self.b = None
        self.beta = None

    def _transform(self, x: np.ndarray) -> np.ndarray:
        z = (x @ self.w + self.b) * self.cfg.input_scale
        return activation_fn(self.cfg.activation, z)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ELMRegressor":
        rng = np.random.default_rng(self.seed)
        in_dim = x.shape[1]
        hid = self.cfg.hidden_units
        self.w = rng.normal(0.0, 1.0, size=(in_dim, hid)).astype(np.float64)
        self.b = rng.normal(0.0, 1.0, size=(hid,)).astype(np.float64)

        h = self._transform(x.astype(np.float64))
        ht = h.T
        reg = self.cfg.ridge_alpha * np.eye(h.shape[1], dtype=np.float64)
        a = ht @ h + reg
        b = ht @ y.astype(np.float64)
        self.beta = np.linalg.solve(a, b)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = self._transform(x.astype(np.float64))
        y = h @ self.beta
        return y.astype(np.float32)

    @property
    def active_params_m(self) -> float:
        if self.w is None or self.b is None or self.beta is None:
            return 0.0
        n_params = self.w.size + self.b.size + self.beta.size
        return float(n_params / 1_000_000.0)

    @property
    def model_size_mib(self) -> float:
        if self.w is None or self.b is None or self.beta is None:
            return 0.0
        n_bytes = (self.w.nbytes + self.b.nbytes + self.beta.nbytes)
        return float(n_bytes / (1024.0 * 1024.0))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    return {
        "R2": float(r2_score(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
    }


def inject_gaussian(x: np.ndarray, sigma_eff: float, seed: int) -> np.ndarray:
    if sigma_eff <= 0:
        return x.copy()
    rng = np.random.default_rng(seed)
    std = np.std(x, axis=(1, 2), keepdims=True) + 1e-12
    noise = rng.normal(0.0, sigma_eff, size=x.shape) * std
    return (x + noise).astype(np.float32)


def inject_spike(x: np.ndarray, severity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = x.copy()
    n, l, c = y.shape
    total = n * l * c
    n_spike = max(1, int(0.03 * total))
    flat = y.reshape(-1)
    idx = rng.choice(total, size=n_spike, replace=False)
    amp = severity * (np.std(flat) + 1e-12) * 3.0
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_spike)
    flat[idx] = flat[idx] + signs * amp
    return y


def decode_cfg(vec: np.ndarray) -> ELMConfig:
    activations = ["tanh", "relu", "sigmoid"]
    hid = int(np.clip(np.round(vec[0]), 32, 384))
    log_alpha = float(np.clip(vec[1], -8.0, -1.0))
    alpha = 10.0 ** log_alpha
    act_id = int(np.clip(np.round(vec[2]), 0, len(activations) - 1))
    inp_scale = float(np.clip(vec[3], 0.4, 2.5))
    return ELMConfig(
        hidden_units=hid,
        ridge_alpha=alpha,
        activation=activations[act_id],
        input_scale=inp_scale,
    )


def ssa_optimize(
    objective_fn,
    bounds: List[Tuple[float, float]],
    pop_size: int = 12,
    iters: int = 12,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)

    pop = rng.uniform(lb, ub, size=(pop_size, dim))
    fit = np.array([objective_fn(ind) for ind in pop], dtype=np.float64)

    trace = []
    for it in range(iters):
        order = np.argsort(fit)
        pop = pop[order]
        fit = fit[order]
        best = pop[0].copy()
        best_score = float(fit[0])
        trace.append(
            {
                "iter": int(it),
                "best_score": best_score,
                "best_params": best.tolist(),
            }
        )

        n_prod = max(1, int(0.2 * pop_size))
        n_warn = max(1, int(0.1 * pop_size))

        for i in range(n_prod):
            r2 = rng.uniform()
            if r2 < 0.8:
                pop[i] = pop[i] * np.exp(-i / (rng.uniform() * max(it + 1, 1)))
            else:
                pop[i] = pop[i] + rng.normal(0.0, 0.1, size=dim)

        for i in range(n_prod, pop_size):
            if i > pop_size / 2:
                pop[i] = rng.normal(0.0, 1.0, size=dim) * np.exp((pop[-1] - pop[i]) / ((i + 1) ** 2))
            else:
                pop[i] = best + np.abs(pop[i] - best) * rng.normal(0.0, 1.0, size=dim)

        warn_idx = rng.choice(pop_size, size=n_warn, replace=False)
        for wi in warn_idx:
            if fit[wi] > best_score:
                pop[wi] = best + rng.normal(0.0, 1.0, size=dim) * np.abs(pop[wi] - best)
            else:
                pop[wi] = pop[wi] + rng.uniform(-1.0, 1.0, size=dim) * np.abs(pop[wi] - pop[-1])

        pop = np.clip(pop, lb, ub)
        fit = np.array([objective_fn(ind) for ind in pop], dtype=np.float64)

    idx = int(np.argmin(fit))
    return pop[idx], float(fit[idx]), trace


def measure_latency_ms(model: ELMRegressor, x_sample: np.ndarray, runs: int = 200) -> Dict[str, float]:
    for _ in range(20):
        _ = model.predict(x_sample)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(x_sample)
        ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ts, dtype=np.float64)
    return {
        "latency_mean_ms": float(np.mean(arr)),
        "latency_std_ms": float(np.std(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run auditable SSA-ELM suite on Wind(CN)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--out_dir", type=str, default=os.path.join("results", "supplementary_evidence"))
    ap.add_argument("--audit_tag", type=str, default="ssa_elm_auditable_suite")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.audit_tag

    df = pd.read_csv("wind_final.csv")
    num = df.select_dtypes(include=[np.number]).copy()
    if "power" not in num.columns:
        raise RuntimeError("power column not found in wind_final.csv")

    cols = [c for c in num.columns if c != "power"] + ["power"]
    arr = num[cols].values.astype(np.float64)
    target_idx = len(cols) - 1
    n = len(arr)
    n_train = int(0.8 * n)

    train_raw = arr[:n_train]
    test_raw = arr[n_train:]
    x_scaler = StandardScaler().fit(train_raw)
    y_scaler = StandardScaler().fit(train_raw[:, target_idx].reshape(-1, 1))

    train_x = x_scaler.transform(train_raw).astype(np.float32)
    test_x = x_scaler.transform(test_raw).astype(np.float32)
    train_y = y_scaler.transform(train_raw[:, target_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    test_y = y_scaler.transform(test_raw[:, target_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)

    xtr, ytr = create_windows(train_x, train_y, args.seq_len, args.pred_len)
    xte, yte = create_windows(test_x, test_y, args.seq_len, args.pred_len)

    in_dim = xtr.shape[1] * xtr.shape[2]
    xtr_flat = xtr.reshape(xtr.shape[0], in_dim).astype(np.float64)
    xte_flat = xte.reshape(xte.shape[0], in_dim).astype(np.float64)

    tr_obj = min(12000, int(0.75 * len(xtr_flat)))
    va_obj = min(3000, len(xtr_flat) - tr_obj)
    if va_obj <= 100:
        raise RuntimeError("Not enough data for SSA objective split.")
    x_obj_tr = xtr_flat[:tr_obj]
    y_obj_tr = ytr[:tr_obj]
    x_obj_va = xtr_flat[tr_obj : tr_obj + va_obj]
    y_obj_va = ytr[tr_obj : tr_obj + va_obj]

    bounds = [
        (48.0, 320.0),   # hidden_units
        (-7.0, -2.0),    # log10(ridge_alpha)
        (0.0, 2.0),      # activation idx
        (0.5, 2.0),      # input_scale
    ]

    def objective(v: np.ndarray) -> float:
        cfg = decode_cfg(v)
        model = ELMRegressor(cfg=cfg, seed=args.seed)
        model.fit(x_obj_tr, y_obj_tr)
        pred = model.predict(x_obj_va)
        r2 = r2_score(y_obj_va.reshape(-1), pred.reshape(-1))
        return float(-r2)

    best_vec, best_score, trace = ssa_optimize(
        objective_fn=objective,
        bounds=bounds,
        pop_size=10,
        iters=10,
        seed=args.seed,
    )
    best_cfg = decode_cfg(best_vec)

    t0 = time.time()
    model = ELMRegressor(cfg=best_cfg, seed=args.seed)
    model.fit(xtr_flat, ytr)
    train_time = time.time() - t0

    pred_clean_s = model.predict(xte_flat)
    pred_clean = y_scaler.inverse_transform(pred_clean_s.reshape(-1, 1)).reshape(pred_clean_s.shape)
    true_clean = y_scaler.inverse_transform(yte.reshape(-1, 1)).reshape(yte.shape)
    m_clean = metrics(true_clean, pred_clean)

    xte_g = inject_gaussian(xte, sigma_eff=0.60, seed=args.seed).reshape(xte.shape[0], in_dim)
    pred_g_s = model.predict(xte_g.astype(np.float64))
    pred_g = y_scaler.inverse_transform(pred_g_s.reshape(-1, 1)).reshape(pred_g_s.shape)
    m_g = metrics(true_clean, pred_g)

    xte_sp = inject_spike(xte, severity=0.6, seed=args.seed + 1).reshape(xte.shape[0], in_dim)
    pred_sp_s = model.predict(xte_sp.astype(np.float64))
    pred_sp = y_scaler.inverse_transform(pred_sp_s.reshape(-1, 1)).reshape(pred_sp_s.shape)
    m_sp = metrics(true_clean, pred_sp)

    lat = measure_latency_ms(model, xte_flat[:1], runs=220)

    clean_rows = [
        {
            "dataset": "wind_cn",
            "model": "SSA-ELM",
            "R2": m_clean["R2"],
            "RMSE": m_clean["RMSE"],
            "MAE": m_clean["MAE"],
            "train_time_s": train_time,
        }
    ]
    robust_rows = [
        {
            "dataset": "wind_cn",
            "model": "SSA-ELM",
            "fault_type": "gaussian",
            "severity": 0.6,
            "sigma_eff": 0.6,
            "R2": m_g["R2"],
            "relative_drop_pct": 100.0 * (m_clean["R2"] - m_g["R2"]) / max(abs(m_clean["R2"]), 1e-8),
        },
        {
            "dataset": "wind_cn",
            "model": "SSA-ELM",
            "fault_type": "spike",
            "severity": 0.6,
            "sigma_eff": np.nan,
            "R2": m_sp["R2"],
            "relative_drop_pct": 100.0 * (m_clean["R2"] - m_sp["R2"]) / max(abs(m_clean["R2"]), 1e-8),
        },
    ]
    latency_rows = [
        {
            "dataset": "wind_cn",
            "model": "SSA-ELM",
            "latency_mean_ms": lat["latency_mean_ms"],
            "latency_std_ms": lat["latency_std_ms"],
            "latency_p95_ms": lat["latency_p95_ms"],
            "active_params_m": model.active_params_m,
            "model_size_mib": model.model_size_mib,
        }
    ]

    trace_rows = []
    for row in trace:
        trace_rows.append(
            {
                "model": "SSA-ELM",
                "iter": row["iter"],
                "best_score": row["best_score"],
                "best_params": json.dumps(row["best_params"]),
            }
        )

    clean_path = os.path.join(args.out_dir, "ssa_elm_clean_windcn.csv")
    robust_path = os.path.join(args.out_dir, "ssa_elm_robustness_windcn.csv")
    latency_path = os.path.join(args.out_dir, "ssa_elm_latency_windcn.csv")
    trace_path = os.path.join(args.out_dir, "ssa_elm_search_trace.csv")
    meta_path = os.path.join(args.out_dir, "ssa_elm_meta.json")

    pd.DataFrame(clean_rows).to_csv(clean_path, index=False)
    pd.DataFrame(robust_rows).to_csv(robust_path, index=False)
    pd.DataFrame(latency_rows).to_csv(latency_path, index=False)
    pd.DataFrame(trace_rows).to_csv(trace_path, index=False)

    meta = {
        "script": "run_ssa_elm_auditable_suite.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset": "wind_final.csv",
        "split": "80_20_chronological",
        "window": {"seq_len": args.seq_len, "pred_len": args.pred_len},
        "best_cfg": {
            "hidden_units": best_cfg.hidden_units,
            "ridge_alpha": best_cfg.ridge_alpha,
            "activation": best_cfg.activation,
            "input_scale": best_cfg.input_scale,
        },
        "search": {
            "ssa_pop_size": 10,
            "ssa_iters": 10,
            "objective_best_score": best_score,
        },
        "outputs": {
            "clean_csv": clean_path,
            "robust_csv": robust_path,
            "latency_csv": latency_path,
            "trace_csv": trace_path,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("SSA-ELM auditable suite completed.")
    print("run_id:", run_id)
    print("outputs:")
    print(clean_path)
    print(robust_path)
    print(latency_path)
    print(trace_path)
    print(meta_path)


if __name__ == "__main__":
    main()
