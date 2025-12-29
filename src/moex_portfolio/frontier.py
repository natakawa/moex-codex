from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class FrontierConfig:
    points: int = 30


@dataclass(frozen=True)
class FrontierConstraints:
    min_weight_rf: float
    max_weight_any: float
    min_obs_days: int = 504


def _prep_panel(returns: pd.DataFrame, constraints: FrontierConstraints) -> pd.DataFrame:
    # Keep columns with sufficient history, then use a common intersection.
    non_na = returns.notna().sum(axis=0)
    keep_cols = non_na[non_na >= constraints.min_obs_days].index.tolist()
    x = returns[keep_cols].dropna(how="any")
    if x.empty:
        raise ValueError("No overlapping return history after filtering")
    return x


def _linear_extreme_mu(mu: np.ndarray, rf_idx: int, c: FrontierConstraints, maximize: bool) -> np.ndarray:
    # Solve max/min mu^T w subject to bounds + sum=1 with a greedy allocator (linear objective).
    n = len(mu)
    w = np.zeros(n, dtype=float)
    w[rf_idx] = c.min_weight_rf
    remaining = 1.0 - w[rf_idx]
    order = np.argsort(mu)
    if maximize:
        order = order[::-1]
    for i in order:
        cap = c.max_weight_any
        if i == rf_idx:
            cap = c.max_weight_any
        take = min(cap - w[i], remaining)
        if take > 0:
            w[i] += take
            remaining -= take
        if remaining <= 1e-12:
            break
    if abs(w.sum() - 1.0) > 1e-6:
        # Fallback: distribute leftover to rf (within cap) if possible.
        slack = 1.0 - w.sum()
        if slack > 0 and w[rf_idx] + slack <= c.max_weight_any + 1e-9:
            w[rf_idx] += slack
    return w


def _min_variance_for_target(mu: np.ndarray, cov: np.ndarray, target: float, rf_idx: int, c: FrontierConstraints):
    n = len(mu)

    def obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    bounds = [(0.0, c.max_weight_any) for _ in range(n)]
    bounds[rf_idx] = (c.min_weight_rf, c.max_weight_any)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(mu @ w) - float(target)},
    ]

    # Start from a feasible-ish point: mix of min and max mu portfolios.
    w_low = _linear_extreme_mu(mu, rf_idx, c, maximize=False)
    w_high = _linear_extreme_mu(mu, rf_idx, c, maximize=True)
    mu_low = float(mu @ w_low)
    mu_high = float(mu @ w_high)
    if not (min(mu_low, mu_high) - 1e-12 <= target <= max(mu_low, mu_high) + 1e-12):
        return None
    t = 0.5 if mu_high == mu_low else (target - mu_low) / (mu_high - mu_low)
    w0 = (1 - t) * w_low + t * w_high
    w0 = np.clip(w0, 0.0, c.max_weight_any)
    w0[rf_idx] = max(w0[rf_idx], c.min_weight_rf)
    w0 = w0 / w0.sum()

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 50_000})
    if not res.success:
        return None
    return res.x


def efficient_frontier(
    returns: pd.DataFrame,
    rf_col: str,
    constraints: FrontierConstraints,
    cfg: FrontierConfig = FrontierConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      frontier_df: rows for target returns with realized (ret, vol) and feasibility
      weights_df: matching weights for feasible points
    """
    if rf_col not in returns.columns:
        raise ValueError(f"rf_col={rf_col} not found")

    x = _prep_panel(returns, constraints)
    cols = list(x.columns)
    rf_idx = cols.index(rf_col)

    mu = x.mean().to_numpy()
    cov = x.cov().to_numpy()

    w_min = _linear_extreme_mu(mu, rf_idx, constraints, maximize=False)
    w_max = _linear_extreme_mu(mu, rf_idx, constraints, maximize=True)
    mu_min = float(mu @ w_min)
    mu_max = float(mu @ w_max)
    if mu_max < mu_min:
        mu_min, mu_max = mu_max, mu_min

    targets = np.linspace(mu_min, mu_max, num=max(cfg.points, 10))
    rows = []
    w_rows = []

    for target in targets:
        w = _min_variance_for_target(mu, cov, float(target), rf_idx, constraints)
        if w is None:
            continue
        ret = float(mu @ w)
        var = float(w @ cov @ w)
        vol = float(np.sqrt(max(var, 0.0)))
        rows.append({"target_return_daily": float(target), "ret_daily": ret, "vol_daily": vol})
        w_rows.append(pd.Series(w, index=cols, name=float(target)))

    frontier_df = pd.DataFrame(rows).sort_values("ret_daily").reset_index(drop=True)
    weights_df = pd.DataFrame(w_rows)
    return frontier_df, weights_df

