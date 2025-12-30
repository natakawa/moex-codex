from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .estimation import EstimationConfig, estimate_mu_cov, filter_and_align_returns


@dataclass(frozen=True)
class Constraints:
    min_weight_rf: float
    max_weight_any: float
    min_obs_days: int = 504


def _portfolio_stats(mu: np.ndarray, cov: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    ret = float(mu @ w)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol


def turnover_l1(w: np.ndarray, w_ref: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(w - w_ref)))


def _soft_abs(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Smooth |x| for SLSQP stability.
    return np.sqrt(x * x + eps)


def robust_near_max_sharpe_long_only(
    returns: pd.DataFrame,
    rf_col: str,
    constraints: Constraints,
    *,
    eps_sharpe_relative: float,
    turnover_lambda: float,
    l2_lambda: float,
    hhi_lambda: float,
    reference_weights: pd.Series,
    estimation: EstimationConfig,
) -> pd.Series:
    if constraints.min_weight_rf > constraints.max_weight_any:
        raise ValueError(
            f"Infeasible bounds: min_weight_rf={constraints.min_weight_rf} > max_weight_any={constraints.max_weight_any}"
        )
    """
    Two-stage approach:
      1) Find max-Sharpe portfolio under constraints.
      2) Find a "near-max Sharpe" portfolio: Sharpe >= (1-eps)*Sharpe_max,
         minimizing stability penalties (turnover + L2 deviation + concentration HHI).

    NOTE: Sharpe is computed on daily returns with RF treated as an asset; the excess
    is taken vs the RF asset mean return.
    """
    w_max = max_sharpe_long_only(returns, rf_col=rf_col, constraints=constraints, estimation=estimation)

    x = filter_and_align_returns(returns[w_max.index], min_obs_days=constraints.min_obs_days)
    cols = list(x.columns)
    mu_s, cov_df = estimate_mu_cov(x, estimation)
    mu = mu_s.to_numpy()
    cov = cov_df.to_numpy()
    n = len(cols)
    rf_idx = cols.index(rf_col)
    w_max_np = w_max.to_numpy(dtype=float)

    ret_max, vol_max = _portfolio_stats(mu, cov, w_max_np)
    rf_mu = float(mu[rf_idx])
    sr_max = (ret_max - rf_mu) / vol_max if vol_max > 0 else float("nan")
    if not np.isfinite(sr_max):
        return w_max

    sr_floor = (1.0 - float(eps_sharpe_relative)) * sr_max

    w_ref = reference_weights.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    if w_ref.sum() != 0:
        w_ref = w_ref / w_ref.sum()
    else:
        w_ref = np.full(n, 1.0 / n)

    def sharpe_excess(w: np.ndarray) -> float:
        ret, vol = _portfolio_stats(mu, cov, w)
        if vol <= 0:
            return -1e9
        return float((ret - rf_mu) / vol)

    def objective(w: np.ndarray) -> float:
        dw = w - w_ref
        turnover = 0.5 * float(np.sum(_soft_abs(dw)))
        l2 = float(np.sum(dw * dw))
        hhi = float(np.sum(w * w))
        return float(turnover_lambda * turnover + l2_lambda * l2 + hhi_lambda * hhi)

    bounds = [(0.0, constraints.max_weight_any) for _ in range(n)]
    bounds[rf_idx] = (constraints.min_weight_rf, constraints.max_weight_any)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: sharpe_excess(w) - sr_floor},
    ]

    # Start from max-sharpe solution (feasible by definition).
    w0 = w_max_np.copy()
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 50_000})
    if not res.success:
        # Fallback: if the second stage fails, return max-sharpe.
        return w_max
    return pd.Series(res.x, index=cols, name="weight")


def max_sharpe_long_only(
    returns: pd.DataFrame,
    rf_col: str,
    constraints: Constraints,
    estimation: EstimationConfig,
) -> pd.Series:
    if constraints.min_weight_rf > constraints.max_weight_any:
        raise ValueError(
            f"Infeasible bounds: min_weight_rf={constraints.min_weight_rf} > max_weight_any={constraints.max_weight_any}"
        )
    if rf_col not in returns.columns:
        raise ValueError(f"rf_col={rf_col} not in returns columns")
    # Avoid shrinking the entire sample to the intersection across illiquid/new instruments:
    # 1) drop columns with too little history
    x = filter_and_align_returns(returns, min_obs_days=constraints.min_obs_days)

    cols = list(x.columns)
    mu_s, cov_df = estimate_mu_cov(x, estimation)
    mu = mu_s.to_numpy()
    cov = cov_df.to_numpy()

    n = len(cols)
    rf_idx = cols.index(rf_col)
    rf_mu = float(mu[rf_idx])

    def objective(w: np.ndarray) -> float:
        ret, vol = _portfolio_stats(mu, cov, w)
        sr = ((ret - rf_mu) / vol) if vol > 0 else -1e9
        return -sr

    bounds = [(0.0, constraints.max_weight_any) for _ in range(n)]
    bounds[rf_idx] = (constraints.min_weight_rf, constraints.max_weight_any)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    w0 = np.full(n, 1.0 / n)
    w0[rf_idx] = max(w0[rf_idx], constraints.min_weight_rf)
    w0 = w0 / w0.sum()

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 10_000})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    w = res.x
    return pd.Series(w, index=cols, name="weight")
