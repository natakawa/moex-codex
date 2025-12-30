from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EstimationConfig:
    mean_method: str = "sample"  # sample | ewma
    ewma_lambda: float = 0.97
    cov_method: str = "sample"  # sample | shrinkage_diag
    shrinkage_alpha: float = 0.20


def filter_and_align_returns(returns: pd.DataFrame, *, min_obs_days: int) -> pd.DataFrame:
    non_na = returns.notna().sum(axis=0)
    keep_cols = non_na[non_na >= int(min_obs_days)].index.tolist()
    x = returns[keep_cols].dropna(how="any")
    if x.empty:
        raise ValueError("No overlapping return history after filtering")
    return x


def estimate_mean(x: pd.DataFrame, cfg: EstimationConfig) -> pd.Series:
    if cfg.mean_method == "sample":
        return x.mean()
    if cfg.mean_method == "ewma":
        lam = float(cfg.ewma_lambda)
        if not (0.0 < lam < 1.0):
            raise ValueError("ewma_lambda must be in (0,1)")
        # EWMA mean with exponentially decaying weights (most recent gets highest weight).
        n = len(x)
        w = np.array([(1 - lam) * (lam ** (n - 1 - i)) for i in range(n)], dtype=float)
        w = w / w.sum()
        return pd.Series((x.to_numpy().T @ w), index=x.columns)
    raise ValueError(f"Unknown mean_method: {cfg.mean_method}")


def estimate_cov(x: pd.DataFrame, cfg: EstimationConfig) -> pd.DataFrame:
    s = x.cov()
    if cfg.cov_method == "sample":
        return s
    if cfg.cov_method == "shrinkage_diag":
        a = float(cfg.shrinkage_alpha)
        if not (0.0 <= a <= 1.0):
            raise ValueError("shrinkage_alpha must be in [0,1]")
        d = np.diag(np.diag(s.to_numpy()))
        shr = (1.0 - a) * s.to_numpy() + a * d
        return pd.DataFrame(shr, index=s.index, columns=s.columns)
    raise ValueError(f"Unknown cov_method: {cfg.cov_method}")


def estimate_mu_cov(x: pd.DataFrame, cfg: EstimationConfig) -> tuple[pd.Series, pd.DataFrame]:
    mu = estimate_mean(x, cfg)
    cov = estimate_cov(x, cfg)
    return mu, cov

