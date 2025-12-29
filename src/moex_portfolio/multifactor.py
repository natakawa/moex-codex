from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    name: str
    secid: str
    excess_vs_rf: bool = False


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, float]:
    # X is (n, k) WITHOUT intercept; we add intercept.
    n = len(y)
    if n == 0:
        return np.array([]), float("nan"), float("nan")
    X1 = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    resid = y - y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    resid_std = float(np.sqrt(ss_res / max(n - X1.shape[1], 1)))
    return beta, float(r2), resid_std


def run_multifactor(
    returns: pd.DataFrame,
    rf: pd.Series,
    factors: list[FactorSpec],
    min_obs: int = 252,
) -> pd.DataFrame:
    """
    Regress asset excess returns on a configurable factor set.
      y = asset - rf
      X_j = factor_j (optionally factor_j - rf if excess_vs_rf)
    Returns a table with alpha + factor loadings + R².
    """
    if returns.empty:
        return pd.DataFrame()

    factor_cols = []
    X_parts = []
    base = pd.DataFrame(index=returns.index)
    base["rf"] = rf

    for f in factors:
        if f.secid not in returns.columns:
            continue
        s = returns[f.secid].rename(f.name)
        if f.excess_vs_rf:
            s = (s - rf).rename(f.name)
        base[f.name] = s
        factor_cols.append(f.name)

    if not factor_cols:
        return pd.DataFrame()

    out_rows = []
    for col in returns.columns:
        if col not in returns.columns:
            continue
        df = base.copy()
        df["asset"] = returns[col]
        df = df.dropna()
        if len(df) < min_obs:
            continue
        y = (df["asset"] - df["rf"]).to_numpy()
        X = df[factor_cols].to_numpy()
        beta, r2, resid_std = _ols(y, X)
        if beta.size == 0:
            continue
        row = {"secid": col, "alpha": float(beta[0]), "r2": r2, "resid_std": resid_std}
        for i, name in enumerate(factor_cols, start=1):
            row[name] = float(beta[i])
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows).set_index("secid").sort_index()
    return df_out

