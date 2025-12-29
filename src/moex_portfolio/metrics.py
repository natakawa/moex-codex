from __future__ import annotations

import numpy as np
import pandas as pd


def annualize_return(mean_periodic: float, periods_per_year: int) -> float:
    return float(mean_periodic * periods_per_year)


def annualize_vol(std_periodic: float, periods_per_year: int) -> float:
    return float(std_periodic * np.sqrt(periods_per_year))


def sharpe_ratio(mean_periodic: float, std_periodic: float, rf_periodic: float) -> float:
    if std_periodic == 0 or np.isnan(std_periodic):
        return float("nan")
    return float((mean_periodic - rf_periodic) / std_periodic)


def max_drawdown_from_returns(r: pd.Series) -> float:
    v = (1.0 + r.fillna(0.0)).cumprod()
    peak = v.cummax()
    dd = v / peak - 1.0
    return float(dd.min())


def historical_var_es(r: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    # Returns VaR and ES as positive numbers (loss magnitudes), consistent with risk reporting.
    x = r.dropna().to_numpy()
    if x.size == 0:
        return float("nan"), float("nan")
    losses = -x
    q = float(np.quantile(losses, alpha))
    tail = losses[losses >= q]
    es = float(tail.mean()) if tail.size else q
    return q, es


def capm_regression(excess_asset: pd.Series, excess_mkt: pd.Series) -> dict[str, float]:
    df = pd.concat([excess_asset.rename("y"), excess_mkt.rename("x")], axis=1).dropna()
    if len(df) < 30:
        return {"alpha": float("nan"), "beta": float("nan"), "r2": float("nan")}
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    beta = float(((x - x_mean) * (y - y_mean)).sum() / denom) if denom != 0 else float("nan")
    alpha = float(y_mean - beta * x_mean) if not np.isnan(beta) else float("nan")
    y_hat = alpha + beta * x
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"alpha": alpha, "beta": beta, "r2": float(r2)}

