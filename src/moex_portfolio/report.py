from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import annualize_return, annualize_vol, historical_var_es, max_drawdown_from_returns, sharpe_ratio, capm_regression


def asset_report(returns: pd.DataFrame, rf_series: pd.Series, periods_per_year: int) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        if r.empty:
            continue
        mu = float(r.mean())
        sig = float(r.std(ddof=1))
        rf_mu = float(rf_series.reindex(r.index).dropna().mean()) if not rf_series.empty else 0.0
        sr = sharpe_ratio(mu, sig, rf_mu)
        mdd = max_drawdown_from_returns(r)
        var95, es95 = historical_var_es(r, alpha=0.95)
        rows.append(
            {
                "secid": col,
                "mean_daily": mu,
                "vol_daily": sig,
                "ret_ann": annualize_return(mu, periods_per_year),
                "vol_ann": annualize_vol(sig, periods_per_year),
                "sharpe_daily_excess": sr,
                "mdd": mdd,
                "var95_1d": var95,
                "es95_1d": es95,
            }
        )
    return pd.DataFrame(rows).set_index("secid").sort_index()


def capm_report(returns: pd.DataFrame, rf: pd.Series, market: pd.Series) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        aligned = pd.concat(
            [
                returns[col].rename("asset"),
                rf.rename("rf"),
                market.rename("mkt"),
            ],
            axis=1,
        ).dropna()
        if aligned.empty:
            continue
        excess_asset = aligned["asset"] - aligned["rf"]
        excess_mkt = aligned["mkt"] - aligned["rf"]
        out = capm_regression(excess_asset, excess_mkt)
        rows.append({"secid": col, **out})
    return pd.DataFrame(rows).set_index("secid").sort_index()


def write_df(df: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
