from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .optimizer import Constraints, max_sharpe_long_only


@dataclass(frozen=True)
class BacktestSpec:
    lookback_days: int
    rebalance_step_days: int
    transaction_cost_bps: float


def run_backtest(
    returns: pd.DataFrame,
    rf_col: str,
    constraints: Constraints,
    spec: BacktestSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple walk-forward backtest:
      - estimate weights on trailing lookback window
      - rebalance every `rebalance_step_days`
      - apply transaction costs: tc_bps * turnover (turnover = 0.5 * sum |dw|)
    Returns:
      daily_df: portfolio path and exposures
      rebalance_df: weights and turnover per rebalance date
    """
    x_all = returns.sort_index()
    if len(x_all) <= spec.lookback_days + 2:
        raise ValueError("Not enough data for backtest")

    dates = x_all.index
    investable_cols = list(x_all.columns)
    if rf_col not in investable_cols:
        raise ValueError("rf_col missing")

    w_prev = pd.Series(0.0, index=investable_cols)
    w_prev[rf_col] = 1.0

    value = 1.0
    daily_rows = []
    rebalance_rows = []

    tc = float(spec.transaction_cost_bps) / 10_000.0

    last_reb_idx = spec.lookback_days
    for reb_idx in range(spec.lookback_days, len(dates) - 1, spec.rebalance_step_days):
        window = x_all.iloc[reb_idx - spec.lookback_days : reb_idx]
        w = max_sharpe_long_only(window, rf_col=rf_col, constraints=constraints)
        w = w.reindex(investable_cols).fillna(0.0)

        turnover = 0.5 * float((w - w_prev).abs().sum())
        cost = tc * turnover
        value *= (1.0 - cost)

        reb_date = dates[reb_idx]
        rebalance_rows.append(
            {
                "date": reb_date,
                "turnover": turnover,
                "tc_cost": cost,
                **{f"w_{k}": float(v) for k, v in w.items()},
            }
        )

        # Apply weights until next rebalance (exclusive of next rebalance date)
        end_idx = min(reb_idx + spec.rebalance_step_days, len(dates) - 1)
        for t in range(reb_idx, end_idx):
            r_t = x_all.iloc[t].fillna(0.0)
            rp = float((r_t * w).sum())
            value *= (1.0 + rp)
            daily_rows.append({"date": dates[t], "rp": rp, "value": value, "turnover": turnover if t == reb_idx else 0.0})

        w_prev = w
        last_reb_idx = reb_idx

    daily_df = pd.DataFrame(daily_rows).set_index("date")
    rebalance_df = pd.DataFrame(rebalance_rows).set_index("date")
    return daily_df, rebalance_df


def backtest_summary(daily_df: pd.DataFrame) -> dict[str, float]:
    if daily_df.empty:
        return {}
    v = daily_df["value"].astype(float)
    rp = daily_df["rp"].astype(float)
    dd = v / v.cummax() - 1.0
    return {
        "final_value": float(v.iloc[-1]),
        "mdd": float(dd.min()),
        "mean_daily": float(rp.mean()),
        "vol_daily": float(rp.std(ddof=1)),
        "turnover_total": float(daily_df["turnover"].sum()) if "turnover" in daily_df.columns else 0.0,
    }

