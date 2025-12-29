from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .optimizer import Constraints, max_sharpe_long_only


@dataclass(frozen=True)
class RollingSpec:
    window_days: int
    step_days: int


def rolling_weights(
    returns: pd.DataFrame,
    rf_col: str,
    constraints: Constraints,
    spec: RollingSpec,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    # Uses time index ordering; assumes returns are daily.
    x_all = returns.sort_index()
    if len(x_all) < spec.window_days:
        raise ValueError("Not enough data for rolling windows (raw)")

    weights = []
    dates = []
    for end_idx in range(spec.window_days, len(x_all) + 1, spec.step_days):
        window0 = x_all.iloc[end_idx - spec.window_days : end_idx]

        # Keep columns with sufficient coverage within the window; keep rf_col always.
        coverage = window0.notna().mean(axis=0)
        keep_cols = coverage[coverage >= min_coverage].index.tolist()
        if rf_col not in keep_cols:
            continue
        # Feasibility under max weight cap: need enough assets to sum to 1.
        min_assets_needed = int((1.0 / constraints.max_weight_any) + 0.999999)
        if len(keep_cols) < min_assets_needed:
            continue
        window = window0[keep_cols].dropna(how="any")
        if window.empty:
            continue
        try:
            w = max_sharpe_long_only(window, rf_col=rf_col, constraints=constraints)
        except Exception:
            continue

        # Expand weights back to full column set for stable output.
        w_full = pd.Series(0.0, index=returns.columns, name="weight")
        w_full.loc[w.index] = w.values
        weights.append(w_full)
        dates.append(window.index[-1])
    out = pd.DataFrame(weights, index=pd.to_datetime(dates))
    out.index.name = "asof"
    return out
