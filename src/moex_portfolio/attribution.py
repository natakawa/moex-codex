from __future__ import annotations

import numpy as np
import pandas as pd


def concentration_metrics(weights: pd.Series) -> dict[str, float]:
    w = weights.fillna(0.0).to_numpy(dtype=float)
    w = np.clip(w, 0.0, 1.0)
    total = float(w.sum())
    if total == 0:
        return {
            "sum_w": 0.0,
            "max_w": float("nan"),
            "top2_w": float("nan"),
            "hhi": float("nan"),
            "effective_n": float("nan"),
        }
    w = w / total
    w_sorted = np.sort(w)[::-1]
    hhi = float(np.sum(w**2))
    return {
        "sum_w": float(np.sum(w)),
        "max_w": float(w_sorted[0]) if len(w_sorted) else float("nan"),
        "top2_w": float(w_sorted[:2].sum()) if len(w_sorted) >= 2 else float("nan"),
        "hhi": hhi,
        "effective_n": float(1.0 / hhi) if hhi > 0 else float("nan"),
    }


def risk_contributions(cov: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Variance-based risk attribution:
      sigma_p = sqrt(w^T Σ w)
      MC_i = (Σ w)_i / sigma_p
      RC_i = w_i * MC_i
    Returns a table with MC/RC and RC share.
    """
    cols = [c for c in cov.columns if c in weights.index]
    if not cols:
        return pd.DataFrame()
    s = weights.reindex(cols).fillna(0.0)
    w = s.to_numpy(dtype=float)
    sigma2 = float(w @ cov.loc[cols, cols].to_numpy() @ w)
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    if sigma == 0:
        out = pd.DataFrame({"weight": s, "mc_vol": 0.0, "rc_vol": 0.0, "rc_share": 0.0})
        out.index.name = "secid"
        return out
    sw = cov.loc[cols, cols].to_numpy() @ w
    mc = sw / sigma
    rc = w * mc
    out = pd.DataFrame(
        {
            "weight": s.values,
            "mc_vol": mc,
            "rc_vol": rc,
        },
        index=cols,
    )
    out["rc_share"] = out["rc_vol"] / sigma
    out.index.name = "secid"
    return out.sort_values("rc_share", ascending=False)

