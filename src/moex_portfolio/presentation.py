from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReportInputs:
    root: Path

    @property
    def analytics_dir(self) -> Path:
        return self.root / "data" / "analytics"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """
    Minimal markdown table renderer (avoids optional 'tabulate' dependency).
    """
    if df is None or df.empty:
        return "_Not available_\n"
    if max_rows is not None:
        df = df.head(max_rows)
    df2 = df.copy()
    df2 = df2.fillna("").astype(str)
    header = [""] + list(df2.columns)
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for idx, row in df2.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [row[c] for c in df2.columns]) + " |")
    return "\n".join(lines) + "\n"


def _md_series(s: pd.Series, name: str = "value", max_rows: int | None = None) -> str:
    df = s.to_frame(name=name)
    return _md_table(df, max_rows=max_rows)


def write_defense_markdown(
    inputs: ReportInputs,
    *,
    rf_secid: str,
    benchmark_secid: str,
    annualization_days: int,
) -> Path:
    """
    Generates a “2 slides + appendix” markdown file referencing CSVs for traceability.
    """
    inputs.reports_dir.mkdir(parents=True, exist_ok=True)
    out = inputs.reports_dir / "defense.md"

    weights = _read_csv(inputs.analytics_dir / "weights_max_sharpe.csv", index_col=0)
    weights_rob_path = inputs.analytics_dir / "weights_robust.csv"
    weights_rob = _read_csv(weights_rob_path, index_col=0) if weights_rob_path.exists() else pd.DataFrame()
    asset_metrics = _read_csv(inputs.analytics_dir / "asset_metrics.csv", index_col=0)
    capm_path = inputs.analytics_dir / "capm.csv"
    capm = _read_csv(capm_path, index_col=0) if capm_path.exists() else pd.DataFrame()

    frontier_path = inputs.analytics_dir / "frontier.csv"
    frontier = _read_csv(frontier_path) if frontier_path.exists() else pd.DataFrame()

    rc_max_path = inputs.analytics_dir / "risk_contributions_max_sharpe.csv"
    rc_max = _read_csv(rc_max_path, index_col=0) if rc_max_path.exists() else pd.DataFrame()
    rc_rob_path = inputs.analytics_dir / "risk_contributions_robust.csv"
    rc_rob = _read_csv(rc_rob_path, index_col=0) if rc_rob_path.exists() else pd.DataFrame()

    conc_path = inputs.analytics_dir / "concentration.csv"
    conc = _read_csv(conc_path, index_col=0) if conc_path.exists() else pd.DataFrame()

    mf_path = inputs.analytics_dir / "multifactor.csv"
    mf = _read_csv(mf_path, index_col=0) if mf_path.exists() else pd.DataFrame()

    bt_sum_path = inputs.analytics_dir / "backtest_summary.csv"
    bt_sum = _read_csv(bt_sum_path, index_col=0) if bt_sum_path.exists() else pd.DataFrame()

    lines: list[str] = []
    lines.append("# Portfolio Defense Pack (auto-generated)\n")
    lines.append("All numbers are traceable to CSV artifacts under `data/`.\n")
    lines.append(f"- Base currency: RUB (price returns)\n")
    lines.append(f"- Annualization: {annualization_days} trading days\n")
    lines.append(f"- RF proxy: `{rf_secid}`\n")
    lines.append(f"- Benchmark: `{benchmark_secid}`\n")
    lines.append("\n---\n")
    lines.append("## Slide 1 — Weights, mandate compliance, key metrics\n")
    lines.append("**Weights (max Sharpe, constrained)** — source: `data/analytics/weights_max_sharpe.csv`\n")
    w = weights["weight"].sort_values(ascending=False)
    lines.append(_md_series(w[w.abs() > 1e-6], name="weight", max_rows=50))
    if not weights_rob.empty and "weight" in weights_rob.columns:
        lines.append("\n**Weights (robust near-max Sharpe)** — source: `data/analytics/weights_robust.csv`\n")
        wr = weights_rob["weight"].sort_values(ascending=False)
        lines.append(_md_series(wr[wr.abs() > 1e-6], name="weight", max_rows=50))
    lines.append("\n\n**Concentration** — source: `data/analytics/concentration.csv`\n")
    if not conc.empty:
        lines.append(_md_table(conc, max_rows=20))
    else:
        lines.append("_Not available_\n")
    lines.append("\n\n**Risk contributions (variance-based)** — sources:\n- `data/analytics/risk_contributions_max_sharpe.csv`\n- `data/analytics/risk_contributions_robust.csv`\n")
    if not rc_max.empty:
        lines.append("\n_Max Sharpe_\n")
        lines.append(_md_table(rc_max, max_rows=15))
    if not rc_rob.empty:
        lines.append("\n_Robust_\n")
        lines.append(_md_table(rc_rob, max_rows=15))
    if rc_max.empty and rc_rob.empty:
        lines.append("_Not available_\n")
    lines.append("\n\n---\n")
    lines.append("## Slide 2 — Efficient frontier position + factors + governance\n")
    lines.append("**Efficient frontier (constrained)** — source: `data/analytics/frontier.csv`\n")
    if not frontier.empty:
        lines.append(_md_table(frontier.set_index(frontier.index + 1), max_rows=15))
    else:
        lines.append("_Not available_\n")
    pts_path = inputs.analytics_dir / "portfolio_points.csv"
    pts = _read_csv(pts_path) if pts_path.exists() else pd.DataFrame()
    lines.append("\n\n**Chosen portfolio point(s)** — source: `data/analytics/portfolio_points.csv`\n")
    if not pts.empty:
        lines.append(_md_table(pts.set_index(pts.index + 1), max_rows=10))
    else:
        lines.append("_Not available_\n")
    lines.append("\n\n**CAPM (alpha/beta/R²)** — source: `data/analytics/capm.csv`\n")
    if not capm.empty:
        lines.append(_md_table(capm, max_rows=50))
    else:
        lines.append("_Not available_\n")
    lines.append("\n\n**Multi-factor** — source: `data/analytics/multifactor.csv`\n")
    if not mf.empty:
        lines.append(_md_table(mf, max_rows=50))
    else:
        lines.append("_Not available_\n")
    lines.append("\n\n**Backtest summary** — source: `data/analytics/backtest_summary.csv`\n")
    if not bt_sum.empty:
        lines.append(_md_table(bt_sum, max_rows=10))
    else:
        lines.append("_Not available_\n")
    lines.append("\n\n---\n")
    lines.append("## Appendix — Methods and auditability\n")
    lines.append("### Data\n")
    lines.append("- Raw MOEX dumps: `data/raw/candles/*.csv` (+ `*.meta.json` request metadata)\n")
    lines.append("- Processed panel: `data/processed/prices_raw.csv`, `data/processed/prices.csv`, `data/processed/returns.csv`\n")
    lines.append("- Backtests: `data/analytics/backtest_daily_max_sharpe.csv`, `data/analytics/backtest_daily_robust.csv` (+ `*_rebalances_*.csv`)\n")
    lines.append("\n### µ and Σ estimation\n")
    lines.append("- µ: sample mean of daily returns over the chosen window.\n")
    lines.append("- Σ: sample covariance matrix of daily returns.\n")
    lines.append("\n### Risk metrics\n")
    lines.append("- Max Drawdown computed from value index `V_t = Π(1+r_t)`.\n")
    lines.append("- VaR/ES: historical method at 95% on daily returns.\n")
    lines.append("\n### Factors\n")
    lines.append("- CAPM: `(r_asset - r_f) = α + β (r_m - r_f) + ε`.\n")
    lines.append("- Multi-factor: `(r_asset - r_f) = α + Σ β_j * factor_j + ε` (factor list is configurable).\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
