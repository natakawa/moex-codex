from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from moex_portfolio.metrics import historical_var_es, max_drawdown_from_returns
from moex_portfolio.optimizer import Constraints, max_sharpe_long_only


st.set_page_config(page_title="MOEX Portfolio", layout="wide")
st.title("MOEX Portfolio")


def repo_root() -> Path:
    # src/moex_portfolio/web.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def config_universe(self) -> Path:
        return self.root / "config" / "universe.yml"

    @property
    def config_strategy(self) -> Path:
        return self.root / "config" / "strategy.yml"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def analytics_dir(self) -> Path:
        return self.data_dir / "analytics"


PATHS = AppPaths(repo_root())


@st.cache_data(show_spinner=False)
def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _maybe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_latest_asof(raw_dir: Path) -> str | None:
    metas = list((raw_dir / "candles").glob("*.meta.json"))
    asofs = []
    for m in metas:
        j = _maybe_read_json(m)
        if not j:
            continue
        if "asof" in j:
            asofs.append(str(j["asof"]))
    return max(asofs) if asofs else None


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    cols = [c for c in weights.index if c in returns.columns and weights[c] != 0]
    if not cols:
        return pd.Series(dtype=float)
    w = weights.reindex(cols).fillna(0.0)
    r = returns[cols].fillna(0.0)
    rp = (r * w.values).sum(axis=1)
    rp.name = "portfolio"
    return rp


def cagr_from_returns(r: pd.Series, periods_per_year: int) -> float:
    x = r.dropna()
    if x.empty:
        return float("nan")
    v = (1.0 + x).cumprod()
    years = len(x) / periods_per_year
    if years <= 0:
        return float("nan")
    return float(v.iloc[-1] ** (1.0 / years) - 1.0)


def annualized_vol(r: pd.Series, periods_per_year: int) -> float:
    x = r.dropna()
    if x.empty:
        return float("nan")
    return float(x.std(ddof=1) * np.sqrt(periods_per_year))


def annualized_mean(r: pd.Series, periods_per_year: int) -> float:
    x = r.dropna()
    if x.empty:
        return float("nan")
    return float(x.mean() * periods_per_year)


def sharpe_annual(rp: pd.Series, rf: pd.Series, periods_per_year: int) -> float:
    df = pd.concat([rp.rename("rp"), rf.rename("rf")], axis=1).dropna()
    if df.empty:
        return float("nan")
    excess = df["rp"] - df["rf"]
    mu = excess.mean() * periods_per_year
    vol = excess.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / vol) if vol != 0 else float("nan")


def download_button(label: str, path: Path) -> None:
    if not path.exists():
        st.caption(f"Missing: {path}")
        return
    st.download_button(
        label=label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
        width="stretch",
    )


with st.sidebar:
    st.header("Context")
    st.caption(f"Repo: `{PATHS.root}`")
    latest_asof = find_latest_asof(PATHS.raw_dir)
    st.caption(f"Data as-of: `{latest_asof or 'unknown'}`")

    st.header("Constraints (interactive)")
    strategy = load_yaml(PATHS.config_strategy) if PATHS.config_strategy.exists() else {}
    opt0 = strategy.get("optimization", {})
    dq0 = strategy.get("data_quality", {})

    min_weight_rf = st.slider("Min weight RF", 0.0, 0.5, float(opt0.get("min_weight_rf", 0.10)), 0.01)
    max_weight_any = st.slider("Max weight (any)", 0.05, 0.7, float(opt0.get("max_weight_any", 0.20)), 0.01)
    min_obs_days = st.slider("Min obs days", 60, 1260, int(dq0.get("min_obs_days", 504)), 21)
    recompute = st.toggle("Recompute weights now", value=False)

    st.header("Downloads")
    download_button("Download `weights_max_sharpe.csv`", PATHS.analytics_dir / "weights_max_sharpe.csv")
    download_button("Download `weights_robust.csv`", PATHS.analytics_dir / "weights_robust.csv")
    download_button("Download `asset_metrics.csv`", PATHS.analytics_dir / "asset_metrics.csv")
    download_button("Download `capm.csv`", PATHS.analytics_dir / "capm.csv")
    download_button("Download `frontier.csv`", PATHS.analytics_dir / "frontier.csv")
    download_button("Download `portfolio_points.csv`", PATHS.analytics_dir / "portfolio_points.csv")
    download_button("Download `risk_contributions_max_sharpe.csv`", PATHS.analytics_dir / "risk_contributions_max_sharpe.csv")
    download_button("Download `risk_contributions_robust.csv`", PATHS.analytics_dir / "risk_contributions_robust.csv")
    download_button("Download `multifactor.csv`", PATHS.analytics_dir / "multifactor.csv")
    download_button("Download `backtest_summary.csv`", PATHS.analytics_dir / "backtest_summary.csv")
    download_button("Download `returns.csv`", PATHS.processed_dir / "returns.csv")


required = [
    PATHS.processed_dir / "returns.csv",
    PATHS.processed_dir / "prices.csv",
    PATHS.analytics_dir / "asset_metrics.csv",
    PATHS.analytics_dir / "corr.csv",
    PATHS.analytics_dir / "weights_max_sharpe.csv",
]
missing = [p for p in required if not p.exists()]
if missing:
    st.error("Missing required files. Generate data first: `moexpf fetch && moexpf build && moexpf analyze && moexpf optimize`")
    for p in missing:
        st.code(str(p))
    st.stop()

universe = load_yaml(PATHS.config_universe) if PATHS.config_universe.exists() else {}
rf_secid = universe.get("risk_free", {}).get("secid", "SU26219RMFS4")
benchmark_secid = universe.get("benchmark", {}).get("secid", "IMOEX")
investable_secids = [rf_secid] + [a["secid"] for a in universe.get("assets", [])]

prices = load_csv(PATHS.processed_dir / "prices.csv", index_col=0, parse_dates=True)
returns_all = load_csv(PATHS.processed_dir / "returns.csv", index_col=0, parse_dates=True)
metrics = load_csv(PATHS.analytics_dir / "asset_metrics.csv", index_col=0)
corr = load_csv(PATHS.analytics_dir / "corr.csv", index_col=0)
capm_path = PATHS.analytics_dir / "capm.csv"
capm = load_csv(capm_path, index_col=0) if capm_path.exists() else None
frontier_path = PATHS.analytics_dir / "frontier.csv"
frontier = load_csv(frontier_path) if frontier_path.exists() else None
pt_path = PATHS.analytics_dir / "portfolio_point.csv"
portfolio_point = load_csv(pt_path) if pt_path.exists() else None
rc_max_path = PATHS.analytics_dir / "risk_contributions_max_sharpe.csv"
risk_contrib_max = load_csv(rc_max_path, index_col=0) if rc_max_path.exists() else None
rc_rob_path = PATHS.analytics_dir / "risk_contributions_robust.csv"
risk_contrib_rob = load_csv(rc_rob_path, index_col=0) if rc_rob_path.exists() else None
conc_path = PATHS.analytics_dir / "concentration.csv"
concentration = load_csv(conc_path, index_col=0) if conc_path.exists() else None
mf_path = PATHS.analytics_dir / "multifactor.csv"
multifactor = load_csv(mf_path, index_col=0) if mf_path.exists() else None
bt_sum_path = PATHS.analytics_dir / "backtest_summary.csv"
backtest_summary = load_csv(bt_sum_path, index_col=0) if bt_sum_path.exists() else None

weights_max = load_csv(PATHS.analytics_dir / "weights_max_sharpe.csv", index_col=0)["weight"].reindex(investable_secids).fillna(0.0)
weights_rob_path = PATHS.analytics_dir / "weights_robust.csv"
weights_rob = (
    load_csv(weights_rob_path, index_col=0)["weight"].reindex(investable_secids).fillna(0.0)
    if weights_rob_path.exists()
    else None
)

mode = st.sidebar.selectbox("Weights view", options=["robust" if weights_rob is not None else "max_sharpe", "max_sharpe"])

if recompute:
    returns = returns_all.reindex(columns=investable_secids).copy()
    c = Constraints(min_weight_rf=min_weight_rf, max_weight_any=max_weight_any, min_obs_days=min_obs_days)
    w = max_sharpe_long_only(returns, rf_col=rf_secid, constraints=c)
    weights = w.reindex(investable_secids).fillna(0.0)
    mode = "max_sharpe (recomputed)"
else:
    if mode == "robust" and weights_rob is not None:
        weights = weights_rob
    else:
        weights = weights_max

returns_investable = returns_all.reindex(columns=investable_secids)
rp = portfolio_returns(returns_investable, weights)
rf = returns_investable[rf_secid].reindex(rp.index).fillna(0.0)
mkt = returns_all[benchmark_secid].reindex(rp.index).fillna(0.0) if benchmark_secid in returns_all.columns else None

periods_per_year = int(strategy.get("estimation", {}).get("annualization_days", 252))

tabs = st.tabs(["Overview", "Allocation", "Risk", "Factors", "Data"])

with tabs[0]:
    st.subheader("Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{cagr_from_returns(rp, periods_per_year):.2%}")
    c2.metric("Ann. Vol", f"{annualized_vol(rp, periods_per_year):.2%}")
    c3.metric("Sharpe (excess vs RF)", f"{sharpe_annual(rp, rf, periods_per_year):.2f}")
    c4.metric("Max Drawdown", f"{max_drawdown_from_returns(rp):.2%}")
    var95, es95 = historical_var_es(rp, alpha=0.95)
    c5.metric("VaR/ES 95% (1d)", f"{var95:.2%} / {es95:.2%}")

    st.caption(f"RF: `{rf_secid}` | Benchmark: `{benchmark_secid}` | Window: ~{len(rp)} obs | Base: RUB | Returns: price return")

    st.subheader("Performance (normalized)")
    v_p = (1.0 + rp.fillna(0.0)).cumprod() * 100
    series = {"Portfolio": v_p}
    if mkt is not None:
        series["IMOEX"] = (1.0 + mkt.fillna(0.0)).cumprod() * 100
    series["RF"] = (1.0 + rf.fillna(0.0)).cumprod() * 100
    st.line_chart(pd.DataFrame(series))

with tabs[1]:
    st.subheader("Allocation")
    w_tbl = weights[weights.abs() > 1e-6].sort_values(ascending=False).to_frame("weight")
    st.dataframe(w_tbl.style.format({"weight": "{:.2%}"}), width="stretch")
    st.bar_chart(w_tbl)

    st.subheader("Concentration")
    if concentration is not None and not concentration.empty:
        st.dataframe(concentration, width="stretch")
    else:
        st.info("Concentration report not found. Run `moexpf analyze`.")

    st.subheader("Risk contributions (variance-based)")
    rc = risk_contrib_rob if (mode.startswith("robust") or mode == "robust") else risk_contrib_max
    if rc is not None and not rc.empty:
        st.dataframe(rc, width="stretch")
        st.bar_chart(rc[["rc_share"]].rename(columns={"rc_share": "RC share"}))
    else:
        st.info("Risk contributions not found. Run `moexpf analyze`.")

with tabs[2]:
    st.subheader("Risk")
    st.caption("Drawdown")
    v = (1.0 + rp.fillna(0.0)).cumprod()
    dd = v / v.cummax() - 1.0
    st.line_chart(dd.rename("drawdown"))

    st.caption("Efficient frontier (constrained)")
    if frontier is not None and not frontier.empty:
        frontier_plot = frontier[["vol_daily", "ret_daily"]].copy()
        frontier_plot = frontier_plot.rename(columns={"vol_daily": "vol", "ret_daily": "ret"})
        st.scatter_chart(frontier_plot.rename(columns={"vol": "x", "ret": "y"}))
        if portfolio_point is not None and not portfolio_point.empty:
            st.caption("Chosen portfolio point (from `data/analytics/portfolio_point.csv`)")
            st.dataframe(portfolio_point, width="stretch")
    else:
        st.info("Frontier not found. Run `moexpf analyze`.")

    st.caption("Correlation (asset returns)")
    st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}"), width="stretch")

with tabs[3]:
    st.subheader("Factors")
    if capm is not None and not capm.empty:
        st.caption("CAPM (alpha/beta/R²) from `data/analytics/capm.csv`")
        st.dataframe(capm, width="stretch")
        if {"alpha", "beta"}.issubset(capm.columns):
            st.scatter_chart(capm[["beta", "alpha"]].rename(columns={"beta": "x", "alpha": "y"}))
    else:
        st.info("CAPM report not found. Run `moexpf analyze`.")

    st.subheader("Multi-factor")
    if multifactor is not None and not multifactor.empty:
        st.dataframe(multifactor, width="stretch")
        if "r2" in multifactor.columns:
            st.bar_chart(multifactor[["r2"]])
    else:
        st.info("Multi-factor report not found (or disabled). See `config/strategy.yml` and run `moexpf analyze`.")

    st.subheader("Governance (backtest summary)")
    if backtest_summary is not None and not backtest_summary.empty:
        st.dataframe(backtest_summary, width="stretch")
    else:
        st.info("Backtest summary not found. Run `moexpf backtest`.")

with tabs[4]:
    st.subheader("Data / Audit")
    st.caption("Raw candle dumps (one CSV per instrument)")
    raw_files = sorted((PATHS.raw_dir / "candles").glob("*.csv"))
    st.write(f"Found {len(raw_files)} raw files in `{PATHS.raw_dir / 'candles'}`")
    st.dataframe(pd.DataFrame({"file": [f.name for f in raw_files]}), width="stretch", height=240)

    st.caption("Processed outputs")
    st.dataframe(
        pd.DataFrame(
            {
                "path": [
                    str(PATHS.processed_dir / "prices_raw.csv"),
                    str(PATHS.processed_dir / "prices.csv"),
                    str(PATHS.processed_dir / "returns.csv"),
                    str(PATHS.analytics_dir / "asset_metrics.csv"),
                    str(PATHS.analytics_dir / "capm.csv"),
                    str(PATHS.analytics_dir / "corr.csv"),
                    str(PATHS.analytics_dir / "weights_max_sharpe.csv"),
                    str(PATHS.analytics_dir / "rolling_weights.csv"),
                    str(PATHS.analytics_dir / "frontier.csv"),
                    str(PATHS.analytics_dir / "frontier_weights.csv"),
                    str(PATHS.analytics_dir / "portfolio_point.csv"),
                    str(PATHS.analytics_dir / "concentration.csv"),
                    str(PATHS.analytics_dir / "risk_contributions_max_sharpe.csv"),
                    str(PATHS.analytics_dir / "risk_contributions_robust.csv"),
                    str(PATHS.analytics_dir / "portfolio_points.csv"),
                    str(PATHS.analytics_dir / "weights_robust.csv"),
                    str(PATHS.analytics_dir / "multifactor.csv"),
                    str(PATHS.analytics_dir / "backtest_daily.csv"),
                    str(PATHS.analytics_dir / "backtest_rebalances.csv"),
                    str(PATHS.analytics_dir / "backtest_summary.csv"),
                ]
            }
        ),
        width="stretch",
        height=260,
    )
