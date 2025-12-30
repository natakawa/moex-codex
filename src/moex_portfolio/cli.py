from __future__ import annotations

import argparse
import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import load_strategy, load_universe
from .logging_utils import setup_logging
from .presentation import ReportInputs, write_defense_markdown
from .moex import fetch_and_store_universe
from .optimizer import Constraints, max_sharpe_long_only, robust_near_max_sharpe_long_only
from .estimation import EstimationConfig
from .paths import Paths
from .processing import build_panel
from .report import asset_report, capm_report, write_df
from .robustness import RollingSpec, rolling_weights
from .frontier import FrontierConfig, FrontierConstraints, efficient_frontier
from .attribution import concentration_metrics, risk_contributions
from .multifactor import FactorSpec, run_multifactor
from .backtest import BacktestSpec, backtest_summary, run_backtest
from .estimation import filter_and_align_returns, estimate_mu_cov


logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path.cwd()


def cmd_fetch(args: argparse.Namespace) -> None:
    paths = Paths(_repo_root())
    paths.ensure()
    universe = load_universe(args.universe)

    instruments = universe.all_instruments
    fetch_and_store_universe(
        instruments=instruments,
        raw_dir=paths.raw_dir,
        years=universe.history_years,
        asof=dt.datetime.now(dt.timezone.utc),
        force=bool(args.force),
    )


def cmd_build(args: argparse.Namespace) -> None:
    paths = Paths(_repo_root())
    paths.ensure()
    universe = load_universe(args.universe)
    secids = [i.secid for i in universe.all_instruments]
    build_panel(paths.raw_dir, paths.processed_dir, secids=secids)
    logger.info("Wrote %s and %s", paths.processed_dir / "prices.csv", paths.processed_dir / "returns.csv")


def cmd_analyze(args: argparse.Namespace) -> None:
    paths = Paths(_repo_root())
    paths.ensure()
    universe = load_universe(args.universe)
    strategy = load_strategy(args.strategy)

    returns = pd.read_csv(paths.processed_dir / "returns.csv", index_col=0, parse_dates=True)
    rf = returns[universe.risk_free.secid]
    rep = asset_report(returns, rf_series=rf, periods_per_year=strategy.annualization_days)
    write_df(rep, paths.analytics_dir / "asset_metrics.csv")

    mkt = returns[universe.benchmark.secid]
    capm = capm_report(returns, rf=rf, market=mkt)
    write_df(capm, paths.analytics_dir / "capm.csv")

    corr = returns.corr()
    write_df(corr, paths.analytics_dir / "corr.csv")

    # Frontier (investable only, excludes benchmark)
    investable = [universe.risk_free.secid] + [a.secid for a in universe.assets]
    inv_returns = returns[investable]
    f_cfg = FrontierConfig(points=strategy.frontier_points)
    f_cons = FrontierConstraints(
        min_weight_rf=strategy.min_weight_rf,
        max_weight_any=strategy.max_weight_any,
        min_obs_days=strategy.min_obs_days,
    )
    est_cfg = EstimationConfig(
        mean_method=strategy.mean_method,
        ewma_lambda=strategy.ewma_lambda,
        cov_method=strategy.covariance_method,
        shrinkage_alpha=strategy.shrinkage_alpha,
    )

    frontier_df, frontier_w = efficient_frontier(
        inv_returns,
        rf_col=universe.risk_free.secid,
        constraints=f_cons,
        cfg=f_cfg,
        estimation=est_cfg,
    )
    write_df(frontier_df, paths.analytics_dir / "frontier.csv", index=False)
    write_df(frontier_w, paths.analytics_dir / "frontier_weights.csv")

    # Attribution for saved portfolios (max_sharpe / robust), using the same µ/Σ estimator.
    x = filter_and_align_returns(inv_returns, min_obs_days=strategy.min_obs_days)
    mu, cov = estimate_mu_cov(x, est_cfg)

    portfolios = {
        "max_sharpe": paths.analytics_dir / "weights_max_sharpe.csv",
        "robust": paths.analytics_dir / "weights_robust.csv",
    }
    conc_rows = []
    pt_rows = []
    for name, p in portfolios.items():
        if not p.exists():
            continue
        w = pd.read_csv(p, index_col=0)["weight"]
        conc = concentration_metrics(w)
        conc_rows.append({"portfolio": name, **conc})

        rc = risk_contributions(cov, w)
        write_df(rc, paths.analytics_dir / f"risk_contributions_{name}.csv")

        w_aligned = w.reindex(mu.index).fillna(0.0)
        ret = float(mu @ w_aligned)
        vol = float((w_aligned.to_numpy() @ cov.to_numpy() @ w_aligned.to_numpy()) ** 0.5)
        rf_mu = float(mu.get(universe.risk_free.secid, 0.0))
        sharpe = float((ret - rf_mu) / vol) if vol > 0 else float("nan")
        pt_rows.append(
            {
                "portfolio": name,
                "ret_daily": ret,
                "vol_daily": vol,
                "rf_daily": rf_mu,
                "sharpe_daily_excess": sharpe,
            }
        )

    if conc_rows:
        write_df(pd.DataFrame(conc_rows).set_index("portfolio"), paths.analytics_dir / "concentration.csv")
    if pt_rows:
        write_df(pd.DataFrame(pt_rows), paths.analytics_dir / "portfolio_points.csv", index=False)

    # Multi-factor (optional; configurable)
    if strategy.multifactor_enabled and strategy.multifactor_factors:
        factors = []
        for f in strategy.multifactor_factors:
            try:
                factors.append(FactorSpec(name=str(f["name"]), secid=str(f["secid"]), excess_vs_rf=bool(f.get("excess_vs_rf", False))))
            except Exception:
                continue
        cols = list(dict.fromkeys(investable + [universe.benchmark.secid]))
        mf = run_multifactor(returns[cols].copy(), rf=rf, factors=factors, min_obs=strategy.min_obs_days)
        write_df(mf, paths.analytics_dir / "multifactor.csv")
    logger.info("Wrote analytics into %s", paths.analytics_dir)


def cmd_optimize(args: argparse.Namespace) -> None:
    paths = Paths(_repo_root())
    paths.ensure()
    universe = load_universe(args.universe)
    strategy = load_strategy(args.strategy)

    returns_all = pd.read_csv(paths.processed_dir / "returns.csv", index_col=0, parse_dates=True)
    investable = [universe.risk_free.secid] + [a.secid for a in universe.assets]
    returns = returns_all[investable]
    c = Constraints(
        min_weight_rf=strategy.min_weight_rf,
        max_weight_any=strategy.max_weight_any,
        min_obs_days=strategy.min_obs_days,
    )
    est_cfg = EstimationConfig(
        mean_method=strategy.mean_method,
        ewma_lambda=strategy.ewma_lambda,
        cov_method=strategy.covariance_method,
        shrinkage_alpha=strategy.shrinkage_alpha,
    )
    w = max_sharpe_long_only(returns, rf_col=universe.risk_free.secid, constraints=c, estimation=est_cfg)
    write_df(w.to_frame(), paths.analytics_dir / "weights_max_sharpe.csv")
    logger.info("Wrote weights into %s", paths.analytics_dir / "weights_max_sharpe.csv")

    if strategy.robust_enabled:
        # Reference weights for turnover/stability:
        # 1) previous robust (if exists) for continuity across runs,
        # 2) otherwise current max-sharpe (keeps us "near-max" by construction).
        ref = None
        if strategy.turnover_reference == "previous":
            p = paths.analytics_dir / "weights_robust.csv"
            if p.exists():
                ref = pd.read_csv(p, index_col=0)["weight"]
            else:
                p2 = paths.analytics_dir / "weights_max_sharpe.csv"
                if p2.exists():
                    ref = pd.read_csv(p2, index_col=0)["weight"]
        if ref is None:
            ref = w

        w_rob = robust_near_max_sharpe_long_only(
            returns,
            rf_col=universe.risk_free.secid,
            constraints=c,
            eps_sharpe_relative=strategy.robust_eps_sharpe_relative,
            turnover_lambda=strategy.robust_turnover_lambda,
            l2_lambda=strategy.robust_l2_lambda,
            hhi_lambda=strategy.robust_hhi_lambda,
            reference_weights=ref,
            estimation=est_cfg,
        )
        write_df(w_rob.to_frame(), paths.analytics_dir / "weights_robust.csv")
        logger.info("Wrote robust weights into %s", paths.analytics_dir / "weights_robust.csv")


def cmd_robustness(args: argparse.Namespace) -> None:
    paths = Paths(_repo_root())
    paths.ensure()
    universe = load_universe(args.universe)
    strategy = load_strategy(args.strategy)
    if not strategy.rolling_enabled:
        logger.info("Rolling robustness disabled in strategy config")
        return

    returns_all = pd.read_csv(paths.processed_dir / "returns.csv", index_col=0, parse_dates=True)
    investable = [universe.risk_free.secid] + [a.secid for a in universe.assets]
    returns = returns_all[investable]
    c = Constraints(
        min_weight_rf=strategy.min_weight_rf,
        max_weight_any=strategy.max_weight_any,
        min_obs_days=strategy.min_obs_days,
    )
    spec = RollingSpec(window_days=strategy.rolling_window_days, step_days=strategy.rolling_step_days)
    est_cfg = EstimationConfig(
        mean_method=strategy.mean_method,
        ewma_lambda=strategy.ewma_lambda,
        cov_method=strategy.covariance_method,
        shrinkage_alpha=strategy.shrinkage_alpha,
    )
    w = rolling_weights(
        returns,
        rf_col=universe.risk_free.secid,
        constraints=c,
        spec=spec,
        min_coverage=strategy.min_coverage,
        estimation=est_cfg,
    )
    write_df(w, paths.analytics_dir / "rolling_weights.csv")
    logger.info("Wrote rolling weights into %s", paths.analytics_dir / "rolling_weights.csv")


def cmd_backtest(args: argparse.Namespace) -> None:
    paths = Paths(_repo_root())
    paths.ensure()
    universe = load_universe(args.universe)
    strategy = load_strategy(args.strategy)
    if not strategy.backtest_enabled:
        logger.info("Backtest disabled in strategy config")
        return

    returns_all = pd.read_csv(paths.processed_dir / "returns.csv", index_col=0, parse_dates=True)
    investable = [universe.risk_free.secid] + [a.secid for a in universe.assets]
    returns = returns_all[investable]

    c = Constraints(
        min_weight_rf=strategy.min_weight_rf,
        max_weight_any=strategy.max_weight_any,
        min_obs_days=strategy.min_obs_days,
    )
    est_cfg = EstimationConfig(
        mean_method=strategy.mean_method,
        ewma_lambda=strategy.ewma_lambda,
        cov_method=strategy.covariance_method,
        shrinkage_alpha=strategy.shrinkage_alpha,
    )
    def run_one(name: str, use_robust: bool):
        spec = BacktestSpec(
            lookback_days=strategy.backtest_lookback_days,
            rebalance_step_days=strategy.backtest_rebalance_step_days,
            transaction_cost_bps=strategy.backtest_tc_bps,
            use_robust=use_robust,
            eps_sharpe_relative=strategy.robust_eps_sharpe_relative,
            turnover_lambda=strategy.robust_turnover_lambda,
            l2_lambda=strategy.robust_l2_lambda,
            hhi_lambda=strategy.robust_hhi_lambda,
        )
        daily, rebs = run_backtest(returns, rf_col=universe.risk_free.secid, constraints=c, spec=spec, estimation=est_cfg)
        write_df(daily, paths.analytics_dir / f"backtest_daily_{name}.csv")
        write_df(rebs, paths.analytics_dir / f"backtest_rebalances_{name}.csv")
        summ = backtest_summary(daily)
        summ["run"] = name
        return summ

    rows = []
    rows.append(run_one("max_sharpe", use_robust=False))
    if strategy.robust_enabled:
        rows.append(run_one("robust", use_robust=True))

    write_df(pd.DataFrame(rows).set_index("run"), paths.analytics_dir / "backtest_summary.csv")
    logger.info("Wrote backtest outputs into %s", paths.analytics_dir)


def cmd_report(args: argparse.Namespace) -> None:
    universe = load_universe(args.universe)
    strategy = load_strategy(args.strategy)
    inputs = ReportInputs(_repo_root())
    out = write_defense_markdown(
        inputs,
        rf_secid=universe.risk_free.secid,
        benchmark_secid=universe.benchmark.secid,
        annualization_days=strategy.annualization_days,
    )
    logger.info("Wrote report to %s", out)


def cmd_web(args: argparse.Namespace) -> None:
    web_script = Path(__file__).parent / "web.py"
    if not web_script.exists():
        logger.error("Web script not found at %s", web_script)
        sys.exit(1)

    logger.info("Starting web view...")
    try:
        subprocess.run(["streamlit", "run", str(web_script)], check=True)
    except FileNotFoundError:
        logger.error("Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("Streamlit exited with error: %s", e)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="moexpf")
    p.add_argument("--universe", default="config/universe.yml")
    p.add_argument("--strategy", default="config/strategy.yml")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)
    sp_fetch = sub.add_parser("fetch")
    sp_fetch.add_argument("--force", action="store_true", help="Re-download even if raw CSV already exists")
    sub.add_parser("build")
    sub.add_parser("analyze")
    sub.add_parser("optimize")
    sub.add_parser("robustness")
    sub.add_parser("backtest")
    sub.add_parser("report")
    sub.add_parser("web")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "optimize":
        cmd_optimize(args)
    elif args.cmd == "robustness":
        cmd_robustness(args)
    elif args.cmd == "web":
        cmd_web(args)
    elif args.cmd == "backtest":
        cmd_backtest(args)
    elif args.cmd == "report":
        cmd_report(args)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")
