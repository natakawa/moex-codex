from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class InstrumentSpec:
    secid: str
    type: str  # share | etf | bond | index


@dataclass(frozen=True)
class UniverseConfig:
    history_years: int
    benchmark: InstrumentSpec
    risk_free: InstrumentSpec
    assets: list[InstrumentSpec]

    @property
    def all_instruments(self) -> list[InstrumentSpec]:
        return [self.benchmark, self.risk_free, *self.assets]


def load_universe(path: str | Path) -> UniverseConfig:
    raw = load_yaml(path)
    benchmark = InstrumentSpec(**raw["benchmark"])
    risk_free = InstrumentSpec(**raw["risk_free"])
    assets = [InstrumentSpec(**a) for a in raw["assets"]]
    return UniverseConfig(
        history_years=int(raw["history_years"]),
        benchmark=benchmark,
        risk_free=risk_free,
        assets=assets,
    )


@dataclass(frozen=True)
class StrategyConfig:
    annualization_days: int
    covariance_method: str
    min_obs_days: int
    min_coverage: float
    min_weight_rf: float
    max_weight_any: float
    turnover_enabled: bool
    turnover_reference: str
    turnover_lambda: float
    robust_enabled: bool
    robust_eps_sharpe_relative: float
    robust_turnover_lambda: float
    robust_l2_lambda: float
    robust_hhi_lambda: float
    rolling_enabled: bool
    rolling_window_days: int
    rolling_step_days: int
    frontier_points: int
    multifactor_enabled: bool
    multifactor_factors: list[dict[str, object]]
    backtest_enabled: bool
    backtest_lookback_days: int
    backtest_rebalance_step_days: int
    backtest_tc_bps: float


def load_strategy(path: str | Path) -> StrategyConfig:
    raw = load_yaml(path)
    opt = raw.get("optimization", {})
    turnover = opt.get("turnover", {})
    robust = opt.get("robust_near_max_sharpe", {})
    return StrategyConfig(
        annualization_days=int(raw["estimation"]["annualization_days"]),
        covariance_method=str(raw["estimation"]["covariance"]["method"]),
        min_obs_days=int(raw["data_quality"]["min_obs_days"]),
        min_coverage=float(raw["data_quality"]["min_coverage"]),
        min_weight_rf=float(opt["min_weight_rf"]),
        max_weight_any=float(opt["max_weight_any"]),
        turnover_enabled=bool(turnover.get("enabled", False)),
        turnover_reference=str(turnover.get("reference_weights", "equal")),
        turnover_lambda=float(turnover.get("penalty_lambda", 0.0)),
        robust_enabled=bool(robust.get("enabled", False)),
        robust_eps_sharpe_relative=float(robust.get("eps_sharpe_relative", 0.05)),
        robust_turnover_lambda=float(robust.get("turnover_lambda", 0.0)),
        robust_l2_lambda=float(robust.get("l2_lambda", 0.0)),
        robust_hhi_lambda=float(robust.get("hhi_lambda", 0.0)),
        rolling_enabled=bool(raw["robustness"]["rolling_windows"]["enabled"]),
        rolling_window_days=int(raw["robustness"]["rolling_windows"]["window_trading_days"]),
        rolling_step_days=int(raw["robustness"]["rolling_windows"]["step_trading_days"]),
        frontier_points=int(raw.get("frontier", {}).get("points", 30)),
        multifactor_enabled=bool(raw.get("factors", {}).get("multifactor", {}).get("enabled", False)),
        multifactor_factors=list(raw.get("factors", {}).get("multifactor", {}).get("factors", [])),
        backtest_enabled=bool(raw.get("backtest", {}).get("enabled", False)),
        backtest_lookback_days=int(raw.get("backtest", {}).get("lookback_trading_days", 504)),
        backtest_rebalance_step_days=int(raw.get("backtest", {}).get("rebalance_step_trading_days", 21)),
        backtest_tc_bps=float(raw.get("backtest", {}).get("transaction_cost_bps", 0.0)),
    )
