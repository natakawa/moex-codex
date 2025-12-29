from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Panel:
    prices: pd.DataFrame  # index=date, columns=secid
    returns: pd.DataFrame  # index=date, columns=secid


def load_raw_candles_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df = df.dropna(subset=["end", "close"])
    df["date"] = df["end"].dt.date
    df = df.sort_values("end")
    return df


def build_price_series(raw_candles_dir: Path, secids: list[str]) -> pd.DataFrame:
    series = {}
    for secid in secids:
        p = raw_candles_dir / f"{secid}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing raw candles CSV: {p}")
        df = load_raw_candles_csv(p)
        s = df.groupby("date")["close"].last()
        series[secid] = s
    prices = pd.DataFrame(series).sort_index()
    return prices


def price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets


def build_panel(raw_dir: Path, processed_dir: Path, secids: list[str]) -> Panel:
    processed_dir.mkdir(parents=True, exist_ok=True)
    prices = build_price_series(raw_dir / "candles", secids=secids)
    prices.to_csv(processed_dir / "prices_raw.csv")

    # For a daily, common-calendar panel we forward-fill prices.
    # With price returns this implies 0 return on missing-quote days (stale close).
    prices_ffill = prices.sort_index().ffill().dropna(how="all")
    returns = price_returns(prices_ffill)
    prices_ffill.to_csv(processed_dir / "prices.csv")
    returns.to_csv(processed_dir / "returns.csv")
    return Panel(prices=prices, returns=returns)
