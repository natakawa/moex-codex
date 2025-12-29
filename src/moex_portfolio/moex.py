from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

import moexapi
from moexapi import utils as moex_utils

from .config import InstrumentSpec


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandleFrame:
    secid: str
    df: pd.DataFrame  # columns: begin, end, open, high, low, close, volume, value


_TYPE_TO_MARKET = {
    "share": moexapi.Markets.SHARES,
    "etf": moexapi.Markets.ETFS,
    "bond": moexapi.Markets.BONDS,
    "index": moexapi.Markets.INDEX,
}


def _market_for(spec: InstrumentSpec) -> moexapi.Market:
    try:
        return _TYPE_TO_MARKET[spec.type]
    except KeyError as e:
        raise ValueError(f"Unknown instrument type: {spec.type}") from e


def fetch_daily_candles(
    spec: InstrumentSpec,
    start: dt.datetime,
    end: dt.datetime,
) -> CandleFrame:
    market = _market_for(spec)
    try:
        ticker = moexapi.Ticker.from_secid(spec.secid, market=market)
    except Exception as e:
        # moexapi's ticker listing may miss some valid instruments (e.g., certain OFZ boards).
        logger.warning("Ticker.from_secid failed for %s (%s): %s; using fallback discovery", spec.secid, spec.type, e)
        ticker = _ticker_from_secid_fallback(spec.secid, market=market)
    candles = _get_candles_safe(ticker, start_date=start, end_date=end, interval=24)
    rows = []
    for c in candles:
        rows.append(
            {
                "begin": c.start,
                "end": c.end,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "value": c.value,
            }
        )
    df = pd.DataFrame(rows).sort_values("end").reset_index(drop=True)
    return CandleFrame(secid=spec.secid, df=df)


def _ticker_from_secid_fallback(secid: str, market: moexapi.Market) -> moexapi.Ticker:
    """
    Build a minimal moexapi.Ticker from /iss/securities/{secid}.json, selecting traded boards
    that match the given engine/market.
    """
    info = moexapi.get_ticker_info_dict(secid)
    resp = moex_utils.json_api_call(f"https://iss.moex.com/iss/securities/{secid}.json")
    boards = moex_utils.prepare_dict(resp, "boards")
    engine = list(market.engines)[0] if market.engines else None
    mkt = list(market.markets)[0] if market.markets else None
    board_ids: list[str] = []
    for b in boards:
        if b.get("is_traded") != 1:
            continue
        if engine is not None and b.get("engine") != engine:
            continue
        if mkt is not None and b.get("market") != mkt:
            continue
        board_id = b.get("boardid")
        if not board_id:
            continue
        if market.boards and board_id not in market.boards:
            continue
        board_ids.append(board_id)
    if not board_ids:
        raise RuntimeError(f"Failed to discover traded boards for {secid} in market={market}")

    return moexapi.Ticker(
        secid=secid,
        alias=secid,
        is_traded=True,
        market=market,
        shortname=info.get("SHORTNAME"),
        isin=info.get("ISIN"),
        subtype=info.get("SECSUBTYPE"),
        listlevel=int(info["LISTLEVEL"]) if "LISTLEVEL" in info and str(info["LISTLEVEL"]).isdigit() else None,
        boards=board_ids,
    )


def _get_candles_safe(
    ticker: moexapi.Ticker,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
    interval: int | None = None,
):
    """
    moexapi.get_candles() currently compares `candle.end` (datetime) with `split.date` (date),
    which raises TypeError on instruments with splits. We reproduce the same logic with a
    safe comparison.
    """
    from moexapi import candles as _candles
    from moexapi import changeover as _changeover
    from moexapi import splits as _splits

    ticker = _changeover.get_current_ticker(ticker)
    prev_tickers = _changeover.get_prev_tickers(ticker)
    ticker_splits = [s for s in _splits.get_splits() if s.secid in [t.secid for t in prev_tickers]]

    parsed = []
    for t in prev_tickers:
        parsed.append(_candles._parse_candles(t, start_date=start_date, end_date=end_date, interval=interval))
    result = _candles._merge_candles_list(parsed)

    for split in ticker_splits:
        for candle in result:
            # split.date is a datetime.date; candle.end is datetime.datetime
            if candle.end.date() < split.date:
                candle.mult(1 / split.mult)
    return result


def write_raw_candles_csv(
    candle_frame: CandleFrame,
    out_path: Path,
    meta: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candle_frame.df.to_csv(out_path, index=False)
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_and_store_universe(
    instruments: Iterable[InstrumentSpec],
    raw_dir: Path,
    years: int,
    asof: dt.datetime | None = None,
    force: bool = False,
) -> None:
    asof = asof or dt.datetime.now(dt.timezone.utc)
    start = asof - dt.timedelta(days=int(365.25 * years) + 10)
    end = asof
    logger.info("Fetching candles: start=%s end=%s years=%s", start.date(), end.date(), years)

    for spec in instruments:
        out_csv = raw_dir / "candles" / f"{spec.secid}.csv"
        if out_csv.exists() and not force:
            logger.info("Skipping %s (%s): raw CSV exists", spec.secid, spec.type)
            continue
        logger.info("Fetching %s (%s)", spec.secid, spec.type)
        cf = fetch_daily_candles(spec, start=start, end=end)
        meta = {
            "secid": spec.secid,
            "type": spec.type,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": 24,
            "asof": asof.isoformat(),
        }
        write_raw_candles_csv(cf, out_csv, meta=meta)
