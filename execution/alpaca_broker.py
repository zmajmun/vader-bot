"""
Alpaca Markets Broker Adapter
Supports both paper and live trading via alpaca-py SDK.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partial
from typing import Optional

from datetime import datetime, timedelta

import pandas as pd
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide as AlpacaOrderSide,
    TimeInForce as AlpacaTIF,
)
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from execution.broker import (
    AccountInfo,
    BrokerBase,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

log = logging.getLogger(__name__)

# ── Timeframe string → alpaca TimeFrame mapping ────────────────────────────
_TF_MAP = {
    "1Min":   TimeFrame(1,  TimeFrameUnit.Minute),
    "5Min":   TimeFrame(5,  TimeFrameUnit.Minute),
    "15Min":  TimeFrame(15, TimeFrameUnit.Minute),
    "30Min":  TimeFrame(30, TimeFrameUnit.Minute),
    "1Hour":  TimeFrame(1,  TimeFrameUnit.Hour),
    "4Hour":  TimeFrame(4,  TimeFrameUnit.Hour),
    "1Day":   TimeFrame(1,  TimeFrameUnit.Day),
}

_SIDE_MAP = {
    OrderSide.BUY:  AlpacaOrderSide.BUY,
    OrderSide.SELL: AlpacaOrderSide.SELL,
}

_TIF_MAP = {
    TimeInForce.DAY: AlpacaTIF.DAY,
    TimeInForce.GTC: AlpacaTIF.GTC,
    TimeInForce.IOC: AlpacaTIF.IOC,
    TimeInForce.FOK: AlpacaTIF.FOK,
}


def _map_status(alpaca_status) -> OrderStatus:
    s = str(alpaca_status).lower()
    if "filled" in s and "partial" in s:
        return OrderStatus.PARTIALLY_FILLED
    if "filled" in s:
        return OrderStatus.FILLED
    if "canceled" in s or "cancelled" in s:
        return OrderStatus.CANCELLED
    if "rejected" in s or "expired" in s:
        return OrderStatus.REJECTED
    if "new" in s or "accepted" in s or "pending" in s:
        return OrderStatus.OPEN
    return OrderStatus.PENDING


def _to_order(o) -> Order:
    return Order(
        order_id=str(o.id),
        symbol=o.symbol,
        side=OrderSide.BUY if str(o.side).lower() == "buy" else OrderSide.SELL,
        qty=int(o.qty or 0),
        order_type=OrderType.MARKET,
        status=_map_status(o.status),
        limit_price=float(o.limit_price) if o.limit_price else None,
        stop_price=float(o.stop_price) if o.stop_price else None,
        filled_qty=int(o.filled_qty or 0),
        filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
        client_order_id=o.client_order_id,
    )


async def _run(fn, *args):
    """Run a blocking Alpaca SDK call off the event loop thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn if not args else partial(fn, *args))


class AlpacaBroker(BrokerBase):
    """
    Live / paper trading adapter for Alpaca Markets.
    paper=True  → uses paper API endpoint (default, safe)
    paper=False → uses live endpoint (real money)
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._data = StockHistoricalDataClient(api_key, secret_key)
        self._paper = paper
        log.info("AlpacaBroker initialised — %s mode", "PAPER" if paper else "LIVE")

    # ── Account ────────────────────────────────────────────────────────────

    async def get_account(self) -> AccountInfo:
        acct = await _run(self._trading.get_account)
        return AccountInfo(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            daily_pnl=float(acct.equity) - float(acct.last_equity),
        )

    # ── Positions ──────────────────────────────────────────────────────────

    async def get_position(self, symbol: str) -> Optional[Position]:
        try:
            p = await _run(self._trading.get_open_position, symbol)
            return Position(
                symbol=p.symbol,
                qty=int(p.qty),
                avg_entry=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                side=str(p.side).lower(),
            )
        except Exception:
            return None

    async def get_positions(self) -> list[Position]:
        positions = await _run(self._trading.get_all_positions)
        return [
            Position(
                symbol=p.symbol,
                qty=int(p.qty),
                avg_entry=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                side=str(p.side).lower(),
            )
            for p in positions
        ]

    # ── Orders ─────────────────────────────────────────────────────────────

    async def submit_market_order(
        self, symbol: str, qty: int, side: OrderSide,
        tif: TimeInForce = TimeInForce.DAY,
    ) -> Order:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=_SIDE_MAP[side],
            time_in_force=_TIF_MAP[tif],
        )
        o = await _run(self._trading.submit_order, req)
        log.info("Market order submitted: %s %s %d", side.value.upper(), symbol, qty)
        return _to_order(o)

    async def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        limit_price: float,
        tif: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
    ) -> Order:
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=_SIDE_MAP[side],
            limit_price=round(limit_price, 2),
            time_in_force=_TIF_MAP[tif],
            client_order_id=client_order_id or str(uuid.uuid4()),
        )
        o = await _run(self._trading.submit_order, req)
        log.info("Limit order submitted: %s %s %d @ %.4f", side.value.upper(), symbol, qty, limit_price)
        return _to_order(o)

    async def submit_stop_order(
        self, symbol: str, qty: int, side: OrderSide, stop_price: float,
        tif: TimeInForce = TimeInForce.GTC,
    ) -> Order:
        req = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=_SIDE_MAP[side],
            stop_price=round(stop_price, 2),
            time_in_force=_TIF_MAP[tif],
        )
        o = await _run(self._trading.submit_order, req)
        log.info("Stop order submitted: %s %s %d stop=%.4f", side.value.upper(), symbol, qty, stop_price)
        return _to_order(o)

    async def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        entry_price: Optional[float],
        stop_price: float,
        take_profit_price: float,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Bracket order = entry + attached stop-loss + take-profit.
        Alpaca handles the OCO (one-cancels-other) logic server-side.
        """
        if entry_price is not None:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=_SIDE_MAP[side],
                limit_price=round(entry_price, 2),
                time_in_force=AlpacaTIF.DAY,
                order_class="bracket",
                stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
                client_order_id=client_order_id or str(uuid.uuid4()),
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=_SIDE_MAP[side],
                time_in_force=AlpacaTIF.DAY,
                order_class="bracket",
                stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
                client_order_id=client_order_id or str(uuid.uuid4()),
            )
        o = await _run(self._trading.submit_order, req)
        log.info(
            "Bracket order submitted: %s %s %d entry=%.4f stop=%.4f tp=%.4f",
            side.value.upper(), symbol, qty,
            entry_price or 0, stop_price, take_profit_price,
        )
        return _to_order(o)

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await _run(self._trading.cancel_order_by_id, order_id)
            return True
        except Exception as e:
            log.warning("Cancel order %s failed: %s", order_id, e)
            return False

    async def close_position(self, symbol: str) -> Optional[Order]:
        try:
            o = await _run(self._trading.close_position, symbol)
            log.info("Closed position: %s", symbol)
            return _to_order(o)
        except Exception as e:
            log.warning("Close position %s failed: %s", symbol, e)
            return None

    async def get_order(self, order_id: str) -> Optional[Order]:
        try:
            o = await _run(self._trading.get_order_by_id, order_id)
            return _to_order(o)
        except Exception:
            return None

    # ── Market Data ────────────────────────────────────────────────────────

    async def get_bars(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        tf = _TF_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(_TF_MAP)}")

        # IEX free feed requires date range, not limit param.
        # Estimate lookback: minutes_per_bar × limit / 390 trading mins per day × 1.5 buffer
        minutes = {"1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30,
                   "1Hour": 60, "4Hour": 240, "1Day": 390}.get(timeframe, 5)
        calendar_days = max(7, int(limit * minutes / 390 * 2))
        end = datetime.now(pytz.utc)
        start = end - timedelta(days=calendar_days)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed="iex",
        )
        bars = await _run(self._data.get_stock_bars, req)
        df = bars.df

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols].tail(limit)

    async def get_latest_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestQuoteRequest
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = await _run(self._data.get_stock_latest_quote, req)
        q = quote[symbol]
        return float((q.ask_price + q.bid_price) / 2)
