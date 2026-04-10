"""
Order Manager
─────────────
Handles the full lifecycle of a trade:
  1. Entry (limit or market bracket)
  2. TP1 partial close + breakeven trail
  3. TP2 full close
  4. Stop-loss (server-side bracket or manual fallback)
  5. Fill timeout → cancel / fallback to market
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from execution.broker import BrokerBase, Order, OrderSide, OrderStatus, TimeInForce
from strategy.smc_strategy import TradeSetup
from core.signals import Direction

log = logging.getLogger(__name__)


@dataclass
class ManagedTrade:
    setup: TradeSetup
    entry_order: Order
    shares: int
    entry_time: datetime = field(default_factory=datetime.utcnow)

    # Attached orders (from bracket)
    stop_order_id: Optional[str] = None
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None

    # State flags
    tp1_hit: bool = False
    be_moved: bool = False
    closed: bool = False

    # Tracking
    realized_pnl: float = 0.0


class OrderManager:
    def __init__(
        self,
        broker: BrokerBase,
        fill_timeout: int = 30,
        limit_offset_pct: float = 0.001,
        use_limit_entry: bool = True,
        fallback_to_market: bool = False,
        tp1_size: float = 0.5,
        be_after_tp1: bool = True,
    ):
        self.broker = broker
        self.fill_timeout = fill_timeout
        self.limit_offset_pct = limit_offset_pct
        self.use_limit_entry = use_limit_entry
        self.fallback_to_market = fallback_to_market
        self.tp1_size = tp1_size
        self.be_after_tp1 = be_after_tp1

        self._trades: Dict[str, ManagedTrade] = {}   # symbol → ManagedTrade

    # ──────────────────────────────────────────────────────────────────────
    #  Enter
    # ──────────────────────────────────────────────────────────────────────

    async def enter(self, setup: TradeSetup, shares: int) -> Optional[ManagedTrade]:
        symbol = setup.symbol
        if symbol in self._trades:
            log.warning("Already in a trade for %s — skip entry", symbol)
            return None

        side = OrderSide.BUY if setup.is_long else OrderSide.SELL
        entry_px = setup.entry_price

        # Slightly offset limit price to improve fill probability
        if self.use_limit_entry:
            offset = entry_px * self.limit_offset_pct
            limit_px = entry_px + offset if setup.is_long else entry_px - offset
        else:
            limit_px = None

        try:
            # Use bracket order: entry + server-side stop + tp2
            order = await self.broker.submit_bracket_order(
                symbol=symbol,
                qty=shares,
                side=side,
                entry_price=limit_px,
                stop_price=setup.stop_price,
                take_profit_price=setup.tp2_price,
                client_order_id=f"vader_{symbol}_{int(datetime.utcnow().timestamp())}",
            )
        except Exception as e:
            log.error("Entry order failed for %s: %s", symbol, e)
            return None

        trade = ManagedTrade(setup=setup, entry_order=order, shares=shares)
        self._trades[symbol] = trade

        log.info(
            "ENTRY %s %s | %d sh @ %.4f | stop=%.4f tp1=%.4f tp2=%.4f | RR=%.2f",
            "LONG" if setup.is_long else "SHORT",
            symbol, shares, entry_px,
            setup.stop_price, setup.tp1_price, setup.tp2_price,
            setup.rr_ratio,
        )

        # Monitor fill in the background
        asyncio.create_task(self._monitor_fill(trade))
        return trade

    # ──────────────────────────────────────────────────────────────────────
    #  Update (called each bar)
    # ──────────────────────────────────────────────────────────────────────

    async def update(self, symbol: str, current_price: float):
        trade = self._trades.get(symbol)
        if not trade or trade.closed:
            return

        # TP1 check: partial close at TP1 level
        if not trade.tp1_hit:
            tp1_hit = (
                (trade.setup.is_long and current_price >= trade.setup.tp1_price)
                or (not trade.setup.is_long and current_price <= trade.setup.tp1_price)
            )
            if tp1_hit:
                await self._handle_tp1(trade, current_price)

    async def _handle_tp1(self, trade: ManagedTrade, price: float):
        symbol = trade.setup.symbol
        tp1_qty = max(1, int(trade.shares * self.tp1_size))
        close_side = OrderSide.SELL if trade.setup.is_long else OrderSide.BUY

        log.info("TP1 hit for %s @ %.4f — closing %d/%d shares", symbol, price, tp1_qty, trade.shares)

        try:
            await self.broker.submit_market_order(symbol, tp1_qty, close_side, TimeInForce.DAY)
            trade.tp1_hit = True
            trade.realized_pnl += tp1_qty * abs(price - trade.setup.entry_price) * (1 if trade.setup.is_long else -1)
        except Exception as e:
            log.error("TP1 close failed for %s: %s", symbol, e)
            return

        # Move stop to breakeven
        if self.be_after_tp1 and not trade.be_moved:
            await self._move_stop_to_be(trade)

    async def _move_stop_to_be(self, trade: ManagedTrade):
        """Cancel existing stop and re-submit at entry price."""
        symbol = trade.setup.symbol
        be_price = trade.setup.entry_price
        remaining = trade.shares - max(1, int(trade.shares * self.tp1_size))
        if remaining < 1:
            return

        close_side = OrderSide.SELL if trade.setup.is_long else OrderSide.BUY
        try:
            # Cancel old server-side stop (bracket leg)
            if trade.stop_order_id:
                await self.broker.cancel_order(trade.stop_order_id)
            # Place new stop at breakeven
            await self.broker.submit_stop_order(
                symbol=symbol,
                qty=remaining,
                side=close_side,
                stop_price=be_price,
                tif=TimeInForce.GTC,
            )
            trade.be_moved = True
            log.info("Breakeven stop set for %s @ %.4f", symbol, be_price)
        except Exception as e:
            log.error("Breakeven stop failed for %s: %s", symbol, e)

    # ──────────────────────────────────────────────────────────────────────
    #  Emergency / manual close
    # ──────────────────────────────────────────────────────────────────────

    async def close_all(self):
        for symbol in list(self._trades):
            await self.close(symbol, reason="close_all")

    async def close(self, symbol: str, reason: str = "manual"):
        trade = self._trades.get(symbol)
        if not trade or trade.closed:
            return
        log.info("Closing %s (%s)", symbol, reason)
        try:
            order = await self.broker.close_position(symbol)
            if order:
                trade.closed = True
                self._trades.pop(symbol, None)
        except Exception as e:
            log.error("Close failed for %s: %s", symbol, e)

    def register_closed(self, symbol: str, pnl: float):
        trade = self._trades.pop(symbol, None)
        if trade:
            trade.closed = True
            trade.realized_pnl += pnl

    # ──────────────────────────────────────────────────────────────────────
    #  Fill monitoring
    # ──────────────────────────────────────────────────────────────────────

    async def _monitor_fill(self, trade: ManagedTrade):
        await asyncio.sleep(self.fill_timeout)
        order = await self.broker.get_order(trade.entry_order.order_id)
        if order and order.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            log.warning(
                "Entry order for %s not filled after %ds — cancelling",
                trade.setup.symbol, self.fill_timeout,
            )
            await self.broker.cancel_order(trade.entry_order.order_id)
            if self.fallback_to_market:
                log.info("Submitting market fallback for %s", trade.setup.symbol)
                side = OrderSide.BUY if trade.setup.is_long else OrderSide.SELL
                await self.broker.submit_market_order(trade.setup.symbol, trade.shares, side)
            else:
                self._trades.pop(trade.setup.symbol, None)

    # ──────────────────────────────────────────────────────────────────────
    #  Queries
    # ──────────────────────────────────────────────────────────────────────

    @property
    def open_trades(self) -> Dict[str, ManagedTrade]:
        return {k: v for k, v in self._trades.items() if not v.closed}

    def has_position(self, symbol: str) -> bool:
        return symbol in self._trades and not self._trades[symbol].closed
