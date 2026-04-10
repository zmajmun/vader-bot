"""
Trading Engine
──────────────
Main async event loop:
  - Consumes live bar events from DataFeed
  - Computes SMC snapshots (LTF + HTF)
  - Runs SMCStrategy.evaluate() on each bar
  - Delegates execution to OrderManager
  - Updates Dashboard and sends Alerts
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from core.signals import compute_snapshot, SMCSnapshot
from data.feed import DataFeed
from execution.broker import BrokerBase
from execution.order_manager import OrderManager
from monitoring.alerts import Alerter
from monitoring.dashboard import Dashboard
from risk.position_sizer import PositionSizer, RiskConfig
from strategy.smc_strategy import SMCStrategy, StrategyConfig, TradeSetup

log = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        broker: BrokerBase,
        feed: DataFeed,
        strategy: SMCStrategy,
        order_manager: OrderManager,
        sizer: PositionSizer,
        dashboard: Dashboard,
        alerter: Alerter,
        symbols: List[str],
        max_trades_per_day: int = 3,
        htf_timeframe: str = "1Hour",
        swing_length: int = 10,
        min_fvg_size_pct: float = 0.001,
        min_ob_strength: float = 30.0,
        liquidity_range_pct: float = 0.005,
        session: str = "New York kill zone",
        paper: bool = True,
    ):
        self.broker = broker
        self.feed = feed
        self.strategy = strategy
        self.order_manager = order_manager
        self.sizer = sizer
        self.dashboard = dashboard
        self.alerter = alerter
        self.symbols = symbols
        self.max_trades_per_day = max_trades_per_day
        self.htf_timeframe = htf_timeframe
        self.swing_length = swing_length
        self.min_fvg_size_pct = min_fvg_size_pct
        self.min_ob_strength = min_ob_strength
        self.liquidity_range_pct = liquidity_range_pct
        self.session = session
        self.paper = paper

        # Per-symbol trade count for the day
        self._daily_trades: Dict[str, int] = defaultdict(int)
        self._daily_date: Optional[date] = None

        # Latest prices for dashboard
        self._prices: Dict[str, float] = {}
        self._signals: Dict[str, str] = {}

        # HTF snapshots (refreshed less frequently)
        self._htf_snaps: Dict[str, SMCSnapshot] = {}
        self._htf_bar_count: Dict[str, int] = defaultdict(int)
        self._htf_refresh_every = 12   # refresh HTF every N LTF bars

        self._running = False

    # ──────────────────────────────────────────────────────────────────────
    #  Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        log.info("TradingEngine starting — %s mode", "PAPER" if self.paper else "LIVE")

        # Warmup historical data
        await self.feed.warmup()

        # Initial HTF snapshots
        for sym in self.symbols:
            await self._refresh_htf(sym)

        # Start dashboard
        self.dashboard.start()

        # Subscribe to live bars
        self.feed.subscribe(self._on_bar)

        # Dashboard refresh loop (parallel to stream)
        asyncio.create_task(self._dashboard_loop())

        # Start live stream (blocks until stopped)
        await self.feed.stream()

    async def stop(self):
        self._running = False
        await self.order_manager.close_all()
        await self.feed.stop()
        self.dashboard.stop()
        log.info("TradingEngine stopped.")

    # ──────────────────────────────────────────────────────────────────────
    #  Bar handler
    # ──────────────────────────────────────────────────────────────────────

    async def _on_bar(self, symbol: str, ltf_df: pd.DataFrame):
        """Called by DataFeed on every new completed bar."""
        self._reset_daily_counts_if_new_day()

        # Update latest price
        current_price = float(ltf_df["close"].iloc[-1])
        self._prices[symbol] = current_price

        # Update open trade (TP1 check, breakeven)
        await self.order_manager.update(symbol, current_price)

        # Refresh HTF periodically
        self._htf_bar_count[symbol] += 1
        if self._htf_bar_count[symbol] % self._htf_refresh_every == 0:
            await self._refresh_htf(symbol)

        # Risk gate: halted?
        acct = await self.broker.get_account()
        if self.sizer.is_halted(acct.equity):
            self._signals[symbol] = "HALTED — daily loss limit"
            return

        # Already in position?
        if self.order_manager.has_position(symbol):
            self._signals[symbol] = f"in position @ {self._prices[symbol]:.4f}"
            return

        # Trade count gate
        if self._daily_trades[symbol] >= self.max_trades_per_day:
            self._signals[symbol] = f"max {self.max_trades_per_day} trades/day reached"
            return

        # Compute LTF SMC snapshot
        ltf_snap = compute_snapshot(
            ohlc=ltf_df,
            symbol=symbol,
            swing_length=self.swing_length,
            min_fvg_size_pct=self.min_fvg_size_pct,
            min_ob_strength=self.min_ob_strength,
            liquidity_range_pct=self.liquidity_range_pct,
            session=self.session,
        )

        htf_snap = self._htf_snaps.get(symbol)

        # Evaluate strategy
        setup: Optional[TradeSetup] = self.strategy.evaluate(ltf_snap, htf_snap, current_price)

        if setup is None:
            direction = "▲" if ltf_snap.trend.value == 1 else "▼" if ltf_snap.trend.value == -1 else "—"
            in_sess = "✓ sess" if ltf_snap.in_session else "✗ sess"
            struct = f"choch={setup}" if setup else "scanning"
            self._signals[symbol] = f"{direction} trend  {in_sess}  {struct}"
            return

        # Size the trade
        shares = self.sizer.size_trade(
            symbol=symbol,
            equity=acct.equity,
            risk_per_share=setup.risk_per_share,
            entry_price=setup.entry_price,
        )
        if shares == 0:
            self._signals[symbol] = "sized to 0 — skip"
            return

        # Execute
        trade = await self.order_manager.enter(setup, shares)
        if trade:
            self._daily_trades[symbol] += 1
            self.sizer.register_open(symbol, shares)
            self._signals[symbol] = f"{'LONG' if setup.is_long else 'SHORT'} entered {shares}sh @ {setup.entry_price:.4f}"
            await self.alerter.trade_entered(setup, shares)
            log.info(
                "NEW TRADE | %s | %s | %d shares | entry=%.4f stop=%.4f tp2=%.4f | score=%.0f",
                symbol, "LONG" if setup.is_long else "SHORT",
                shares, setup.entry_price, setup.stop_price, setup.tp2_price, setup.confidence,
            )

    # ──────────────────────────────────────────────────────────────────────
    #  HTF refresh
    # ──────────────────────────────────────────────────────────────────────

    async def _refresh_htf(self, symbol: str):
        htf_df = self.feed.get_htf(symbol)
        if htf_df is None or len(htf_df) < 20:
            return
        try:
            self._htf_snaps[symbol] = compute_snapshot(
                ohlc=htf_df,
                symbol=symbol,
                swing_length=self.swing_length,
                session=self.session,
            )
        except Exception as e:
            log.warning("HTF snapshot failed for %s: %s", symbol, e)

    # ──────────────────────────────────────────────────────────────────────
    #  Dashboard refresh loop
    # ──────────────────────────────────────────────────────────────────────

    async def _dashboard_loop(self):
        while self._running:
            try:
                acct = await self.broker.get_account()
                self.dashboard.update(
                    equity=acct.equity,
                    prices=dict(self._prices),
                    trades=dict(self.order_manager.open_trades),
                    signals=dict(self._signals),
                )
            except Exception as e:
                log.debug("Dashboard update error: %s", e)
            await asyncio.sleep(2)

    # ──────────────────────────────────────────────────────────────────────
    #  Daily reset
    # ──────────────────────────────────────────────────────────────────────

    def _reset_daily_counts_if_new_day(self):
        today = date.today()
        if self._daily_date != today:
            self._daily_date = today
            self._daily_trades = defaultdict(int)
            log.info("New trading day — daily counters reset")
