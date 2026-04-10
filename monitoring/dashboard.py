"""
Rich Terminal Dashboard
Shows live P&L, open positions, SMC signals, and bot status.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from execution.order_manager import ManagedTrade
from risk.position_sizer import PositionSizer


class Dashboard:
    def __init__(self, symbols: List[str], sizer: PositionSizer, paper: bool = True):
        self.symbols = symbols
        self.sizer = sizer
        self.paper = paper
        self._console = Console()
        self._live: Optional[Live] = None
        self._prices: Dict[str, float] = {}
        self._signals: Dict[str, str] = {}
        self._equity: float = 0.0
        self._last_update = datetime.utcnow()

    def start(self):
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=2,
            screen=True,
        )
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()

    def update(
        self,
        equity: float,
        prices: Dict[str, float],
        trades: Dict[str, ManagedTrade],
        signals: Dict[str, str],
    ):
        self._equity = equity
        self._prices = prices
        self._trades = trades
        self._signals = signals
        self._last_update = datetime.utcnow()
        if self._live:
            self._live.update(self._render())

    # ──────────────────────────────────────────────────────────────────────

    def _render(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(self._header(),     name="header",    size=3),
            Layout(self._positions(),  name="positions", size=14),
            Layout(self._signals_panel(), name="signals", size=10),
            Layout(self._footer(),     name="footer",    size=3),
        )
        return Panel(layout, title="[bold cyan]VADER — SMC Algo Trading Bot[/]", border_style="cyan")

    def _header(self) -> Panel:
        mode = "[bold red]⚠  LIVE[/]" if not self.paper else "[bold green]PAPER[/]"
        daily_pnl = self.sizer.daily_pnl
        pnl_color = "green" if daily_pnl >= 0 else "red"
        pnl_str = f"[{pnl_color}]${daily_pnl:+.2f}[/]"
        equity_str = f"[white]Equity: [bold]${self._equity:,.2f}[/][/]"
        time_str = self._last_update.strftime("%H:%M:%S UTC")
        halted = "[bold red] HALTED [/]" if self.sizer.is_halted(self._equity) else ""
        txt = f"  {mode}  {equity_str}  Daily P&L: {pnl_str}  Positions: {self.sizer.open_position_count}/{self.sizer.cfg.max_positions}  {halted} [{time_str}]"
        return Panel(Text.from_markup(txt), box=box.SIMPLE)

    def _positions(self) -> Panel:
        t = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False)
        t.add_column("Symbol",    style="bold white",  width=8)
        t.add_column("Dir",       style="bold",        width=6)
        t.add_column("Shares",    justify="right",     width=8)
        t.add_column("Entry",     justify="right",     width=10)
        t.add_column("Current",   justify="right",     width=10)
        t.add_column("Stop",      justify="right",     width=10)
        t.add_column("TP1",       justify="right",     width=10)
        t.add_column("TP2",       justify="right",     width=10)
        t.add_column("Unr. P&L",  justify="right",     width=12)
        t.add_column("Setup",     style="dim",         width=30)

        trades = getattr(self, "_trades", {})
        for sym, trade in trades.items():
            price = self._prices.get(sym, trade.setup.entry_price)
            pnl = (price - trade.setup.entry_price) * trade.shares * (1 if trade.setup.is_long else -1)
            pnl_color = "green" if pnl >= 0 else "red"
            dir_str = "[green]▲ LONG[/]" if trade.setup.is_long else "[red]▼ SHORT[/]"
            tp1_done = "[dim]✓[/] " if trade.tp1_hit else ""
            be_done  = "[cyan]BE[/] " if trade.be_moved else ""
            t.add_row(
                sym,
                dir_str,
                str(trade.shares),
                f"${trade.setup.entry_price:.4f}",
                f"${price:.4f}",
                f"${trade.setup.stop_price:.4f}",
                f"{tp1_done}${trade.setup.tp1_price:.4f}",
                f"{be_done}${trade.setup.tp2_price:.4f}",
                f"[{pnl_color}]${pnl:+.2f}[/]",
                trade.setup.reason[:30],
            )

        if not trades:
            t.add_row("[dim]—", "—", "—", "—", "—", "—", "—", "—", "—", "No open positions")

        return Panel(t, title="[bold]Open Positions", border_style="blue")

    def _signals_panel(self) -> Panel:
        t = Table(box=box.SIMPLE, expand=True, show_edge=False)
        t.add_column("Symbol",  style="bold white", width=8)
        t.add_column("Price",   justify="right",    width=10)
        t.add_column("Trend",   width=8)
        t.add_column("Signal",  width=50)

        for sym in self.symbols:
            price = self._prices.get(sym, 0)
            sig = self._signals.get(sym, "scanning…")
            trend_str = "—"
            if "LONG" in sig.upper():
                trend_str = "[green]▲ Bull[/]"
            elif "SHORT" in sig.upper():
                trend_str = "[red]▼ Bear[/]"
            t.add_row(sym, f"${price:.4f}" if price else "—", trend_str, sig)

        return Panel(t, title="[bold]SMC Signal Scanner", border_style="magenta")

    def _footer(self) -> Panel:
        txt = "  [dim]q[/] quit   [dim]c[/] close all   [dim]r[/] refresh   Built with Smart Money Concepts · Alpaca Markets"
        return Panel(Text.from_markup(txt), box=box.SIMPLE)
