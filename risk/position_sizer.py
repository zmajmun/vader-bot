"""
Position Sizer
──────────────
Fixed-fractional risk model:
    shares = (account_equity × risk_pct) / risk_per_share

Additional guards:
  - Max position size cap (% of equity)
  - Daily loss limit check
  - Maximum concurrent positions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    account_risk_pct: float = 0.01     # 1% per trade
    max_position_pct: float = 0.20     # no single position > 20% equity
    max_positions: int = 5
    max_daily_loss_pct: float = 0.03   # halt at -3% daily


@dataclass
class DailyStats:
    date: date = field(default_factory=date.today)
    realized_pnl: float = 0.0
    trade_count: int = 0

    def reset_if_new_day(self):
        today = date.today()
        if self.date != today:
            self.date = today
            self.realized_pnl = 0.0
            self.trade_count = 0


class PositionSizer:
    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()
        self._daily = DailyStats()
        self._open_positions: Dict[str, int] = {}   # symbol → shares

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def size_trade(
        self,
        symbol: str,
        equity: float,
        risk_per_share: float,
        entry_price: float,
    ) -> int:
        """
        Return number of shares to trade.
        Returns 0 if any risk check fails.
        """
        self._daily.reset_if_new_day()

        if not self._pre_trade_checks(equity):
            return 0

        if risk_per_share <= 0:
            log.warning("risk_per_share <= 0 for %s — skip", symbol)
            return 0

        dollar_risk = equity * self.cfg.account_risk_pct
        raw_shares = dollar_risk / risk_per_share

        # Cap by max position size
        max_shares_by_pct = (equity * self.cfg.max_position_pct) / entry_price
        shares = int(min(raw_shares, max_shares_by_pct))

        if shares < 1:
            log.debug("%s: position size rounded to 0 — skip", symbol)
            return 0

        log.info(
            "Size %s: equity=%.0f risk_pct=%.1f%% risk/sh=%.4f → %d shares",
            symbol, equity, self.cfg.account_risk_pct * 100, risk_per_share, shares,
        )
        return shares

    def register_open(self, symbol: str, shares: int):
        self._open_positions[symbol] = self._open_positions.get(symbol, 0) + shares

    def register_close(self, symbol: str, pnl: float):
        self._open_positions.pop(symbol, None)
        self._daily.realized_pnl += pnl
        self._daily.trade_count += 1

    def is_halted(self, equity: float) -> bool:
        self._daily.reset_if_new_day()
        loss_pct = abs(self._daily.realized_pnl) / equity if equity else 0
        if self._daily.realized_pnl < 0 and loss_pct >= self.cfg.max_daily_loss_pct:
            log.warning("Daily loss limit hit (%.2f%%) — trading halted", loss_pct * 100)
            return True
        return False

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def daily_pnl(self) -> float:
        return self._daily.realized_pnl

    # ──────────────────────────────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────────────────────────────

    def _pre_trade_checks(self, equity: float) -> bool:
        if self.is_halted(equity):
            return False
        if self.open_position_count >= self.cfg.max_positions:
            log.info("Max positions (%d) reached — no new trades", self.cfg.max_positions)
            return False
        return True
