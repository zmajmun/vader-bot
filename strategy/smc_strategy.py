"""
VADER SMC Strategy
──────────────────
Entry logic mirrors what you see in the screenshots:

  LONG SETUP
  1. HTF bias is bullish (HH/HL structure)
  2. CHoCH or BOS to the upside detected on LTF
  3. Price displaces upward, creating a bullish FVG
  4. Price retraces into the FVG (or nearest bullish OB)
  5. Entry at FVG bottom (or OB top).  Stop below swing low.
  6. TP1 = 1.5R (partial close).  TP2 = 3R or prev liquidity high.

  SHORT SETUP (mirror image)
  1. HTF bias is bearish (LH/LL structure)
  2. Distribution zone → CHoCH or BOS to the downside
  3. Bearish FVG formed.  Price retraces into it.
  4. Entry at FVG top.  Stop above swing high.
  5. TP1 = 1.5R, TP2 = 3R or equal-lows liquidity target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.signals import (
    Direction,
    FVGSignal,
    OrderBlockSignal,
    SMCSnapshot,
    nearest_fvg_to_price,
    nearest_ob_to_price,
)

log = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    symbol: str
    direction: Direction
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    risk_per_share: float
    rr_ratio: float
    reason: str                       # human-readable setup label
    # Source signal
    fvg: Optional[FVGSignal] = None
    ob: Optional[OrderBlockSignal] = None
    # Quality score 0-100
    confidence: float = 50.0

    @property
    def is_long(self) -> bool:
        return self.direction == Direction.LONG


@dataclass
class StrategyConfig:
    require_choch: bool = True
    htf_trend_filter: bool = True
    min_rr: float = 1.5
    tp1_r: float = 1.5
    tp2_r: float = 3.0
    stop_buffer_pct: float = 0.002   # add 0.2% to stop distance


class SMCStrategy:
    """
    Stateless strategy: evaluate() takes two SMC snapshots (HTF + LTF)
    and the current price, returning a TradeSetup or None.
    """

    def __init__(self, config: StrategyConfig | None = None):
        self.cfg = config or StrategyConfig()

    def evaluate(
        self,
        ltf_snap: SMCSnapshot,
        htf_snap: Optional[SMCSnapshot],
        current_price: float,
    ) -> Optional[TradeSetup]:
        """
        Main evaluation entry point.  Called on every new bar close.
        Returns a TradeSetup if entry conditions are met, else None.
        """
        symbol = ltf_snap.symbol

        # ── Session gate ─────────────────────────────────────────────────
        if not ltf_snap.in_session:
            return None

        # ── HTF trend filter ─────────────────────────────────────────────
        htf_trend = Direction.FLAT
        if htf_snap is not None:
            htf_trend = htf_snap.trend
        elif not self.cfg.htf_trend_filter:
            htf_trend = ltf_snap.trend  # fall back to LTF trend

        if self.cfg.htf_trend_filter and htf_trend == Direction.FLAT:
            return None

        # ── Require CHoCH / BOS ──────────────────────────────────────────
        if self.cfg.require_choch:
            if ltf_snap.structure is None:
                return None
            struct_dir = ltf_snap.structure.choch or ltf_snap.structure.bos
            if struct_dir is None:
                return None
            # Structure must align with HTF trend
            if self.cfg.htf_trend_filter and struct_dir != htf_trend:
                return None
        else:
            struct_dir = htf_trend if htf_trend != Direction.FLAT else ltf_snap.trend

        # ── Attempt to build a LONG setup ────────────────────────────────
        if struct_dir == Direction.LONG and (not self.cfg.htf_trend_filter or htf_trend == Direction.LONG):
            setup = self._long_setup(symbol, ltf_snap, current_price)
            if setup:
                return setup

        # ── Attempt to build a SHORT setup ───────────────────────────────
        if struct_dir == Direction.SHORT and (not self.cfg.htf_trend_filter or htf_trend == Direction.SHORT):
            setup = self._short_setup(symbol, ltf_snap, current_price)
            if setup:
                return setup

        return None

    # ──────────────────────────────────────────────────────────────────────
    #  LONG
    # ──────────────────────────────────────────────────────────────────────
    def _long_setup(self, symbol: str, snap: SMCSnapshot, price: float) -> Optional[TradeSetup]:
        # Price must be BELOW the nearest bullish FVG / OB (i.e. retracing into it)
        fvg = nearest_fvg_to_price(snap.bullish_fvgs, price, Direction.LONG)
        ob = nearest_ob_to_price(snap.bullish_obs, price, Direction.LONG)

        entry, stop, reason, source_fvg, source_ob = None, None, "", None, None

        # Prefer FVG entry
        if fvg and fvg.bottom <= price <= fvg.top:
            entry = fvg.bottom       # enter at bottom of gap
            reason = f"Bullish FVG entry [{fvg.bottom:.4f}–{fvg.top:.4f}]"
            source_fvg = fvg
        # Fallback to OB
        elif ob and ob.bottom <= price <= ob.top:
            entry = ob.top           # enter at top of OB (demand zone)
            reason = f"Bullish OB entry [{ob.bottom:.4f}–{ob.top:.4f}] str={ob.strength_pct:.0f}%"
            source_ob = ob

        if entry is None:
            return None

        # Stop: below the nearest swing low with a buffer
        if snap.swing_low is None:
            return None
        raw_stop = snap.swing_low
        stop = raw_stop * (1 - self.cfg.stop_buffer_pct)

        if stop >= entry:
            log.debug("%s: long stop above entry — skip", symbol)
            return None

        risk = entry - stop
        tp1 = entry + risk * self.cfg.tp1_r
        tp2 = entry + risk * self.cfg.tp2_r

        # Override TP2 with nearest buy-side liquidity above if available
        buy_liq = [l for l in snap.liquidity_pools if l.direction == Direction.LONG and l.level > entry and not l.swept]
        if buy_liq:
            liq_target = min(buy_liq, key=lambda l: l.level).level
            if liq_target > tp1:
                tp2 = liq_target

        rr = (tp2 - entry) / risk
        if rr < self.cfg.min_rr:
            log.debug("%s: long RR %.2f < %.2f — skip", symbol, rr, self.cfg.min_rr)
            return None

        confidence = self._score_long(snap, fvg, ob)

        return TradeSetup(
            symbol=symbol,
            direction=Direction.LONG,
            entry_price=round(entry, 4),
            stop_price=round(stop, 4),
            tp1_price=round(tp1, 4),
            tp2_price=round(tp2, 4),
            risk_per_share=round(risk, 4),
            rr_ratio=round(rr, 2),
            reason=reason,
            fvg=source_fvg,
            ob=source_ob,
            confidence=confidence,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  SHORT
    # ──────────────────────────────────────────────────────────────────────
    def _short_setup(self, symbol: str, snap: SMCSnapshot, price: float) -> Optional[TradeSetup]:
        fvg = nearest_fvg_to_price(snap.bearish_fvgs, price, Direction.SHORT)
        ob = nearest_ob_to_price(snap.bearish_obs, price, Direction.SHORT)

        entry, stop, reason, source_fvg, source_ob = None, None, "", None, None

        if fvg and fvg.bottom <= price <= fvg.top:
            entry = fvg.top
            reason = f"Bearish FVG entry [{fvg.bottom:.4f}–{fvg.top:.4f}]"
            source_fvg = fvg
        elif ob and ob.bottom <= price <= ob.top:
            entry = ob.bottom
            reason = f"Bearish OB entry [{ob.bottom:.4f}–{ob.top:.4f}] str={ob.strength_pct:.0f}%"
            source_ob = ob

        if entry is None:
            return None

        if snap.swing_high is None:
            return None
        stop = snap.swing_high * (1 + self.cfg.stop_buffer_pct)

        if stop <= entry:
            log.debug("%s: short stop below entry — skip", symbol)
            return None

        risk = stop - entry
        tp1 = entry - risk * self.cfg.tp1_r
        tp2 = entry - risk * self.cfg.tp2_r

        sell_liq = [l for l in snap.liquidity_pools if l.direction == Direction.SHORT and l.level < entry and not l.swept]
        if sell_liq:
            liq_target = max(sell_liq, key=lambda l: l.level).level
            if liq_target < tp1:
                tp2 = liq_target

        rr = (entry - tp2) / risk
        if rr < self.cfg.min_rr:
            log.debug("%s: short RR %.2f < %.2f — skip", symbol, rr, self.cfg.min_rr)
            return None

        confidence = self._score_short(snap, fvg, ob)

        return TradeSetup(
            symbol=symbol,
            direction=Direction.SHORT,
            entry_price=round(entry, 4),
            stop_price=round(stop, 4),
            tp1_price=round(tp1, 4),
            tp2_price=round(tp2, 4),
            risk_per_share=round(risk, 4),
            rr_ratio=round(rr, 2),
            reason=reason,
            fvg=source_fvg,
            ob=source_ob,
            confidence=confidence,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Confidence scoring (higher = stronger setup)
    # ──────────────────────────────────────────────────────────────────────
    def _score_long(self, snap: SMCSnapshot, fvg, ob) -> float:
        score = 50.0
        if snap.structure and snap.structure.choch == Direction.LONG:
            score += 20   # CHoCH > BOS
        elif snap.structure and snap.structure.bos == Direction.LONG:
            score += 10
        if fvg:
            score += 15
        if ob:
            score += ob.strength_pct * 0.1
        if snap.prev_day_low and snap.swing_low and snap.swing_low > snap.prev_day_low:
            score += 5    # swing low above PDL — bullish context
        return min(score, 100.0)

    def _score_short(self, snap: SMCSnapshot, fvg, ob) -> float:
        score = 50.0
        if snap.structure and snap.structure.choch == Direction.SHORT:
            score += 20
        elif snap.structure and snap.structure.bos == Direction.SHORT:
            score += 10
        if fvg:
            score += 15
        if ob:
            score += ob.strength_pct * 0.1
        if snap.prev_day_high and snap.swing_high and snap.swing_high < snap.prev_day_high:
            score += 5
        return min(score, 100.0)
