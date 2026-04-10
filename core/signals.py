"""
SMC Signal Engine
Wraps the raw smc library and produces clean, typed signal objects
from OHLCV DataFrames.  All methods are pure functions — no side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd

from core.smc import smc


class Direction(IntEnum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class FVGSignal:
    direction: Direction
    top: float
    bottom: float
    midpoint: float
    bar_index: int
    mitigated: bool = False

    @property
    def size(self) -> float:
        return self.top - self.bottom


@dataclass
class OrderBlockSignal:
    direction: Direction
    top: float
    bottom: float
    strength_pct: float   # 0-100 — OB quality score
    bar_index: int
    mitigated: bool = False

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class StructureSignal:
    bos: Optional[Direction]     # Break of Structure
    choch: Optional[Direction]   # Change of Character
    level: float
    broken_bar: int


@dataclass
class LiquidityLevel:
    direction: Direction          # 1 = buy-side liquidity, -1 = sell-side
    level: float
    swept: bool
    bar_index: int


@dataclass
class SMCSnapshot:
    """Full SMC picture for a single symbol/timeframe at the latest bar."""
    symbol: str
    timestamp: pd.Timestamp

    # Market structure
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    trend: Direction = Direction.FLAT

    # Active FVGs (not yet mitigated)
    bullish_fvgs: list[FVGSignal] = field(default_factory=list)
    bearish_fvgs: list[FVGSignal] = field(default_factory=list)

    # Active Order Blocks
    bullish_obs: list[OrderBlockSignal] = field(default_factory=list)
    bearish_obs: list[OrderBlockSignal] = field(default_factory=list)

    # Most recent structure event
    structure: Optional[StructureSignal] = None

    # Liquidity pools
    liquidity_pools: list[LiquidityLevel] = field(default_factory=list)

    # Previous day reference
    prev_day_high: Optional[float] = None
    prev_day_low: Optional[float] = None

    # Session state
    in_session: bool = False
    session_high: Optional[float] = None
    session_low: Optional[float] = None


def compute_snapshot(
    ohlc: pd.DataFrame,
    symbol: str,
    swing_length: int = 10,
    min_fvg_size_pct: float = 0.001,
    min_ob_strength: float = 30.0,
    liquidity_range_pct: float = 0.005,
    session: str = "New York kill zone",
) -> SMCSnapshot:
    """
    Compute a full SMC snapshot from the provided OHLCV DataFrame.
    ohlc must have columns: open, high, low, close, volume
    with a DatetimeIndex.
    """
    if len(ohlc) < swing_length * 4 + 10:
        return SMCSnapshot(symbol=symbol, timestamp=ohlc.index[-1])

    snap = SMCSnapshot(symbol=symbol, timestamp=ohlc.index[-1])

    # ── Swing Highs / Lows ───────────────────────────────────────────────
    shl = smc.swing_highs_lows(ohlc, swing_length=swing_length)

    highs = shl[shl["HighLow"] == 1]["Level"].dropna()
    lows = shl[shl["HighLow"] == -1]["Level"].dropna()
    if len(highs):
        snap.swing_high = highs.iloc[-1]
    if len(lows):
        snap.swing_low = lows.iloc[-1]

    # Trend bias: compare last two swing highs and lows
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs.iloc[-1] > highs.iloc[-2]
        hl = lows.iloc[-1] > lows.iloc[-2]
        lh = highs.iloc[-1] < highs.iloc[-2]
        ll = lows.iloc[-1] < lows.iloc[-2]
        if hh and hl:
            snap.trend = Direction.LONG
        elif lh and ll:
            snap.trend = Direction.SHORT

    # ── BOS / CHoCH ──────────────────────────────────────────────────────
    struct = smc.bos_choch(ohlc, shl, close_break=True)
    # Find the most recent structure event
    bos_mask = struct["BOS"].notna() & (struct["BOS"] != 0)
    choch_mask = struct["CHOCH"].notna() & (struct["CHOCH"] != 0)
    all_struct = pd.concat([
        struct[bos_mask][["BOS", "Level", "BrokenIndex"]].rename(columns={"BOS": "val"}),
        struct[choch_mask][["CHOCH", "Level", "BrokenIndex"]].rename(columns={"CHOCH": "val"}),
    ]).sort_index()

    if len(all_struct):
        last = all_struct.iloc[-1]
        is_bos = last.name in struct[bos_mask].index
        snap.structure = StructureSignal(
            bos=Direction(int(last["val"])) if is_bos else None,
            choch=Direction(int(last["val"])) if not is_bos else None,
            level=float(last["Level"]),
            broken_bar=int(last["BrokenIndex"]) if not np.isnan(last["BrokenIndex"]) else -1,
        )

    # ── Fair Value Gaps ───────────────────────────────────────────────────
    fvg_df = smc.fvg(ohlc, join_consecutive=False)
    current_price = ohlc["close"].iloc[-1]

    for i, row in fvg_df.iterrows():
        if np.isnan(row["FVG"]):
            continue
        top, bottom = float(row["Top"]), float(row["Bottom"])
        if top <= 0 or bottom <= 0:
            continue
        size_pct = (top - bottom) / current_price
        if size_pct < min_fvg_size_pct:
            continue
        mitigated = not np.isnan(row["MitigatedIndex"]) and row["MitigatedIndex"] > 0
        sig = FVGSignal(
            direction=Direction(int(row["FVG"])),
            top=top,
            bottom=bottom,
            midpoint=(top + bottom) / 2,
            bar_index=ohlc.index.get_loc(i) if i in ohlc.index else -1,
            mitigated=mitigated,
        )
        if not mitigated:
            if row["FVG"] == 1:
                snap.bullish_fvgs.append(sig)
            else:
                snap.bearish_fvgs.append(sig)

    # ── Order Blocks ──────────────────────────────────────────────────────
    ob_df = smc.ob(ohlc, shl, close_mitigation=False)
    for i, row in ob_df.iterrows():
        if np.isnan(row["OB"]):
            continue
        top, bottom = float(row["Top"]), float(row["Bottom"])
        if top <= 0 or bottom <= 0:
            continue
        if float(row["Percentage"]) < min_ob_strength:
            continue
        mitigated = not np.isnan(row["MitigatedIndex"]) and row["MitigatedIndex"] > 0
        sig = OrderBlockSignal(
            direction=Direction(int(row["OB"])),
            top=top,
            bottom=bottom,
            strength_pct=float(row["Percentage"]),
            bar_index=ohlc.index.get_loc(i) if i in ohlc.index else -1,
            mitigated=mitigated,
        )
        if not mitigated:
            if row["OB"] == 1:
                snap.bullish_obs.append(sig)
            else:
                snap.bearish_obs.append(sig)

    # Keep only the most recent N OBs / FVGs (avoid stale noise)
    snap.bullish_fvgs = snap.bullish_fvgs[-5:]
    snap.bearish_fvgs = snap.bearish_fvgs[-5:]
    snap.bullish_obs = snap.bullish_obs[-3:]
    snap.bearish_obs = snap.bearish_obs[-3:]

    # ── Liquidity Pools ───────────────────────────────────────────────────
    liq_df = smc.liquidity(ohlc, shl, range_percent=liquidity_range_pct)
    for i, row in liq_df.iterrows():
        if np.isnan(row["Liquidity"]):
            continue
        swept = not np.isnan(row["Swept"]) and row["Swept"] > 0
        snap.liquidity_pools.append(LiquidityLevel(
            direction=Direction(int(row["Liquidity"])),
            level=float(row["Level"]),
            swept=swept,
            bar_index=ohlc.index.get_loc(i) if i in ohlc.index else -1,
        ))

    # ── Previous Day High / Low ───────────────────────────────────────────
    try:
        pdhl = smc.previous_high_low(ohlc, time_frame="1D")
        snap.prev_day_high = float(pdhl["PreviousHigh"].iloc[-1])
        snap.prev_day_low = float(pdhl["PreviousLow"].iloc[-1])
    except Exception:
        pass

    # ── Session ───────────────────────────────────────────────────────────
    try:
        sess = smc.sessions(ohlc, session=session, time_zone="UTC")
        snap.in_session = bool(sess["Active"].iloc[-1])
        snap.session_high = float(sess["High"].iloc[-1]) or None
        snap.session_low = float(sess["Low"].iloc[-1]) or None
    except Exception:
        pass

    return snap


def nearest_fvg_to_price(
    fvgs: list[FVGSignal], price: float, direction: Direction
) -> Optional[FVGSignal]:
    """Return the closest unmitigated FVG in the given direction to price."""
    candidates = [f for f in fvgs if f.direction == direction and not f.mitigated]
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs(f.midpoint - price))


def nearest_ob_to_price(
    obs: list[OrderBlockSignal], price: float, direction: Direction
) -> Optional[OrderBlockSignal]:
    """Return the closest unmitigated OB in the given direction to price."""
    candidates = [o for o in obs if o.direction == direction and not o.mitigated]
    if not candidates:
        return None
    return min(candidates, key=lambda o: abs(o.midpoint - price))
