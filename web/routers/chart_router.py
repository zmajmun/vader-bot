"""
Chart Data API
──────────────
Returns OHLCV bars + fully computed SMC signals so the frontend
Lightweight Charts canvas can draw:
  • Candlesticks + volume
  • FVG zones (bullish / bearish)
  • Order Block zones (bullish / bearish)
  • BOS / CHoCH horizontal lines
  • Swing High / Low labels (HH, HL, LH, LL)
  • Previous Day High / Low
  • Liquidity pool levels
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from web.auth import get_current_user
from web.models import User

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chart", tags=["chart"])

# Map frontend TF codes → Alpaca timeframe strings
_TF_MAP = {
    "1":   "1Min",
    "3":   "5Min",    # Alpaca doesn't have 3-min; fall back to 5
    "5":   "5Min",
    "15":  "15Min",
    "30":  "30Min",
    "60":  "1Hour",
    "240": "4Hour",
    "D":   "1Day",
}


@router.get("/data/{symbol:path}")
async def chart_data(
    symbol: str,
    tf: str = "5",
    limit: int = 300,
    user: User = Depends(get_current_user),
):
    """
    Return OHLCV bars (unix-second timestamps) plus SMC signal overlays
    ready for consumption by the TradingView Lightweight Charts canvas.
    """
    if not user.alpaca_key or not user.alpaca_secret:
        raise HTTPException(
            status_code=400,
            detail="Alpaca API keys required — go to Settings → Connect Exchange.",
        )

    alpaca_tf = _TF_MAP.get(tf, "5Min")

    try:
        from core.signals import compute_snapshot
        from core.smc import smc as _smc
        from execution.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(
            user.alpaca_key, user.alpaca_secret, paper=user.alpaca_paper
        )
        df = await broker.get_bars(symbol, alpaca_tf, limit=limit)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No bars returned for {symbol}")

        # ── OHLCV bars for the chart ─────────────────────────────────────
        bars = [
            {
                "time":   int(ts.timestamp()),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row["volume"]),
            }
            for ts, row in df.iterrows()
        ]

        # bar_index → unix timestamp lookup (for zone time mapping)
        idx2ts = {i: int(df.index[i].timestamp()) for i in range(len(df))}
        last_ts = bars[-1]["time"] if bars else 0

        # ── SMC Snapshot ─────────────────────────────────────────────────
        snap = compute_snapshot(df, symbol)

        def _safe_ts(bar_idx: int) -> int:
            return idx2ts.get(bar_idx, last_ts)

        def fvg_dict(f):
            return {
                "top":        f.top,
                "bottom":     f.bottom,
                "midpoint":   f.midpoint,
                "time_start": _safe_ts(f.bar_index),
                "time_end":   last_ts,
                "direction":  int(f.direction),
            }

        def ob_dict(o):
            return {
                "top":          o.top,
                "bottom":       o.bottom,
                "strength_pct": o.strength_pct,
                "time_start":   _safe_ts(o.bar_index),
                "time_end":     last_ts,
                "direction":    int(o.direction),
            }

        def liq_dict(lv):
            return {
                "level":     lv.level,
                "direction": int(lv.direction),
                "swept":     lv.swept,
                "time":      _safe_ts(lv.bar_index),
            }

        # ── Swing points with HH/HL/LH/LL labels ────────────────────────
        swing_points: list[dict] = []
        try:
            shl_df = _smc.swing_highs_lows(df, swing_length=10)
            prev_h = prev_l = None
            for ts_idx, row in shl_df.iterrows():
                hl = row.get("HighLow")
                if pd.isna(hl):
                    continue
                lvl = float(row.get("Level", 0))
                if lvl <= 0:
                    continue
                ts_unix = int(ts_idx.timestamp()) if hasattr(ts_idx, "timestamp") else last_ts
                if int(hl) == 1:
                    label = "HH" if (prev_h is None or lvl > prev_h) else "LH"
                    prev_h = lvl
                    swing_points.append({"time": ts_unix, "price": lvl, "type": label, "dir": 1})
                elif int(hl) == -1:
                    label = "HL" if (prev_l is None or lvl > prev_l) else "LL"
                    prev_l = lvl
                    swing_points.append({"time": ts_unix, "price": lvl, "type": label, "dir": -1})
            swing_points = swing_points[-24:]   # keep last 24 labels
        except Exception as e:
            log.debug("Swing points error: %s", e)

        # ── Structure info ───────────────────────────────────────────────
        structure = None
        if snap.structure:
            s = snap.structure
            structure = {
                "bos":   int(s.bos)   if s.bos   else None,
                "choch": int(s.choch) if s.choch else None,
                "level": s.level,
                "time":  _safe_ts(s.broken_bar) if s.broken_bar >= 0 else None,
            }

        signals = {
            "bullish_fvgs":  [fvg_dict(f) for f in snap.bullish_fvgs],
            "bearish_fvgs":  [fvg_dict(f) for f in snap.bearish_fvgs],
            "bullish_obs":   [ob_dict(o)  for o in snap.bullish_obs],
            "bearish_obs":   [ob_dict(o)  for o in snap.bearish_obs],
            "swing_high":    snap.swing_high,
            "swing_low":     snap.swing_low,
            "swing_points":  swing_points,
            "pdh":           snap.prev_day_high,
            "pdl":           snap.prev_day_low,
            "trend":         int(snap.trend),
            "structure":     structure,
            "liquidity":     [liq_dict(lv) for lv in snap.liquidity_pools if not lv.swept][:12],
            "session_high":  snap.session_high,
            "session_low":   snap.session_low,
            "in_session":    snap.in_session,
        }

        return {"bars": bars, "signals": signals, "symbol": symbol, "tf": tf}

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Chart data error for %s: %s", symbol, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
