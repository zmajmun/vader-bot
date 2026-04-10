"""
Market Data Feed
─────────────────
Provides:
  - Historical OHLCV bars (warmup on startup)
  - Live bar streaming via Alpaca WebSocket
  - Per-symbol ring buffers (deque) for rolling indicator windows
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Callable, Coroutine, Dict, List, Optional

import pandas as pd
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar

log = logging.getLogger(__name__)

# Type alias for bar callback
BarCallback = Callable[[str, pd.DataFrame], Coroutine]


class DataFeed:
    """
    Manages historical + live data for a list of symbols.

    Usage:
        feed = DataFeed(api_key, secret_key, symbols, broker, warmup_bars=500)
        await feed.warmup()
        feed.subscribe(on_bar_callback)
        await feed.stream()
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbols: List[str],
        broker,               # AlpacaBroker instance
        timeframe: str = "5Min",
        htf_timeframe: str = "1Hour",
        warmup_bars: int = 500,
        max_buffer: int = 2000,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self.symbols = symbols
        self._broker = broker
        self.timeframe = timeframe
        self.htf_timeframe = htf_timeframe
        self.warmup_bars = warmup_bars
        self.max_buffer = max_buffer

        # Ring buffers: symbol → deque of OHLCV rows
        self._ltf_buffers: Dict[str, deque] = {s: deque(maxlen=max_buffer) for s in symbols}
        self._htf_buffers: Dict[str, deque] = {s: deque(maxlen=max_buffer) for s in symbols}

        self._callbacks: List[BarCallback] = []
        self._stream: Optional[StockDataStream] = None

    # ──────────────────────────────────────────────────────────────────────
    #  Warmup
    # ──────────────────────────────────────────────────────────────────────

    async def warmup(self):
        """Load historical bars for all symbols before live trading starts."""
        log.info("Warming up data feed for %d symbols…", len(self.symbols))
        tasks = [self._warmup_symbol(s) for s in self.symbols]
        await asyncio.gather(*tasks)
        log.info("Warmup complete.")

    async def _warmup_symbol(self, symbol: str):
        try:
            ltf = await self._broker.get_bars(symbol, self.timeframe, limit=self.warmup_bars)
            for ts, row in ltf.iterrows():
                self._ltf_buffers[symbol].append({
                    "timestamp": ts,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                })
            log.info("  %s LTF: %d bars loaded", symbol, len(ltf))

            htf = await self._broker.get_bars(symbol, self.htf_timeframe, limit=self.warmup_bars)
            for ts, row in htf.iterrows():
                self._htf_buffers[symbol].append({
                    "timestamp": ts,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                })
            log.info("  %s HTF: %d bars loaded", symbol, len(htf))

        except Exception as e:
            log.error("Warmup failed for %s: %s", symbol, e)

    # ──────────────────────────────────────────────────────────────────────
    #  Live stream
    # ──────────────────────────────────────────────────────────────────────

    def subscribe(self, callback: BarCallback):
        self._callbacks.append(callback)

    async def stream(self):
        """Start the live WebSocket stream. Blocks until stopped."""
        self._stream = StockDataStream(self._api_key, self._secret_key)

        async def _on_bar(bar: Bar):
            symbol = bar.symbol
            if symbol not in self._ltf_buffers:
                return

            row = {
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            self._ltf_buffers[symbol].append(row)

            df = self.get_ltf(symbol)
            if df is None or len(df) < 20:
                return

            for cb in self._callbacks:
                try:
                    await cb(symbol, df)
                except Exception as e:
                    log.error("Callback error for %s: %s", symbol, e)

        self._stream.subscribe_bars(_on_bar, *self.symbols)
        log.info("Live stream started for: %s", self.symbols)
        await self._stream.run()

    async def stop(self):
        if self._stream:
            self._stream.stop()

    # ──────────────────────────────────────────────────────────────────────
    #  Buffer access
    # ──────────────────────────────────────────────────────────────────────

    def get_ltf(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._buf_to_df(self._ltf_buffers.get(symbol))

    def get_htf(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._buf_to_df(self._htf_buffers.get(symbol))

    @staticmethod
    def _buf_to_df(buf: Optional[deque]) -> Optional[pd.DataFrame]:
        if not buf:
            return None
        df = pd.DataFrame(list(buf))
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df = df[~df.index.duplicated(keep="last")]
        return df
