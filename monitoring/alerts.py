"""
Alerts — Discord webhook and Telegram notifications.
All sends are best-effort (failures are logged, not raised).
"""
from __future__ import annotations

import logging
from typing import Optional

import aiohttp

from strategy.smc_strategy import TradeSetup
from execution.order_manager import ManagedTrade

log = logging.getLogger(__name__)


class Alerter:
    def __init__(
        self,
        discord_webhook: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ):
        self._discord = discord_webhook
        self._tg_token = telegram_token
        self._tg_chat = telegram_chat_id

    async def trade_entered(self, setup: TradeSetup, shares: int):
        direction = "LONG" if setup.is_long else "SHORT"
        emoji = "🟢" if setup.is_long else "🔴"
        msg = (
            f"{emoji} **{direction} ENTRY — {setup.symbol}**\n"
            f"Shares: {shares}\n"
            f"Entry:  ${setup.entry_price:.4f}\n"
            f"Stop:   ${setup.stop_price:.4f}\n"
            f"TP1:    ${setup.tp1_price:.4f} ({setup.rr_ratio * 0.5:.1f}R)\n"
            f"TP2:    ${setup.tp2_price:.4f} ({setup.rr_ratio:.1f}R)\n"
            f"Reason: {setup.reason}\n"
            f"Score:  {setup.confidence:.0f}/100"
        )
        await self._send(msg)

    async def tp1_hit(self, symbol: str, price: float, pnl: float):
        msg = f"🎯 **TP1 hit — {symbol}** @ ${price:.4f}  P&L: ${pnl:+.2f}"
        await self._send(msg)

    async def trade_closed(self, symbol: str, pnl: float, reason: str):
        emoji = "✅" if pnl >= 0 else "❌"
        msg = f"{emoji} **CLOSED — {symbol}**  P&L: ${pnl:+.2f}  ({reason})"
        await self._send(msg)

    async def daily_halt(self, daily_pnl: float):
        msg = f"🛑 **DAILY LOSS LIMIT HIT** — P&L: ${daily_pnl:.2f} — trading halted for today."
        await self._send(msg)

    async def _send(self, message: str):
        await self._discord_send(message)
        await self._telegram_send(message)

    async def _discord_send(self, message: str):
        if not self._discord:
            return
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"content": message}
                async with session.post(self._discord, json=payload) as resp:
                    if resp.status not in (200, 204):
                        log.warning("Discord alert failed: %s", resp.status)
        except Exception as e:
            log.warning("Discord send error: %s", e)

    async def _telegram_send(self, message: str):
        if not self._tg_token or not self._tg_chat:
            return
        # Strip markdown bold markers for Telegram plain mode
        plain = message.replace("**", "")
        url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"chat_id": self._tg_chat, "text": plain}
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        log.warning("Telegram alert failed: %s", resp.status)
        except Exception as e:
            log.warning("Telegram send error: %s", e)
