#!/usr/bin/env python3
"""
VADER — SMC Algorithmic Trading Bot
────────────────────────────────────
Usage:
    python main.py                    # paper mode (from .env / settings.yaml)
    ALPACA_PAPER=false python main.py # live mode  ⚠  real money

Press Ctrl+C to stop gracefully.
"""
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))
from logs.logger import setup_logging

# Load config
with open(ROOT / "config" / "settings.yaml") as f:
    cfg = yaml.safe_load(f)

bot_cfg = cfg["bot"]
setup_logging(log_level=bot_cfg.get("log_level", "INFO"), log_dir=str(ROOT / "logs"))
log = logging.getLogger("vader.main")

# ── Imports (after path setup) ────────────────────────────────────────────
from core.engine import TradingEngine
from data.feed import DataFeed
from execution.alpaca_broker import AlpacaBroker
from execution.order_manager import OrderManager
from monitoring.alerts import Alerter
from monitoring.dashboard import Dashboard
from risk.position_sizer import PositionSizer, RiskConfig
from strategy.smc_strategy import SMCStrategy, StrategyConfig


def build_engine() -> TradingEngine:
    # ── Credentials ───────────────────────────────────────────────────────
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        log.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        sys.exit(1)

    paper_env = os.environ.get("ALPACA_PAPER", "true").lower()
    paper = paper_env not in ("false", "0", "no", "live")

    if not paper:
        log.warning("=" * 60)
        log.warning("  ⚠  LIVE MODE ENABLED — REAL MONEY AT RISK  ⚠")
        log.warning("  Make sure you have tested thoroughly in paper mode.")
        log.warning("=" * 60)

    # ── Config sections ───────────────────────────────────────────────────
    uni_cfg  = cfg["universe"]
    smc_cfg  = cfg["smc"]
    strat_cfg = cfg["strategy"]
    risk_cfg = cfg["risk"]
    exec_cfg = cfg["execution"]

    symbols = uni_cfg["symbols"]

    # ── Build dependencies ────────────────────────────────────────────────
    broker = AlpacaBroker(api_key, secret_key, paper=paper)

    feed = DataFeed(
        api_key=api_key,
        secret_key=secret_key,
        symbols=symbols,
        broker=broker,
        timeframe=uni_cfg["timeframe"],
        htf_timeframe=uni_cfg["htf_timeframe"],
        warmup_bars=uni_cfg["warmup_bars"],
    )

    strategy = SMCStrategy(StrategyConfig(
        require_choch=strat_cfg["require_choch"],
        htf_trend_filter=strat_cfg["htf_trend_filter"],
        min_rr=strat_cfg["min_rr"],
        tp1_r=strat_cfg["tp1_r"],
        tp2_r=strat_cfg["tp2_r"],
        stop_buffer_pct=risk_cfg["stop_buffer_pct"],
    ))

    order_mgr = OrderManager(
        broker=broker,
        fill_timeout=exec_cfg["fill_timeout"],
        limit_offset_pct=exec_cfg["limit_offset_pct"],
        use_limit_entry=exec_cfg["limit_entry"],
        fallback_to_market=exec_cfg["fallback_to_market"],
        tp1_size=strat_cfg["tp1_size"],
        be_after_tp1=strat_cfg["be_after_tp1"],
    )

    sizer = PositionSizer(RiskConfig(
        account_risk_pct=float(os.environ.get("ACCOUNT_RISK_PCT", risk_cfg["account_risk_pct"])),
        max_position_pct=risk_cfg["max_position_pct"],
        max_positions=int(os.environ.get("MAX_POSITIONS", risk_cfg["max_positions"])),
        max_daily_loss_pct=float(os.environ.get("MAX_DAILY_LOSS_PCT", risk_cfg["max_daily_loss_pct"])),
    ))

    dashboard = Dashboard(symbols=symbols, sizer=sizer, paper=paper)

    alerter = Alerter(
        discord_webhook=os.environ.get("DISCORD_WEBHOOK_URL") or None,
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None,
    )

    engine = TradingEngine(
        broker=broker,
        feed=feed,
        strategy=strategy,
        order_manager=order_mgr,
        sizer=sizer,
        dashboard=dashboard,
        alerter=alerter,
        symbols=symbols,
        max_trades_per_day=strat_cfg["max_trades_per_day"],
        htf_timeframe=uni_cfg["htf_timeframe"],
        swing_length=smc_cfg["swing_length"],
        min_fvg_size_pct=smc_cfg["min_fvg_size_pct"],
        min_ob_strength=smc_cfg["min_ob_strength"],
        liquidity_range_pct=smc_cfg["liquidity_range_pct"],
        session=strat_cfg["session"],
        paper=paper,
    )
    return engine


async def main():
    engine = build_engine()

    loop = asyncio.get_event_loop()

    def _shutdown():
        log.info("Shutdown signal received — stopping gracefully…")
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    await engine.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted — goodbye.")
