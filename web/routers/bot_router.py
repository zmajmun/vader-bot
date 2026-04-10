"""
Bot control and status API endpoints.
Provides start/stop, live status, and trade history.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from web.auth import get_current_user
from web.models import SessionLocal, TradeLog, User, get_db

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bot", tags=["bot"])

# ── In-memory bot registry (one engine per user_id) ────────────────────────
# In production: use Redis or a process manager (Celery, etc.)
_bot_tasks: Dict[int, asyncio.Task] = {}
_bot_status: Dict[int, dict] = {}


# ── Schemas ────────────────────────────────────────────────────────────────

class BotStatusResponse(BaseModel):
    running: bool
    mode: str                   # paper | live
    symbols: List[str]
    equity: Optional[float]
    daily_pnl: Optional[float]
    open_positions: int
    signals: Dict[str, str]
    started_at: Optional[str]


class StartBotRequest(BaseModel):
    symbols: Optional[List[str]] = None  # override config symbols


class TradeResponse(BaseModel):
    id: int
    symbol: str
    direction: str
    shares: int
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    status: str
    reason: Optional[str]
    confidence: Optional[float]
    opened_at: str
    closed_at: Optional[str]


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/start")
async def start_bot(
    body: StartBotRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not user.alpaca_key or not user.alpaca_secret:
        raise HTTPException(
            status_code=400,
            detail="Alpaca API keys not configured. Go to Settings → Connect Exchange."
        )

    if user.id in _bot_tasks and not _bot_tasks[user.id].done():
        raise HTTPException(status_code=409, detail="Bot is already running")

    # Build and start engine for this user
    task = asyncio.create_task(_run_bot_for_user(user, body.symbols))
    _bot_tasks[user.id] = task
    _bot_status[user.id] = {
        "running": True,
        "mode": "paper" if user.alpaca_paper else "live",
        "symbols": body.symbols or [],
        "equity": None,
        "daily_pnl": None,
        "open_positions": 0,
        "signals": {},
        "started_at": datetime.utcnow().isoformat(),
    }

    mode = "paper" if user.alpaca_paper else "LIVE"
    return {"message": f"Bot started ({mode} mode)", "started_at": _bot_status[user.id]["started_at"]}


@router.post("/stop")
async def stop_bot(user: User = Depends(get_current_user)):
    task = _bot_tasks.get(user.id)
    if not task or task.done():
        raise HTTPException(status_code=404, detail="No running bot found")
    task.cancel()
    _bot_tasks.pop(user.id, None)
    status = _bot_status.get(user.id, {})
    status["running"] = False
    return {"message": "Bot stopped"}


@router.get("/status", response_model=BotStatusResponse)
def bot_status(user: User = Depends(get_current_user)):
    task = _bot_tasks.get(user.id)
    running = task is not None and not task.done()
    status = _bot_status.get(user.id, {})
    return BotStatusResponse(
        running=running,
        mode=status.get("mode", "paper"),
        symbols=status.get("symbols", []),
        equity=status.get("equity"),
        daily_pnl=status.get("daily_pnl"),
        open_positions=status.get("open_positions", 0),
        signals=status.get("signals", {}),
        started_at=status.get("started_at"),
    )


@router.get("/trades", response_model=List[TradeResponse])
def get_trades(
    limit: int = 50,
    days: int = 30,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    trades = (
        db.query(TradeLog)
        .filter(TradeLog.user_id == user.id, TradeLog.opened_at >= since)
        .order_by(TradeLog.opened_at.desc())
        .limit(limit)
        .all()
    )
    return [
        TradeResponse(
            id=t.id,
            symbol=t.symbol,
            direction=t.direction,
            shares=t.shares,
            entry_price=t.entry_price,
            stop_price=t.stop_price,
            tp1_price=t.tp1_price,
            tp2_price=t.tp2_price,
            exit_price=t.exit_price,
            pnl=t.pnl,
            status=t.status,
            reason=t.reason,
            confidence=t.confidence,
            opened_at=t.opened_at.isoformat(),
            closed_at=t.closed_at.isoformat() if t.closed_at else None,
        )
        for t in trades
    ]


@router.get("/stats")
def get_stats(
    days: int = 30,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(days=days)
    trades = (
        db.query(TradeLog)
        .filter(
            TradeLog.user_id == user.id,
            TradeLog.opened_at >= since,
            TradeLog.status == "closed",
        )
        .all()
    )
    if not trades:
        return {"message": "No closed trades yet", "trades": 0}

    wins = [t for t in trades if (t.pnl or 0) > 0]
    losses = [t for t in trades if (t.pnl or 0) <= 0]
    total_pnl = sum(t.pnl or 0 for t in trades)
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else 0

    return {
        "period_days": days,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "best_trade": round(max((t.pnl or 0) for t in trades), 2),
        "worst_trade": round(min((t.pnl or 0) for t in trades), 2),
    }


# ── Background bot runner ──────────────────────────────────────────────────

async def _run_bot_for_user(user: User, symbol_override: Optional[List[str]]):
    """Builds and runs a TradingEngine instance for a specific user."""
    import yaml
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from core.engine import TradingEngine
    from data.feed import DataFeed
    from execution.alpaca_broker import AlpacaBroker
    from execution.order_manager import OrderManager, ManagedTrade
    from monitoring.alerts import Alerter
    from monitoring.dashboard import Dashboard
    from risk.position_sizer import PositionSizer, RiskConfig
    from strategy.smc_strategy import SMCStrategy, StrategyConfig, TradeSetup

    class LoggingOrderManager(OrderManager):
        """Extends OrderManager to persist trades to the database."""

        def __init__(self, user_id: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._user_id = user_id
            self._trade_log_ids: Dict[str, int] = {}  # symbol → TradeLog.id

        async def enter(self, setup: TradeSetup, shares: int) -> Optional[ManagedTrade]:
            trade = await super().enter(setup, shares)
            if trade is None:
                return None
            db = SessionLocal()
            try:
                row = TradeLog(
                    user_id=self._user_id,
                    symbol=setup.symbol,
                    direction="LONG" if setup.is_long else "SHORT",
                    shares=shares,
                    entry_price=setup.entry_price,
                    stop_price=setup.stop_price,
                    tp1_price=setup.tp1_price,
                    tp2_price=setup.tp2_price,
                    status="open",
                    reason=setup.reason,
                    confidence=setup.confidence,
                    opened_at=datetime.utcnow(),
                )
                db.add(row)
                db.commit()
                db.refresh(row)
                self._trade_log_ids[setup.symbol] = row.id
            except Exception as e:
                log.error("Failed to persist trade entry for %s: %s", setup.symbol, e)
                db.rollback()
            finally:
                db.close()
            return trade

        async def close(self, symbol: str, reason: str = "manual"):
            trade = self._trades.get(symbol)
            exit_price = None
            pnl = None
            if trade:
                exit_price = trade.setup.tp2_price if trade.tp1_hit else trade.setup.stop_price
                pnl = trade.realized_pnl
            await super().close(symbol, reason)
            log_id = self._trade_log_ids.pop(symbol, None)
            if log_id:
                db = SessionLocal()
                try:
                    row = db.query(TradeLog).filter(TradeLog.id == log_id).first()
                    if row:
                        row.exit_price = exit_price
                        row.pnl = pnl
                        row.status = "closed"
                        row.closed_at = datetime.utcnow()
                        db.commit()
                except Exception as e:
                    log.error("Failed to persist trade close for %s: %s", symbol, e)
                    db.rollback()
                finally:
                    db.close()

    cfg_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    uni_cfg = cfg["universe"]
    smc_cfg = cfg["smc"]
    strat_cfg = cfg["strategy"]
    risk_cfg = cfg["risk"]
    exec_cfg = cfg["execution"]

    symbols = symbol_override or uni_cfg["symbols"]
    paper = user.alpaca_paper

    broker = AlpacaBroker(user.alpaca_key, user.alpaca_secret, paper=paper)
    feed = DataFeed(
        api_key=user.alpaca_key,
        secret_key=user.alpaca_secret,
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
    ))
    order_mgr = LoggingOrderManager(
        user_id=user.id,
        broker=broker,
        fill_timeout=exec_cfg["fill_timeout"],
        limit_offset_pct=exec_cfg["limit_offset_pct"],
        use_limit_entry=exec_cfg["limit_entry"],
        fallback_to_market=exec_cfg["fallback_to_market"],
        tp1_size=strat_cfg["tp1_size"],
        be_after_tp1=strat_cfg["be_after_tp1"],
    )
    sizer = PositionSizer(RiskConfig(
        account_risk_pct=risk_cfg["account_risk_pct"],
        max_position_pct=risk_cfg["max_position_pct"],
        max_positions=risk_cfg["max_positions"],
        max_daily_loss_pct=risk_cfg["max_daily_loss_pct"],
    ))

    # Alerter (no Discord/Telegram per-user for now — extend as needed)
    alerter = Alerter()

    # Headless dashboard (web mode — no Rich terminal)
    class WebDashboard:
        def start(self): pass
        def stop(self): pass
        def update(self, equity, prices, trades, signals):
            status = _bot_status.get(user.id, {})
            status.update({
                "equity": equity,
                "daily_pnl": sizer.daily_pnl,
                "open_positions": len(trades),
                "signals": signals,
                "prices": prices,
            })
            _bot_status[user.id] = status

    engine = TradingEngine(
        broker=broker, feed=feed, strategy=strategy,
        order_manager=order_mgr, sizer=sizer,
        dashboard=WebDashboard(), alerter=alerter,
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

    try:
        await engine.start()
    except asyncio.CancelledError:
        await engine.stop()
        log.info("Bot stopped for user %d", user.id)
    except Exception as e:
        log.error("Bot crashed for user %d: %s", user.id, e)
        _bot_status[user.id]["running"] = False
        _bot_status[user.id]["error"] = str(e)
