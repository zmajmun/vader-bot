"""
Abstract Broker Interface
All execution adapters must implement this protocol.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType
    status: OrderStatus
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_qty: int = 0
    filled_avg_price: Optional[float] = None
    client_order_id: Optional[str] = None


@dataclass
class Position:
    symbol: str
    qty: int           # positive = long, negative = short
    avg_entry: float
    market_value: float
    unrealized_pnl: float
    side: str          # "long" | "short"


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    daily_pnl: float


class BrokerBase(ABC):
    """Abstract broker — implement for Alpaca, IBKR, TD Ameritrade, etc."""

    @abstractmethod
    async def get_account(self) -> AccountInfo: ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]: ...

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    @abstractmethod
    async def submit_market_order(
        self, symbol: str, qty: int, side: OrderSide, tif: TimeInForce = TimeInForce.DAY
    ) -> Order: ...

    @abstractmethod
    async def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        limit_price: float,
        tif: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
    ) -> Order: ...

    @abstractmethod
    async def submit_stop_order(
        self, symbol: str, qty: int, side: OrderSide, stop_price: float,
        tif: TimeInForce = TimeInForce.GTC,
    ) -> Order: ...

    @abstractmethod
    async def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        entry_price: Optional[float],   # None = market
        stop_price: float,
        take_profit_price: float,
        client_order_id: Optional[str] = None,
    ) -> Order: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    async def close_position(self, symbol: str) -> Optional[Order]: ...

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]: ...

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> "pd.DataFrame": ...

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> float: ...
