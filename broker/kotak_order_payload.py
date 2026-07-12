"""Pure Kotak order payload construction; this module never calls a broker."""

from __future__ import annotations

from typing import Any

from models import OrderRequest, OrderSide, OrderType
from utils.exceptions import OrderValidationError

_EXCHANGES = {"NSE", "BSE"}
_SEGMENTS = {"CASH", "CM", "FO", "F&O", "NFO"}
_PRODUCTS = {"MIS", "NRML", "CNC"}
_VALIDITIES = {"DAY", "IOC"}


def build_kotak_order_payload(order: OrderRequest) -> dict[str, Any]:
    """Validate and convert an OrderRequest to the stable Kotak adapter schema."""
    if not isinstance(order, OrderRequest):
        raise OrderValidationError("order must be an OrderRequest")
    meta = order.metadata
    exchange = str(meta.get("exchange", "")).upper()
    segment = str(meta.get("segment", "")).upper()
    token = str(meta.get("token", "")).strip()
    symbol = order.symbol.strip()
    lot_size = meta.get("lot_size")
    disclosed = meta.get("disclosed_quantity", 0)
    if exchange not in _EXCHANGES:
        raise OrderValidationError("Unsupported exchange")
    if segment not in _SEGMENTS:
        raise OrderValidationError("Unsupported segment")
    if not token or not symbol:
        raise OrderValidationError("token and symbol are required")
    if order.side not in {OrderSide.BUY, OrderSide.SELL}:
        raise OrderValidationError("Unsupported side")
    if order.quantity <= 0 or not isinstance(lot_size, int) or lot_size <= 0 or order.quantity % lot_size:
        raise OrderValidationError("quantity must be a positive lot multiple")
    product = order.product_type.upper()
    validity = order.validity.upper()
    if product not in _PRODUCTS or validity not in _VALIDITIES:
        raise OrderValidationError("Unsupported product type or validity")
    if order.order_type == OrderType.MARKET and not order.is_emergency_exit:
        raise OrderValidationError("MARKET is restricted to emergency exits")
    if order.order_type not in {OrderType.LIMIT, OrderType.SL_LIMIT, OrderType.MARKET}:
        raise OrderValidationError("Unsupported order type")
    if order.order_type in {OrderType.LIMIT, OrderType.SL_LIMIT} and (order.limit_price is None or order.limit_price <= 0):
        raise OrderValidationError("positive limit_price is required")
    if order.order_type == OrderType.SL_LIMIT:
        if order.trigger_price is None or order.trigger_price <= 0:
            raise OrderValidationError("positive trigger_price is required")
        if order.side == OrderSide.BUY and order.limit_price < order.trigger_price:
            raise OrderValidationError("BUY SL-LIMIT limit must be >= trigger")
        if order.side == OrderSide.SELL and order.limit_price > order.trigger_price:
            raise OrderValidationError("SELL SL-LIMIT limit must be <= trigger")
    if not isinstance(disclosed, int) or disclosed < 0 or disclosed > order.quantity:
        raise OrderValidationError("invalid disclosed quantity")
    return {
        "exchange_segment": f"{exchange}_{segment}", "instrument_token": token,
        "trading_symbol": symbol, "transaction_type": order.side.value,
        "quantity": order.quantity, "order_type": "SL" if order.order_type == OrderType.SL_LIMIT else order.order_type.value,
        "price": 0.0 if order.order_type == OrderType.MARKET else float(order.limit_price),
        "trigger_price": float(order.trigger_price or 0.0), "product": product,
        "validity": validity, "disclosed_quantity": disclosed, "tag": order.tag,
    }


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a log-safe payload without credentials or approval material."""
    sensitive = {"risk_approval_token", "order_validator_approval_token", "authorization", "api_key"}
    return {key: ("***" if key.lower() in sensitive else value) for key, value in payload.items()}
