"""Custom exception hierarchy for the trading system."""

from __future__ import annotations


class TradingSystemError(Exception):
    """Base class for application-specific errors."""


class ConfigError(ValueError, TradingSystemError):
    """Raised when configuration is missing, malformed, or unsafe."""


class BrokerError(TradingSystemError):
    """Raised for broker API and broker state failures."""


class RiskRejectedError(TradingSystemError):
    """Raised when risk checks reject a trade, strategy, or order."""


class DataValidationError(TradingSystemError):
    """Raised when incoming or persisted data fails validation."""


class OrderValidationError(TradingSystemError):
    """Raised when an order request is invalid before execution."""


class ExecutionError(TradingSystemError):
    """Raised when order execution fails after validation."""


class ModelError(TradingSystemError):
    """Raised for prediction, regime, or model artifact failures."""
