"""Retry utilities for execution operations."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import sleep
from typing import Callable, Generic, TypeVar

from utils.exceptions import ExecutionError

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class RetryablePreSubmissionError(ExecutionError):
    """Failure proven to have occurred before a broker submission call."""


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy for broker/execution calls."""

    attempts: int = 3
    backoff_seconds: float = 1.0
    retry_exceptions: tuple[type[Exception], ...] = (RetryablePreSubmissionError,)


class RetryHandler(Generic[T]):
    """Small retry wrapper with structured logging."""

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        """Create a retry handler."""

        self.policy = policy or RetryPolicy()

    def run(self, operation: Callable[[], T], *, operation_name: str) -> T:
        """Run operation with retry policy."""

        last_error: Exception | None = None
        attempts = max(1, self.policy.attempts)
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except self.policy.retry_exceptions as exc:
                last_error = exc
                LOGGER.warning(
                    "Execution operation failed",
                    extra={"operation": operation_name, "attempt": attempt, "attempts": attempts},
                )
                if attempt < attempts and self.policy.backoff_seconds > 0:
                    sleep(self.policy.backoff_seconds)
        raise ExecutionError(f"{operation_name} failed after {attempts} attempts") from last_error
