"""
Retry policies and backpressure handling for event processing
"""
import time
import logging
from typing import Callable, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BackpressureError(Exception):
    """Raised when backpressure threshold is exceeded"""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    backoff_seconds: List[float] = None
    exponential_backoff: bool = True
    max_backoff_seconds: float = 60.0

    def __post_init__(self):
        if self.backoff_seconds is None:
            self.backoff_seconds = [1, 5, 15]


class RetryPolicy:
    """
    Handles retry logic with exponential backoff and DLQ
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.retry_counts = {}  # message_id -> retry_count

    def should_retry(self, message_id: str, attempt: int) -> bool:
        """
        Determine if message should be retried

        Args:
            message_id: Unique message identifier
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry, False if should go to DLQ
        """
        return attempt < self.config.max_attempts

    def get_backoff_delay(self, attempt: int) -> float:
        """
        Calculate backoff delay for retry attempt

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds before retry
        """
        if self.config.exponential_backoff:
            # Exponential backoff: 2^attempt seconds
            delay = min(2 ** attempt, self.config.max_backoff_seconds)
        else:
            # Use configured backoff schedule
            if attempt < len(self.config.backoff_seconds):
                delay = self.config.backoff_seconds[attempt]
            else:
                delay = self.config.backoff_seconds[-1]

        return delay

    def execute_with_retry(
        self,
        func: Callable,
        message_id: str,
        *args,
        **kwargs
    ) -> bool:
        """
        Execute function with retry logic

        Args:
            func: Function to execute
            message_id: Unique message identifier for tracking
            *args, **kwargs: Arguments to pass to function

        Returns:
            True if successful, False if all retries exhausted

        Raises:
            Exception: Re-raises exception if all retries fail
        """
        attempt = 0
        last_exception = None

        while attempt < self.config.max_attempts:
            try:
                func(*args, **kwargs)
                # Success - reset retry count
                if message_id in self.retry_counts:
                    del self.retry_counts[message_id]
                return True

            except Exception as e:
                last_exception = e
                attempt += 1
                self.retry_counts[message_id] = attempt

                if not self.should_retry(message_id, attempt):
                    logger.error(
                        f"Max retries ({self.config.max_attempts}) exceeded for {message_id}",
                        exc_info=True
                    )
                    raise e

                backoff = self.get_backoff_delay(attempt)
                logger.warning(
                    f"Retry {attempt}/{self.config.max_attempts} for {message_id} "
                    f"after {backoff}s. Error: {str(e)}"
                )
                time.sleep(backoff)

        # All retries exhausted
        if last_exception:
            raise last_exception

        return False


class BackpressureManager:
    """
    Manages backpressure to prevent overwhelming downstream services
    """

    def __init__(
        self,
        max_pending: int = 1000,
        block_threshold: int = 800,
        enabled: bool = True
    ):
        """
        Args:
            max_pending: Maximum pending messages before hard block
            block_threshold: Threshold to start blocking new messages
            enabled: Whether backpressure is enabled
        """
        self.max_pending = max_pending
        self.block_threshold = block_threshold
        self.enabled = enabled

    def check_backpressure(self, current_pending: int) -> None:
        """
        Check if backpressure should be applied

        Args:
            current_pending: Current number of pending messages

        Raises:
            BackpressureError: If backpressure threshold exceeded
        """
        if not self.enabled:
            return

        if current_pending >= self.max_pending:
            raise BackpressureError(
                f"Hard limit reached: {current_pending}/{self.max_pending} pending messages"
            )

        if current_pending >= self.block_threshold:
            logger.warning(
                f"Backpressure activated: {current_pending}/{self.max_pending} pending messages"
            )
            raise BackpressureError(
                f"Backpressure threshold exceeded: {current_pending}/{self.block_threshold}"
            )

    def should_block(self, current_pending: int) -> bool:
        """
        Check if should block new messages (without raising exception)

        Returns:
            True if should block, False otherwise
        """
        if not self.enabled:
            return False

        return current_pending >= self.block_threshold
