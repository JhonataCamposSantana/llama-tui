import threading
import time
from typing import Optional


class CancelledError(Exception):
    """Raised when a user-requested cancellation should unwind an action."""


class CancelToken:
    def __init__(self):
        self._event = threading.Event()
        self._reason = 'cancelled'

    def cancel(self, reason: str = 'cancelled'):
        self._reason = reason or 'cancelled'
        self._event.set()

    @property
    def reason(self) -> str:
        return self._reason

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self):
        if self.is_cancelled():
            raise CancelledError(self.reason)

    def wait(self, seconds: float) -> bool:
        return self._event.wait(max(0.0, seconds))


def check_cancelled(cancel_token: Optional[CancelToken]):
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()


def sleep_with_cancel(seconds: float, cancel_token: Optional[CancelToken] = None, step: float = 0.1):
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        check_cancelled(cancel_token)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        wait_for = min(max(0.01, step), remaining)
        if cancel_token is not None and cancel_token.wait(wait_for):
            check_cancelled(cancel_token)
        elif cancel_token is None:
            time.sleep(wait_for)
