"""ActionRunner: thread + CancelToken pair used by ``tui()`` background workers.

First extraction of audit finding #8 (split ``ui.py:tui()``). The
6.4k-line ``tui()`` function juggles two background worker lifecycles
(the benchmark/action thread and the try-out chat thread) plus their
``CancelToken``s as four separate closure-local variables. That made
the closure noisy (every helper had to ``nonlocal`` four names) and
hid the abstraction: each pair is really one ``ActionRunner``.

This module is intentionally minimal — just enough to collapse those
four closure variables into two ``ActionRunner`` instances. The
``shutdown_workers`` helper from audit finding #5 keeps accepting bare
threads/tokens via its interleaved-args interface; you call it with
``runner.token, runner.thread`` and the existing logic does the right
thing because ActionRunner doesn't change the underlying types.
"""

import threading
from dataclasses import dataclass
from typing import Optional

from .control import CancelToken


@dataclass
class ActionRunner:
    """Owns the (thread, token) pair for one background worker slot.

    Attributes are mutated directly by callers — there's no ``start``
    method because the existing call sites already build the
    ``threading.Thread`` with their own ``target=runner`` closure. This
    class only exists to:

    * give the pair a name so the closure carries one slot instead of
      two unrelated variables;
    * provide ``is_running()`` so the predicate is testable in isolation
      and reused across views;
    * provide ``cancel()`` that's safe against ``None`` and any internal
      cancel-token errors (matches the swallow-and-continue idiom the
      shutdown path already uses);
    * provide ``reset()`` to clear the slot when a worker finishes or
      a view is left.
    """

    thread: Optional[threading.Thread] = None
    token: Optional[CancelToken] = None

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def cancel(self, reason: str = 'cancelled') -> None:
        if self.token is None:
            return
        try:
            self.token.cancel(reason)
        except Exception:
            pass

    def reset(self) -> None:
        self.thread = None
        self.token = None
