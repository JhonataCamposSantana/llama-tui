"""Curses modal-window scaffolding shared by the ``prompt_*`` helpers.

Second slice of audit #8 step 2. The previous slice (`ui_modals.py`) was
a flat extraction: it moved the helpers out of ``ui.py`` but left each
one open-coding the same setup/teardown:

  - ``stdscr.getmaxyx()`` size check (different floors per helper)
  - ``curses.newwin(...)`` centered placement
  - ``keypad(True)`` + ``stdscr.nodelay(False)`` so the modal can block
  - ``curses.curs_set`` save/restore for text-entry modals
  - ``stdscr.touchwin()`` + ``stdscr.nodelay(True)`` on every return path

This module owns that scaffolding via the ``open_modal`` context
manager. Migration is intentionally one-helper-at-a-time so behaviour
stays bit-identical — the audit calls this out specifically because
modal layout drift would surface as visible regressions.
"""

import curses
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class ModalSession:
    """Live curses sub-window plus the parent ``stdscr`` it was opened on."""
    stdscr: object
    window: object
    box_h: int
    box_w: int


@contextmanager
def open_modal(
    stdscr,
    *,
    box_h: int,
    box_w: int,
    min_h: Optional[int] = None,
    min_w: Optional[int] = None,
    show_cursor: bool = False,
) -> Iterator[Optional[ModalSession]]:
    """Yield a ``ModalSession`` for a centered curses sub-window.

    The caller computes ``box_h`` and ``box_w`` (the dimensions passed
    straight to ``curses.newwin``). ``min_h``/``min_w`` gate whether the
    terminal is large enough — defaulting to ``box_h + 1`` / ``box_w + 2``
    so the modal never overflows the screen. When the terminal is too
    small ``None`` is yielded so callers can bail without an
    ``except curses.error`` swamp.

    On exit, the previous cursor visibility (when ``show_cursor`` was
    requested) and the parent ``stdscr.nodelay`` state are restored, and
    the parent is told to redraw via ``touchwin``.
    """
    h, w = stdscr.getmaxyx()
    effective_min_h = int(min_h) if min_h is not None else int(box_h) + 1
    effective_min_w = int(min_w) if min_w is not None else int(box_w) + 2
    if h < effective_min_h or w < effective_min_w:
        yield None
        return
    box_x = max(1, (w - int(box_w)) // 2)
    box_y = max(1, (h - int(box_h)) // 2)
    window = curses.newwin(int(box_h), int(box_w), box_y, box_x)
    window.keypad(True)
    stdscr.nodelay(False)
    previous_cursor: Optional[int] = None
    if show_cursor:
        try:
            previous_cursor = curses.curs_set(1)
        except curses.error:
            previous_cursor = 0
    try:
        yield ModalSession(stdscr=stdscr, window=window, box_h=int(box_h), box_w=int(box_w))
    finally:
        if show_cursor and previous_cursor is not None:
            try:
                curses.curs_set(previous_cursor)
            except curses.error:
                pass
        try:
            stdscr.touchwin()
        except curses.error:
            pass
        stdscr.nodelay(True)
