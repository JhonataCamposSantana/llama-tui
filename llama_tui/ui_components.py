"""Reusable curses rendering primitives for the llama-tui cockpit UI.

Keeping these here (instead of inline in ui.py) lets panels share a single
look for boxes, cards, badges, status chips and key-hint bars.
"""

import curses
from typing import Dict, List, Sequence, Tuple

from .textutil import ellipsize
from .ui_theme import kind_style, style


# Kind -> leading symbol mapping for status-coloured lines. Audit finding
# #16: status chips in the MTP Doctor / Benchmark Plan / similar overlays
# previously relied on color alone to distinguish success/warning/error.
# Prefixing a small ASCII-safe glyph keeps the signal visible on dumb
# terminals, in screenshots, and for color-blind users.
KIND_STATUS_SYMBOLS: Dict[str, str] = {
    'success': '✓',
    'error': '✗',
    'warning': '⚠',
    'muted': '·',
}


def kind_status_symbol(kind: str) -> str:
    """Return a one-character glyph for ``kind``, or '' for heading/normal."""
    return KIND_STATUS_SYMBOLS.get(str(kind or '').strip().lower(), '')


def kind_status_prefix(text: str, kind: str) -> str:
    """Prepend a kind-derived symbol to ``text`` when one applies.

    Headings and plain ``normal`` lines pass through unchanged so layout
    is preserved.
    """
    symbol = kind_status_symbol(kind)
    if not symbol:
        return text
    return f'{symbol} {text}'


def safe_addch(stdscr, y: int, x: int, ch, attr: int = 0):
    try:
        stdscr.addch(y, x, ch, attr)
    except curses.error:
        pass


def safe_addstr(stdscr, y: int, x: int, text: str, attr: int = 0):
    try:
        stdscr.addstr(y, x, text, attr)
    except curses.error:
        pass


def truncate(text: str, width: int) -> str:
    """Hard-cut a string to ``width`` columns (no ellipsis)."""
    if width <= 0:
        return ''
    return str(text or '')[:width]


def wrap_card_lines(lines: Sequence[object], width: int) -> List[Tuple[str, str]]:
    """Normalize and hard-wrap card content lines to ``width`` columns.

    Each input line may be a plain string or a ``(text, kind)`` tuple. The
    result is a flat list of ``(text, kind)`` pairs where every text fits
    within ``width``. Pure/string-only so it is unit testable.
    """
    width = max(1, int(width or 1))
    wrapped: List[Tuple[str, str]] = []
    for entry in lines or ():
        if isinstance(entry, (tuple, list)):
            text = str(entry[0] if len(entry) > 0 else '')
            kind = str(entry[1] if len(entry) > 1 else 'normal')
        else:
            text, kind = str(entry), 'normal'
        if not text:
            wrapped.append(('', kind))
            continue
        while len(text) > width:
            wrapped.append((text[:width], kind))
            text = text[width:]
        wrapped.append((text, kind))
    return wrapped


def draw_box(stdscr, y: int, x: int, h: int, w: int, title: str, title_attr: int = curses.A_BOLD, border_attr: int = 0):
    if h < 2 or w < 4:
        return
    safe_addstr(stdscr, y, x + 2, f' {title} ', title_attr)
    for i in range(x, x + w):
        safe_addch(stdscr, y + 1, i, curses.ACS_HLINE, border_attr)
    for i in range(y + 1, y + h):
        safe_addch(stdscr, i, x, curses.ACS_VLINE, border_attr)
        safe_addch(stdscr, i, x + w - 1, curses.ACS_VLINE, border_attr)
    safe_addch(stdscr, y + 1, x, curses.ACS_ULCORNER, border_attr)
    safe_addch(stdscr, y + 1, x + w - 1, curses.ACS_URCORNER, border_attr)
    safe_addch(stdscr, y + h, x, curses.ACS_LLCORNER, border_attr)
    safe_addch(stdscr, y + h, x + w - 1, curses.ACS_LRCORNER, border_attr)
    for i in range(x + 1, x + w - 1):
        safe_addch(stdscr, y + h, i, curses.ACS_HLINE, border_attr)












_SPARK_CHARS = '▁▂▃▄▅▆▇█'


def sparkline(values: Sequence[float], width: int) -> str:
    """Render a numeric series as a fixed-width Unicode block sparkline.

    Pure/string-only so it is unit testable. Right-aligned to the most recent
    ``width`` samples; a flat series renders as a mid-height band.
    """
    width = max(0, int(width or 0))
    nums = [float(v) for v in (values or []) if v is not None]
    if not nums or width <= 0:
        return ' ' * width
    series = nums[-width:]
    lo, hi = min(series), max(series)
    span = hi - lo
    if span <= 0:
        return _SPARK_CHARS[len(_SPARK_CHARS) // 2] * len(series)
    cells = []
    for value in series:
        idx = int(round((value - lo) / span * (len(_SPARK_CHARS) - 1)))
        cells.append(_SPARK_CHARS[max(0, min(len(_SPARK_CHARS) - 1, idx))])
    return ''.join(cells)


def gauge_bar(fraction: float, width: int) -> str:
    """Return a ``[||||    ]`` style bar of the given inner width. Pure."""
    width = max(0, int(width or 0))
    if width <= 0:
        return ''
    frac = max(0.0, min(1.0, float(fraction or 0.0)))
    filled = int(round(frac * width))
    return '[' + ('|' * filled) + (' ' * (width - filled)) + ']'
