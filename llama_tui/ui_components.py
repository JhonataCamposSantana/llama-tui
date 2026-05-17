"""Reusable curses rendering primitives for the llama-tui cockpit UI.

Keeping these here (instead of inline in ui.py) lets panels share a single
look for boxes, cards, badges, status chips and key-hint bars.
"""

import curses
from typing import Dict, List, Sequence, Tuple

from .textutil import ellipsize
from .ui_theme import kind_style, style


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


def draw_section_title(stdscr, y: int, x: int, w: int, title: str, colors: Dict[str, int]):
    """Render a lightweight section heading with a trailing rule."""
    label = f' {str(title or "").strip().upper()} '
    safe_addstr(stdscr, y, x, ellipsize(label, max(1, w)), style(colors, 'active', bold=True))
    rule_start = x + min(len(label), max(0, w))
    for i in range(rule_start, x + w):
        safe_addch(stdscr, y, i, curses.ACS_HLINE, style(colors, 'muted'))


def draw_badge(stdscr, y: int, x: int, label: str, attr: int = 0) -> int:
    """Render a ``[label]`` badge. Returns the x cursor after the badge."""
    text = f'[{str(label or "").strip()}]'
    safe_addstr(stdscr, y, x, text, attr)
    return x + len(text) + 1


def draw_status_chip(stdscr, y: int, x: int, label: str, attr: int = 0) -> int:
    """Render a padded status chip (e.g. run state). Returns the next x."""
    text = f' {str(label or "").strip()} '
    safe_addstr(stdscr, y, x, text, attr)
    return x + len(text) + 1


def draw_key_hint_bar(stdscr, y: int, x: int, w: int, hints: Sequence[Tuple[str, str]], colors: Dict[str, int]):
    """Render a ``[k] label`` hint bar, stopping when it runs out of width."""
    cursor = x
    limit = x + max(0, w)
    for key, label in hints or ():
        chunk = f'[{key}] {label}'
        if cursor + len(chunk) + 1 > limit:
            break
        safe_addstr(stdscr, y, cursor, f'[{key}]', style(colors, 'active', bold=True))
        safe_addstr(stdscr, y, cursor + len(key) + 2, f' {label}', style(colors, 'muted'))
        cursor += len(chunk) + 2


def draw_card(
    stdscr,
    y: int,
    x: int,
    h: int,
    w: int,
    title: str,
    lines: Sequence[object],
    colors: Dict[str, int],
):
    """Render a titled card box with wrapped, kind-styled content lines."""
    if h < 3 or w < 6:
        return
    draw_box(stdscr, y, x, h - 1, w, title, style(colors, 'active', bold=True), style(colors, 'muted'))
    inner_w = max(1, w - 4)
    body_rows = max(0, h - 3)
    for idx, (text, kind) in enumerate(wrap_card_lines(lines, inner_w)[:body_rows]):
        safe_addstr(stdscr, y + 2 + idx, x + 2, text, kind_style(colors, kind))


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
