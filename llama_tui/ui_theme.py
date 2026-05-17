"""Semantic style helpers for the curses UI.

The curses palette built by ``init_colors()`` uses raw keys (``accent``,
``panel``...). This module gives the rest of the UI a small semantic
vocabulary so renderers ask for ``success`` / ``badge_mtp`` instead of
hard-coding palette keys, keeping a consistent look across panels.
"""

import curses
from typing import Dict


# Semantic style name -> init_colors() palette key.
_STYLE_COLOR_KEY = {
    'normal': 'panel',
    'muted': 'muted',
    'success': 'success',
    'warning': 'warning',
    'error': 'error',
    'selected': 'selection',
    'active': 'accent',
    'badge_engine': 'accent',
    'badge_mtp': 'banner',
    'badge_benchmark': 'success',
    'badge_health': 'muted',
}

# Styles that read better bold by default.
_BOLD_BY_DEFAULT = frozenset({'success', 'error', 'active', 'badge_engine', 'badge_mtp'})

SEMANTIC_STYLES = tuple(_STYLE_COLOR_KEY.keys())


def style(colors: Dict[str, int], name: str, bold: bool = False) -> int:
    """Return the curses attribute for a semantic style name."""
    normalized = str(name or '').strip().lower()
    key = _STYLE_COLOR_KEY.get(normalized, 'panel')
    attr = int(colors.get(key, 0) or 0)
    if bold or normalized in _BOLD_BY_DEFAULT:
        attr |= curses.A_BOLD
    return attr


# Model health label -> semantic style.
_HEALTH_STYLE = {
    'OK': 'success',
    'READY': 'success',
    'STALE': 'warning',
    'WARN': 'warning',
    'FAIL': 'error',
}


def health_style_name(health: str) -> str:
    return _HEALTH_STYLE.get(str(health or '').strip().upper(), 'muted')


def health_style(colors: Dict[str, int], health: str) -> int:
    return style(colors, health_style_name(health))


# Capability-driven MTP status -> semantic style.
_MTP_STYLE = {
    'ready': 'success',
    'usable': 'success',
    'capable': 'active',
    'testing': 'active',
    'risky': 'warning',
    'blocked': 'error',
    'failed': 'error',
    'unsupported': 'muted',
    'off': 'muted',
    'unknown': 'muted',
}


def mtp_style_name(status: str) -> str:
    return _MTP_STYLE.get(str(status or '').strip().lower(), 'muted')


def mtp_style(colors: Dict[str, int], status: str) -> int:
    return style(colors, mtp_style_name(status))


# Run state -> palette chip key.
_STATE_CHIP_KEY = {
    'running': 'chip_ready',
    'starting': 'chip_loading',
    'stopped': 'chip_stopped',
    'error': 'error',
}


def state_chip_style(colors: Dict[str, int], state: str) -> int:
    """Return the curses attribute for a run-state status chip."""
    key = _STATE_CHIP_KEY.get(str(state or '').strip().lower(), 'chip_stopped')
    attr = int(colors.get(key, 0) or 0)
    if key == 'error':
        attr |= curses.A_BOLD
    return attr


def kind_style(colors: Dict[str, int], kind: str) -> int:
    """Map a generic line ``kind`` (success/warning/error/heading/muted...) to an attribute."""
    normalized = str(kind or '').strip().lower()
    if normalized in ('heading', 'title', 'header'):
        return style(colors, 'active', bold=True)
    if normalized in _STYLE_COLOR_KEY:
        return style(colors, normalized)
    if normalized == 'success':
        return style(colors, 'success')
    return style(colors, 'normal')
