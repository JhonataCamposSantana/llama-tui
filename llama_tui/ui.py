import curses
import textwrap
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

from .app import AppConfig, CONTINUE_MERGE_MODES, context_per_slot
from .benchmark import (
    append_model_log,
    apply_measured_profile,
    benchmark_all_models_deep,
    benchmark_best_optimization,
    benchmark_fast_profiles,
    benchmark_profile_is_fresh,
    deep_benchmark_model_decision,
    estimate_text_tokens,
    launch_hermes_stack,
    launch_opencode_stack,
    launch_with_failsafe,
    machine_best_summary,
    record_matches_profile,
    start_model_with_progress,
    sync_opencode_after_tuning,
)
from .chat import stream_chat_events
from .constants import LOGO, REFRESH_SECONDS
from .control import CancelToken, CancelledError
from .discovery import classify_model_type, display_offload, display_runtime, extract_quant
from .gguf import architecture_detail
from .hermes_benchmark import benchmark_hermes_workflow
from .hardware import HardwareProfile
from .models import ModelConfig
from .opencode_benchmark import benchmark_opencode_workflow
from .optimize import apply_best_optimization, select_best_tier
from .textutil import compact_message, ellipsize, important_log_excerpt, is_error_message, wrap_display_lines

PROFILE_LABELS = {
    'best': 'Auto Profile',
    'auto': 'Auto Profile',
    'max_context': 'Long Context',
    'max_context_q8_kv': 'Long Context q8 KV',
    'max_context_safe': 'Safe Context',
    'tokens_per_sec': 'Fast Chat',
    'tokens_per_sec_q8_kv': 'Fast Chat q8 KV',
    'manual': 'Manual',
    'measured_auto': 'Measured Auto',
    'measured_fast_chat': 'Measured Fast Chat',
    'measured_long_context': 'Measured Long Context',
    'measured_opencode_ready': 'Measured OpenCode',
    'fast_chat': 'Fast Chat',
    'long_context': 'Long Context',
    'opencode_ready': 'OpenCode Ready',
    'winner': 'Winner',
}

TIER_LABELS = {
    'auto': 'Auto',
    'safe': 'Safe',
    'moderate': 'Balanced',
    'extreme': 'Aggressive',
    'measured': 'Measured',
}

SIMPLE_PROFILE_ACTIONS = {
    'auto_profile': ('best', 'auto', 'Auto profile'),
    'balanced_chat': ('tokens_per_sec', 'moderate', 'Balanced chat'),
    'fast_chat': ('tokens_per_sec', 'extreme', 'Fast chat'),
    'long_context': ('max_context', 'moderate', 'Long context'),
}

TRY_INPUT_ROWS = 5
TRY_TRANSCRIPT_SCROLL_KEYS = {
    16: 'older',
    14: 'newer',
    2: 'page_older',
    6: 'page_newer',
    1: 'oldest',
    5: 'newest',
}
BENCHMARK_FEED_LIMIT = 80
BENCHMARK_RECORD_LIMIT = 120
BENCHMARK_COMMAND_LIMIT = 12
HEADER_DASHBOARD_MIN_WIDTH = 124
HEADER_DASHBOARD_MIN_PANEL_WIDTH = 42
HEADER_DASHBOARD_HEIGHT = 10
RIGHT_PANE_SCROLL_KEYS = {
    curses.KEY_UP: 'older',
    ord('k'): 'older',
    curses.KEY_DOWN: 'newer',
    ord('j'): 'newer',
    curses.KEY_PPAGE: 'page_older',
    curses.KEY_NPAGE: 'page_newer',
    curses.KEY_HOME: 'oldest',
    curses.KEY_END: 'newest',
}
RIGHT_TABS = {
    'detail': ['summary', 'logs', 'errors', 'command', 'benchmarks'],
    'benchmark': ['progress', 'results', 'commands', 'logs', 'errors'],
    'try': ['profile', 'logs', 'errors', 'stats', 'command'],
    'results': ['run_summary', 'rankings', 'failures'],
    'machine_results': ['overview', 'rankings', 'failures'],
}
RIGHT_TAB_LABELS = {
    'summary': 'Summary',
    'logs': 'Logs',
    'errors': 'Errors',
    'command': 'Command',
    'commands': 'Commands',
    'benchmarks': 'Benchmarks',
    'progress': 'Progress',
    'results': 'Results',
    'profile': 'Profile',
    'stats': 'Stats',
    'run_summary': 'Run Summary',
    'rankings': 'Rankings',
    'failures': 'Failures',
    'overview': 'Overview',
}
SERVER_WINNER_LABELS = {
    'fast_chat': ('Winner', 'Fastest'),
    'long_context': ('Winner', 'Highest Context'),
    'opencode_ready': ('Winner', 'OpenCode-ready'),
    'auto': ('Winner', 'Ideal'),
}
RANK_ROLE_PRIORITY = {
    'Winner': 0,
    'Runner-up': 1,
    'Fastest': 2,
    'Highest Context': 3,
    'OpenCode-ready': 4,
    'Ideal': 5,
    'Auto': 5,
    'Possible': 6,
    'Measured': 7,
    'Passed': 7,
    'Probe': 8,
    'Skipped': 20,
    'Failed': 30,
    'Break Point': 31,
}
RIGHT_DEFAULT_TAB = {
    'detail': 'summary',
    'benchmark': 'progress',
    'try': 'profile',
    'results': 'run_summary',
    'machine_results': 'overview',
}

VIEW_LABELS = {
    'list': 'Models',
    'detail': 'Model Details',
    'benchmark': 'Benchmark',
    'try': 'Try It Out',
    'results': 'Results',
    'machine_results': 'Machine Rankings',
}

BENCHMARK_WIKI_SECTIONS = [
    (
        'What is a benchmark?',
        'A benchmark is a safe test run. llama-tui starts the model with one set of settings, checks that the server is ready, asks it to write a short answer, records speed and stability, then stops that server before trying the next set.',
    ),
    (
        'What the numbers mean',
        'ctx is how much conversation or code the model can keep in memory. ctx/slot is how much of that memory each simultaneous request gets. parallel is how many requests the server can handle at once. tok/s is how fast the model writes. threads is CPU worker count. ngl is how many llama.cpp layers go to the GPU. Headroom is RAM or VRAM left after the test.',
    ),
    (
        'Extra table labels',
        'variant shows runtime tweaks, usually default or q8 KV. measurement_type tells you whether a row was a quick probe or a full speed measurement. planner_reason explains why llama-tui tested that row, such as frontier, speed_knee, chat_parallel, opencode_floor, or q8_probe.',
    ),
    (
        'Deep benchmark: B',
        'B is the careful benchmark. It first finds the safe edge for context, then fully measures only the settings that could realistically win. This is the smart bounded path: less waste than testing everything, but winners still come from real measurements.',
    ),
    (
        'Fast benchmark: F',
        'F is the quick benchmark. It tests a small set of practical settings and gives you a useful first profile faster. Use it when you want a good starting point; use B when you want higher confidence.',
    ),
    (
        'OpenCode benchmark: O',
        'O is the OpenCode check. It runs headless, meaning no terminal opens. llama-tui uses throwaway test projects, captures logs and exit codes, checks the result with python -m unittest -q, then cleans up OpenCode and the model server.',
    ),
    (
        'Reading results',
        'Winner is the saved setting for that category. Runner-up is the next best measured option. Failed means the server did not start, did not become ready, or could not finish the sample. Break Point means a setting failed twice, so llama-tui stopped trying larger or heavier settings in that direction.',
    ),
    (
        'Which result should I use?',
        'Fast Chat is for snappy replies. Long Context is for large files and long sessions. OpenCode-ready is for coding workflows with OpenCode. Auto is the balanced everyday choice when you just want llama-tui to pick a sensible profile.',
    ),
]

TRUTHY_VALUES = {'1', 'true', 'yes', 'y', 'on'}
FALSY_VALUES = {'0', 'false', 'no', 'n', 'off'}
VALID_RUNTIMES = ('llama.cpp', 'vllm')
VALID_OPTIMIZE_MODES = ('max_context_safe', 'manual', 'best', 'max_context', 'tokens_per_sec', 'opencode_ready')
VALID_OPTIMIZE_TIERS = ('safe', 'moderate', 'extreme')
SORT_OPTIONS = [
    ('favorites', 'Favorites'),
    ('recent', 'Recent'),
    ('name', 'Name'),
    ('benchmark', 'Best Benchmark'),
    ('context', 'Highest Context'),
    ('port', 'Port'),
]
DETAIL_DENSITY_OPTIONS = [('simple', 'Simple'), ('advanced', 'Advanced')]
FILTER_RUNTIME_OPTIONS = [
    ('all', 'All runtimes'),
    ('llama.cpp', 'llama.cpp'),
    ('vllm', 'vLLM'),
]
FILTER_SOURCE_OPTIONS = [
    ('all', 'All sources'),
    ('manual', 'Manual'),
    ('huggingface', 'Hugging Face'),
    ('llmfit', 'llmfit'),
    ('llm-models', 'llm-models'),
    ('lm-studio', 'LM Studio'),
]
FILTER_STATUS_OPTIONS = [
    ('all', 'All server states'),
    ('READY', 'Ready'),
    ('LOADING', 'Loading'),
    ('STARTING', 'Starting'),
    ('STOPPED', 'Stopped'),
    ('ERROR', 'Error'),
    ('fresh', 'Fresh benchmark'),
    ('stale', 'Stale benchmark'),
    ('missing', 'Missing benchmark'),
    ('failed', 'Failed benchmark'),
    ('pending', 'Pending benchmark'),
    ('running', 'Running benchmark'),
]
BENCHMARK_FRESHNESS_LABELS = {
    'fresh': 'Fresh',
    'stale': 'Stale',
    'missing': 'Missing',
    'failed': 'Failed',
    'pending': 'Pending',
    'running': 'Running',
}


def profile_label(value: str) -> str:
    raw = (value or '').strip()
    key = raw.lower()
    return PROFILE_LABELS.get(key, raw.replace('_', ' ').title() if raw else '-')


def tier_label(value: str) -> str:
    raw = (value or '').strip()
    key = raw.lower()
    return TIER_LABELS.get(key, raw.replace('_', ' ').title() if raw else '-')


def simple_profile_action(value: str) -> Tuple[str, str, str]:
    return SIMPLE_PROFILE_ACTIONS[value]


def benchmark_wiki_lines(width: int) -> List[str]:
    width = max(24, int(width or 24))
    lines: List[str] = []
    for title, body in BENCHMARK_WIKI_SECTIONS:
        if lines:
            lines.append('')
        lines.append(title)
        lines.extend(wrap_display_lines(body, width))
    return lines


def parse_bool_text(value: str, field_label: str = 'value') -> bool:
    normalized = str(value or '').strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise ValueError(f'{field_label} must be true/false')


def normalize_choice(value: str, allowed: Tuple[str, ...], default: str) -> str:
    normalized = str(value or '').strip().lower() or default
    return normalized if normalized in allowed else default


def sort_mode_label(value: str) -> str:
    normalized = normalize_choice(value, tuple(key for key, _label in SORT_OPTIONS), 'port')
    return dict(SORT_OPTIONS).get(normalized, 'Port')


def detail_density_label(value: str) -> str:
    normalized = normalize_choice(value, tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'simple')
    return dict(DETAIL_DENSITY_OPTIONS).get(normalized, 'Simple')


def benchmark_freshness_label(app: AppConfig, model: ModelConfig) -> str:
    if benchmark_profile_is_fresh(app, model):
        return 'fresh'
    status = (getattr(model, 'default_benchmark_status', '') or '').strip().lower()
    if status in ('running', 'failed', 'pending', 'aborted'):
        return 'failed' if status == 'aborted' else status
    if status == 'done':
        return 'stale'
    if float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0) > 0.0:
        return 'stale'
    return 'missing'


def benchmark_freshness_short(app: AppConfig, model: ModelConfig) -> str:
    mapping = {
        'fresh': 'FRSH',
        'stale': 'STAL',
        'missing': 'MISS',
        'failed': 'FAIL',
        'pending': 'PEND',
        'running': 'RUN',
    }
    return mapping.get(benchmark_freshness_label(app, model), 'MISS')


def benchmark_freshness_display(app: AppConfig, model: ModelConfig) -> str:
    return BENCHMARK_FRESHNESS_LABELS.get(benchmark_freshness_label(app, model), 'Missing')


def filter_option_label(options: List[Tuple[str, str]], value: str) -> str:
    return dict(options).get(value, value or '-')


def iso_recent_key(value: str) -> float:
    if not value:
        return 0.0
    try:
        return time.mktime(time.strptime(value, '%Y-%m-%dT%H:%M:%S'))
    except Exception:
        return 0.0


def model_matches_search(model: ModelConfig, status: str, search: str) -> bool:
    needle = str(search or '').strip().lower()
    if not needle:
        return True
    haystack = ' '.join([
        str(getattr(model, 'id', '') or ''),
        str(getattr(model, 'name', '') or ''),
        str(getattr(model, 'alias', '') or ''),
        str(getattr(model, 'path', '') or ''),
        str(display_runtime(model) or ''),
        str(getattr(model, 'source', '') or ''),
        str(status or ''),
        str(extract_quant(model) or ''),
        str(classify_model_type(model) or ''),
        ' '.join(str(tag) for tag in list(getattr(model, 'tags', []) or [])),
    ]).lower()
    return needle in haystack


def model_matches_browser_filters(
    app: AppConfig,
    model: ModelConfig,
    status: str,
    search: str = '',
    runtime_filter: str = 'all',
    source_filter: str = 'all',
    status_filter: str = 'all',
    tag_filter: str = 'all',
) -> bool:
    runtime_filter = str(runtime_filter or 'all').strip().lower() or 'all'
    source_filter = str(source_filter or 'all').strip().lower() or 'all'
    status_filter = str(status_filter or 'all').strip() or 'all'
    tag_filter = str(tag_filter or 'all').strip().lower() or 'all'
    if runtime_filter != 'all' and str(getattr(model, 'runtime', 'llama.cpp') or '').strip().lower() != runtime_filter:
        return False
    if source_filter != 'all' and str(getattr(model, 'source', 'manual') or '').strip().lower() != source_filter:
        return False
    if status_filter != 'all':
        freshness = benchmark_freshness_label(app, model)
        if status_filter in ('fresh', 'stale', 'missing', 'failed', 'pending', 'running'):
            if freshness != status_filter:
                return False
        elif status != status_filter:
            return False
    if tag_filter != 'all':
        tags = {str(tag).strip().lower() for tag in list(getattr(model, 'tags', []) or []) if str(tag).strip()}
        if tag_filter not in tags:
            return False
    return model_matches_search(model, status, search)


def model_sort_key(model: ModelConfig, sort_mode: str) -> Tuple[Any, ...]:
    normalized = normalize_choice(sort_mode, tuple(key for key, _label in SORT_OPTIONS), 'port')
    name_key = (str(getattr(model, 'name', '') or getattr(model, 'id', '') or '')).lower()
    model_id = str(getattr(model, 'id', '') or '')
    recent_key = iso_recent_key(str(getattr(model, 'last_used_at', '') or ''))
    benchmark_key = float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0)
    context_key = int(getattr(model, 'ctx', 0) or 0) // max(1, int(getattr(model, 'parallel', 1) or 1))
    favorite_key = 0 if bool(getattr(model, 'favorite', False)) else 1
    if normalized == 'favorites':
        return favorite_key, name_key, model_id
    if normalized == 'recent':
        return -recent_key, favorite_key, name_key, model_id
    if normalized == 'name':
        return name_key, model_id
    if normalized == 'benchmark':
        return -benchmark_key, favorite_key, name_key, model_id
    if normalized == 'context':
        return -context_key, favorite_key, name_key, model_id
    return int(getattr(model, 'port', 0) or 0), favorite_key, name_key, model_id


def browser_models(
    app: AppConfig,
    statuses: Dict[str, Tuple[str, str]],
    search: str = '',
    runtime_filter: str = 'all',
    source_filter: str = 'all',
    status_filter: str = 'all',
    tag_filter: str = 'all',
    sort_mode: str = 'port',
) -> List[ModelConfig]:
    filtered: List[ModelConfig] = []
    for model in list(getattr(app, 'models', []) or []):
        status = str((statuses.get(model.id) or ('STOPPED', ''))[0] or 'STOPPED')
        if model_matches_browser_filters(
            app,
            model,
            status,
            search=search,
            runtime_filter=runtime_filter,
            source_filter=source_filter,
            status_filter=status_filter,
            tag_filter=tag_filter,
        ):
            filtered.append(model)
    return sorted(filtered, key=lambda item: model_sort_key(item, sort_mode))


def clamp_scroll(scroll: int, total_lines: int, visible_rows: int) -> int:
    max_scroll = max(0, int(total_lines or 0) - max(1, int(visible_rows or 1)))
    return max(0, min(max_scroll, int(scroll or 0)))


def scrollable_pane_wrapped_items(items: List[object], width: int, default_attr: int = 0) -> List[Tuple[str, int]]:
    width = max(1, int(width or 1))
    wrapped: List[Tuple[str, int]] = []
    for item in items or ['']:
        if isinstance(item, tuple):
            text, attr = item[0], int(item[1] or default_attr)
        else:
            text, attr = item, default_attr
        lines = wrap_display_item_lines(str(text), width) or ['']
        wrapped.extend((line, attr) for line in lines)
    return wrapped or [('', default_attr)]


def wrap_display_item_lines(text: str, width: int, continuation_indent: str = '  ') -> List[str]:
    width = max(1, int(width or 1))
    subsequent = continuation_indent if width > len(continuation_indent) + 4 else ''
    wrapped: List[str] = []
    for paragraph in str(text).splitlines() or ['']:
        if not paragraph:
            wrapped.append('')
            continue
        wrapped.extend(textwrap.wrap(
            paragraph,
            width=width,
            replace_whitespace=False,
            break_long_words=True,
            subsequent_indent=subsequent,
        ) or [''])
    return wrapped


def scrollable_pane_wrapped_lines(lines: List[str], width: int) -> List[str]:
    return [line for line, _attr in scrollable_pane_wrapped_items(list(lines or []), width)]


def scrollable_pane_max_scroll(lines: List[str], width: int, rows: int) -> int:
    return max(0, len(scrollable_pane_wrapped_lines(lines, width)) - max(1, int(rows or 1)))


def scrollable_pane_view(lines: List[str], width: int, rows: int, scroll: int) -> Tuple[List[str], int, bool, bool]:
    items, clamped, has_older, has_newer, _total = scrollable_pane_item_view(lines, width, rows, scroll)
    return [line for line, _attr in items], clamped, has_older, has_newer


def scrollable_pane_item_view(
    items: List[object],
    width: int,
    rows: int,
    scroll: int,
    default_attr: int = 0,
) -> Tuple[List[Tuple[str, int]], int, bool, bool, int]:
    visible_rows = max(1, int(rows or 1))
    wrapped = scrollable_pane_wrapped_items(items, width, default_attr)
    total = len(wrapped)
    clamped = clamp_scroll(scroll, total, visible_rows)
    start = max(0, total - visible_rows - clamped)
    end = min(total, start + visible_rows)
    visible = wrapped[start:end]
    has_older = start > 0
    has_newer = end < total
    while len(visible) < visible_rows:
        visible.append(('', default_attr))
    return visible, clamped, has_older, has_newer, total


def adjust_scroll_offset(scroll: int, action: str, total_lines: int, visible_rows: int) -> int:
    page = max(1, int(visible_rows or 1))
    if action == 'older':
        scroll += 1
    elif action == 'newer':
        scroll -= 1
    elif action == 'page_older':
        scroll += page
    elif action == 'page_newer':
        scroll -= page
    elif action == 'oldest':
        scroll = max(0, int(total_lines or 0))
    elif action == 'newest':
        scroll = 0
    return clamp_scroll(scroll, total_lines, visible_rows)


def read_display_file_lines(path: Path) -> List[str]:
    if not path.exists():
        return ['<no log file yet>']
    try:
        return path.read_text(errors='replace').splitlines() or ['<empty log>']
    except Exception as exc:
        return [f'<failed to read log: {exc}>']


def right_tabs_for_view(view_mode: str) -> List[str]:
    return list(RIGHT_TABS.get(view_mode, []))


def default_right_tab(view_mode: str) -> str:
    tabs = right_tabs_for_view(view_mode)
    return RIGHT_DEFAULT_TAB.get(view_mode, tabs[0] if tabs else '')


def normalize_right_tab(view_mode: str, tab: str) -> str:
    tabs = right_tabs_for_view(view_mode)
    if tab in tabs:
        return tab
    return default_right_tab(view_mode)


def cycle_right_tab(view_mode: str, current_tab: str, direction: int = 1) -> str:
    tabs = right_tabs_for_view(view_mode)
    if not tabs:
        return ''
    current = normalize_right_tab(view_mode, current_tab)
    try:
        index = tabs.index(current)
    except ValueError:
        index = 0
    return tabs[(index + int(direction or 1)) % len(tabs)]


def right_tab_key_direction(key: int) -> int:
    if key in (9, ord(']')):
        return 1
    if key in (getattr(curses, 'KEY_BTAB', -999), ord('[')):
        return -1
    return 0


def right_tab_scroll_key(view_mode: str, tab: str) -> str:
    return f'{view_mode}:{normalize_right_tab(view_mode, tab)}'


def right_scroll_action_for_view(view_mode: str, key: int) -> str:
    action = RIGHT_PANE_SCROLL_KEYS.get(key, '')
    if not action:
        return ''
    if view_mode in ('detail', 'benchmark'):
        return action
    if view_mode in ('try', 'results', 'machine_results') and key in (curses.KEY_PPAGE, curses.KEY_NPAGE, curses.KEY_HOME, curses.KEY_END):
        return action
    return ''


def right_tab_label(tab: str, error_count: int = 0) -> str:
    if tab == 'errors' and int(error_count or 0) > 0:
        return f'Errors {int(error_count or 0)}'
    return RIGHT_TAB_LABELS.get(tab, tab.replace('_', ' ').title())


def build_log_items(
    log_lines: List[str],
    log_attr: int = 0,
    muted_attr: int = 0,
) -> List[Tuple[str, int]]:
    if log_lines:
        return [(str(line), log_attr) for line in log_lines]
    return [('<no log lines>', muted_attr)]


def build_error_items(
    error_lines: List[str],
    error_attr: int = 0,
    muted_attr: int = 0,
) -> List[Tuple[str, int]]:
    if error_lines:
        return [(str(line), error_attr) for line in error_lines]
    return [('No errors captured for this model/run.', muted_attr)]


def header_dashboard_layout(width: int) -> Tuple[bool, int, int, int]:
    total_width = max(1, int(width or 1))
    left_w = max(76, min(112, (total_width // 2) + 8))
    right_x = left_w + 2
    available_right_w = max(0, total_width - right_x - 2)
    right_w = max(38, available_right_w)
    enabled = total_width >= HEADER_DASHBOARD_MIN_WIDTH and available_right_w >= HEADER_DASHBOARD_MIN_PANEL_WIDTH
    return enabled, left_w, right_x, right_w


def body_pane_layout(width: int) -> Tuple[int, int, int]:
    total_width = max(1, int(width or 1))
    left_x = 1
    gap = 2
    right_margin = 1
    usable = max(1, total_width - left_x - right_margin)
    min_left = 44
    min_right = 32
    preferred_left = max(76, min(112, (total_width // 2) + 8))

    if usable >= min_left + gap + min_right:
        left_w = min(preferred_left, usable - gap - min_right)
        left_w = max(min_left, left_w)
        right_x = left_x + left_w + gap
        right_w = max(1, usable - left_w - gap)
    else:
        left_w = max(1, min(preferred_left, max(1, usable - gap - 1)))
        right_x = min(total_width - 2, left_x + left_w + gap)
        right_w = max(1, total_width - right_x - right_margin)

    if right_x + right_w > total_width:
        right_w = max(1, total_width - right_x)
    return left_w, right_x, right_w


def body_pane_height(screen_height: int, box_top: int) -> int:
    return max(2, int(screen_height or 0) - int(box_top or 0) - 4)


def body_content_rows(screen_height: int, box_top: int) -> int:
    return max(0, body_pane_height(screen_height, box_top) - 2)


def body_content_bottom(screen_height: int, box_top: int) -> int:
    return int(box_top or 0) + body_pane_height(screen_height, box_top) - 1


def try_input_row_count(content_rows: int, max_rows: int = TRY_INPUT_ROWS) -> int:
    rows = max(0, int(content_rows or 0))
    if rows < 3:
        return 0
    return min(max(1, int(max_rows or 1)), rows - 2)


def visible_selection_window(total: int, selected: int, rows: int) -> Tuple[int, int]:
    total = max(0, int(total or 0))
    rows = max(0, int(rows or 0))
    if total <= 0 or rows <= 0:
        return 0, 0
    rows = min(total, rows)
    selected = max(0, min(int(selected or 0), total - 1))
    start = max(0, min(selected - rows // 2, total - rows))
    return start, start + rows


def build_error_source_lines(
    error_history: List[str],
    benchmark_errors: Optional[List[str]] = None,
    benchmark_mode: bool = False,
    status_error: str = '',
    last_error_message: str = '',
) -> List[str]:
    if benchmark_mode and benchmark_errors:
        source = [compact_message(str(item)) for item in benchmark_errors if compact_message(str(item))]
    else:
        source = [compact_message(str(item)) for item in error_history if compact_message(str(item))]
        status_line = compact_message(status_error)
        if status_line and (not source or source[-1] != status_line):
            source.append(status_line)
        elif compact_message(last_error_message) and not source:
            source.append(compact_message(last_error_message))
    return source[-BENCHMARK_FEED_LIMIT:]


def header_dashboard_title(view_mode: str) -> str:
    if view_mode == 'benchmark':
        return 'Benchmark Status'
    if view_mode == 'try':
        return 'Try-It-Out Status'
    return 'System Status'


def summarize_roots(paths: List[Path], width: int) -> str:
    values = [str(path) for path in paths]
    if not values:
        return '-'
    text = ', '.join(values)
    return ellipsize(text, max(8, int(width or 8)))


def build_header_config_items(app: AppConfig, message: str, width: int) -> List[Tuple[str, str]]:
    body_width = max(12, int(width or 12))
    continue_path = getattr(getattr(app, 'continue_settings', None), 'path', '') or '<unset>'
    ui_summary = (
        f'sort={sort_mode_label(getattr(getattr(app, "ui", None), "preferred_sort", "port"))} '
        f'detail={detail_density_label(getattr(getattr(app, "ui", None), "detail_density", "simple"))}'
    )
    roots_summary = (
        f'hf={app.hf_cache_root} | llmfit={app.llmfit_cache_root} | '
        f'local={app.llm_models_cache_root} | lm-studio={summarize_roots(app.lm_studio_roots(), body_width)}'
    )
    lines = [
        (f'config: {app.config_path}', 'muted'),
        (f'llama-server: {app.llama_server}', 'muted'),
        (f'vllm: {app.vllm_command}', 'muted'),
        (f'opencode: {app.opencode.path or "<unset>"}  continue: {continue_path}  hermes: {getattr(app.hermes, "command", "hermes")}', 'muted'),
        (f'roots: {roots_summary}', 'muted'),
        (f'ui: {ui_summary}', 'muted'),
        (f'message: {compact_message(message)}', 'message'),
    ]
    if getattr(app, 'load_warnings', None):
        lines.insert(-1, (f'recovery: {compact_message(str(app.load_warnings[0]))}', 'message'))
    return [(ellipsize(text, body_width), kind) for text, kind in lines]


def build_header_dashboard_items(
    statuses: Dict[str, Tuple[str, str]],
    active_model: Optional[ModelConfig],
    active_status: Tuple[str, str],
    view_mode: str,
    benchmark_state: Dict[str, object],
    action_active: bool,
    action_label: str,
    hardware_summary: str,
    error_history: List[str],
    width: int,
) -> List[Tuple[str, str]]:
    body_width = max(12, int(width or 12))
    counts = {'READY': 0, 'LOADING': 0, 'STARTING': 0, 'STOPPED': 0, 'ERROR': 0}
    for status, _detail in statuses.values():
        if status in counts:
            counts[status] += 1
    loading = counts['LOADING'] + counts['STARTING']
    status, detail = active_status
    if active_model:
        active_line = f'active: {active_model.id} {status}'
        detail_text = compact_message(str(detail or ''))
        if detail_text:
            active_line += f' ({detail_text})'
    else:
        active_line = 'active: none'

    view_line = f'view: {VIEW_LABELS.get(view_mode, view_mode or "Models")}'

    run_kind = str(benchmark_state.get('run_kind') or '')
    benchmark_active = bool(benchmark_state.get('active')) or view_mode == 'benchmark' or bool(run_kind)
    if benchmark_active:
        completed = int(benchmark_state.get('completed', 0) or 0)
        total = int(benchmark_state.get('total', 0) or 0)
        pct = int(round(benchmark_progress_fraction(completed, total) * 100)) if total else 0
        phase = str(benchmark_state.get('phase') or '-')
        candidate = str(benchmark_state.get('candidate') or '-')
        bench_line = f'bench: {run_kind or "server"} {phase} {completed}/{total if total else "?"} {pct}% {candidate}'
    else:
        bench_line = 'bench: idle'

    latest_error = compact_message(str(error_history[-1])) if error_history else 'none'
    lines = [
        (f'counts: READY:{counts["READY"]} LOADING:{loading} STOPPED:{counts["STOPPED"]} ERROR:{counts["ERROR"]}', 'counts'),
        (active_line, 'status' if status == 'READY' else 'error' if status == 'ERROR' else 'muted'),
        (view_line, 'muted'),
        (bench_line, 'benchmark' if benchmark_active else 'muted'),
        (f'hardware: {hardware_summary or "-"}', 'muted'),
        (f'last error: {latest_error}', 'error' if error_history else 'muted'),
    ]
    return [(ellipsize(text, body_width), kind) for text, kind in lines]


def build_benchmark_progress_items(
    model: ModelConfig,
    state: Dict[str, object],
    status: str,
    detail: str,
    pid: object,
    width: int,
    accent_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    completed = int(state.get('completed', 0) or 0)
    total = int(state.get('total', 0) or 0)
    fraction = benchmark_progress_fraction(completed, total)
    records = list(state.get('records', []) or [])
    latest = records[-1] if records else {}
    current_slot = int(getattr(model, 'ctx', 0) or 0) // max(1, int(getattr(model, 'parallel', 1) or 1))
    run_kind = str(state.get('run_kind') or 'server')
    if run_kind == 'server_all':
        status_detail = (
            f'batch skipped={int(state.get("batch_skipped", 0) or 0)} '
            f'failed={int(state.get("batch_failed", 0) or 0)} '
            f'restored={int(state.get("batch_restored", 0) or 0)}'
        )
        model_line = f'model: {state.get("model_id") or "all managed models"}'
        runtime_line = 'runtime: deep benchmark all'
        pid_line = status_detail
    else:
        model_line = f'model: {model.id}'
        runtime_line = (
            f'arch: {classify_model_type(model)}  runtime: {display_runtime(model)}  '
            f'offload: {display_offload(model)}  ctx/slot={current_slot}  par={getattr(model, "parallel", 1)}'
        )
        pid_line = f'pid: {pid or "-"}  {detail}'
    items = [
        (model_line, normal_attr),
        (f'run: {run_kind}', normal_attr),
        (f'status: {state.get("status") or "idle"} / server {status}', accent_attr),
        (f'elapsed: {benchmark_elapsed_text(state)}', normal_attr),
        (f'progress: {progress_bar_text(completed, total, max(8, width - 18))} {int(round(fraction * 100))}%', accent_attr),
        ('', normal_attr),
        (f'phase: {state.get("phase") or "-"}', normal_attr),
        (f'candidate: {state.get("candidate") or "-"}', normal_attr),
        (f'profile: {model_profile_summary(model)}', normal_attr),
        (runtime_line, normal_attr),
        (pid_line, normal_attr),
    ]
    pressure_detail = str(latest.get('process_pressure_detail', '') or state.get('process_pressure_detail', '') or '')
    if pressure_detail:
        items.append((f'process pressure: {compact_message(pressure_detail)}', normal_attr))
    if latest:
        items.extend([
            ('', normal_attr),
            ('latest result:', accent_attr),
            (benchmark_row_text(latest), normal_attr),
        ])
        latest_detail = compact_message(str(latest.get('detail', '') or ''))
        if latest_detail:
            items.append((f'detail: {latest_detail}', normal_attr))
        if latest.get('status') == 'not Hermes-ready':
            required = int(latest.get('required_context', 0) or 0)
            actual = int(latest.get('actual_ctx_per_slot', 0) or 0)
            items.append((f'Hermes readiness: needs {required}, actual ctx/slot {actual}', normal_attr))
        if latest.get('experimental_context_override'):
            configured = int(latest.get('configured_context_length', 0) or 0)
            actual = int(latest.get('actual_ctx_per_slot', 0) or 0)
            items.append((f'Hermes experimental override: config={configured}, actual={actual}', normal_attr))
    else:
        items.extend([
            ('', normal_attr),
            ('latest result: waiting for first row', normal_attr),
        ])
    return items


def should_prompt_quit_keepalive(managed_running: bool, action_active: bool) -> bool:
    return bool(managed_running) and not bool(action_active)


def apply_quit_policy(app: AppConfig, policy: str) -> Tuple[bool, str]:
    if policy == 'cancel':
        return False, 'Quit cancelled.'
    if policy == 'leave':
        app.leave_managed_processes_running()
        return True, 'Leaving managed model servers running.'
    return True, 'Stopping managed model servers on quit.'


def model_profile_summary(model: ModelConfig) -> str:
    mode = profile_label(getattr(model, 'optimize_mode', 'max_context_safe'))
    tier = tier_label(getattr(model, 'optimize_tier', 'moderate'))
    reserve = int(getattr(model, 'memory_reserve_percent', 25) or 25)
    measured = getattr(model, 'measured_profiles', {}) or {}
    suffix = f' / measured {len(measured)}' if measured else ''
    return f'{mode} / {tier} / reserve {reserve}%{suffix}'


def stop_try_model(app: AppConfig, model: ModelConfig) -> Tuple[bool, str]:
    return app.stop(model)


def should_stop_try_model(try_launched_model_id: str, model: Optional[ModelConfig]) -> bool:
    return bool(model and try_launched_model_id == getattr(model, 'id', ''))


def new_try_live_metrics() -> Dict[str, object]:
    return {
        'active': False,
        'started_at': 0.0,
        'first_chunk_at': 0.0,
        'latest_chunk_at': 0.0,
        'text': '',
        'tokens': 0,
        'last_tokens': 0,
        'last_seconds': 0.0,
        'last_tokens_per_sec': 0.0,
    }


def clear_try_live_metrics(metrics: Dict[str, object]):
    metrics.clear()
    metrics.update(new_try_live_metrics())


def reset_try_live_metrics(metrics: Dict[str, object], now: Optional[float] = None):
    started_at = time.monotonic() if now is None else now
    clear_try_live_metrics(metrics)
    metrics.update({
        'active': True,
        'started_at': started_at,
    })


def update_try_live_metrics(metrics: Dict[str, object], chunk: str, now: Optional[float] = None):
    timestamp = time.monotonic() if now is None else now
    if not metrics.get('active'):
        reset_try_live_metrics(metrics, timestamp)
    if chunk and not float(metrics.get('first_chunk_at') or 0.0):
        metrics['first_chunk_at'] = timestamp
    metrics['latest_chunk_at'] = timestamp
    metrics['text'] = f'{metrics.get("text", "")}{chunk}'
    metrics['tokens'] = estimate_text_tokens(str(metrics.get('text') or ''))


def finish_try_live_metrics(metrics: Dict[str, object], now: Optional[float] = None):
    timestamp = time.monotonic() if now is None else now
    started_at = float(metrics.get('started_at') or timestamp)
    latest_at = float(metrics.get('latest_chunk_at') or timestamp)
    end_at = max(timestamp, latest_at)
    tokens = int(metrics.get('tokens') or 0)
    seconds = max(0.0, end_at - started_at)
    metrics['active'] = False
    metrics['last_tokens'] = tokens
    metrics['last_seconds'] = seconds
    metrics['last_tokens_per_sec'] = (tokens / seconds) if tokens > 0 and seconds > 0 else 0.0


def try_live_metric_snapshot(metrics: Dict[str, object], now: Optional[float] = None) -> Dict[str, float]:
    timestamp = time.monotonic() if now is None else now
    if metrics.get('active'):
        started_at = float(metrics.get('started_at') or timestamp)
        seconds = max(0.0, timestamp - started_at)
        tokens = int(metrics.get('tokens') or 0)
        tokens_per_sec = (tokens / seconds) if tokens > 0 and seconds > 0 else 0.0
        return {
            'tokens': float(tokens),
            'seconds': seconds,
            'tokens_per_sec': tokens_per_sec,
            'active': 1.0,
        }
    return {
        'tokens': float(int(metrics.get('last_tokens') or 0)),
        'seconds': float(metrics.get('last_seconds') or 0.0),
        'tokens_per_sec': float(metrics.get('last_tokens_per_sec') or 0.0),
        'active': 0.0,
    }


def build_try_live_stat_lines(
    model: ModelConfig,
    try_status: str,
    pid: Optional[int],
    metrics: Dict[str, object],
    now: Optional[float] = None,
) -> List[str]:
    benchmark_score = float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0)
    benchmark_profile = (getattr(model, 'last_benchmark_profile', '') or '').strip()
    benchmark_text = (
        f'{benchmark_score:.2f} tok/s {benchmark_profile}'.strip()
        if benchmark_score > 0
        else 'not run'
    )
    snapshot = try_live_metric_snapshot(metrics, now=now)
    live_prefix = 'live' if snapshot['active'] else 'last'
    if snapshot['tokens_per_sec'] > 0:
        live_text = (
            f'{live_prefix}: {snapshot["tokens_per_sec"]:.2f} tok/s / '
            f'{int(snapshot["tokens"])} tok / {snapshot["seconds"]:.1f}s'
        )
    else:
        live_text = f'{live_prefix}: waiting / {int(snapshot["tokens"])} tok / {snapshot["seconds"]:.1f}s'
    return [
        f'model: {getattr(model, "name", "") or model.id}',
        f'profile: {model_profile_summary(model)}',
        f'benchmark: {benchmark_text}',
        live_text,
        f'status: {try_status} pid={pid or "-"}',
        f'ctx/output: {model.ctx}/{model.output}',
    ]


def build_try_transcript_items(
    model: ModelConfig,
    messages: List[Dict[str, str]],
    try_status: str,
    width: int,
    user_attr: int = 0,
    assistant_attr: int = 0,
    muted_attr: int = 0,
) -> List[Tuple[str, int]]:
    body_width = max(1, int(width or 1))
    transcript_items: List[Tuple[str, int]] = []
    if not messages:
        intro = (
            'Type a prompt when the server is ready. Esc stops this model and returns to details.'
            if try_status == 'ready'
            else 'Starting the selected model. Input opens when /v1/models is ready.'
        )
        for line in wrap_display_lines(intro, body_width):
            transcript_items.append((line, muted_attr))
        return transcript_items

    model_label = model.alias or model.id
    for item in messages:
        role = str(item.get('role', '') or '')
        content = str(item.get('content', '') or '')
        reasoning = str(item.get('reasoning', '') or '')
        final_notice = str(item.get('final_notice', '') or '')
        if role == 'user':
            prefix = 'you> '
            attr = user_attr
        else:
            prefix = f'{model_label}> '
            attr = assistant_attr
        if reasoning:
            for line in wrap_display_lines(f'{prefix}[reasoning] {reasoning}', body_width):
                transcript_items.append((line, muted_attr))
        body_text = content or ('...' if not final_notice else '')
        if body_text:
            wrapped = wrap_display_lines(prefix + body_text, body_width)
            for line in wrapped:
                transcript_items.append((line, attr))
        if final_notice:
            for line in wrap_display_lines(f'{prefix}{final_notice}', body_width):
                transcript_items.append((line, muted_attr))
        transcript_items.append(('', assistant_attr))
    return transcript_items


def try_input_wrapped_lines(text: str, width: int) -> List[str]:
    prompt_text = f'> {text}' if text else '> '
    return wrap_display_lines(prompt_text, max(1, width)) or ['> ']


def try_input_max_scroll(text: str, width: int, rows: int) -> int:
    return max(0, len(try_input_wrapped_lines(text, width)) - max(1, rows))


def try_input_view(text: str, width: int, rows: int, scroll: int) -> Tuple[List[str], int, bool, bool]:
    visible_rows = max(1, rows)
    lines = try_input_wrapped_lines(text, width)
    max_scroll = max(0, len(lines) - visible_rows)
    clamped_scroll = max(0, min(scroll, max_scroll))
    visible = lines[clamped_scroll: clamped_scroll + visible_rows]
    while len(visible) < visible_rows:
        visible.append('')
    return visible, clamped_scroll, clamped_scroll > 0, clamped_scroll < max_scroll


def try_transcript_scroll_action(key: int) -> str:
    return TRY_TRANSCRIPT_SCROLL_KEYS.get(key, '')


def new_benchmark_run_state(
    model_id: str = '',
    run_kind: str = '',
    label: str = '',
    now: Optional[float] = None,
) -> Dict[str, object]:
    timestamp = time.monotonic() if now is None else now
    return {
        'active': False,
        'model_id': model_id,
        'run_kind': run_kind,
        'label': label,
        'status': 'idle',
        'phase': '',
        'candidate': '',
        'message': '',
        'completed': 0,
        'total': 0,
        'started_at': timestamp,
        'ended_at': 0.0,
        'updated_at': timestamp,
        'records': [],
        'feed': [],
        'commands': [],
        'current_command': '',
        'errors': [],
    }


def benchmark_progress_fraction(completed: object, total: object) -> float:
    try:
        completed_value = max(0.0, float(completed or 0))
        total_value = max(0.0, float(total or 0))
    except Exception:
        return 0.0
    if total_value <= 0:
        return 0.0
    return max(0.0, min(1.0, completed_value / total_value))


def progress_bar_text(completed: object, total: object, width: int) -> str:
    width = max(4, int(width or 4))
    fraction = benchmark_progress_fraction(completed, total)
    filled = max(0, min(width, int(round(width * fraction))))
    return '[' + ('#' * filled) + ('-' * (width - filled)) + ']'


def benchmark_command_lines(state: Dict[str, object], width: int, max_rows: int) -> List[Tuple[str, str]]:
    width = max(8, int(width or 8))
    max_rows = max(1, int(max_rows or 1))
    current = str(state.get('current_command', '') or '')
    commands = [str(item) for item in list(state.get('commands', []) or []) if str(item)]
    if not current and not commands:
        return [('waiting for first command...', 'muted')]
    lines: List[Tuple[str, str]] = []
    if current:
        lines.append((f'current: {current}', 'current'))
    else:
        lines.append(('current: -', 'muted'))
    remaining = max_rows - len(lines)
    if remaining > 0:
        recent = commands[-remaining:]
        for command in recent:
            prefix = 'recent: '
            lines.append((prefix + command, 'muted'))
    return lines[:max_rows]


def benchmark_elapsed_text(state: Dict[str, object], now: Optional[float] = None) -> str:
    ended_at = float(state.get('ended_at') or 0.0)
    timestamp = ended_at if ended_at > 0 else (time.monotonic() if now is None else now)
    started_at = float(state.get('started_at') or timestamp)
    elapsed = max(0.0, timestamp - started_at)
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f'{minutes:02d}:{seconds:02d}'


def benchmark_record_score(record: Dict[str, object]) -> Tuple[str, float]:
    if 'score' in record:
        return 'score', float(record.get('score', 0.0) or 0.0)
    return 'tok/s', float(record.get('tokens_per_sec', 0.0) or 0.0)


def benchmark_row_text(record: Dict[str, object]) -> str:
    label = str(record.get('spectrum_label') or record.get('objective') or record.get('preset') or '-')
    score_label, score = benchmark_record_score(record)
    seconds = float(record.get('seconds', 0.0) or 0.0)
    ctx = int(record.get('ctx', 0) or 0)
    parallel = int(record.get('parallel', 0) or 0)
    slot = int(record.get('ctx_per_slot', 0) or 0) or (ctx // max(1, parallel or 1))
    status = str(record.get('status', '-') or '-')
    suffix_parts = []
    scan_level = str(record.get('scan_level', '') or '')
    if scan_level:
        suffix_parts.append(scan_level)
    if 'exit_code' in record:
        suffix_parts.append(f'exit={int(record.get("exit_code", -1) or -1)}')
    timeout_type = str(record.get('timeout_type', '') or '')
    if timeout_type:
        suffix_parts.append(f'timeout={timeout_type}')
    context_required = int(record.get('context_required', 0) or 0)
    if context_required:
        suffix_parts.append(f'needs~{context_required}tok')
    pressure = str(record.get('process_pressure_level', '') or '')
    if pressure:
        suffix_parts.append(f'pressure={pressure}')
    suffix = (' ' + ' '.join(suffix_parts)) if suffix_parts else ''
    return (
        f'{label[:18]:18} {score:7.2f} {score_label:5} {seconds:6.1f}s '
        f'ctx={ctx:<6} slot={slot:<6} par={parallel:<2} {status}{suffix}'
    )


def benchmark_record_display_items(record: Dict[str, object], attr: int = 0) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = [(benchmark_row_text(record), attr)]
    detail = compact_message(str(record.get('detail', '') or ''))
    if detail:
        items.append((f'  detail: {detail}', attr))
    pressure_detail = compact_message(str(record.get('process_pressure_detail', '') or ''))
    if pressure_detail:
        items.append((f'  process pressure: {pressure_detail}', attr))
    architecture = str(record.get('architecture_label', '') or '')
    if architecture:
        source = str(record.get('classification_source', '') or '')
        items.append((f'  architecture: {architecture}' + (f' from {source}' if source else ''), attr))
    if any(key in record for key in ('required_context', 'configured_context_length', 'actual_ctx_per_slot')):
        required = int(record.get('required_context', 0) or 0)
        configured = int(record.get('configured_context_length', 0) or 0)
        actual = int(record.get('actual_ctx_per_slot', 0) or 0)
        experimental = bool(record.get('experimental_context_override', False))
        suffix = ' experimental override' if experimental else ''
        items.append((f'  context: required={required} configured={configured} actual_slot={actual}{suffix}', attr))
    samples = list(record.get('samples', []) or [])
    for sample in samples[:3]:
        if not isinstance(sample, dict):
            continue
        sample_line = (
            f'  task {sample.get("task", "-")}: {sample.get("status", "-")} '
            f'exit={sample.get("exit_code", "-")} '
            f'timeout={sample.get("timeout_type", "") or "-"} '
            f'unittest_seen={bool(sample.get("unittest_command_seen"))}'
        )
        items.append((sample_line, attr))
        command = compact_message(str(sample.get('command_preview', '') or ''))
        if command:
            items.append((f'    command: {command}', attr))
        config_path = str(sample.get('config_path', '') or '')
        if config_path:
            items.append((f'    config: {config_path}', attr))
        stderr = compact_message(' | '.join(str(line) for line in list(sample.get('stderr_tail', []) or [])[-4:]))
        stdout = compact_message(' | '.join(str(line) for line in list(sample.get('stdout_tail', []) or [])[-4:]))
        if stderr:
            items.append((f'    stderr: {stderr}', attr))
        if stdout:
            items.append((f'    stdout: {stdout}', attr))
    return items


def benchmark_runs_for_model(model: ModelConfig) -> List[Dict[str, object]]:
    runs = list(getattr(model, 'benchmark_runs', []) or [])
    if runs:
        return runs
    rows = list(getattr(model, 'last_benchmark_results', []) or [])
    if not rows:
        return []
    return [{
        'id': 'legacy-latest',
        'kind': 'server',
        'status': getattr(model, 'default_benchmark_status', '') or 'done',
        'summary': getattr(model, 'last_benchmark_profile', '') or 'legacy benchmark',
        'records': rows,
        'winners': getattr(model, 'measured_profiles', {}) or {},
        'started_at': getattr(model, 'default_benchmark_at', '') or '',
        'ended_at': getattr(model, 'default_benchmark_at', '') or '',
        'elapsed_seconds': float(getattr(model, 'last_benchmark_seconds', 0.0) or 0.0),
    }]


def benchmark_run_line(run: Dict[str, object], index: int, selected: bool = False) -> str:
    marker = '>' if selected else ' '
    status = str(run.get('status', '-') or '-')[:8]
    run_id = str(run.get('id', f'run-{index + 1}') or f'run-{index + 1}')
    summary = str(run.get('summary', '') or 'no summary')
    return f'{marker} {index + 1:02d} {status:8} {run_id[:18]:18} {summary}'


def benchmark_record_status_kind(record: Dict[str, object]) -> str:
    status = str(record.get('status', '') or '').lower()
    if record.get('break_point'):
        return 'error'
    if status in ('ok', 'tests passed'):
        return 'success'
    if status in ('probe ok', 'time budget exhausted', 'context too small', 'tests failed', 'not hermes-ready', 'skipped'):
        return 'warning'
    if not status or status == '-':
        return 'normal'
    return 'error'


def agent_record_matches_winner(record: Dict[str, object], winner: Dict[str, object]) -> bool:
    if not record or not winner:
        return False
    comparable = ('ctx', 'ctx_per_slot', 'parallel', 'preset', 'tier', 'status')
    for key in comparable:
        if key in winner and key in record and str(record.get(key)) != str(winner.get(key)):
            return False
    record_score = float(record.get('score', 0.0) or 0.0)
    winner_score = float(winner.get('score', 0.0) or 0.0)
    return winner_score <= 0 or abs(record_score - winner_score) < 0.05


def benchmark_record_roles(record: Dict[str, object], winners: Dict[str, object], run_kind: str) -> List[str]:
    labels: List[str] = []
    raw_labels = str(record.get('spectrum_label', '') or '')
    for label in raw_labels.split(','):
        clean = label.strip()
        if clean and clean not in labels:
            labels.append(clean)
    if run_kind in ('opencode', 'hermes') or 'score' in record:
        winner = {}
        if isinstance(winners, dict):
            winner = winners.get(run_kind) or winners.get('agent') or {}
        if isinstance(winner, dict) and agent_record_matches_winner(record, winner):
            labels.append('Winner')
        if str(record.get('status', '') or '') in ('ok', 'tests passed'):
            labels.append('Passed')
    elif isinstance(winners, dict):
        for key, role_labels in SERVER_WINNER_LABELS.items():
            winner = winners.get(key) or {}
            if isinstance(winner, dict) and record_matches_profile(record, winner):
                labels.extend(role_labels)
    status_kind = benchmark_record_status_kind(record)
    if status_kind == 'warning' and str(record.get('status', '') or '').lower() in ('skipped', 'not hermes-ready', 'context too small'):
        labels.append('Skipped')
    if status_kind == 'error':
        labels.append('Break Point' if record.get('break_point') else 'Failed')
    if not labels:
        measurement = str(record.get('measurement_type', '') or '')
        labels.append('Probe' if measurement == 'probe' or str(record.get('status', '') or '') == 'probe ok' else 'Measured')
    deduped: List[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped


def benchmark_role_priority(labels: List[str]) -> int:
    return min((RANK_ROLE_PRIORITY.get(label, 12) for label in labels), default=12)


def ranked_benchmark_records(run: Dict[str, object]) -> List[Tuple[Dict[str, object], List[str]]]:
    winners = run.get('winners') or {}
    records = list(run.get('records', []) or [])
    kind = str(run.get('kind', '') or '')
    ranked: List[Tuple[Dict[str, object], List[str]]] = [
        (record, benchmark_record_roles(record, winners if isinstance(winners, dict) else {}, kind))
        for record in records
        if isinstance(record, dict)
    ]
    agent_run = kind in ('opencode', 'hermes') or any('score' in row for row in records if isinstance(row, dict))

    def sort_key(item: Tuple[Dict[str, object], List[str]]) -> Tuple[object, ...]:
        record, labels = item
        status_kind = benchmark_record_status_kind(record)
        status_group = 0 if status_kind == 'success' else 1 if status_kind == 'warning' else 2 if status_kind == 'error' else 1
        measurement = str(record.get('measurement_type', 'full') or 'full')
        if not agent_run and status_kind == 'success' and measurement == 'probe':
            status_group = 1
        score_label, score = benchmark_record_score(record)
        ctx_slot = int(record.get('ctx_per_slot', 0) or 0)
        seconds = float(record.get('seconds', 0.0) or 0.0)
        if agent_run:
            return (status_group, -score, -ctx_slot, seconds)
        return (status_group, benchmark_role_priority(labels), -score, -ctx_slot, seconds, score_label)

    return sorted(ranked, key=sort_key)


def benchmark_run_is_agent(run: Dict[str, object]) -> bool:
    kind = str((run or {}).get('kind', '') or '')
    records = list((run or {}).get('records', []) or [])
    return kind in ('opencode', 'hermes') or any(isinstance(row, dict) and 'score' in row for row in records)


def _table_row(values: List[object], widths: List[int]) -> str:
    cells = []
    for value, width in zip(values, widths):
        text = ellipsize(str(value), max(1, int(width or 1)))
        cells.append(f'{text:{max(1, int(width or 1))}}')
    return ' '.join(cells).rstrip()


def _table_rule(widths: List[int]) -> str:
    return ' '.join('-' * max(1, int(width or 1)) for width in widths).rstrip()


def _table_widths(fixed: List[int], total_width: int) -> List[int]:
    width = max(1, int(total_width or 1))
    widths = [max(1, int(item or 1)) for item in fixed]
    spaces = len(widths)
    detail_w = max(1, width - sum(widths) - spaces)
    widths.append(detail_w)
    while sum(widths) + len(widths) - 1 > width:
        shrinkable = [idx for idx, value in enumerate(widths[:-1]) if value > 1]
        if not shrinkable:
            break
        target = max(shrinkable, key=lambda idx: widths[idx])
        widths[target] -= 1
    return widths


def _status_attr_for_record(
    record: Dict[str, object],
    success_attr: int,
    warning_attr: int,
    error_attr: int,
    normal_attr: int,
) -> int:
    status_kind = benchmark_record_status_kind(record)
    if status_kind == 'success':
        return success_attr
    if status_kind == 'warning':
        return warning_attr
    if status_kind == 'error':
        return error_attr
    return normal_attr


def benchmark_rank_line(rank: int, record: Dict[str, object], labels: List[str]) -> str:
    score_label, score = benchmark_record_score(record)
    seconds = float(record.get('seconds', 0.0) or 0.0)
    ctx = int(record.get('ctx', 0) or 0)
    parallel = int(record.get('parallel', 0) or 0)
    slot = int(record.get('ctx_per_slot', 0) or 0) or (ctx // max(1, parallel or 1))
    status = str(record.get('status', '-') or '-')
    role_text = ', '.join(labels or ['Measured'])
    detail = compact_message(str(record.get('detail', '') or ''))
    left = (
        f'#{rank:02d} [{role_text}] {status} '
        f'{score:.2f} {score_label} {seconds:.1f}s '
        f'ctx={ctx} slot={slot} par={parallel}'
    )
    return f'{left}  {detail}' if detail else left


def benchmark_rank_table_items(
    run: Dict[str, object],
    width: int = 120,
    success_attr: int = 0,
    warning_attr: int = 0,
    error_attr: int = 0,
    heading_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    if not run:
        return [('No benchmark run selected.', warning_attr)]
    ranked = ranked_benchmark_records(run)
    if not ranked:
        return [('No benchmark rows yet.', warning_attr)]

    width = max(24, int(width or 120))
    agent_run = benchmark_run_is_agent(run)
    items: List[Tuple[str, int]] = []
    if agent_run:
        if width >= 100:
            widths = _table_widths([4, 18, 22, 8, 6, 7, 7, 7], width)
            headers = ['Rank', 'Role', 'Status', 'Score', 'Sec', 'Pass', 'Ctx', 'Slot', 'Detail']
            columns = ('rank', 'role', 'status', 'score', 'seconds', 'pass', 'ctx', 'slot', 'detail')
        elif width >= 72:
            widths = _table_widths([4, 14, 12, 7, 5, 5, 6], width)
            headers = ['Rank', 'Role', 'Status', 'Score', 'Sec', 'Pass', 'Ctx', 'Detail']
            columns = ('rank', 'role', 'status', 'score', 'seconds', 'pass', 'ctx', 'detail')
        elif width >= 40:
            widths = _table_widths([4, 10, 10, 6, 5], width)
            headers = ['Rank', 'Role', 'Status', 'Score', 'Pass', 'Detail']
            columns = ('rank', 'role', 'status', 'score', 'pass', 'detail')
        else:
            widths = _table_widths([4, 8, 6], width)
            headers = ['Rank', 'Role', 'Score', 'Detail']
            columns = ('rank', 'role', 'score', 'detail')
        items.append((_table_row(headers, widths), heading_attr))
        items.append((_table_rule(widths), heading_attr))
        for index, (record, labels) in enumerate(ranked, 1):
            score_label, score = benchmark_record_score(record)
            seconds = float(record.get('seconds', 0.0) or 0.0)
            ctx = int(record.get('ctx', 0) or 0)
            parallel = max(1, int(record.get('parallel', 1) or 1))
            slot = int(record.get('ctx_per_slot', 0) or 0) or (ctx // parallel)
            passed = record.get('passed')
            tasks = record.get('tasks')
            pass_text = f'{int(passed or 0)}/{int(tasks or 0)}' if passed is not None or tasks is not None else '-'
            detail = compact_message(str(record.get('detail', '') or ''))
            record_values = {
                'rank': f'{index:02d}',
                'role': ', '.join(labels or ['Measured']),
                'status': str(record.get('status', '-') or '-'),
                'score': f'{score:.2f}' if score_label == 'score' else '-',
                'seconds': f'{seconds:.1f}' if seconds > 0 else '-',
                'pass': pass_text,
                'ctx': ctx or '-',
                'slot': slot or '-',
                'detail': detail,
            }
            values = [record_values[column] for column in columns]
            items.append((_table_row(values, widths), _status_attr_for_record(record, success_attr, warning_attr, error_attr, normal_attr)))
        return items

    if width >= 112:
        widths = _table_widths([4, 20, 14, 8, 6, 7, 7, 3, 8, 14], width)
        headers = ['Rank', 'Role', 'Status', 'Tok/s', 'Sec', 'Ctx', 'Slot', 'Par', 'Variant', 'Reason', 'Detail']
        columns = ('rank', 'role', 'status', 'score', 'seconds', 'ctx', 'slot', 'parallel', 'variant', 'reason', 'detail')
    elif width >= 72:
        widths = _table_widths([4, 14, 12, 7, 5, 6, 6], width)
        headers = ['Rank', 'Role', 'Status', 'Tok/s', 'Sec', 'Ctx', 'Slot', 'Detail']
        columns = ('rank', 'role', 'status', 'score', 'seconds', 'ctx', 'slot', 'detail')
    elif width >= 40:
        widths = _table_widths([4, 10, 9, 6, 6], width)
        headers = ['Rank', 'Role', 'Status', 'Tok/s', 'Ctx', 'Detail']
        columns = ('rank', 'role', 'status', 'score', 'ctx', 'detail')
    else:
        widths = _table_widths([4, 8, 5], width)
        headers = ['Rank', 'Role', 'Tok/s', 'Detail']
        columns = ('rank', 'role', 'score', 'detail')
    items.append((_table_row(headers, widths), heading_attr))
    items.append((_table_rule(widths), heading_attr))
    for index, (record, labels) in enumerate(ranked, 1):
        _score_label, score = benchmark_record_score(record)
        seconds = float(record.get('seconds', 0.0) or 0.0)
        ctx = int(record.get('ctx', 0) or 0)
        parallel = int(record.get('parallel', 0) or 0)
        slot = int(record.get('ctx_per_slot', 0) or 0) or (ctx // max(1, parallel or 1))
        reason = str(record.get('selection_reason') or record.get('planner_reason') or record.get('scan_level') or record.get('measurement_type') or '')
        detail = compact_message(str(record.get('detail', '') or ''))
        detail_text = detail or (reason if width < 112 else '')
        record_values = {
            'rank': f'{index:02d}',
            'role': ', '.join(labels or ['Measured']),
            'status': str(record.get('status', '-') or '-'),
            'score': f'{score:.2f}' if score > 0 else '-',
            'seconds': f'{seconds:.1f}' if seconds > 0 else '-',
            'ctx': ctx or '-',
            'slot': slot or '-',
            'parallel': parallel or '-',
            'variant': str(record.get('variant', '') or 'default'),
            'reason': reason or '-',
            'detail': detail_text,
        }
        values = [record_values[column] for column in columns]
        items.append((_table_row(values, widths), _status_attr_for_record(record, success_attr, warning_attr, error_attr, normal_attr)))
    return items


def benchmark_ranking_items(
    run: Dict[str, object],
    width: int = 120,
    success_attr: int = 0,
    warning_attr: int = 0,
    error_attr: int = 0,
    heading_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    return benchmark_rank_table_items(
        run,
        width=width,
        success_attr=success_attr,
        warning_attr=warning_attr,
        error_attr=error_attr,
        heading_attr=heading_attr,
        normal_attr=normal_attr,
    )


def benchmark_ranking_rows(run: Dict[str, object]) -> List[str]:
    return [line for line, _attr in benchmark_ranking_items(run)]


def machine_category_items(
    summary: Dict[str, object],
    accent_attr: int = 0,
    success_attr: int = 0,
    warning_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    rows = list((summary or {}).get('rows', []) or [])
    if not rows:
        return [('No fresh benchmarked models yet. Press D to run Deep Benchmark All.', warning_attr)]
    categories = dict((summary or {}).get('categories', {}) or {})
    order = ('machine_pick', 'fastest_chat', 'longest_context', 'opencode_ready')
    items: List[Tuple[str, int]] = [
        ('Best model for this machine', accent_attr),
        (f'fresh benchmarked models: {len(rows)}', normal_attr),
        ('', normal_attr),
    ]
    for key in order:
        winner = categories.get(key) or {}
        if not winner:
            continue
        label = str(winner.get('label', key.replace('_', ' ').title()) or '')
        model_id = str(winner.get('model_id', '') or '-')
        metric = str(winner.get('metric', '') or '-')
        reason = compact_message(str(winner.get('reason', '') or ''))
        attr = success_attr if key == 'machine_pick' else normal_attr
        items.append((f'{label}: {model_id}  {metric}', attr))
        if reason:
            items.append((f'  {reason}', warning_attr if reason.startswith('fallback') else normal_attr))
    return items


def machine_row_badges(row: Dict[str, object], summary: Dict[str, object]) -> List[str]:
    categories = dict((summary or {}).get('categories', {}) or {})
    badge_specs = (
        ('machine_pick', 'Pick'),
        ('fastest_chat', 'Fast'),
        ('longest_context', 'Ctx'),
        ('opencode_ready', 'Code'),
    )
    model_id = str(row.get('model_id', '') or '')
    badges: List[str] = []
    for key, label in badge_specs:
        winner = categories.get(key) or {}
        if str(winner.get('model_id', '') or '') == model_id and label not in badges:
            badges.append(label)
    return badges


def machine_ranking_items(
    summary: Dict[str, object],
    width: int = 120,
    success_attr: int = 0,
    heading_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    rows = list((summary or {}).get('rows', []) or [])
    if not rows:
        return [('No machine rankings yet. Run Deep Benchmark All first.', normal_attr)]
    width = max(24, int(width or 120))
    if width >= 112:
        widths = _table_widths([4, 13, 18, 7, 7, 7, 7, 7], width)
        headers = ['Rank', 'Badge', 'Model', 'Score', 'Auto', 'Fast', 'Ctx', 'OCtx', 'Reason']
        columns = ('rank', 'badge', 'model', 'score', 'auto', 'fast', 'ctx', 'octx', 'reason')
    elif width >= 72:
        widths = _table_widths([4, 10, 16, 7, 7, 7], width)
        headers = ['Rank', 'Badge', 'Model', 'Score', 'Auto', 'Ctx', 'Reason']
        columns = ('rank', 'badge', 'model', 'score', 'auto', 'ctx', 'reason')
    elif width >= 40:
        widths = _table_widths([4, 14, 6, 7], width)
        headers = ['Rank', 'Model', 'Score', 'Auto', 'Detail']
        columns = ('rank', 'model', 'score', 'auto', 'detail')
    else:
        widths = _table_widths([4, 10, 6], width)
        headers = ['Rank', 'Model', 'Score', 'Detail']
        columns = ('rank', 'model', 'score', 'detail')

    items: List[Tuple[str, int]] = [(_table_row(headers, widths), heading_attr), (_table_rule(widths), heading_attr)]
    pick_id = str(((summary or {}).get('machine_pick') or {}).get('model_id', '') or '')
    for index, row in enumerate(rows, 1):
        badges = ','.join(machine_row_badges(row, summary)) or '-'
        model_id = str(row.get('model_id', '') or '-')
        arch = str(row.get('architecture', '') or '')
        pressure = str(row.get('process_pressure_level', '') or '')
        auto_tps = float(row.get('auto_tokens_per_sec', 0.0) or 0.0)
        fast_tps = float(row.get('fast_tokens_per_sec', 0.0) or 0.0)
        machine_score = float(row.get('machine_score', 0.0) or 0.0)
        ctx_slot = int(row.get('auto_ctx_per_slot', 0) or 0)
        long_ctx = int(row.get('long_ctx_per_slot', 0) or 0)
        opencode_ctx = int(row.get('opencode_ctx_per_slot', 0) or 0)
        reason = compact_message(str(row.get('machine_reason') or row.get('selection_reason') or ''))
        if arch and width >= 72:
            reason = compact_message(f'{arch} {reason}')
        values_by_column = {
            'rank': f'{index:02d}',
            'badge': badges,
            'model': model_id,
            'score': f'{machine_score:.2f}',
            'auto': f'{auto_tps:.2f}',
            'fast': f'{fast_tps:.2f}',
            'ctx': long_ctx or ctx_slot or '-',
            'octx': opencode_ctx or '-',
            'reason': reason,
            'detail': f'{badges} {arch} {auto_tps:.2f}t/s ctx={ctx_slot} pressure={pressure or "-"}',
        }
        values = [values_by_column[column] for column in columns]
        attr = success_attr if model_id == pick_id else normal_attr
        items.append((_table_row(values, widths), attr))
    return items


def machine_gap_items(
    app: AppConfig,
    summary: Dict[str, object],
    warning_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    fresh_ids = {str(row.get('model_id', '') or '') for row in list((summary or {}).get('rows', []) or [])}
    items: List[Tuple[str, int]] = []
    for model in getattr(app, 'models', []) or []:
        if model.id in fresh_ids:
            continue
        include, reason = deep_benchmark_model_decision(app, model, force=False)
        status = (getattr(model, 'default_benchmark_status', '') or 'unbenchmarked').strip() or 'unbenchmarked'
        attr = warning_attr if include or not benchmark_profile_is_fresh(app, model) else normal_attr
        items.append((f'{model.id}: {status} - {reason}', attr))
    if not items:
        return [('All enabled models have fresh machine-ranking data.', normal_attr)]
    return items


def reduce_benchmark_event(
    state: Dict[str, object],
    payload: Dict[str, object],
    now: Optional[float] = None,
) -> Dict[str, object]:
    timestamp = time.monotonic() if now is None else now
    event = str(payload.get('event', '') or '')
    if event == 'benchmark_started':
        state.clear()
        state.update(new_benchmark_run_state(
            model_id=str(payload.get('model_id', '') or ''),
            run_kind=str(payload.get('run_kind', '') or ''),
            label=str(payload.get('message', '') or 'benchmark'),
            now=timestamp,
        ))
        state['active'] = True
        state['status'] = 'running'
    elif not state:
        state.update(new_benchmark_run_state(now=timestamp))

    state['updated_at'] = timestamp
    for key in (
        'model_id',
        'run_kind',
        'phase',
        'candidate',
        'message',
        'batch_completed',
        'batch_total',
        'batch_skipped',
        'batch_failed',
        'batch_restored',
    ):
        value = payload.get(key)
        if value not in (None, ''):
            state[key] = value
    if 'completed' in payload:
        state['completed'] = max(0, int(payload.get('completed') or 0))
    if 'total' in payload:
        state['total'] = max(0, int(payload.get('total') or 0))

    command = compact_message(str(payload.get('command') or payload.get('command_preview') or ''))
    if command:
        state['current_command'] = command
        commands = list(state.get('commands', []) or [])
        if not commands or commands[-1] != command:
            commands.append(command)
        state['commands'] = commands[-BENCHMARK_COMMAND_LIMIT:]

    message = compact_message(str(payload.get('message', '') or ''))
    if message:
        pure_command = command and message == command
        if not pure_command:
            feed = list(state.get('feed', []) or [])
            if not feed or feed[-1] != message:
                feed.append(message)
            state['feed'] = feed[-BENCHMARK_FEED_LIMIT:]
    if 'records' in payload and isinstance(payload.get('records'), list):
        state['records'] = list(payload.get('records') or [])[-BENCHMARK_RECORD_LIMIT:]
    elif event == 'benchmark_result' and isinstance(payload.get('record'), dict):
        records = list(state.get('records', []) or [])
        records.append(dict(payload.get('record') or {}))
        state['records'] = records[-BENCHMARK_RECORD_LIMIT:]

    if event in ('benchmark_error', 'benchmark_aborted') or is_error_message(message):
        errors = list(state.get('errors', []) or [])
        if message:
            errors.append(message)
        state['errors'] = errors[-BENCHMARK_FEED_LIMIT:]
    if event == 'benchmark_done':
        state['active'] = False
        state['status'] = 'done'
        state['ended_at'] = timestamp
        if int(state.get('total', 0) or 0) <= 0:
            state['total'] = int(state.get('completed', 0) or 0)
    elif event == 'benchmark_error':
        state['active'] = False
        state['status'] = 'failed'
        state['ended_at'] = timestamp
    elif event == 'benchmark_aborted':
        state['active'] = False
        state['status'] = 'aborted'
        state['ended_at'] = timestamp
    elif event and event != 'benchmark_started':
        state['status'] = 'running' if state.get('active') else str(state.get('status') or 'idle')
    return state


def form_field(key: str, label: str, default: str = '', hint: str = '') -> Dict[str, str]:
    return {
        'key': key,
        'label': label,
        'default': str(default or ''),
        'hint': hint,
    }


def _clip_form_value(value: str, width: int, cursor: int) -> Tuple[str, int]:
    text = str(value or '')
    width = max(1, int(width or 1))
    cursor = max(0, min(int(cursor or 0), len(text)))
    if len(text) <= width:
        return text.ljust(width), cursor
    start = max(0, cursor - width + 1)
    end = min(len(text), start + width)
    if end - start < width:
        start = max(0, end - width)
    return text[start:end].ljust(width), cursor - start


def prompt_form(
    stdscr,
    colors,
    title: str,
    fields: List[Dict[str, str]],
    validator: Callable[[Dict[str, str]], Tuple[Optional[object], Dict[str, str]]],
    footer_hint: str = '',
    presets: Optional[List[str]] = None,
) -> Optional[object]:
    h, w = stdscr.getmaxyx()
    box_w = min(108, max(64, w - 6))
    box_h = min(max(14, h - 4), 26)
    if h < 16 or w < 68:
        return None
    box_x = max(2, (w - box_w) // 2)
    box_y = max(1, (h - box_h) // 2)
    modal = curses.newwin(box_h, box_w, box_y, box_x)
    modal.keypad(True)
    values = {field['key']: str(field.get('default', '') or '') for field in fields}
    selected = 0
    cursor = len(values[fields[0]['key']]) if fields else 0
    scroll = 0
    errors: Dict[str, str] = {}
    status = 'Edit values in place. Ctrl+S/F2 saves. Esc cancels.'
    preset_values = list(presets or [])[:9]
    label_w = max(16, min(28, max(len(field['label']) for field in fields) + 1 if fields else 18))
    stdscr.nodelay(False)
    try:
        previous_cursor = curses.curs_set(1)
    except curses.error:
        previous_cursor = 0
    try:
        while True:
            scroll = clamp_scroll(scroll, len(fields), max(1, box_h - 9))
            if selected < scroll:
                scroll = selected
            elif selected >= scroll + max(1, box_h - 9):
                scroll = selected - max(1, box_h - 9) + 1
            modal.erase()
            draw_box(modal, 0, 0, box_h - 1, box_w, title, colors['accent'] | curses.A_BOLD, colors['accent'])
            body_rows = max(1, box_h - 9)
            visible = fields[scroll: scroll + body_rows]
            for idx, field in enumerate(visible):
                absolute = scroll + idx
                key = field['key']
                label = field['label']
                line_y = 2 + idx
                label_text = f'{label}:'
                attr = colors['selection'] | curses.A_BOLD if absolute == selected else colors['panel']
                safe_addstr(modal, line_y, 2, ellipsize(label_text, label_w), attr)
                value_width = max(8, box_w - label_w - 6)
                display, cursor_x = _clip_form_value(values.get(key, ''), value_width, cursor if absolute == selected else len(values.get(key, '')))
                value_attr = colors['selection'] | curses.A_BOLD if absolute == selected else curses.A_NORMAL
                safe_addstr(modal, line_y, 2 + label_w, display[:value_width], value_attr)
                if absolute == selected:
                    try:
                        modal.move(line_y, 2 + label_w + cursor_x)
                    except curses.error:
                        pass
            active_field = fields[selected] if fields else {'key': '', 'hint': '', 'label': ''}
            hint = active_field.get('hint', '') or footer_hint or 'Use Tab/Shift-Tab or arrows to move.'
            safe_addstr(modal, box_h - 6, 2, ellipsize(f'Field: {active_field.get("label", "-")} | {hint}', box_w - 4), colors['muted'])
            if preset_values:
                preset_text = '  '.join(f'[{idx + 1}] {ellipsize(value, 18)}' for idx, value in enumerate(preset_values))
                safe_addstr(modal, box_h - 5, 2, ellipsize(f'Presets: {preset_text}', box_w - 4), colors['muted'])
            else:
                safe_addstr(modal, box_h - 5, 2, ellipsize(status, box_w - 4), colors['muted'])
            error_lines = list(errors.items())[:2]
            for idx in range(2):
                line = ''
                attr = colors['muted']
                if idx < len(error_lines):
                    field_name, error = error_lines[idx]
                    line = f'{field_name}: {error}'
                    attr = colors['error'] | curses.A_BOLD
                safe_addstr(modal, box_h - 4 + idx, 2, ellipsize(line, box_w - 4), attr)
            safe_addstr(modal, box_h - 2, 2, ellipsize('[Ctrl+S/F2] save  [Esc] cancel  [Ctrl+U] clear field', box_w - 4), colors['accent'] | curses.A_BOLD)
            modal.refresh()

            key = modal.getch()
            if key in (27,):
                return None
            if key in (19, getattr(curses, 'KEY_F2', -1)):
                result, validation_errors = validator(dict(values))
                if validation_errors:
                    errors = validation_errors
                    first_error_key = next(iter(validation_errors))
                    for idx, field in enumerate(fields):
                        if field['key'] == first_error_key:
                            selected = idx
                            cursor = len(values.get(field['key'], ''))
                            break
                    status = 'Fix the highlighted field errors and save again.'
                    continue
                return result
            if key in (curses.KEY_UP,):
                selected = max(0, selected - 1)
                cursor = len(values.get(fields[selected]['key'], ''))
                continue
            if key in (curses.KEY_DOWN, 9, 10, 13, curses.KEY_ENTER):
                selected = min(len(fields) - 1, selected + 1)
                cursor = len(values.get(fields[selected]['key'], ''))
                continue
            if key in (getattr(curses, 'KEY_BTAB', -999),):
                selected = max(0, selected - 1)
                cursor = len(values.get(fields[selected]['key'], ''))
                continue
            if preset_values and ord('1') <= key <= ord(str(min(9, len(preset_values)))):
                values[fields[selected]['key']] = preset_values[key - ord('1')]
                cursor = len(values[fields[selected]['key']])
                errors.pop(fields[selected]['key'], None)
                continue
            current_key = fields[selected]['key']
            current_value = values.get(current_key, '')
            if key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
                continue
            if key == curses.KEY_RIGHT:
                cursor = min(len(current_value), cursor + 1)
                continue
            if key == curses.KEY_HOME:
                cursor = 0
                continue
            if key == curses.KEY_END:
                cursor = len(current_value)
                continue
            if key in (21,):
                values[current_key] = ''
                cursor = 0
                errors.pop(current_key, None)
                continue
            if key in (curses.KEY_BACKSPACE, 127, 8):
                if cursor > 0:
                    values[current_key] = current_value[: cursor - 1] + current_value[cursor:]
                    cursor -= 1
                    errors.pop(current_key, None)
                continue
            if key == curses.KEY_DC:
                if cursor < len(current_value):
                    values[current_key] = current_value[:cursor] + current_value[cursor + 1:]
                    errors.pop(current_key, None)
                continue
            if 32 <= key <= 126:
                values[current_key] = current_value[:cursor] + chr(key) + current_value[cursor:]
                cursor += 1
                errors.pop(current_key, None)
                continue
    finally:
        try:
            curses.curs_set(previous_cursor)
        except curses.error:
            pass
        stdscr.touchwin()
        stdscr.nodelay(True)


def model_form_fields(initial: ModelConfig) -> List[Dict[str, str]]:
    return [
        form_field('id', 'id', initial.id, 'required; stable short id'),
        form_field('name', 'name', initial.name, 'required; user-facing label'),
        form_field('path', 'path', initial.path, 'GGUF path or vLLM model reference'),
        form_field('alias', 'alias', initial.alias or initial.id, 'OpenAI-style model name served on the port'),
        form_field('runtime', 'runtime', getattr(initial, 'runtime', 'llama.cpp'), 'llama.cpp or vllm'),
        form_field('optimize_mode', 'optimize_mode', getattr(initial, 'optimize_mode', 'max_context_safe'), 'max_context_safe/manual/best/max_context/tokens_per_sec/opencode_ready'),
        form_field('optimize_tier', 'optimize_tier', getattr(initial, 'optimize_tier', 'moderate'), 'safe/moderate/extreme'),
        form_field('port', 'port', str(initial.port), 'required integer port'),
        form_field('host', 'host', initial.host, 'bind host, usually 127.0.0.1'),
        form_field('ctx', 'ctx', str(initial.ctx), 'total context length'),
        form_field('ctx_min', 'ctx_min', str(getattr(initial, 'ctx_min', 2048)), 'minimum ctx for adaptive tuning'),
        form_field('ctx_max', 'ctx_max', str(getattr(initial, 'ctx_max', 131072)), 'maximum ctx for adaptive tuning'),
        form_field('threads', 'threads', str(initial.threads), 'CPU worker threads'),
        form_field('ngl', 'ngl', str(initial.ngl), 'GPU layers for llama.cpp'),
        form_field('temp', 'temp', str(initial.temp), 'sampling temperature'),
        form_field('parallel', 'parallel', str(initial.parallel), 'parallel request slots'),
        form_field('memory_reserve_percent', 'memory_reserve_percent', str(getattr(initial, 'memory_reserve_percent', 25)), 'RAM/VRAM headroom percent'),
        form_field('cache_ram', 'cache_ram', str(initial.cache_ram), 'cache size in MiB, 0 to auto'),
        form_field('output', 'output', str(initial.output), 'max output tokens'),
        form_field('enabled', 'enabled', str(initial.enabled).lower(), 'true/false'),
        form_field('flash_attn', 'flash_attn', str(initial.flash_attn).lower(), 'true/false'),
        form_field('jinja', 'jinja', str(initial.jinja).lower(), 'true/false'),
        form_field('favorite', 'favorite', str(getattr(initial, 'favorite', False)).lower(), 'true/false'),
        form_field('tags', 'tags', ', '.join(list(getattr(initial, 'tags', []) or [])), 'comma-separated: coding/autocomplete/long-context/fast-chat/custom'),
        form_field('extra_args', 'extra_args', ' '.join(initial.extra_args), 'space-separated extra runtime flags'),
    ]


def parse_model_form_answers(answers: Dict[str, str], initial: Optional[ModelConfig] = None) -> Tuple[Optional[ModelConfig], Dict[str, str]]:
    initial = initial or ModelConfig(id='', name='', path='', alias='', port=8080)
    cleaned = {key: str(value or '').strip() for key, value in answers.items()}
    errors: Dict[str, str] = {}
    if not cleaned.get('id'):
        errors['id'] = 'id is required'
    if not cleaned.get('name'):
        errors['name'] = 'name is required'
    if not cleaned.get('path'):
        errors['path'] = 'path is required'
    runtime = normalize_choice(cleaned.get('runtime', ''), VALID_RUNTIMES, 'llama.cpp')
    if runtime != cleaned.get('runtime', '').strip().lower():
        errors['runtime'] = 'runtime must be llama.cpp or vllm'
    optimize_mode = cleaned.get('optimize_mode', '').strip().lower() or 'max_context_safe'
    if optimize_mode not in VALID_OPTIMIZE_MODES:
        errors['optimize_mode'] = 'unsupported optimize mode'
    optimize_tier = cleaned.get('optimize_tier', '').strip().lower() or 'moderate'
    if optimize_tier not in VALID_OPTIMIZE_TIERS:
        errors['optimize_tier'] = 'tier must be safe, moderate, or extreme'

    def parse_int(name: str, minimum: int = 0) -> int:
        try:
            value = int(cleaned.get(name, '0') or 0)
        except ValueError:
            errors[name] = 'must be an integer'
            return 0
        if value < minimum:
            errors[name] = f'must be >= {minimum}'
        return value

    def parse_float(name: str, minimum: float = 0.0) -> float:
        try:
            value = float(cleaned.get(name, '0') or 0.0)
        except ValueError:
            errors[name] = 'must be a number'
            return 0.0
        if value < minimum:
            errors[name] = f'must be >= {minimum}'
        return value

    port = parse_int('port', 1)
    ctx = parse_int('ctx', 1)
    ctx_min = parse_int('ctx_min', 1)
    ctx_max = parse_int('ctx_max', ctx_min if ctx_min > 0 else 1)
    threads = parse_int('threads', 1)
    ngl = parse_int('ngl', 0)
    parallel = parse_int('parallel', 1)
    memory_reserve = parse_int('memory_reserve_percent', 0)
    cache_ram = parse_int('cache_ram', 0)
    output = parse_int('output', 1)
    temp = parse_float('temp', 0.0)
    if ctx_max and ctx_min and ctx_max < ctx_min:
        errors['ctx_max'] = 'ctx_max must be >= ctx_min'
    host = cleaned.get('host', '').strip()
    if not host:
        errors['host'] = 'host is required'
    for key in ('enabled', 'flash_attn', 'jinja', 'favorite'):
        try:
            parse_bool_text(cleaned.get(key, ''), key)
        except ValueError as exc:
            errors[key] = str(exc)
    if errors:
        return None, errors
    model = ModelConfig(
        id=cleaned['id'],
        name=cleaned['name'],
        path=cleaned['path'],
        alias=cleaned.get('alias', '').strip() or cleaned['id'],
        port=port,
        host=host,
        ctx=ctx,
        ctx_min=ctx_min,
        ctx_max=ctx_max,
        threads=threads,
        ngl=ngl,
        temp=temp,
        parallel=parallel,
        optimize_mode=optimize_mode,
        optimize_tier=optimize_tier,
        memory_reserve_percent=memory_reserve,
        cache_ram=cache_ram,
        output=output,
        enabled=parse_bool_text(cleaned['enabled'], 'enabled'),
        runtime=runtime,
        flash_attn=parse_bool_text(cleaned['flash_attn'], 'flash_attn'),
        jinja=parse_bool_text(cleaned['jinja'], 'jinja'),
        favorite=parse_bool_text(cleaned['favorite'], 'favorite'),
        source=getattr(initial, 'source', 'manual'),
        last_used_at=str(getattr(initial, 'last_used_at', '') or ''),
        sort_rank=int(getattr(initial, 'sort_rank', 0) or 0),
        tags=[item.strip() for item in cleaned.get('tags', '').split(',') if item.strip()],
        verification_status=str(getattr(initial, 'verification_status', 'unknown') or 'unknown'),
        verification_at=str(getattr(initial, 'verification_at', '') or ''),
        verification_fingerprint=str(getattr(initial, 'verification_fingerprint', '') or ''),
        verification_summary=str(getattr(initial, 'verification_summary', '') or ''),
        verification_results=dict(getattr(initial, 'verification_results', {}) or {}),
        extra_args=cleaned.get('extra_args', '').split() if cleaned.get('extra_args') else [],
    )
    return model, {}


SETTINGS_SECTIONS = [
    ('runtime', 'Runtime'),
    ('roots', 'Model Roots'),
    ('opencode', 'OpenCode'),
    ('continue', 'Continue'),
    ('hermes', 'Hermes'),
    ('ui', 'UI'),
]


def settings_form_fields(app: AppConfig, section: str = 'all') -> List[Dict[str, str]]:
    o = app.opencode
    continue_settings = app.continue_settings
    hermes = app.hermes
    groups = {
        'runtime': [
        form_field('llama_server', 'llama_server', app.llama_server, 'command or path to llama.cpp server'),
        form_field('vllm_command', 'vllm_command', app.vllm_command, 'command used to start vLLM'),
        ],
        'roots': [
        form_field('hf_cache_root', 'hf_cache_root', app.hf_cache_root, 'Hugging Face GGUF cache root'),
        form_field('llm_models_cache_root', 'llm_models_cache_root', app.llm_models_cache_root, 'local model cache root'),
        form_field('llmfit_cache_root', 'llmfit_cache_root', app.llmfit_cache_root, 'llmfit cache root'),
        form_field('lm_studio_model_roots', 'lm_studio_model_roots', getattr(app, 'lm_studio_model_roots', ''), 'comma-separated LM Studio model roots'),
        ],
        'opencode': [
        form_field('opencode_path', 'opencode_path', o.path, 'path to opencode config.json'),
        form_field('opencode_backup_dir', 'opencode_backup_dir', o.backup_dir, 'backup directory for generated OpenCode configs'),
        form_field('default_model_id', 'default_model_id', o.default_model_id, 'OpenCode main model id'),
        form_field('small_model_id', 'small_model_id', o.small_model_id, 'OpenCode small/autocomplete model id'),
        form_field('build_model_id', 'build_model_id', o.build_model_id, 'OpenCode build model id'),
        form_field('plan_model_id', 'plan_model_id', o.plan_model_id, 'OpenCode plan model id'),
        form_field('instructions', 'instructions', ', '.join(o.instructions), 'comma-separated OpenCode instruction files'),
        form_field('build_prompt', 'build_prompt', o.build_prompt, 'OpenCode build agent prompt'),
        form_field('plan_prompt', 'plan_prompt', o.plan_prompt, 'OpenCode plan agent prompt'),
        form_field('timeout', 'timeout', str(o.timeout), 'OpenCode request timeout in ms'),
        form_field('chunk_timeout', 'chunk_timeout', str(o.chunk_timeout), 'OpenCode chunk timeout in ms'),
        form_field('terminal_command', 'terminal_command', getattr(o, 'terminal_command', ''), 'custom terminal template using {title} {cwd} {cmd}'),
        form_field('last_workspace_path', 'last_workspace_path', getattr(o, 'last_workspace_path', ''), 'last OpenCode workspace path'),
        ],
        'continue': [
        form_field('continue_path', 'continue_path', getattr(continue_settings, 'path', ''), 'path to Continue config.yaml'),
        form_field('continue_backup_dir', 'continue_backup_dir', getattr(continue_settings, 'backup_dir', ''), 'backup directory for generated Continue configs'),
        form_field('continue_default_model_id', 'continue_default_model_id', getattr(continue_settings, 'default_model_id', ''), 'Continue chat model id; blank uses OpenCode main'),
        form_field('continue_edit_model_id', 'continue_edit_model_id', getattr(continue_settings, 'edit_model_id', ''), 'Continue edit/apply model id; blank uses OpenCode build'),
        form_field('continue_autocomplete_model_id', 'continue_autocomplete_model_id', getattr(continue_settings, 'autocomplete_model_id', ''), 'Continue autocomplete model id; blank uses OpenCode small'),
        form_field('continue_merge_mode', 'continue_merge_mode', getattr(continue_settings, 'merge_mode', 'preserve_sections'), 'preserve_sections or managed_file'),
        ],
        'hermes': [
        form_field('hermes_command', 'hermes_command', getattr(hermes, 'command', 'hermes'), 'Hermes command name or path'),
        form_field('hermes_home_root', 'hermes_home_root', getattr(hermes, 'home_root', ''), 'Hermes isolated home root'),
        form_field('hermes_default_model_id', 'hermes_default_model_id', getattr(hermes, 'default_model_id', ''), 'Hermes default model id'),
        form_field('hermes_code_model_id', 'hermes_code_model_id', getattr(hermes, 'code_model_id', ''), 'Hermes coding model id'),
        form_field('hermes_toolsets', 'hermes_toolsets', ', '.join(getattr(hermes, 'toolsets', []) or []), 'comma-separated Hermes toolsets'),
        form_field('hermes_max_turns', 'hermes_max_turns', str(getattr(hermes, 'max_turns', 20)), 'maximum Hermes turns'),
        form_field('hermes_quiet', 'hermes_quiet', str(getattr(hermes, 'quiet', True)).lower(), 'true/false'),
        form_field('hermes_min_context_tokens', 'hermes_min_context_tokens', str(getattr(hermes, 'min_context_tokens', 64000)), 'required ctx/slot for Hermes readiness'),
        form_field('hermes_allow_experimental_context_override', 'hermes_allow_experimental_context_override', str(getattr(hermes, 'allow_experimental_context_override', False)).lower(), 'true/false'),
        form_field('hermes_experimental_context_override_tokens', 'hermes_experimental_context_override_tokens', str(getattr(hermes, 'experimental_context_override_tokens', 0)), '0 disables override'),
        form_field('hermes_terminal_command', 'hermes_terminal_command', getattr(hermes, 'terminal_command', ''), 'custom Hermes terminal template'),
        form_field('hermes_last_workspace_path', 'hermes_last_workspace_path', getattr(hermes, 'last_workspace_path', ''), 'last Hermes workspace path'),
        ],
        'ui': [
        form_field('preferred_sort', 'preferred_sort', getattr(app.ui, 'preferred_sort', 'port'), 'favorites/recent/name/benchmark/context/port'),
        form_field('detail_density', 'detail_density', getattr(app.ui, 'detail_density', 'simple'), 'simple or advanced'),
        ],
    }
    if section == 'all':
        fields: List[Dict[str, str]] = []
        for key, _label in SETTINGS_SECTIONS:
            fields.extend(groups[key])
        return fields
    return list(groups.get(section, []))


def settings_form_answers_from_app(app: AppConfig) -> Dict[str, str]:
    return {field['key']: str(field.get('default', '') or '') for field in settings_form_fields(app, 'all')}


def parse_settings_form_answers(answers: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
    cleaned = {key: str(value or '').strip() for key, value in answers.items()}
    errors: Dict[str, str] = {}

    def parse_int(name: str, minimum: int = 0) -> int:
        try:
            value = int(cleaned.get(name, '0') or 0)
        except ValueError:
            errors[name] = 'must be an integer'
            return 0
        if value < minimum:
            errors[name] = f'must be >= {minimum}'
        return value

    for key in ('hermes_quiet', 'hermes_allow_experimental_context_override'):
        try:
            parse_bool_text(cleaned.get(key, ''), key)
        except ValueError as exc:
            errors[key] = str(exc)
    preferred_sort = normalize_choice(cleaned.get('preferred_sort', ''), tuple(key for key, _label in SORT_OPTIONS), 'port')
    if preferred_sort != (cleaned.get('preferred_sort', '').lower() or 'port'):
        errors['preferred_sort'] = 'unsupported sort mode'
    detail_density = normalize_choice(cleaned.get('detail_density', ''), tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'simple')
    if detail_density != (cleaned.get('detail_density', '').lower() or 'simple'):
        errors['detail_density'] = 'detail_density must be simple or advanced'
    continue_merge_mode = cleaned.get('continue_merge_mode', 'preserve_sections') or 'preserve_sections'
    if continue_merge_mode not in CONTINUE_MERGE_MODES:
        errors['continue_merge_mode'] = 'must be preserve_sections or managed_file'
    payload = {
        'llama_server': cleaned.get('llama_server', ''),
        'vllm_command': cleaned.get('vllm_command', ''),
        'hf_cache_root': cleaned.get('hf_cache_root', ''),
        'llm_models_cache_root': cleaned.get('llm_models_cache_root', ''),
        'llmfit_cache_root': cleaned.get('llmfit_cache_root', ''),
        'lm_studio_model_roots': ', '.join([item.strip() for item in cleaned.get('lm_studio_model_roots', '').split(',') if item.strip()]),
        'opencode': {
            'path': cleaned.get('opencode_path', ''),
            'backup_dir': cleaned.get('opencode_backup_dir', ''),
            'default_model_id': cleaned.get('default_model_id', ''),
            'small_model_id': cleaned.get('small_model_id', ''),
            'build_model_id': cleaned.get('build_model_id', ''),
            'plan_model_id': cleaned.get('plan_model_id', ''),
            'instructions': [item.strip() for item in cleaned.get('instructions', '').split(',') if item.strip()],
            'build_prompt': cleaned.get('build_prompt', ''),
            'plan_prompt': cleaned.get('plan_prompt', ''),
            'timeout': parse_int('timeout', 1),
            'chunk_timeout': parse_int('chunk_timeout', 1),
            'terminal_command': cleaned.get('terminal_command', ''),
            'last_workspace_path': cleaned.get('last_workspace_path', ''),
        },
        'continue': {
            'path': cleaned.get('continue_path', ''),
            'backup_dir': cleaned.get('continue_backup_dir', ''),
            'default_model_id': cleaned.get('continue_default_model_id', ''),
            'edit_model_id': cleaned.get('continue_edit_model_id', ''),
            'autocomplete_model_id': cleaned.get('continue_autocomplete_model_id', ''),
            'merge_mode': continue_merge_mode,
        },
        'hermes': {
            'command': cleaned.get('hermes_command', '') or 'hermes',
            'home_root': cleaned.get('hermes_home_root', ''),
            'default_model_id': cleaned.get('hermes_default_model_id', ''),
            'code_model_id': cleaned.get('hermes_code_model_id', ''),
            'toolsets': [item.strip() for item in cleaned.get('hermes_toolsets', '').split(',') if item.strip()],
            'max_turns': parse_int('hermes_max_turns', 1),
            'quiet': parse_bool_text(cleaned.get('hermes_quiet', 'true') or 'true', 'hermes_quiet') if 'hermes_quiet' not in errors else True,
            'min_context_tokens': parse_int('hermes_min_context_tokens', 1),
            'allow_experimental_context_override': (
                parse_bool_text(cleaned.get('hermes_allow_experimental_context_override', 'false') or 'false', 'hermes_allow_experimental_context_override')
                if 'hermes_allow_experimental_context_override' not in errors
                else False
            ),
            'experimental_context_override_tokens': parse_int('hermes_experimental_context_override_tokens', 0),
            'terminal_command': cleaned.get('hermes_terminal_command', ''),
            'last_workspace_path': cleaned.get('hermes_last_workspace_path', ''),
        },
        'ui': {
            'preferred_sort': preferred_sort,
            'detail_density': detail_density,
        },
    }
    if errors:
        return None, errors
    return payload, {}


def workspace_form_fields(default: str) -> List[Dict[str, str]]:
    return [form_field('workspace', 'workspace', default, 'existing directory path')]


def parse_workspace_form_answers(app: AppConfig, answers: Dict[str, str]) -> Tuple[Optional[str], Dict[str, str]]:
    workspace = str((answers or {}).get('workspace', '') or '').strip()
    valid, workspace_path, reason = app.validate_workspace_path(workspace)
    if not valid or workspace_path is None:
        return None, {'workspace': reason}
    return str(workspace_path), {}


def prompt_model(stdscr, colors, title: str, initial: Optional[ModelConfig] = None) -> Optional[ModelConfig]:
    initial = initial or ModelConfig(id='', name='', path='', alias='', port=8080)
    return prompt_form(
        stdscr,
        colors,
        title,
        model_form_fields(initial),
        lambda answers: parse_model_form_answers(answers, initial=initial),
        footer_hint='Model forms stay in the TUI now, and invalid fields keep your typed values.',
    )


def prompt_settings(stdscr, colors, app: AppConfig) -> bool:
    section = prompt_modal_choice(
        stdscr,
        colors,
        'Settings Section',
        [(str(idx + 1), label, key) for idx, (key, label) in enumerate(SETTINGS_SECTIONS)] + [('q', 'Cancel', 'cancel')],
    )
    if section == 'cancel':
        return False
    base_answers = settings_form_answers_from_app(app)

    def validate_section(answers: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
        merged = dict(base_answers)
        merged.update(answers)
        parsed, errors = parse_settings_form_answers(merged)
        section_keys = {field['key'] for field in settings_form_fields(app, section)}
        visible_errors = {key: value for key, value in errors.items() if key in section_keys}
        hidden_errors = {key: value for key, value in errors.items() if key not in section_keys}
        if visible_errors:
            return None, visible_errors
        if hidden_errors:
            return None, {'settings': 'Open another settings section to fix saved invalid values.'}
        return parsed, {}

    parsed = prompt_form(
        stdscr,
        colors,
        f'Settings: {dict(SETTINGS_SECTIONS).get(section, section)}',
        settings_form_fields(app, section),
        validate_section,
        footer_hint='Settings are split by section; reopen settings to edit another area.',
    )
    if not parsed:
        return False
    app.llama_server = parsed['llama_server']
    app.vllm_command = parsed['vllm_command']
    app.hf_cache_root = parsed['hf_cache_root']
    app.llm_models_cache_root = parsed['llm_models_cache_root']
    app.llmfit_cache_root = parsed['llmfit_cache_root']
    app.lm_studio_model_roots = parsed['lm_studio_model_roots']
    for key, value in parsed['opencode'].items():
        setattr(app.opencode, key, value)
    for key, value in parsed['continue'].items():
        setattr(app.continue_settings, key, value)
    for key, value in parsed['hermes'].items():
        setattr(app.hermes, key, value)
    for key, value in parsed['ui'].items():
        setattr(app.ui, key, value)
    app.save()
    return True


def prompt_workspace(stdscr, colors, app: AppConfig, runtime: str = 'opencode') -> Optional[str]:
    default = getattr(app.workspace_settings(runtime), 'last_workspace_path', '') or str(Path.cwd())
    presets = app.workspace_presets(runtime)
    return prompt_form(
        stdscr,
        colors,
        'Hermes Workspace' if runtime == 'hermes' else 'OpenCode Workspace',
        workspace_form_fields(default),
        lambda answers: parse_workspace_form_answers(app, answers),
        footer_hint='Pick a recent preset with 1-9, or type a path directly.',
        presets=presets,
    )


def prompt_search_query(stdscr, colors, current: str) -> Optional[str]:
    return prompt_form(
        stdscr,
        colors,
        'Search Models',
        [form_field('search', 'search', current, 'match id, name, alias, path, runtime, source, status')],
        lambda answers: (str((answers or {}).get('search', '') or ''), {}),
        footer_hint='Leave blank to clear the current search.',
    )


def browser_filter_fields(runtime_filter: str, source_filter: str, status_filter: str, tag_filter: str = 'all') -> List[Dict[str, str]]:
    return [
        form_field('runtime_filter', 'runtime_filter', runtime_filter, 'all/llama.cpp/vllm'),
        form_field('source_filter', 'source_filter', source_filter, 'all/manual/huggingface/llmfit/llm-models/lm-studio'),
        form_field('status_filter', 'status_filter', status_filter, 'all/READY/LOADING/STARTING/STOPPED/ERROR/fresh/stale/missing/failed/pending/running'),
        form_field('tag_filter', 'tag_filter', tag_filter, 'all/coding/autocomplete/long-context/fast-chat/custom tag'),
    ]


def parse_browser_filter_answers(answers: Dict[str, str]) -> Tuple[Optional[Tuple[str, str, str, str]], Dict[str, str]]:
    cleaned = {key: str(value or '').strip() for key, value in answers.items()}
    runtime = cleaned.get('runtime_filter', '').lower() or 'all'
    source = cleaned.get('source_filter', '').lower() or 'all'
    status = cleaned.get('status_filter', '') or 'all'
    tag = cleaned.get('tag_filter', '').lower() or 'all'
    errors: Dict[str, str] = {}
    if runtime not in dict(FILTER_RUNTIME_OPTIONS):
        errors['runtime_filter'] = 'unsupported runtime filter'
    if source not in dict(FILTER_SOURCE_OPTIONS):
        errors['source_filter'] = 'unsupported source filter'
    if status not in dict(FILTER_STATUS_OPTIONS):
        errors['status_filter'] = 'unsupported status filter'
    if errors:
        return None, errors
    return (runtime, source, status, tag), {}


def prompt_browser_filters(
    stdscr,
    colors,
    runtime_filter: str,
    source_filter: str,
    status_filter: str,
    tag_filter: str = 'all',
) -> Optional[Tuple[str, str, str, str]]:
    return prompt_form(
        stdscr,
        colors,
        'Model Filters',
        browser_filter_fields(runtime_filter, source_filter, status_filter, tag_filter),
        parse_browser_filter_answers,
        footer_hint='Use "all" to clear an individual filter.',
    )


def prompt_sort_mode(stdscr, colors, current: str) -> str:
    options = [
        ('1', 'Favorites', 'favorites'),
        ('2', 'Recent', 'recent'),
        ('3', 'Name', 'name'),
        ('4', 'Best Benchmark', 'benchmark'),
        ('5', 'Highest Context', 'context'),
        ('6', 'Port', 'port'),
        ('q', f'Cancel ({sort_mode_label(current)})', 'cancel'),
    ]
    return prompt_modal_choice(stdscr, colors, 'Sort Models', options)


def prompt_yes_no(stdscr, colors, title: str, body: str) -> bool:
    result = prompt_modal_choice(stdscr, colors, title, [
        ('1', f'Yes: {body}', 'yes'),
        ('q', 'Cancel', 'cancel'),
    ])
    return result == 'yes'


def help_overlay_lines() -> List[str]:
    return [
        'Quick Help',
        '',
        'List view',
        'Enter opens details. / searches. f filters. C clears browser. T sorts. * favorites the selected model.',
        'a adds models, e edits them, d deletes them, x detects GGUFs, X prunes missing entries.',
        '',
        'Detail view',
        'Enter or l opens launch actions. T opens Try It Out. v toggles Simple/Advanced detail density.',
        'B runs deep benchmark, F runs fast benchmark, O runs OpenCode workflow, H runs Hermes workflow.',
        '',
        'Power tools',
        ': opens a compact command palette. g/c/G export OpenCode, Continue, and Hermes configs.',
        'Y verifies the selected model. y runs benchmark-proof verification for stale or missing models.',
        'The Config Doctor checks runtime commands, export paths, terminal launcher, and proof status.',
        'z applies the auto profile. R opens results or rankings depending on the view.',
        '',
        'The new browser stores favorites, recents, sort preference, and workspace presets automatically.',
    ]


def show_help_overlay(stdscr, colors):
    h, w = stdscr.getmaxyx()
    box_w = min(96, max(54, w - 8))
    box_h = min(max(12, h - 6), 22)
    if h < 12 or w < 56:
        return
    box_x = max(2, (w - box_w) // 2)
    box_y = max(2, (h - box_h) // 2)
    modal = curses.newwin(box_h, box_w, box_y, box_x)
    modal.keypad(True)
    content_h = max(1, box_h - 4)
    lines = [line for raw in help_overlay_lines() for line in wrap_display_item_lines(raw, box_w - 4)]
    scroll = 0
    stdscr.nodelay(False)
    try:
        while True:
            scroll = clamp_scroll(scroll, len(lines), content_h)
            modal.erase()
            draw_box(modal, 0, 0, box_h - 1, box_w, 'Help', colors['accent'] | curses.A_BOLD, colors['accent'])
            visible = lines[scroll: scroll + content_h]
            for idx, line in enumerate(visible):
                attr = colors['accent'] | curses.A_BOLD if line in ('Quick Help', 'List view', 'Detail view', 'Power tools') else curses.A_NORMAL
                safe_addstr(modal, 2 + idx, 2, line[: box_w - 4], attr)
            safe_addstr(modal, box_h - 2, 2, '[Up/Down] scroll  [Esc/q] close'[: box_w - 4], colors['muted'])
            modal.refresh()
            key = modal.getch()
            if key in (27, ord('q')):
                return
            if key in (curses.KEY_UP, ord('k')):
                scroll -= 1
            elif key in (curses.KEY_DOWN, ord('j')):
                scroll += 1
            elif key == curses.KEY_PPAGE:
                scroll -= content_h
            elif key == curses.KEY_NPAGE:
                scroll += content_h
    finally:
        stdscr.touchwin()
        stdscr.nodelay(True)


def config_doctor_items(app: AppConfig, active_model: Optional[ModelConfig] = None) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = [('Config Doctor', 'heading')]
    llama_ok = app.command_exists(app.llama_server)
    vllm_ok = app.command_exists(app.vllm_command)
    hermes_ok = app.command_exists(getattr(app.hermes, 'command', 'hermes') or 'hermes')
    items.extend([
        (f'llama-server: {"ok" if llama_ok else "missing"}  {app.llama_server}', 'success' if llama_ok else 'error'),
        (f'vLLM: {"ok" if vllm_ok else "missing"}  {app.vllm_command}', 'success' if vllm_ok else 'warning'),
        (f'Hermes: {"ok" if hermes_ok else "missing"}  {getattr(app.hermes, "command", "hermes") or "hermes"}', 'success' if hermes_ok else 'warning'),
        (f'OpenCode config: {app.opencode.path or "<unset>"}', 'success' if app.opencode.path else 'warning'),
        (f'Continue config: {getattr(app.continue_settings, "path", "") or "<unset>"} mode={getattr(app.continue_settings, "merge_mode", "preserve_sections")}', 'success' if getattr(app.continue_settings, 'path', '') else 'warning'),
        (f'Hermes home: {getattr(app.hermes, "home_root", "") or "<unset>"}', 'success' if getattr(app.hermes, 'home_root', '') else 'warning'),
    ])
    terminal = app.detect_terminal_launcher()
    items.append((f'terminal launcher: {terminal or "<not detected>"}', 'success' if terminal else 'warning'))
    code_ok = app.command_exists('code')
    items.append((f'VS Code CLI: {"ok" if code_ok else "missing"}', 'success' if code_ok else 'warning'))
    statuses: Dict[str, int] = {}
    for model in list(getattr(app, 'models', []) or []):
        status = str(getattr(model, 'verification_status', 'unknown') or 'unknown')
        statuses[status] = statuses.get(status, 0) + 1
    status_line = ' '.join(f'{key}:{value}' for key, value in sorted(statuses.items())) or 'none'
    items.append((f'model verification: {status_line}', 'success' if statuses.get('passed') else 'muted'))
    pending = app.benchmark_proof_model_ids(force=False)
    items.append((f'benchmark proof needed: {len(pending)} model(s)', 'warning' if pending else 'success'))
    if active_model:
        result = getattr(active_model, 'verification_results', {}) or {}
        cap = result.get('cap', {}) if isinstance(result, dict) else {}
        items.extend([
            ('', 'normal'),
            (f'active model: {active_model.id}', 'heading'),
            (f'verification: {getattr(active_model, "verification_status", "unknown")} {getattr(active_model, "verification_summary", "")}', 'normal'),
            (f'cap: factor={cap.get("limiting_factor", "-")} configured={cap.get("configured_ctx", "-")} slot={cap.get("ctx_per_slot", "-")} safe={cap.get("estimated_safe_context", "-")} measured={cap.get("measured_max_context", "-")}', 'normal'),
        ])
    return items


def show_config_doctor_overlay(stdscr, colors, app: AppConfig, active_model: Optional[ModelConfig] = None):
    h, w = stdscr.getmaxyx()
    box_w = min(108, max(62, w - 8))
    box_h = min(max(12, h - 6), 24)
    if h < 12 or w < 64:
        return
    box_x = max(2, (w - box_w) // 2)
    box_y = max(2, (h - box_h) // 2)
    modal = curses.newwin(box_h, box_w, box_y, box_x)
    modal.keypad(True)
    content_h = max(1, box_h - 4)
    rows = config_doctor_items(app, active_model=active_model)
    items = []
    for text, kind in rows:
        attr = (
            colors['accent'] | curses.A_BOLD if kind == 'heading'
            else colors['success'] | curses.A_BOLD if kind == 'success'
            else colors['warning'] if kind == 'warning'
            else colors['error'] | curses.A_BOLD if kind == 'error'
            else colors['muted'] if kind == 'muted'
            else curses.A_NORMAL
        )
        items.extend((line, attr) for line in wrap_display_item_lines(text, box_w - 4))
    scroll = 0
    stdscr.nodelay(False)
    try:
        while True:
            visible, scroll, _older, _newer, _total = scrollable_pane_item_view(items, box_w - 4, content_h, scroll)
            modal.erase()
            draw_box(modal, 0, 0, box_h - 1, box_w, 'Config Doctor', colors['accent'] | curses.A_BOLD, colors['accent'])
            for idx, (line, attr) in enumerate(visible):
                safe_addstr(modal, 2 + idx, 2, line[: box_w - 4], attr)
            safe_addstr(modal, box_h - 2, 2, '[PgUp/PgDn] scroll  [Esc/q] close'[: box_w - 4], colors['muted'])
            modal.refresh()
            key = modal.getch()
            if key in (27, ord('q')):
                return
            action = RIGHT_PANE_SCROLL_KEYS.get(key, '')
            if action:
                scroll = adjust_scroll_offset(scroll, action, len(items), content_h)
    finally:
        stdscr.touchwin()
        stdscr.nodelay(True)


def measured_profile_line(model: ModelConfig, key: str, label: str) -> str:
    profile = dict((getattr(model, 'measured_profiles', {}) or {}).get(key) or {})
    if not profile or str(profile.get('status', 'ok') or 'ok') != 'ok':
        return f'{label}: not measured'
    ctx_slot = int(profile.get('ctx_per_slot', profile.get('ctx', 0)) or 0)
    tps = float(profile.get('tokens_per_sec', 0.0) or 0.0)
    parallel = int(profile.get('parallel', 1) or 1)
    return f'{label}: {tps:.2f} tok/s  ctx/slot={ctx_slot}  par={parallel}'


def compare_overlay_lines(app: AppConfig, left: ModelConfig, right: ModelConfig) -> List[str]:
    return [
        'Model Compare',
        '',
        f'left:  {left.name} [{left.id}]',
        f'right: {right.name} [{right.id}]',
        '',
        f'runtime/source: {display_runtime(left)} / {getattr(left, "source", "manual")}   |   {display_runtime(right)} / {getattr(right, "source", "manual")}',
        f'favorite/freshness: {"yes" if getattr(left, "favorite", False) else "no"} / {benchmark_freshness_display(app, left)}   |   {"yes" if getattr(right, "favorite", False) else "no"} / {benchmark_freshness_display(app, right)}',
        f'ctx/output: {left.ctx}/{left.output}   |   {right.ctx}/{right.output}',
        f'threads/ngl/parallel: {left.threads}/{left.ngl}/{left.parallel}   |   {right.threads}/{right.ngl}/{right.parallel}',
        f'roles: {app.role_badges(left.id)}   |   {app.role_badges(right.id)}',
        f'last used: {getattr(left, "last_used_at", "") or "-"}   |   {getattr(right, "last_used_at", "") or "-"}',
        '',
        measured_profile_line(left, 'auto', 'Auto') + '   |   ' + measured_profile_line(right, 'auto', 'Auto'),
        measured_profile_line(left, 'fast_chat', 'Fast') + '   |   ' + measured_profile_line(right, 'fast_chat', 'Fast'),
        measured_profile_line(left, 'long_context', 'Long') + '   |   ' + measured_profile_line(right, 'long_context', 'Long'),
        measured_profile_line(left, 'opencode_ready', 'Code') + '   |   ' + measured_profile_line(right, 'opencode_ready', 'Code'),
    ]


def show_compare_overlay(stdscr, colors, app: AppConfig, left: Optional[ModelConfig], right: Optional[ModelConfig]):
    if not left or not right:
        return
    h, w = stdscr.getmaxyx()
    box_w = min(108, max(60, w - 8))
    box_h = min(max(12, h - 6), 20)
    if h < 12 or w < 62:
        return
    box_x = max(2, (w - box_w) // 2)
    box_y = max(2, (h - box_h) // 2)
    modal = curses.newwin(box_h, box_w, box_y, box_x)
    modal.keypad(True)
    content_h = max(1, box_h - 4)
    lines = [line for raw in compare_overlay_lines(app, left, right) for line in wrap_display_item_lines(raw, box_w - 4)]
    scroll = 0
    stdscr.nodelay(False)
    try:
        while True:
            scroll = clamp_scroll(scroll, len(lines), content_h)
            modal.erase()
            draw_box(modal, 0, 0, box_h - 1, box_w, 'Compare', colors['accent'] | curses.A_BOLD, colors['accent'])
            visible = lines[scroll: scroll + content_h]
            for idx, line in enumerate(visible):
                attr = colors['accent'] | curses.A_BOLD if line in ('Model Compare',) else curses.A_NORMAL
                safe_addstr(modal, 2 + idx, 2, line[: box_w - 4], attr)
            safe_addstr(modal, box_h - 2, 2, '[Up/Down] scroll  [Esc/q] close'[: box_w - 4], colors['muted'])
            modal.refresh()
            key = modal.getch()
            if key in (27, ord('q')):
                return
            if key in (curses.KEY_UP, ord('k')):
                scroll -= 1
            elif key in (curses.KEY_DOWN, ord('j')):
                scroll += 1
    finally:
        stdscr.touchwin()
        stdscr.nodelay(True)


def prompt_command_palette(stdscr, colors) -> str:
    return prompt_modal_choice(stdscr, colors, 'Command Palette', [
        ('1', 'Search models', 'search'),
        ('2', 'Set filters', 'filters'),
        ('3', 'Sort models', 'sort'),
        ('4', 'Toggle favorite', 'favorite'),
        ('5', 'Toggle detail density', 'density'),
        ('6', 'Open help', 'help'),
        ('7', 'Open settings', 'settings'),
        ('8', 'Detect models', 'detect'),
        ('9', 'Compare with machine pick', 'compare'),
        ('o', 'Export OpenCode config', 'export_opencode'),
        ('c', 'Export Continue config', 'export_continue'),
        ('g', 'Export Hermes config', 'export_hermes'),
        ('v', 'Verify selected model', 'verify_selected'),
        ('a', 'Verify all benchmark proof', 'verify_all'),
        ('!', 'Open config doctor', 'config_doctor'),
        ('q', 'Cancel', 'cancel'),
    ])
def prompt_modal_choice(stdscr, colors, title: str, options: List[Tuple[str, str, str]]) -> str:
    h, w = stdscr.getmaxyx()
    box_w = min(68, max(48, w - 8))
    box_h = max(8, len(options) + 6)
    if h < box_h + 4 or w < box_w + 4:
        return 'cancel'
    box_x = max(2, (w - box_w) // 2)
    box_y = max(2, (h - box_h) // 2)
    modal = curses.newwin(box_h + 1, box_w, box_y, box_x)
    modal.keypad(True)
    stdscr.nodelay(False)
    while True:
        modal.erase()
        draw_box(modal, 0, 0, box_h - 1, box_w, title, colors['accent'] | curses.A_BOLD, colors['accent'])
        modal.addstr(2, 2, 'Choose an option:', colors['panel'] | curses.A_BOLD)
        for idx, (key, label, _val) in enumerate(options):
            modal.addstr(3 + idx, 4, f'[{key}] {label}'[: box_w - 8], colors['panel'])
        modal.addstr(box_h - 1, 2, 'Press key to continue...'[: box_w - 6], colors['muted'])
        modal.refresh()
        key = modal.getch()
        if key == -1:
            continue
        key_str = chr(key).lower() if 0 <= key <= 255 else ''
        for option_key, _label, value in options:
            if key_str == option_key:
                stdscr.touchwin()
                stdscr.nodelay(True)
                return value
def launch_options_for_stopped_model(model: ModelConfig) -> List[Tuple[str, str, str]]:
    return [
        ('1', 'Start server now', 'keep'),
        ('2', 'Auto profile', 'auto_profile'),
        ('3', 'Balanced chat', 'balanced_chat'),
        ('4', 'Fast chat', 'fast_chat'),
        ('5', 'Long context', 'long_context'),
        ('6', 'Advanced profiles', 'advanced'),
        ('7', 'Try it out', 'try'),
        ('8', 'Launch model + OpenCode', 'opencode'),
        ('9', 'Launch full-stack: OpenCode + VS Code', 'full_stack'),
        ('h', 'Launch model + Hermes', 'hermes'),
        ('v', 'Launch full-stack: Hermes + VS Code', 'hermes_full_stack'),
        ('q', 'Cancel', 'cancel'),
    ]


def deep_benchmark_all_options() -> List[Tuple[str, str, str]]:
    return [
        ('1', 'Safer adaptive batch for missing/stale/failed models', 'missing'),
        ('2', 'Safer adaptive batch force refresh for every model', 'force'),
        ('q', 'Cancel', 'cancel'),
    ]


def prompt_launch_optimization(stdscr, model: ModelConfig, colors) -> str:
    return prompt_modal_choice(stdscr, colors, f'Launch {model.id}', launch_options_for_stopped_model(model))
def prompt_deep_benchmark_all(stdscr, colors) -> str:
    return prompt_modal_choice(stdscr, colors, 'Deep Benchmark All', deep_benchmark_all_options())
def prompt_running_model_action(stdscr, model: ModelConfig, colors) -> str:
    return prompt_modal_choice(stdscr, colors, f'{model.id} is running', [
        ('1', 'Stop model', 'stop'),
        ('2', 'Try it out', 'try'),
        ('3', 'Launch OpenCode', 'opencode'),
        ('4', 'Launch full-stack: OpenCode + VS Code', 'full_stack'),
        ('5', 'Launch Hermes', 'hermes'),
        ('6', 'Launch full-stack: Hermes + VS Code', 'hermes_full_stack'),
        ('q', 'Cancel', 'cancel'),
    ])
def prompt_advanced_profile(stdscr, colors) -> str:
    return prompt_modal_choice(stdscr, colors, 'Advanced profile', [
        ('1', 'Long context', 'max_context'),
        ('2', 'Fast responses', 'tokens_per_sec'),
        ('q', 'Cancel', 'cancel'),
    ])
def prompt_optimization_tier(stdscr, colors) -> str:
    return prompt_modal_choice(stdscr, colors, 'Profile aggression', [
        ('1', 'Safe', 'safe'),
        ('2', 'Balanced', 'moderate'),
        ('3', 'Aggressive', 'extreme'),
        ('q', 'Cancel', 'cancel'),
    ])
def prompt_quit_policy(stdscr, colors) -> str:
    return prompt_modal_choice(stdscr, colors, 'Quit llama-tui', [
        ('1', 'Stop managed servers and quit', 'stop'),
        ('2', 'Leave servers running and quit', 'leave'),
        ('q', 'Cancel', 'cancel'),
    ])


def show_benchmark_wiki(stdscr, colors):
    h, w = stdscr.getmaxyx()
    box_w = min(92, max(50, w - 8))
    box_h = min(max(12, h - 6), 28)
    if h < 12 or w < 54:
        return
    box_x = max(2, (w - box_w) // 2)
    box_y = max(2, (h - box_h) // 2)
    modal = curses.newwin(box_h, box_w, box_y, box_x)
    modal.keypad(True)
    content_h = max(1, box_h - 4)
    lines = benchmark_wiki_lines(box_w - 4)
    scroll = 0
    stdscr.nodelay(False)
    try:
        while True:
            scroll = clamp_scroll(scroll, len(lines), content_h)
            modal.erase()
            draw_box(modal, 0, 0, box_h - 1, box_w, 'Benchmark Wiki', colors['accent'] | curses.A_BOLD, colors['accent'])
            visible = lines[scroll: scroll + content_h]
            for idx, line in enumerate(visible):
                attr = colors['accent'] | curses.A_BOLD if line and not line.startswith(' ') and any(line == title for title, _body in BENCHMARK_WIKI_SECTIONS) else curses.A_NORMAL
                modal.addstr(2 + idx, 2, line[: box_w - 4], attr)
            footer = '[Up/Down] scroll  [PgUp/PgDn] page  [Esc/q] close'
            modal.addstr(box_h - 2, 2, footer[: box_w - 4], colors['muted'])
            modal.refresh()
            key = modal.getch()
            if key in (27, ord('q')):
                return
            if key in (curses.KEY_UP, ord('k')):
                scroll -= 1
            elif key in (curses.KEY_DOWN, ord('j')):
                scroll += 1
            elif key == curses.KEY_PPAGE:
                scroll -= content_h
            elif key == curses.KEY_NPAGE:
                scroll += content_h
            elif key == curses.KEY_HOME:
                scroll = 0
            elif key == curses.KEY_END:
                scroll = len(lines)
    finally:
        stdscr.touchwin()
        stdscr.nodelay(True)


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


def draw_scrollable_items(
    stdscr,
    y: int,
    x: int,
    h: int,
    w: int,
    items: List[object],
    scroll: int,
    colors: Dict[str, int],
    default_attr: int = 0,
) -> Tuple[int, int, int]:
    rows = max(1, h - 3)
    width = max(1, w - 4)
    visible, clamped, has_older, has_newer, total = scrollable_pane_item_view(
        items,
        width,
        rows,
        scroll,
        default_attr=default_attr,
    )
    if rows == 1 and has_older and has_newer:
        visible[0] = ('^ older / v newer', colors['muted'])
    else:
        if has_older and visible:
            visible[0] = ('^ older lines above', colors['muted'])
        if has_newer and visible:
            visible[-1] = ('v newer lines below', colors['muted'])
    for idx, (line, attr) in enumerate(visible[:rows]):
        safe_addstr(stdscr, y + 2 + idx, x + 2, str(line)[:width], attr)
    return clamped, total, rows


def draw_tabbed_panel(
    stdscr,
    y: int,
    x: int,
    h: int,
    w: int,
    title: str,
    tabs: List[str],
    active_tab: str,
    colors: Dict[str, int],
    error_count: int = 0,
):
    draw_box(stdscr, y, x, h, w, title, colors['accent'] | curses.A_BOLD, colors['accent'])
    tab_x = x + len(title) + 5
    max_x = x + w - 2
    for tab in tabs:
        label = right_tab_label(tab, error_count)
        text = f'[{label}]' if tab == active_tab else f' {label} '
        if tab_x + len(text) > max_x:
            remaining = max_x - tab_x
            if remaining > 4:
                safe_addstr(stdscr, y, tab_x, ellipsize(text, remaining), colors['muted'])
            break
        attr = colors['selection'] | curses.A_BOLD if tab == active_tab else colors['muted']
        safe_addstr(stdscr, y, tab_x, text, attr)
        tab_x += len(text) + 1


def draw_header_dashboard(
    stdscr,
    y: int,
    x: int,
    h: int,
    w: int,
    title: str,
    items: List[Tuple[str, str]],
    colors: Dict[str, int],
):
    if h < 4 or w < HEADER_DASHBOARD_MIN_PANEL_WIDTH:
        return
    draw_box(stdscr, y, x, h, w, title, colors['accent'] | curses.A_BOLD, colors['accent'])
    max_rows = max(0, h - 2)
    for idx, (line, kind) in enumerate(items[:max_rows]):
        row_y = y + 2 + idx
        row_x = x + 2
        width = max(1, w - 4)
        if kind == 'counts':
            cursor = row_x
            prefix = 'counts:'
            safe_addstr(stdscr, row_y, cursor, prefix, colors['muted'])
            cursor += len(prefix) + 1
            for token in str(line).split()[1:]:
                label = token.split(':', 1)[0]
                text = f' {token} '
                if cursor + len(text) > x + w - 2:
                    break
                safe_addstr(stdscr, row_y, cursor, text, chip_attr(colors, label))
                cursor += len(text) + 1
            continue
        attr = colors['muted']
        if kind == 'error':
            attr = colors['error'] | curses.A_BOLD
        elif kind == 'status':
            attr = colors['success'] | curses.A_BOLD
        elif kind in ('action', 'benchmark'):
            attr = colors['warning'] | curses.A_BOLD
        safe_addstr(stdscr, row_y, row_x, ellipsize(str(line), width), attr)


def draw_header_config_box(
    stdscr,
    y: int,
    x: int,
    h: int,
    w: int,
    items: List[Tuple[str, str]],
    colors: Dict[str, int],
    message_is_error: bool = False,
):
    if h < 4 or w < 24:
        return
    draw_box(stdscr, y, x, h, w, 'Config', colors['accent'] | curses.A_BOLD, colors['accent'])
    max_rows = max(0, h - 2)
    width = max(1, w - 4)
    for idx, (line, kind) in enumerate(items[:max_rows]):
        attr = colors['muted']
        if kind == 'message':
            attr = colors['warning'] | curses.A_BOLD if message_is_error else colors['accent'] | curses.A_BOLD
        safe_addstr(stdscr, y + 2 + idx, x + 2, ellipsize(str(line), width), attr)


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
def init_colors():
    palette = {
        'default': 0,
        'accent': 0,
        'success': 0,
        'warning': 0,
        'error': 0,
        'muted': 0,
        'selection': 0,
        'banner': 0,
        'panel': 0,
        'chip_ready': 0,
        'chip_loading': 0,
        'chip_stopped': 0,
    }
    if not curses.has_colors():
        return palette
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    pairs = [
        ('accent', curses.COLOR_CYAN, -1),
        ('success', curses.COLOR_GREEN, -1),
        ('warning', curses.COLOR_YELLOW, -1),
        ('error', curses.COLOR_RED, -1),
        ('muted', curses.COLOR_BLUE, -1),
        ('selection', curses.COLOR_BLACK, curses.COLOR_CYAN),
        ('banner', curses.COLOR_MAGENTA, -1),
        ('panel', curses.COLOR_WHITE, -1),
        ('chip_ready', curses.COLOR_BLACK, curses.COLOR_GREEN),
        ('chip_loading', curses.COLOR_BLACK, curses.COLOR_YELLOW),
        ('chip_stopped', curses.COLOR_WHITE, curses.COLOR_BLUE),
    ]
    pair_id = 1
    for name, fg, bg in pairs:
        try:
            curses.init_pair(pair_id, fg, bg)
            palette[name] = curses.color_pair(pair_id)
            pair_id += 1
        except curses.error:
            palette[name] = curses.A_BOLD if name in ('accent', 'success', 'warning', 'error', 'banner') else 0
    return palette
def status_attr(colors, status: str):
    mapping = {
        'READY': colors['success'] | curses.A_BOLD,
        'LOADING': colors['warning'] | curses.A_BOLD,
        'STARTING': colors['warning'],
        'STOPPED': colors['muted'],
        'ERROR': colors['error'] | curses.A_BOLD,
    }
    return mapping.get(status, colors['accent'])
def status_symbol(status: str) -> str:
    symbols = {
        'READY': '●',
        'LOADING': '◐',
        'STARTING': '◔',
        'STOPPED': '○',
        'ERROR': '✖',
    }
    return symbols.get(status, '·')
def chip_attr(colors, label: str):
    mapping = {
        'READY': colors['chip_ready'] | curses.A_BOLD,
        'LOADING': colors['chip_loading'] | curses.A_BOLD,
        'STARTING': colors['chip_loading'] | curses.A_BOLD,
        'STOPPED': colors['chip_stopped'] | curses.A_BOLD,
        'ERROR': colors['error'] | curses.A_BOLD,
    }
    return mapping.get(label, colors['accent'] | curses.A_BOLD)
def tui(stdscr, app: AppConfig):
    colors = init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    selected = 0
    list_search = ''
    filter_runtime = 'all'
    filter_source = 'all'
    filter_status = 'all'
    filter_tag = 'all'
    sort_mode = normalize_choice(getattr(app.ui, 'preferred_sort', 'port'), tuple(key for key, _label in SORT_OPTIONS), 'port')
    view_mode = 'list'
    detail_model_id = ''
    message = 'Ready.'
    last_error_message = ''
    error_history: List[str] = []
    right_tab_by_view: Dict[str, str] = {}
    right_tab_scrolls: Dict[str, int] = {}
    right_tab_scroll_total = 0
    right_tab_scroll_rows = 1
    last_refresh = 0.0
    statuses: Dict[str, Tuple[str, str]] = {}
    action_thread: Optional[threading.Thread] = None
    action_token: Optional[CancelToken] = None
    action_queue: Queue = Queue()
    try_thread: Optional[threading.Thread] = None
    try_token: Optional[CancelToken] = None
    try_session = 0
    try_messages: List[Dict[str, str]] = []
    try_input = ''
    try_status = 'idle'
    try_error = ''
    try_response_index: Optional[int] = None
    try_launched_model_id = ''
    try_live_metrics = new_try_live_metrics()
    try_input_scroll = 0
    try_transcript_scroll = 0
    try_transcript_total = 0
    try_transcript_rows = 1
    benchmark_state = new_benchmark_run_state()
    results_run_index = 0
    machine_summary_cache: Dict[str, object] = {}
    machine_summary_cache_at = 0.0
    load_warnings = list(getattr(app, 'load_warnings', []) or [])
    if load_warnings:
        message = load_warnings[0]
        error_history.extend(load_warnings[-BENCHMARK_FEED_LIMIT:])

    def invalidate_machine_summary():
        nonlocal machine_summary_cache_at
        machine_summary_cache_at = 0.0

    def current_machine_summary(force: bool = False) -> Dict[str, object]:
        nonlocal machine_summary_cache, machine_summary_cache_at
        now = time.time()
        if force or not machine_summary_cache or now - machine_summary_cache_at > 15.0:
            machine_summary_cache = machine_best_summary(app)
            machine_summary_cache_at = now
        return machine_summary_cache

    def compare_partner_model(current: Optional[ModelConfig]) -> Optional[ModelConfig]:
        if not current:
            return None
        summary = current_machine_summary()
        pick_id = str(((summary or {}).get('machine_pick') or {}).get('model_id', '') or '')
        if pick_id and pick_id != current.id:
            partner = app.get_model(pick_id)
            if partner:
                return partner
        for row in list((summary or {}).get('rows', []) or []):
            candidate_id = str(row.get('model_id', '') or '')
            if candidate_id and candidate_id != current.id:
                partner = app.get_model(candidate_id)
                if partner:
                    return partner
        for candidate in current_browser_models() + list(getattr(app, 'models', []) or []):
            if candidate.id != current.id:
                return candidate
        return None

    def reset_right_tabs(view: str = ''):
        nonlocal right_tab_by_view, right_tab_scrolls
        if view:
            right_tab_by_view[view] = default_right_tab(view)
        else:
            right_tab_by_view = {}
        right_tab_scrolls = {}

    def remember_error(text: str):
        nonlocal last_error_message
        line = compact_message(text)
        if not line:
            return
        last_error_message = line
        if not error_history or error_history[-1] != line:
            error_history.append(line)
            del error_history[:-BENCHMARK_FEED_LIMIT]

    def action_running() -> bool:
        return action_thread is not None and action_thread.is_alive()

    def current_browser_models() -> List[ModelConfig]:
        return browser_models(
            app,
            statuses,
            search=list_search,
            runtime_filter=filter_runtime,
            source_filter=filter_source,
            status_filter=filter_status,
            tag_filter=filter_tag,
            sort_mode=sort_mode,
        )

    def clamp_selected():
        nonlocal selected
        models = current_browser_models()
        if not models:
            selected = 0
            return
        selected = max(0, min(selected, len(models) - 1))

    def select_model_in_browser(model_id: str):
        nonlocal selected
        models = current_browser_models()
        for idx, model in enumerate(models):
            if model.id == model_id:
                selected = idx
                return

    def start_background_action(
        model: ModelConfig,
        label: str,
        worker: Callable[[Callable[[object], None], CancelToken], Tuple[bool, str]],
        done_event: str = 'done',
        run_kind: str = '',
    ):
        nonlocal action_thread, action_token, message, view_mode, detail_model_id, benchmark_state
        if action_running():
            message = '⏳ Another optimization is still running. Watch the log window for progress.'
            return
        token = CancelToken()
        action_token = token
        if run_kind:
            reset_right_tabs('benchmark')
            view_mode = 'benchmark'
            detail_model_id = model.id
            benchmark_state = new_benchmark_run_state(model.id, run_kind, label)
            reduce_benchmark_event(
                benchmark_state,
                {
                    'event': 'benchmark_started',
                    'run_kind': run_kind,
                    'model_id': model.id,
                    'message': f'{label} started for {model.id}',
                    'phase': 'starting',
                    'completed': 0,
                    'total': 0,
                },
            )

        def progress(payload: object):
            if isinstance(payload, dict):
                event_payload = dict(payload)
                event_payload.setdefault('model_id', model.id)
                event_payload.setdefault('run_kind', run_kind)
                line = compact_message(str(event_payload.get('message') or event_payload.get('phase') or event_payload.get('event') or 'benchmark update'))
                if line:
                    append_model_log(app, model, line)
                event_payload['message'] = line
                action_queue.put(('benchmark_event', event_payload))
                return
            line = compact_message(str(payload))
            append_model_log(app, model, line)
            action_queue.put(('progress', line))

        def runner():
            try:
                progress(f'{label} started for {model.id}')
                _ok, result = worker(progress, token)
            except CancelledError:
                result = '⚠ aborted; managed processes stopped'
                progress(result)
            except Exception as exc:
                result = f'❌ {label} failed: {exc}'
                progress(result)
            if run_kind:
                action_queue.put((
                    'benchmark_event',
                    {
                        'event': 'benchmark_aborted' if str(result).startswith('⚠ aborted') else 'benchmark_error' if is_error_message(str(result)) else 'benchmark_done',
                        'run_kind': run_kind,
                        'model_id': model.id,
                        'message': compact_message(result),
                        'phase': 'complete',
                    },
                ))
            action_queue.put((done_event, compact_message(result)))

        action_thread = threading.Thread(target=runner, daemon=True)
        action_thread.start()
        if run_kind:
            message = f'⏳ {label} started for {model.id}. Benchmark dashboard is open.'
        else:
            message = f'⏳ {label} started for {model.id}. Progress is in the log window.'

    def selected_model() -> Optional[ModelConfig]:
        models = current_browser_models()
        if not models:
            return None
        idx = max(0, min(selected, len(models) - 1))
        return models[idx]

    def active_detail_model() -> Optional[ModelConfig]:
        if view_mode in ('detail', 'try', 'benchmark', 'results') and detail_model_id:
            return app.get_model(detail_model_id) or selected_model()
        return selected_model()

    def model_is_running(model: ModelConfig) -> bool:
        status, _detail = app.health(model)
        return status in ('READY', 'LOADING', 'STARTING') or bool(app.get_pid(model))

    def managed_server_running() -> bool:
        return any(app.get_pid(model, discover=False, managed_only=True) for model in app.models)

    def has_benchmark(model: ModelConfig) -> bool:
        return float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0) > 0.0

    def benchmark_hint(model: ModelConfig) -> str:
        status = (getattr(model, 'default_benchmark_status', '') or '').strip().lower()
        if has_benchmark(model):
            return f'{model.id}: benchmark data loaded.'
        if status == 'pending':
            return f'{model.id}: safe defaults are set. Start now or press B when you want measured settings.'
        if status in ('failed', 'aborted'):
            return f'{model.id}: last benchmark {status}. You can still start now; press B to retry benchmarking.'
        return f'{model.id}: no benchmark yet. Start now or press B from details for measured settings.'

    def show_benchmark_hint(model: ModelConfig):
        nonlocal message
        if action_running():
            return
        if model_is_running(model):
            message = f'{model.id}: server is running. Benchmarking remains optional.'
            return
        message = benchmark_hint(model)

    def open_model_details(model: ModelConfig):
        nonlocal view_mode, detail_model_id, message
        reset_right_tabs('detail')
        view_mode = 'detail'
        detail_model_id = model.id
        message = f'{model.id}: details loaded. Press Enter/l to start or Esc to return.'
        show_benchmark_hint(model)

    def open_try_view(model: ModelConfig):
        nonlocal view_mode, detail_model_id, message, try_thread, try_token, try_session, try_input_scroll, try_transcript_scroll
        nonlocal try_messages, try_input, try_status, try_error, try_response_index, try_launched_model_id
        if action_running():
            message = '⏳ Wait for the current launch or benchmark before opening Try it out.'
            return
        app.mark_model_used(model.id)
        reset_right_tabs('try')
        view_mode = 'try'
        detail_model_id = model.id
        try_session += 1
        session = try_session
        try_messages = []
        try_input = ''
        try_input_scroll = 0
        try_transcript_scroll = 0
        try_error = ''
        try_response_index = None
        try_launched_model_id = ''
        clear_try_live_metrics(try_live_metrics)
        try_token = CancelToken()
        status, detail = app.health(model)
        if status == 'READY':
            try_status = 'ready'
            message = f'{model.id}: try-out ready. Type a prompt and press Enter.'
            return

        try_status = 'starting'
        message = f'{model.id}: starting try-out server...'
        token = try_token
        will_launch = not (status in ('LOADING', 'STARTING') or app.get_pid(model))
        if will_launch:
            try_launched_model_id = model.id

        def progress(text: str):
            line = compact_message(text)
            append_model_log(app, model, line)
            action_queue.put(('try_progress', line, session))

        def runner():
            try:
                if not will_launch:
                    progress(f'{model.id} is starting; waiting for chat readiness...')
                    ok, result = app.wait_until_ready(model, timeout=180, cancel_token=token)
                else:
                    ok, result = launch_with_failsafe(app, model, 'best', 'auto', progress=progress, cancel_token=token)
                action_queue.put(('try_ready' if ok else 'try_error', compact_message(result), session))
            except CancelledError:
                action_queue.put(('try_error', 'try-out start cancelled', session))
            except Exception as exc:
                action_queue.put(('try_error', f'try-out start failed: {exc}', session))

        try_thread = threading.Thread(target=runner, daemon=True)
        try_thread.start()

    def open_results_view(model: ModelConfig):
        nonlocal view_mode, detail_model_id, message, results_run_index
        reset_right_tabs('results')
        view_mode = 'results'
        detail_model_id = model.id
        results_run_index = 0
        run_count = len(benchmark_runs_for_model(model))
        message = f'{model.id}: {run_count} benchmark result run(s).'

    def open_machine_results():
        nonlocal view_mode, detail_model_id, message
        reset_right_tabs('machine_results')
        view_mode = 'machine_results'
        detail_model_id = ''
        summary = current_machine_summary(force=True)
        rows = list(summary.get('rows', []) or [])
        pick = summary.get('machine_pick') or {}
        pick_id = str(pick.get('model_id', '') or '')
        message = (
            f'Machine Rankings: {len(rows)} fresh benchmarked model(s). '
            f'Machine Pick: {pick_id or "-"}'
        )

    def start_try_chat_send():
        nonlocal message, try_thread, try_token, try_input, try_input_scroll, try_status, try_error, try_response_index, try_messages, try_transcript_scroll
        if view_mode != 'try':
            return
        model = active_detail_model()
        if not model:
            return
        if try_status != 'ready':
            message = f'{model.id}: wait until the try-out server is ready.'
            return
        if try_thread is not None and try_thread.is_alive():
            message = f'{model.id}: response is still streaming.'
            return
        prompt = try_input.strip()
        if not prompt:
            return
        if try_token is None:
            try_token = CancelToken()
        token = try_token
        session = try_session
        reset_try_live_metrics(try_live_metrics)
        try_messages.append({'role': 'user', 'content': prompt})
        request_messages = [
            {'role': str(item.get('role', '') or ''), 'content': str(item.get('content', '') or '')}
            for item in try_messages
        ]
        try_messages.append({'role': 'assistant', 'content': '', 'reasoning': '', 'final_notice': ''})
        try_response_index = len(try_messages) - 1
        try_input = ''
        try_input_scroll = 0
        try_transcript_scroll = 0
        try_error = ''
        try_status = 'responding'
        message = f'{model.id}: streaming response...'

        def runner():
            chunks = 0
            try:
                for event_type, chunk in stream_chat_events(model, request_messages, cancel_token=token):
                    if event_type == 'chunk':
                        chunks += 1
                        action_queue.put(('chat_chunk', chunk, session))
                    elif event_type == 'reasoning':
                        action_queue.put(('chat_reasoning', chunk, session))
                action_queue.put(('chat_done', str(chunks), session))
            except CancelledError:
                action_queue.put(('chat_error', 'chat stream cancelled', session))
            except Exception as exc:
                action_queue.put(('chat_error', compact_message(str(exc)), session))

        try_thread = threading.Thread(target=runner, daemon=True)
        try_thread.start()

    def exit_try_view():
        nonlocal view_mode, message, last_refresh, try_thread, try_token, try_session, try_input, try_input_scroll, try_transcript_scroll
        nonlocal try_status, try_error, try_response_index, try_launched_model_id
        model = active_detail_model()
        if try_token is not None:
            try_token.cancel('leaving try-out')
        try_session += 1
        stop_msg = 'no model selected'
        if model:
            if should_stop_try_model(try_launched_model_id, model):
                _ok, stop_msg = stop_try_model(app, model)
            else:
                stop_msg = 'left pre-existing server running'
            append_model_log(app, model, f'try-it-out exit: {stop_msg}')
            message = f'{model.id}: try-out closed; {stop_msg}'
        else:
            message = 'Try-out closed.'
        reset_right_tabs('detail')
        view_mode = 'detail'
        try_thread = None
        try_token = None
        try_input = ''
        try_input_scroll = 0
        try_transcript_scroll = 0
        try_status = 'idle'
        try_error = ''
        try_response_index = None
        try_launched_model_id = ''
        clear_try_live_metrics(try_live_metrics)
        statuses.clear()
        last_refresh = 0.0

    def begin_model_launch(model: ModelConfig):
        nonlocal message
        status, _detail = app.health(model)
        running = status in ('READY', 'LOADING', 'STARTING') or bool(app.get_pid(model))
        if running:
            launch_mode = prompt_running_model_action(stdscr, model, colors)
        else:
            launch_mode = prompt_launch_optimization(stdscr, model, colors)

        if launch_mode == 'stop':
            ok, msg = app.stop(model)
            message = f'{model.id}: {msg}'
            return
        if launch_mode == 'cancel':
            message = 'Launch cancelled.'
            return
        if launch_mode == 'try':
            open_try_view(model)
            return
        if launch_mode in ('opencode', 'full_stack', 'hermes', 'hermes_full_stack'):
            runtime = 'hermes' if launch_mode in ('hermes', 'hermes_full_stack') else 'opencode'
            workspace = prompt_workspace(stdscr, colors, app, runtime=runtime)
            if not workspace:
                message = f'{"Hermes" if runtime == "hermes" else "OpenCode"} launch cancelled.'
                return
            app.remember_workspace_preset(runtime, workspace)
            app.mark_model_used(model.id)
            label = (
                'Hermes full-stack launch'
                if launch_mode == 'hermes_full_stack'
                else 'Hermes launch'
                if launch_mode == 'hermes'
                else 'full-stack launch'
                if launch_mode == 'full_stack'
                else 'OpenCode launch'
            )
            include_vscode = launch_mode in ('full_stack', 'hermes_full_stack')
            launcher = launch_hermes_stack if runtime == 'hermes' else launch_opencode_stack
            start_background_action(
                model,
                label,
                lambda progress, token, model=model, workspace=workspace, include_vscode=include_vscode, launcher=launcher: launcher(
                    app,
                    model,
                    workspace,
                    include_vscode=include_vscode,
                    progress=progress,
                    cancel_token=token,
                ),
                done_event='stack_done',
            )
            return
        if launch_mode in SIMPLE_PROFILE_ACTIONS:
            mode, tier, label = simple_profile_action(launch_mode)
            app.mark_model_used(model.id)
            start_background_action(
                model,
                f'{label} launch',
                lambda progress, token, model=model, mode=mode, tier=tier: launch_with_failsafe(
                    app,
                    model,
                    mode,
                    tier,
                    progress=progress,
                    cancel_token=token,
                ),
            )
        elif launch_mode == 'advanced':
            advanced_mode = prompt_advanced_profile(stdscr, colors)
            if advanced_mode == 'cancel':
                message = 'Launch cancelled.'
                return
            tier = prompt_optimization_tier(stdscr, colors)
            if tier == 'cancel':
                message = 'Launch cancelled.'
                return
            app.mark_model_used(model.id)
            start_background_action(
                model,
                f'{profile_label(advanced_mode)} / {tier_label(tier)} launch',
                lambda progress, token, model=model, advanced_mode=advanced_mode, tier=tier: launch_with_failsafe(
                    app,
                    model,
                    advanced_mode,
                    tier,
                    progress=progress,
                    cancel_token=token,
                ),
            )
        else:
            app.mark_model_used(model.id)
            start_background_action(
                model,
                'model launch',
                lambda progress, token, model=model: start_model_with_progress(app, model, progress=progress, cancel_token=token),
            )

    while True:
        while True:
            try:
                queued_event = action_queue.get_nowait()
            except Empty:
                break
            event = queued_event[0]
            text = queued_event[1] if len(queued_event) > 1 else ''
            event_session = queued_event[2] if len(queued_event) > 2 else None
            if event in ('try_progress', 'try_ready', 'try_error', 'chat_chunk', 'chat_reasoning', 'chat_done', 'chat_error'):
                if event_session != try_session:
                    continue
                if event == 'try_progress':
                    message = text
                    continue
                if event == 'try_ready':
                    try_status = 'ready'
                    try_error = ''
                    try_thread = None
                    message = text or f'{detail_model_id}: try-out ready.'
                    last_refresh = 0.0
                    continue
                if event == 'try_error':
                    try_status = 'error'
                    try_error = text or 'try-out failed'
                    try_thread = None
                    message = f'❌ {try_error}'
                    remember_error(message)
                    continue
                if event == 'chat_chunk':
                    update_try_live_metrics(try_live_metrics, text)
                    if try_response_index is not None and 0 <= try_response_index < len(try_messages):
                        try_messages[try_response_index]['content'] += text
                        try_messages[try_response_index]['final_notice'] = ''
                    message = 'streaming response...'
                    continue
                if event == 'chat_reasoning':
                    if try_response_index is not None and 0 <= try_response_index < len(try_messages):
                        existing = str(try_messages[try_response_index].get('reasoning', '') or '')
                        try_messages[try_response_index]['reasoning'] = existing + text
                    message = 'streaming reasoning...'
                    continue
                if event == 'chat_done':
                    finish_try_live_metrics(try_live_metrics)
                    if try_response_index is not None and 0 <= try_response_index < len(try_messages):
                        content = str(try_messages[try_response_index].get('content', '') or '')
                        reasoning = str(try_messages[try_response_index].get('reasoning', '') or '')
                        if not content.strip() and reasoning.strip():
                            try_messages[try_response_index]['final_notice'] = '[no final answer returned]'
                        elif not content.strip() and not reasoning.strip():
                            try_messages[try_response_index]['content'] = '(no content returned)'
                            try_messages[try_response_index]['final_notice'] = ''
                        else:
                            try_messages[try_response_index]['final_notice'] = ''
                    try_status = 'ready'
                    try_response_index = None
                    try_thread = None
                    message = f'{detail_model_id}: response complete.'
                    continue
                if event == 'chat_error':
                    finish_try_live_metrics(try_live_metrics)
                    try_status = 'error'
                    try_error = text or 'chat stream failed'
                    if try_response_index is not None and 0 <= try_response_index < len(try_messages):
                        content = str(try_messages[try_response_index].get('content', '') or '')
                        try_messages[try_response_index]['final_notice'] = ''
                        if content.strip():
                            try_messages[try_response_index]['content'] = content + f'\n[error] {try_error}'
                        else:
                            try_messages[try_response_index]['content'] = f'[error] {try_error}'
                    try_response_index = None
                    try_thread = None
                    message = f'❌ {try_error}'
                    remember_error(message)
                    continue
            if event == 'benchmark_event':
                payload = text if isinstance(text, dict) else {}
                reduce_benchmark_event(benchmark_state, payload)
                event_message = compact_message(str(payload.get('message', '') or ''))
                if event_message:
                    message = event_message
                    if is_error_message(event_message):
                        remember_error(event_message)
                continue
            if is_error_message(text):
                remember_error(text)
            message = text
            if event in ('done', 'stack_done', 'benchmark_done'):
                action_thread = None
                action_token = None
                last_refresh = 0.0
                invalidate_machine_summary()

        now = time.time()
        if now - last_refresh > REFRESH_SECONDS:
            statuses = {m.id: app.health(m) for m in app.models}
            last_refresh = now

        browser_list = current_browser_models()
        if app.models:
            clamp_selected()
            if view_mode in ('detail', 'try', 'benchmark', 'results'):
                current_detail = app.get_model(detail_model_id)
                if not current_detail:
                    detail_model_id = browser_list[selected].id if browser_list else ''
        else:
            selected = 0
            view_mode = 'list'
            detail_model_id = ''

        active_model = active_detail_model()
        machine_summary = current_machine_summary() if view_mode in ('list', 'machine_results') else {}
        machine_pick_id = str(((machine_summary or {}).get('machine_pick') or {}).get('model_id', '') or '')
        if is_error_message(message):
            remember_error(message)

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 18 or w < 88:
            safe_addstr(stdscr, 1, 2, 'Window too small for llama-tui. Stretch it a bit.', colors['warning'] | curses.A_BOLD)
            safe_addstr(stdscr, 3, 2, f'Current size: {w}x{h}')
            safe_addstr(stdscr, 5, 2, '[q] quit', curses.A_BOLD)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord('q'), 27):
                break
            time.sleep(0.05)
            continue

        y = 0
        dashboard_enabled, header_left_w, header_right_x, header_right_w = header_dashboard_layout(w)
        left_w, right_x, right_w = body_pane_layout(w)
        if w >= 100:
            for line in LOGO:
                safe_addstr(stdscr, y, 2, line[:w-4], colors['banner'] | curses.A_BOLD)
                y += 1
            title_x = min(w - 28, max(30, header_left_w - 28)) if dashboard_enabled else min(w - 28, 60)
            safe_addstr(stdscr, 1, title_x, 'local model control plane', colors['accent'] | curses.A_BOLD)
            header_y = y + 1
        else:
            safe_addstr(stdscr, 0, 2, 'llama-tui', colors['banner'] | curses.A_BOLD)
            safe_addstr(stdscr, 0, 14, 'local model control plane', colors['accent'] | curses.A_BOLD)
            header_y = 2

        counts = {'READY': 0, 'LOADING': 0, 'STARTING': 0, 'STOPPED': 0, 'ERROR': 0}
        for _mid, (st, _detail) in statuses.items():
            if st in counts:
                counts[st] += 1

        message_is_error = is_error_message(message)
        header_message = (
            compact_message(message)
            if message_is_error and view_mode == 'try'
            else 'Error captured in the lower-right Errors box.' if message_is_error else compact_message(message)
        )

        active_status = statuses.get(active_model.id, ('?', '')) if active_model else ('?', '')
        box_top = header_y + (HEADER_DASHBOARD_HEIGHT + 1 if dashboard_enabled else 9)
        config_h = max(4, box_top - header_y - 1)
        dashboard_y = 1 if dashboard_enabled and w >= 100 else header_y
        dashboard_h = max(HEADER_DASHBOARD_HEIGHT, box_top - dashboard_y - 1)
        left_header_width = max(24, (header_right_x - 3) if dashboard_enabled else (w - 3))
        config_items = build_header_config_items(app, header_message, left_header_width - 4)
        draw_header_config_box(
            stdscr,
            header_y,
            1,
            config_h,
            left_header_width,
            config_items,
            colors,
            message_is_error=message_is_error,
        )
        if dashboard_enabled:
            dashboard_items = build_header_dashboard_items(
                statuses,
                active_model,
                active_status,
                view_mode,
                benchmark_state,
                action_running(),
                str(benchmark_state.get('label') or message),
                app.hardware_profile().short_summary(),
                error_history,
                header_right_w - 4,
            )
            draw_header_dashboard(
                stdscr,
                dashboard_y,
                header_right_x,
                dashboard_h,
                header_right_w,
                header_dashboard_title(view_mode),
                dashboard_items,
                colors,
            )
        else:
            chip_y = header_y
            chip_x = max(12, min(left_header_width - 34, w - 34))
            chips = [
                ('READY', counts['READY']),
                ('LOADING', counts['LOADING'] + counts['STARTING']),
                ('STOPPED', counts['STOPPED']),
            ]
            for label, value in chips:
                text = f' {label}:{value} '
                if chip_x + len(text) < w - 2:
                    safe_addstr(stdscr, chip_y, chip_x, text, chip_attr(colors, label))
                    chip_x += len(text) + 1

        pane_h = body_pane_height(h, box_top)
        content_rows = body_content_rows(h, box_top)
        content_bottom = body_content_bottom(h, box_top)
        try_input_rows = try_input_row_count(content_rows)
        visible_rows = max(0, content_rows - 1)
        right_total_h = pane_h
        status_error = f'{active_model.id}: status ERROR ({active_status[1]})' if active_model and active_status[0] == 'ERROR' else ''
        try_mode = view_mode == 'try'
        benchmark_mode = view_mode == 'benchmark'
        results_mode = view_mode == 'results'
        machine_results_mode = view_mode == 'machine_results'
        benchmark_errors = list(benchmark_state.get('errors', []) or [])
        error_source_lines = build_error_source_lines(
            error_history,
            benchmark_errors=benchmark_errors,
            benchmark_mode=benchmark_mode,
            status_error=status_error,
            last_error_message=last_error_message,
        )
        error_text = '\n'.join(error_source_lines)
        right_tabs = right_tabs_for_view(view_mode)
        right_active_tab = normalize_right_tab(view_mode, right_tab_by_view.get(view_mode, '')) if right_tabs else ''
        if right_tabs:
            right_tab_by_view[view_mode] = right_active_tab
        right_panel_h = right_total_h
        right_content_w = max(1, right_w - 4)
        right_tab_key = right_tab_scroll_key(view_mode, right_active_tab)
        right_scroll = int(right_tab_scrolls.get(right_tab_key, 0) or 0)

        left_title = 'Try It Out' if try_mode else 'Benchmark' if benchmark_mode else 'Machine Rankings' if machine_results_mode else 'Results' if results_mode else 'Model Details' if view_mode == 'detail' else 'Models'
        draw_box(stdscr, box_top, 1, pane_h, left_w, left_title, colors['accent'] | curses.A_BOLD, colors['accent'])
        if right_tabs:
            draw_tabbed_panel(
                stdscr,
                box_top,
                right_x,
                right_panel_h,
                right_w,
                'Right Pane',
                right_tabs,
                right_active_tab,
                colors,
                error_count=len(error_source_lines),
            )
        else:
            draw_box(stdscr, box_top, right_x, right_panel_h, right_w, 'Details / Logs / Roles', colors['accent'] | curses.A_BOLD, colors['accent'])

        if view_mode == 'machine_results':
            summary = machine_summary or current_machine_summary()
            category_rows = machine_category_items(
                summary,
                accent_attr=colors['accent'] | curses.A_BOLD,
                success_attr=colors['success'] | curses.A_BOLD,
                warning_attr=colors['warning'],
                normal_attr=curses.A_NORMAL,
            )
            y_cursor = box_top + 2
            for line, attr in category_rows[:content_rows]:
                if y_cursor > content_bottom:
                    break
                safe_addstr(stdscr, y_cursor, 3, line[: left_w - 5], attr)
                y_cursor += 1
            if y_cursor <= content_bottom:
                safe_addstr(stdscr, y_cursor, 3, '[D] deep benchmark all   [Esc] models', colors['muted'])
        elif view_mode == 'results' and active_model:
            model = active_model
            runs = benchmark_runs_for_model(model)
            if runs:
                results_run_index = max(0, min(results_run_index, len(runs) - 1))
            content_h = content_rows
            header_lines = [
                (f'model: {model.name or model.id}', curses.A_BOLD),
                (f'runs: {len(runs)} latest benchmark run(s)', colors['accent'] | curses.A_BOLD),
                ('[Up/Down] select run   [Esc] details', colors['muted']),
                ('', curses.A_NORMAL),
            ]
            y_cursor = box_top + 2
            for line, attr in header_lines[:content_h]:
                safe_addstr(stdscr, y_cursor, 3, line[: left_w - 5], attr)
                y_cursor += 1
            if not runs:
                if y_cursor <= content_bottom:
                    safe_addstr(stdscr, y_cursor, 3, 'No benchmark history yet. Press B from details to run one.', colors['warning'])
            else:
                run_rows = max(0, content_h - min(len(header_lines), content_h))
                start_idx, end_idx = visible_selection_window(len(runs), results_run_index, run_rows)
                for idx in range(start_idx, end_idx):
                    if y_cursor > content_bottom:
                        break
                    run = runs[idx]
                    line = benchmark_run_line(run, idx, selected=(idx == results_run_index))
                    attr = colors['selection'] | curses.A_BOLD if idx == results_run_index else curses.A_NORMAL
                    safe_addstr(stdscr, y_cursor, 3, ellipsize(line, left_w - 5), attr)
                    y_cursor += 1
        elif view_mode == 'benchmark' and active_model:
            model = active_model
            run_kind = str(benchmark_state.get('run_kind') or 'server')
            benchmark_model_label = (
                f'all managed models / current: {benchmark_state.get("model_id") or "-"}'
                if run_kind == 'server_all'
                else (model.name or model.id)
            )
            status_text = str(benchmark_state.get('status') or 'idle')
            phase = str(benchmark_state.get('phase') or '-')
            candidate = str(benchmark_state.get('candidate') or '-')
            completed = int(benchmark_state.get('completed', 0) or 0)
            total = int(benchmark_state.get('total', 0) or 0)
            pct = int(round(benchmark_progress_fraction(completed, total) * 100))
            bar = progress_bar_text(completed, total, max(10, min(34, left_w - 62)))
            feed = list(benchmark_state.get('feed', []) or [])
            content_h = content_rows
            summary_lines = [
                (f'model: {benchmark_model_label}', curses.A_BOLD),
                (f'run: {run_kind}   status: {status_text}   elapsed: {benchmark_elapsed_text(benchmark_state)}', colors['accent'] | curses.A_BOLD),
                (f'phase: {phase}', curses.A_NORMAL),
                (f'candidate: {candidate}', curses.A_NORMAL),
                (f'progress: {bar} {completed}/{total if total else "?"} {pct if total else 0}%', colors['warning'] | curses.A_BOLD if benchmark_state.get('active') else colors['success'] | curses.A_BOLD),
                ('', curses.A_NORMAL),
                ('live feed:', colors['accent'] | curses.A_BOLD),
            ]
            y_cursor = box_top + 2
            for line, attr in summary_lines[:content_h]:
                safe_addstr(stdscr, y_cursor, 3, line[: left_w - 5], attr)
                y_cursor += 1
            rows_available = max(0, content_bottom - y_cursor + 1)
            feed_target = max(0, rows_available)
            if not feed and y_cursor <= content_bottom:
                safe_addstr(stdscr, y_cursor, 3, 'waiting for benchmark updates...', colors['muted'])
                y_cursor += 1
            for line in feed[-feed_target:]:
                if y_cursor > content_bottom:
                    break
                attr = colors['error'] if is_error_message(str(line)) else colors['muted']
                for wrapped in wrap_display_item_lines(str(line), left_w - 5):
                    if y_cursor > content_bottom:
                        break
                    safe_addstr(stdscr, y_cursor, 3, wrapped[: left_w - 5], attr)
                    y_cursor += 1
        elif view_mode == 'try' and active_model:
            model = active_model
            input_block_rows = 1 + try_input_rows if try_input_rows > 0 else 0
            input_y = content_bottom - input_block_rows + 1 if input_block_rows else content_bottom + 1
            transcript_h = max(0, (input_y if input_block_rows else content_bottom + 1) - (box_top + 2))
            transcript_items = build_try_transcript_items(
                model,
                try_messages,
                try_status,
                left_w - 6,
                user_attr=colors['accent'] | curses.A_BOLD,
                assistant_attr=curses.A_NORMAL,
                muted_attr=colors['muted'],
            )
            transcript_visible, try_transcript_scroll, transcript_has_older, transcript_has_newer, try_transcript_total = scrollable_pane_item_view(
                transcript_items,
                left_w - 6,
                transcript_h,
                try_transcript_scroll,
                default_attr=curses.A_NORMAL,
            )
            try_transcript_rows = max(1, transcript_h)
            if transcript_h == 1 and transcript_has_older and transcript_has_newer and transcript_visible:
                transcript_visible[0] = ('^ older / v newer', colors['muted'])
            else:
                if transcript_has_older and transcript_visible:
                    transcript_visible[0] = ('^ older lines above', colors['muted'])
                if transcript_has_newer and transcript_visible:
                    transcript_visible[-1] = ('v newer lines below', colors['muted'])
            for i, (line, attr) in enumerate(transcript_visible[:transcript_h]):
                safe_addstr(stdscr, box_top + 2 + i, 3, line[: left_w - 5], attr)
            if input_block_rows:
                input_width = max(1, left_w - 6)
                input_lines, try_input_scroll, has_more_above, has_more_below = try_input_view(
                    try_input,
                    input_width,
                    try_input_rows,
                    try_input_scroll,
                )
                marker = ''
                if has_more_above:
                    marker += ' ^'
                if has_more_below:
                    marker += ' v'
                divider_label = f' input{marker} '
                divider = (divider_label + '-' * max(1, left_w - 5))[: max(1, left_w - 5)]
                safe_addstr(stdscr, input_y, 3, divider[: left_w - 5], colors['muted'])
                if try_status == 'ready':
                    input_attr = colors['panel'] | curses.A_BOLD
                    for row_idx, input_line in enumerate(input_lines[:try_input_rows]):
                        safe_addstr(stdscr, input_y + 1 + row_idx, 3, input_line[: left_w - 5], input_attr)
                elif try_status == 'responding':
                    input_line = 'streaming response... Esc cancels and stops the model'
                    input_attr = colors['warning'] | curses.A_BOLD
                    safe_addstr(stdscr, input_y + 1, 3, input_line[: left_w - 5], input_attr)
                elif try_status == 'error':
                    input_line = f'error: {try_error or "chat failed"}'
                    input_attr = colors['error'] | curses.A_BOLD
                    safe_addstr(stdscr, input_y + 1, 3, input_line[: left_w - 5], input_attr)
                else:
                    input_line = 'waiting for server readiness...'
                    input_attr = colors['warning'] | curses.A_BOLD
                    safe_addstr(stdscr, input_y + 1, 3, input_line[: left_w - 5], input_attr)
        elif view_mode == 'detail' and active_model:
            model = active_model
            status, detail = statuses.get(model.id, ('?', ''))
            benchmark_score = float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0)
            benchmark_seconds = float(getattr(model, 'last_benchmark_seconds', 0.0) or 0.0)
            opencode_score = float(getattr(model, 'last_opencode_benchmark_score', 0.0) or 0.0)
            opencode_seconds = float(getattr(model, 'last_opencode_benchmark_seconds', 0.0) or 0.0)
            hermes_score = float(getattr(model, 'last_hermes_benchmark_score', 0.0) or 0.0)
            hermes_seconds = float(getattr(model, 'last_hermes_benchmark_seconds', 0.0) or 0.0)
            if benchmark_score > 0:
                benchmark_summary = f'{benchmark_score:.2f} tok/s in {benchmark_seconds:.2f}s'
            else:
                benchmark_summary = 'not run yet; benchmark optional'
            if opencode_score > 0:
                opencode_summary = f'{opencode_score:.2f} score in {opencode_seconds:.2f}s'
            else:
                opencode_summary = 'not run yet; press O for opencode workflow'
            if hermes_score > 0:
                hermes_summary = f'{hermes_score:.2f} score in {hermes_seconds:.2f}s'
            else:
                hermes_summary = 'not run yet; press H for Hermes workflow'
            hardware = app.hardware_profile().short_summary()
            detail_density = normalize_choice(getattr(app.ui, 'detail_density', 'simple'), tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'simple')
            freshness = benchmark_freshness_display(app, model)
            verification_status = getattr(model, 'verification_status', 'unknown') or 'unknown'
            verification_summary = getattr(model, 'verification_summary', '') or 'not verified'
            tags_text = ', '.join(list(getattr(model, 'tags', []) or [])) or '-'
            cap = dict((getattr(model, 'verification_results', {}) or {}).get('cap') or {})
            cap_text = (
                f'cap: {cap.get("limiting_factor", "-")} '
                f'configured={cap.get("configured_ctx", "-")} slot={cap.get("ctx_per_slot", context_per_slot(model))} '
                f'safe={cap.get("estimated_safe_context", "-")} measured={cap.get("measured_max_context", "-")}'
            )
            detail_rows = [
                ('[Esc] back   [Enter/l] actions   [T] try   [B] deep bench   [F] fast bench   [O] opencode bench   [H] hermes bench   [R] results   [z] auto   [v] detail density', colors['accent'] | curses.A_BOLD),
                ('', curses.A_NORMAL),
                (f'name: {model.name}', curses.A_BOLD),
                (f'id/runtime/source: {model.id} / {display_runtime(model)} / {getattr(model, "source", "manual")}', curses.A_NORMAL),
                (f'architecture/runtime/offload: {classify_model_type(model)} / {display_runtime(model)} / {display_offload(model)}', curses.A_NORMAL),
                (f'architecture detail: {architecture_detail(model)}', curses.A_NORMAL),
                (f'quant/source: {extract_quant(model)} / {getattr(model, "source", "manual")}', curses.A_NORMAL),
                (f'favorite/freshness: {"yes" if getattr(model, "favorite", False) else "no"} / {freshness}', curses.A_NORMAL),
                (f'tags: {tags_text}', curses.A_NORMAL),
                (f'verification: {verification_status} / {verification_summary}', colors['success'] | curses.A_BOLD if verification_status == 'passed' else colors['warning'] if verification_status in ('warning', 'needs_benchmark', 'unknown') else colors['error'] | curses.A_BOLD),
                (cap_text, curses.A_NORMAL),
                (f'path: {model.path}', curses.A_NORMAL),
                (f'alias/bind: {model.alias} / http://{model.host}:{model.port}', curses.A_NORMAL),
                (f'status: {status} ({detail})', status_attr(colors, status)),
                (f'pid/roles: {app.get_pid(model) or "-"} / {app.role_badges(model.id)}', curses.A_NORMAL),
                (f'ctx/output: {model.ctx} / {model.output}', curses.A_NORMAL),
                (f'threads/ngl/parallel: {model.threads} / {model.ngl} / {model.parallel}', curses.A_NORMAL),
                (f'temp/cache_ram: {model.temp} / {model.cache_ram}', curses.A_NORMAL),
                (f'profile: {model_profile_summary(model)}', curses.A_NORMAL),
                (f'ctx range: {getattr(model, "ctx_min", 2048)}..{getattr(model, "ctx_max", 131072)}', curses.A_NORMAL),
                (f'last used: {getattr(model, "last_used_at", "") or "-"}', curses.A_NORMAL),
                (f'detail density: {detail_density_label(detail_density)}', curses.A_NORMAL),
                (f'hardware: {hardware}', curses.A_NORMAL),
                (f'last benchmark: {benchmark_summary}', colors['warning'] if benchmark_score <= 0 else colors['success'] | curses.A_BOLD),
                (f'opencode benchmark: {opencode_summary}', colors['warning'] if opencode_score <= 0 else colors['success'] | curses.A_BOLD),
                (f'hermes benchmark: {hermes_summary}', colors['warning'] if hermes_score <= 0 else colors['success'] | curses.A_BOLD),
                ('command preview:', colors['accent'] | curses.A_BOLD),
                (' '.join(app.build_command(model)), curses.A_NORMAL),
                ('', curses.A_NORMAL),
            ]
            if detail_density == 'advanced':
                benchmark_rows = list(getattr(model, 'last_benchmark_results', []) or [])
                if not benchmark_rows and benchmark_score > 0:
                    preset_tier = (getattr(model, 'last_benchmark_profile', '') or 'winner/-').split()[0]
                    preset, _, tier = preset_tier.partition('/')
                    benchmark_rows = [{
                        'preset': preset or 'winner',
                        'tier': tier or '-',
                        'status': 'ok',
                        'tokens_per_sec': benchmark_score,
                        'seconds': benchmark_seconds,
                        'ctx': model.ctx,
                        'parallel': model.parallel,
                        'threads': model.threads,
                        'ngl': model.ngl,
                    }]
                detail_rows.extend([('', curses.A_NORMAL), ('server benchmark table:', colors['accent'] | curses.A_BOLD)])
                if benchmark_rows:
                    detail_rows.extend(benchmark_ranking_items(
                        {
                            'kind': 'server',
                            'records': benchmark_rows,
                            'winners': getattr(model, 'measured_profiles', {}) or {},
                        },
                        width=left_w - 5,
                        success_attr=colors['success'] | curses.A_BOLD,
                        warning_attr=colors['warning'],
                        error_attr=colors['error'],
                        heading_attr=colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD,
                        normal_attr=curses.A_NORMAL,
                    ))
                else:
                    detail_rows.append((' no benchmark rows yet; server start is still available', colors['warning']))

                opencode_rows = list(getattr(model, 'last_opencode_benchmark_results', []) or [])
                detail_rows.extend([('', curses.A_NORMAL), ('opencode workflow table:', colors['accent'] | curses.A_BOLD)])
                if opencode_rows:
                    detail_rows.extend(benchmark_ranking_items(
                        {'kind': 'opencode', 'records': opencode_rows, 'winners': {}},
                        width=left_w - 5,
                        success_attr=colors['success'] | curses.A_BOLD,
                        warning_attr=colors['warning'],
                        error_attr=colors['error'],
                        heading_attr=colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD,
                        normal_attr=curses.A_NORMAL,
                    ))
                else:
                    detail_rows.append((' no opencode workflow rows yet; press O to run', colors['warning']))

                hermes_rows = list(getattr(model, 'last_hermes_benchmark_results', []) or [])
                detail_rows.extend([('', curses.A_NORMAL), ('hermes workflow table:', colors['accent'] | curses.A_BOLD)])
                if hermes_rows:
                    detail_rows.extend(benchmark_ranking_items(
                        {'kind': 'hermes', 'records': hermes_rows, 'winners': {}},
                        width=left_w - 5,
                        success_attr=colors['success'] | curses.A_BOLD,
                        warning_attr=colors['warning'],
                        error_attr=colors['error'],
                        heading_attr=colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD,
                        normal_attr=curses.A_NORMAL,
                    ))
                else:
                    detail_rows.append((' no Hermes workflow rows yet; press H to run', colors['warning']))
            else:
                detail_rows.extend([
                    ('benchmark tables hidden in Simple mode; press v for Advanced details.', colors['muted']),
                    (f'server rows: {len(getattr(model, "last_benchmark_results", []) or [])}', colors['muted']),
                    (f'opencode rows: {len(getattr(model, "last_opencode_benchmark_results", []) or [])}', colors['muted']),
                    (f'hermes rows: {len(getattr(model, "last_hermes_benchmark_results", []) or [])}', colors['muted']),
                ])

            detail_items = scrollable_pane_wrapped_items(detail_rows, left_w - 5)
            for i, (line, attr) in enumerate(detail_items[:content_rows]):
                safe_addstr(stdscr, box_top + 2 + i, 3, line[: left_w - 4], attr)
        elif browser_list:
            header = ' ID              PRT  ST        BN    RLS  RUNTIME    QNT      ARCH   NAME'
            if content_rows > 0:
                summary = (
                    f'search={list_search or "-"}  runtime={filter_option_label(FILTER_RUNTIME_OPTIONS, filter_runtime)}  '
                    f'source={filter_option_label(FILTER_SOURCE_OPTIONS, filter_source)}  '
                    f'status={filter_option_label(FILTER_STATUS_OPTIONS, filter_status)}  '
                    f'tag={filter_tag}  sort={sort_mode_label(sort_mode)}'
                )
                safe_addstr(stdscr, box_top + 2, 3, ellipsize(summary, left_w - 4), colors['muted'])
            if content_rows > 1:
                safe_addstr(stdscr, box_top + 3, 3, header, colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD)
            start_idx, end_idx = visible_selection_window(len(browser_list), selected, max(0, visible_rows - 1))
            for idx in range(start_idx, end_idx):
                model = browser_list[idx]
                status, _ = statuses.get(model.id, ('?', ''))
                roles = app.role_badges(model.id)
                engine = display_runtime(model)[:10]
                quant = extract_quant(model)[:8]
                model_type = classify_model_type(model)[:6]
                freshness = benchmark_freshness_short(app, model)
                name_col_width = max(10, left_w - 75)
                favorite_prefix = '★ ' if getattr(model, 'favorite', False) and name_col_width >= 14 else ''
                best_badge = ' BEST' if model.id == machine_pick_id and name_col_width >= 15 else ''
                display_name = favorite_prefix + (model.name or model.id)
                display_name = display_name[: max(1, name_col_width - len(best_badge))] + best_badge
                line = f' {model.id[:14]:14} {model.port:4}  {status_symbol(status)} {status[:6]:6}  {freshness:4}  {roles:3}  {engine:10} {quant:8} {model_type:6} {display_name}'
                row_y = box_top + 4 + idx - start_idx
                if idx == selected:
                    try:
                        safe_addstr(stdscr, row_y, 3, line[: left_w - 3], colors['selection'] | curses.A_BOLD)
                    except curses.error:
                        safe_addstr(stdscr, row_y, 3, line[: left_w - 3], curses.A_REVERSE)
                else:
                    safe_addstr(stdscr, row_y, 3, line[: left_w - 3])
                    status_x = 3 + 1 + 14 + 1 + 4 + 2
                    safe_addstr(stdscr, row_y, status_x, f'{status_symbol(status)} {status[:6]:6}', status_attr(colors, status))
            if visible_rows > 1 and len(browser_list) > max(1, visible_rows - 1):
                bar_h = max(1, visible_rows - 1)
                track_x = left_w - 1
                for i in range(bar_h):
                    safe_addch(stdscr, box_top + 4 + i, track_x, '│', colors['muted'])
                thumb_h = max(1, int(bar_h * (bar_h / max(1, len(browser_list)))))
                thumb_top = int((start_idx / max(1, len(browser_list) - bar_h)) * max(0, bar_h - thumb_h))
                for i in range(thumb_h):
                    safe_addch(stdscr, box_top + 4 + thumb_top + i, track_x, '█', colors['accent'] | curses.A_BOLD)
        elif app.models:
            if content_rows > 1:
                safe_addstr(stdscr, box_top + 3, 3, 'No models match the current search/filter set.', colors['warning'] | curses.A_BOLD)
                safe_addstr(stdscr, box_top + 5, 3, 'Press / to search, f to filter, and C to clear the browser.', colors['muted'])
        else:
            if content_rows > 2:
                safe_addstr(stdscr, box_top + 3, 3, 'Welcome to llama-tui. No models are configured yet.', colors['warning'] | curses.A_BOLD)
                safe_addstr(stdscr, box_top + 5, 3, 'Press x to detect GGUFs from your managed roots.', colors['muted'])
                safe_addstr(stdscr, box_top + 6, 3, 'Press a to add a manual llama.cpp or vLLM model.', colors['muted'])
                safe_addstr(stdscr, box_top + 7, 3, 'Press o to review paths, exports, and UI defaults.', colors['muted'])

        if view_mode == 'machine_results' and right_tabs:
            summary = machine_summary or current_machine_summary()
            if right_active_tab == 'overview':
                right_items = machine_category_items(
                    summary,
                    accent_attr=colors['accent'] | curses.A_BOLD,
                    success_attr=colors['success'] | curses.A_BOLD,
                    warning_attr=colors['warning'],
                    normal_attr=curses.A_NORMAL,
                )
            elif right_active_tab == 'rankings':
                right_items = machine_ranking_items(
                    summary,
                    width=right_content_w,
                    success_attr=colors['success'] | curses.A_BOLD,
                    heading_attr=colors['accent'] | curses.A_BOLD,
                    normal_attr=curses.A_NORMAL,
                )
            elif right_active_tab == 'failures':
                right_items = machine_gap_items(
                    app,
                    summary,
                    warning_attr=colors['warning'],
                    normal_attr=curses.A_NORMAL,
                )
            else:
                right_items = [('Machine rankings: no content yet.', colors['muted'])]
            right_scroll, right_tab_scroll_total, right_tab_scroll_rows = draw_scrollable_items(
                stdscr,
                box_top,
                right_x,
                right_panel_h,
                right_w,
                right_items,
                right_scroll,
                colors,
                curses.A_NORMAL,
            )
            right_tab_scrolls[right_tab_key] = right_scroll
        elif active_model and right_tabs:
            model = active_model
            status, detail = statuses.get(model.id, ('?', ''))
            pid = app.get_pid(model)
            command_preview = ' '.join(app.build_command(model))
            log_lines = read_display_file_lines(app.logfile(model.id))
            if view_mode == 'detail' and status in ('ERROR', 'STOPPED') and error_text:
                log_lines = important_log_excerpt(app.logfile(model.id), max_lines=400, after_last_launch=True)
            right_items: List[Tuple[str, int]] = []

            if view_mode == 'detail':
                benchmark_score = float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0)
                opencode_score = float(getattr(model, 'last_opencode_benchmark_score', 0.0) or 0.0)
                hermes_score = float(getattr(model, 'last_hermes_benchmark_score', 0.0) or 0.0)
                if right_active_tab == 'summary':
                    right_items = [
                        (f'name: {model.name}', curses.A_BOLD),
                        (f'id/runtime/source: {model.id} / {display_runtime(model)} / {getattr(model, "source", "manual")}', curses.A_NORMAL),
                        (f'path: {model.path}', curses.A_NORMAL),
                        (f'alias/bind: {model.alias} / {model.host}:{model.port}', curses.A_NORMAL),
                        (f'quant/architecture/offload: {extract_quant(model)} / {classify_model_type(model)} / {display_offload(model)}', curses.A_NORMAL),
                        (f'ctx/output: {model.ctx} / {model.output}', curses.A_NORMAL),
                        (f'threads/ngl/parallel: {model.threads} / {model.ngl} / {model.parallel}', curses.A_NORMAL),
                        (f'temp/cache_ram: {model.temp} / {model.cache_ram}', curses.A_NORMAL),
                        (f'profile: {model_profile_summary(model)}', curses.A_NORMAL),
                        (f'status: {status} ({detail})', status_attr(colors, status)),
                        (f'pid/roles: {pid or "-"} / {app.role_badges(model.id)}', curses.A_NORMAL),
                        (f'log: {app.logfile(model.id)}', curses.A_NORMAL),
                    ]
                elif right_active_tab == 'logs':
                    right_items = build_log_items(log_lines, curses.A_NORMAL, colors['muted'])
                elif right_active_tab == 'errors':
                    right_items = build_error_items(error_source_lines, colors['error'], colors['muted'])
                elif right_active_tab == 'command':
                    right_items = [
                        ('command preview:', colors['accent'] | curses.A_BOLD),
                        (command_preview, curses.A_NORMAL),
                    ]
                elif right_active_tab == 'benchmarks':
                    right_items = [
                        (f'benchmark: {benchmark_score:.2f} tok/s {getattr(model, "last_benchmark_profile", "")}', colors['success'] | curses.A_BOLD if benchmark_score > 0 else colors['warning']),
                        (f'opencode: {opencode_score:.2f} score {getattr(model, "last_opencode_benchmark_profile", "")}', colors['success'] | curses.A_BOLD if opencode_score > 0 else colors['warning']),
                        (f'hermes: {hermes_score:.2f} score {getattr(model, "last_hermes_benchmark_profile", "")}', colors['success'] | curses.A_BOLD if hermes_score > 0 else colors['warning']),
                        ('', curses.A_NORMAL),
                        ('server benchmark rows:', colors['accent'] | curses.A_BOLD),
                    ]
                    rows = list(getattr(model, 'last_benchmark_results', []) or [])
                    if rows:
                        right_items.extend(benchmark_ranking_items(
                            {'kind': 'server', 'records': rows, 'winners': getattr(model, 'measured_profiles', {}) or {}},
                            width=right_content_w,
                            success_attr=colors['success'] | curses.A_BOLD,
                            warning_attr=colors['warning'],
                            error_attr=colors['error'],
                            heading_attr=colors['accent'] | curses.A_BOLD,
                            normal_attr=curses.A_NORMAL,
                        ))
                    else:
                        right_items.append(('no server benchmark rows yet', colors['muted']))
                    right_items.extend([('', curses.A_NORMAL), ('opencode workflow rows:', colors['accent'] | curses.A_BOLD)])
                    opencode_rows = list(getattr(model, 'last_opencode_benchmark_results', []) or [])
                    if opencode_rows:
                        right_items.extend(benchmark_ranking_items(
                            {'kind': 'opencode', 'records': opencode_rows, 'winners': {}},
                            width=right_content_w,
                            success_attr=colors['success'] | curses.A_BOLD,
                            warning_attr=colors['warning'],
                            error_attr=colors['error'],
                            heading_attr=colors['accent'] | curses.A_BOLD,
                            normal_attr=curses.A_NORMAL,
                        ))
                    else:
                        right_items.append(('no opencode workflow rows yet', colors['muted']))
                    right_items.extend([('', curses.A_NORMAL), ('hermes workflow rows:', colors['accent'] | curses.A_BOLD)])
                    hermes_rows = list(getattr(model, 'last_hermes_benchmark_results', []) or [])
                    if hermes_rows:
                        right_items.extend(benchmark_ranking_items(
                            {'kind': 'hermes', 'records': hermes_rows, 'winners': {}},
                            width=right_content_w,
                            success_attr=colors['success'] | curses.A_BOLD,
                            warning_attr=colors['warning'],
                            error_attr=colors['error'],
                            heading_attr=colors['accent'] | curses.A_BOLD,
                            normal_attr=curses.A_NORMAL,
                        ))
                    else:
                        right_items.append(('no Hermes workflow rows yet', colors['muted']))

            elif view_mode == 'benchmark':
                records = list(benchmark_state.get('records', []) or [])
                if not records:
                    if str(benchmark_state.get('run_kind') or '') == 'opencode':
                        records = list(getattr(model, 'last_opencode_benchmark_results', []) or [])
                    elif str(benchmark_state.get('run_kind') or '') == 'hermes':
                        records = list(getattr(model, 'last_hermes_benchmark_results', []) or [])
                    else:
                        records = list(getattr(model, 'last_benchmark_results', []) or [])
                if right_active_tab == 'progress':
                    right_items = build_benchmark_progress_items(
                        model,
                        benchmark_state,
                        status,
                        detail,
                        pid,
                        right_content_w,
                        accent_attr=colors['accent'] | curses.A_BOLD,
                    )
                elif right_active_tab == 'results':
                    run = {
                        'kind': str(benchmark_state.get('run_kind') or ''),
                        'records': records,
                        'winners': {},
                    }
                    right_items = benchmark_ranking_items(
                        run,
                        width=right_content_w,
                        success_attr=colors['success'] | curses.A_BOLD,
                        warning_attr=colors['warning'],
                        error_attr=colors['error'],
                        heading_attr=colors['accent'] | curses.A_BOLD,
                        normal_attr=curses.A_NORMAL,
                    )
                elif right_active_tab == 'commands':
                    right_items = [
                        (line, colors['warning'] if kind == 'current' and benchmark_state.get('active') else colors['muted'])
                        for line, kind in benchmark_command_lines(benchmark_state, right_content_w, BENCHMARK_COMMAND_LIMIT + 1)
                    ]
                elif right_active_tab == 'logs':
                    right_items = build_log_items(log_lines, curses.A_NORMAL, colors['muted'])
                elif right_active_tab == 'errors':
                    right_items = build_error_items(error_source_lines, colors['error'], colors['muted'])

            elif view_mode == 'try':
                if right_active_tab == 'profile':
                    right_items = [
                        (f'model: {model.name}', curses.A_BOLD),
                        (f'id/runtime/source: {model.id} / {display_runtime(model)} / {getattr(model, "source", "manual")}', curses.A_NORMAL),
                        (f'status: {status} ({detail})', status_attr(colors, status)),
                        (f'pid: {pid or "-"}', curses.A_NORMAL),
                        (f'url: http://{model.host}:{model.port}', curses.A_NORMAL),
                        (f'ctx/output: {model.ctx} / {model.output}', curses.A_NORMAL),
                        (f'threads/ngl/parallel: {model.threads} / {model.ngl} / {model.parallel}', curses.A_NORMAL),
                        (f'temp/cache_ram: {model.temp} / {model.cache_ram}', curses.A_NORMAL),
                        (f'profile: {model_profile_summary(model)}', curses.A_NORMAL),
                        (f'last bench: {getattr(model, "last_benchmark_tokens_per_sec", 0.0):.2f} tok/s {getattr(model, "last_benchmark_profile", "")}', curses.A_NORMAL),
                        (f'opencode: {getattr(model, "last_opencode_benchmark_score", 0.0):.2f} score {getattr(model, "last_opencode_benchmark_profile", "")}', curses.A_NORMAL),
                        (f'hermes: {getattr(model, "last_hermes_benchmark_score", 0.0):.2f} score {getattr(model, "last_hermes_benchmark_profile", "")}', curses.A_NORMAL),
                        (f'chat: {try_status}', colors['accent'] | curses.A_BOLD),
                    ]
                    if try_error:
                        right_items.append((f'error: {try_error}', colors['error'] | curses.A_BOLD))
                elif right_active_tab == 'logs':
                    right_items = build_log_items(log_lines, curses.A_NORMAL, colors['muted'])
                elif right_active_tab == 'errors':
                    right_items = build_error_items(error_source_lines, colors['error'], colors['muted'])
                elif right_active_tab == 'stats':
                    right_items = [
                        (line, colors['accent'] | curses.A_BOLD if line.startswith(('benchmark:', 'live:', 'last:')) else curses.A_NORMAL)
                        for line in build_try_live_stat_lines(model, try_status, pid, try_live_metrics)
                    ]
                elif right_active_tab == 'command':
                    right_items = [
                        ('command preview:', colors['accent'] | curses.A_BOLD),
                        (command_preview, curses.A_NORMAL),
                    ]

            elif view_mode == 'results':
                runs = benchmark_runs_for_model(model)
                run = runs[results_run_index] if runs and 0 <= results_run_index < len(runs) else {}
                if right_active_tab == 'run_summary':
                    if run:
                        right_items = [
                            (f'run: {run.get("id", "-")}', colors['accent'] | curses.A_BOLD),
                            (f'status: {run.get("status", "-")}  kind: {run.get("kind", "-")}', curses.A_NORMAL),
                            (f'started: {run.get("started_at", "-")}', curses.A_NORMAL),
                            (f'ended: {run.get("ended_at", "-")}', curses.A_NORMAL),
                            (f'elapsed: {float(run.get("elapsed_seconds", 0.0) or 0.0):.1f}s', curses.A_NORMAL),
                            (f'summary: {run.get("summary", "no summary")}', curses.A_NORMAL),
                        ]
                    else:
                        right_items = [('No benchmark run selected.', colors['muted'])]
                elif right_active_tab == 'rankings':
                    right_items = benchmark_ranking_items(
                        run,
                        width=right_content_w,
                        success_attr=colors['success'] | curses.A_BOLD,
                        warning_attr=colors['warning'],
                        error_attr=colors['error'],
                        heading_attr=colors['accent'] | curses.A_BOLD,
                        normal_attr=curses.A_NORMAL,
                    )
                elif right_active_tab == 'failures':
                    records = list(run.get('records', []) or []) if run else []
                    failures = [row for row in records if row.get('status') not in ('ok', 'tests passed') or row.get('break_point')]
                    if failures:
                        for row in failures:
                            detail_line = compact_message(str(row.get('detail', '') or ''))
                            right_items.append((
                                f'{row.get("objective", "-")} {row.get("variant", "-")} ctx={row.get("ctx", 0)} par={row.get("parallel", 0)} {row.get("status", "-")} {detail_line}',
                                colors['error'] if row.get('break_point') else colors['warning'],
                            ))
                    else:
                        right_items = [('No failures or break points in this run.', colors['success'] | curses.A_BOLD)]

            if not right_items:
                right_items = [(f'{right_tab_label(right_active_tab, len(error_source_lines))}: no content yet.', colors['muted'])]
            right_scroll, right_tab_scroll_total, right_tab_scroll_rows = draw_scrollable_items(
                stdscr,
                box_top,
                right_x,
                right_panel_h,
                right_w,
                right_items,
                right_scroll,
                colors,
                curses.A_NORMAL,
            )
            right_tab_scrolls[right_tab_key] = right_scroll
        elif active_model:
            model = active_model
            status, detail = statuses.get(model.id, ('?', ''))
            list_right_items: List[Tuple[str, int]] = [
                (f'name: {model.name}', curses.A_BOLD),
                (f'id/runtime/source: {model.id} / {display_runtime(model)} / {getattr(model, "source", "manual")}', curses.A_NORMAL),
                (f'favorite/freshness: {"yes" if getattr(model, "favorite", False) else "no"} / {benchmark_freshness_display(app, model)}', curses.A_NORMAL),
                (f'tags/verification: {", ".join(list(getattr(model, "tags", []) or [])) or "-"} / {getattr(model, "verification_status", "unknown")}', curses.A_NORMAL),
                (f'alias/bind: {model.alias} / {model.host}:{model.port}', curses.A_NORMAL),
                (f'quant/architecture/offload: {extract_quant(model)} / {classify_model_type(model)} / {display_offload(model)}', curses.A_NORMAL),
                (f'ctx/output: {model.ctx} / {model.output}', curses.A_NORMAL),
                (f'profile: {model_profile_summary(model)}', curses.A_NORMAL),
                (f'benchmark: {getattr(model, "last_benchmark_tokens_per_sec", 0.0):.2f} tok/s {getattr(model, "last_benchmark_profile", "")}', curses.A_NORMAL),
                (f'status: {status} ({detail})', status_attr(colors, status)),
                (f'pid/roles: {app.get_pid(model) or "-"} / {app.role_badges(model.id)}', curses.A_NORMAL),
                (f'browser: search={list_search or "-"} runtime={filter_runtime} source={filter_source} status={filter_status} tag={filter_tag} sort={sort_mode}', colors['muted']),
                ('', curses.A_NORMAL),
                ('last log lines:', colors['accent'] | curses.A_BOLD),
            ]
            list_right_items.extend((line, curses.A_NORMAL) for line in read_display_file_lines(app.logfile(model.id))[-40:])
            draw_scrollable_items(
                stdscr,
                box_top,
                right_x,
                right_panel_h,
                right_w,
                list_right_items,
                0,
                colors,
                curses.A_NORMAL,
            )

        if view_mode == 'try':
            footer = '[Enter] send  Tab/] next tab  Shift-Tab/[ prev tab  [Esc] stop model + exit'
            footer2 = '[Up/Down] prompt  [Ctrl+P/N/B/F/A/E] convo  [PgUp/PgDn/Home/End] right tab.'
        elif view_mode == 'benchmark':
            footer = '[Esc] details  [F] fast bench  [R] results  [W] wiki  Tab/] next  Shift-Tab/[ prev  [A] abort'
            footer2 = '[Up/Down/PgUp/PgDn/Home/End] scroll active right tab.'
        elif view_mode == 'results':
            footer = '[Esc] details  [Up/Down] select run  Tab/] next tab  Shift-Tab/[ prev tab'
            footer2 = '[PgUp/PgDn/Home/End] scroll active right tab.'
        elif view_mode == 'machine_results':
            footer = '[Esc] models  [D] deep all  Tab/] next tab  Shift-Tab/[ prev tab'
            footer2 = '[PgUp/PgDn/Home/End] scroll active right tab  [M] refresh rankings.'
        elif view_mode == 'detail':
            footer = '[Esc] models  [Enter/l] actions  [T] try  [B/F/O/H] benchmarks  [R] results  [v] density'
            footer2 = 'Tab/] next tab  Shift-Tab/[ prev tab  [g/c/G] exports  [Y/y] verify  [:] palette  [?] help'
        else:
            footer = '[Enter] details  [/] search  [f] filter  [T] sort  [C] clear browser  [*] favorite'
            footer2 = '[a/e/d] models  [x/X] detect/prune  [R/M] ranks  [Y/y] verify  [:] palette  [?] help  [q] quit'
        if action_running():
            footer = '[A] abort active action   ' + footer
        safe_addstr(stdscr, h - 2, 2, footer[: w - 4], colors['accent'] | curses.A_BOLD)
        safe_addstr(stdscr, h - 1, 2, footer2[: w - 4], colors['muted'] | curses.A_BOLD)
        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break

        if key == -1:
            time.sleep(0.05)
            continue
        scroll_action = right_scroll_action_for_view(view_mode, key)
        tab_direction = right_tab_key_direction(key)
        if right_tabs and tab_direction:
            right_tab_by_view[view_mode] = cycle_right_tab(view_mode, right_active_tab, tab_direction)
            message = f'Right tab: {right_tab_label(right_tab_by_view[view_mode], len(error_source_lines))}.'
            continue
        if key == ord('?'):
            show_help_overlay(stdscr, colors)
            message = 'Help closed.'
            continue
        if key == ord('V'):
            model = active_detail_model() if view_mode != 'list' else selected_model()
            partner = compare_partner_model(model)
            if model and partner:
                show_compare_overlay(stdscr, colors, app, model, partner)
                message = f'Compared {model.id} against {partner.id}.'
            else:
                message = 'Need at least two models for compare.'
            continue
        if key == ord(':'):
            action = prompt_command_palette(stdscr, colors)
            palette_keys = {
                'search': ord('/'),
                'filters': ord('f'),
                'sort': ord('T'),
                'favorite': ord('*'),
                'density': ord('v'),
                'help': ord('?'),
                'settings': ord('o'),
                'detect': ord('x'),
                'compare': ord('V'),
                'export_opencode': ord('g'),
                'export_continue': ord('c'),
                'export_hermes': ord('G'),
                'verify_selected': ord('Y'),
                'verify_all': ord('y'),
            }
            if action in palette_keys:
                key = palette_keys[action]
            elif action == 'clear_browser':
                key = ord('C')
            elif action == 'config_doctor':
                show_config_doctor_overlay(stdscr, colors, app, active_detail_model() or selected_model())
                message = 'Config Doctor closed.'
                continue
            else:
                message = 'Command palette cancelled.'
                continue
        if view_mode == 'try':
            transcript_scroll_action = try_transcript_scroll_action(key)
            if transcript_scroll_action:
                try_transcript_scroll = adjust_scroll_offset(
                    try_transcript_scroll,
                    transcript_scroll_action,
                    try_transcript_total,
                    try_transcript_rows,
                )
                message = 'Conversation: newest lines.' if try_transcript_scroll == 0 else 'Conversation: scrolled back.'
                continue
            if scroll_action:
                right_tab_scrolls[right_tab_key] = adjust_scroll_offset(right_scroll, scroll_action, right_tab_scroll_total, right_tab_scroll_rows)
                message = 'Right tab: newest lines.' if right_tab_scrolls[right_tab_key] == 0 else 'Right tab: scrolled back.'
                continue
            if key == 27:
                exit_try_view()
                continue
            if key in (curses.KEY_UP, curses.KEY_DOWN):
                if try_status == 'ready':
                    if try_input_rows <= 0:
                        message = 'Try-out input needs a taller terminal.'
                        continue
                    input_width = max(1, left_w - 6)
                    max_scroll = try_input_max_scroll(try_input, input_width, try_input_rows)
                    if key == curses.KEY_UP:
                        try_input_scroll = max(0, try_input_scroll - 1)
                    else:
                        try_input_scroll = min(max_scroll, try_input_scroll + 1)
                    if max_scroll <= 0:
                        message = 'Try-out input fits in the editor.'
                else:
                    message = 'Try-out input is available once the model is ready.'
                continue
            if key == 21:
                try_input = ''
                try_input_scroll = 0
                message = 'Try-out input cleared.'
                continue
            if key in (curses.KEY_BACKSPACE, 127, 8):
                if try_status == 'ready' and try_input:
                    try_input = try_input[:-1]
                    try_input_scroll = try_input_max_scroll(try_input, max(1, left_w - 6), try_input_rows)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                start_try_chat_send()
                continue
            if 32 <= key <= 126:
                if try_status == 'ready':
                    try_input += chr(key)
                    try_input_scroll = try_input_max_scroll(try_input, max(1, left_w - 6), try_input_rows)
                else:
                    message = 'Try-out input is available once the model is ready.'
                continue
            continue
        if action_running() and key == ord('A'):
            if action_token is not None:
                action_token.cancel('user requested abort')
            message = '⏳ Aborting active action and cleaning up managed processes...'
            continue
        if view_mode == 'list' and key == ord('/'):
            updated_search = prompt_search_query(stdscr, colors, list_search)
            if updated_search is not None:
                list_search = updated_search
                selected = 0
                message = f'Search set to: {list_search or "all models"}.'
            else:
                message = 'Search cancelled.'
            continue
        if view_mode == 'list' and key == ord('f'):
            filters = prompt_browser_filters(stdscr, colors, filter_runtime, filter_source, filter_status, filter_tag)
            if filters is not None:
                filter_runtime, filter_source, filter_status, filter_tag = filters
                selected = 0
                message = (
                    f'Filters set: {filter_option_label(FILTER_RUNTIME_OPTIONS, filter_runtime)}, '
                    f'{filter_option_label(FILTER_SOURCE_OPTIONS, filter_source)}, '
                    f'{filter_option_label(FILTER_STATUS_OPTIONS, filter_status)}, tag={filter_tag}.'
                )
            else:
                message = 'Filter changes cancelled.'
            continue
        if view_mode == 'list' and key == ord('C'):
            list_search = ''
            filter_runtime = 'all'
            filter_source = 'all'
            filter_status = 'all'
            filter_tag = 'all'
            selected = 0
            message = 'Browser search and filters cleared.'
            continue
        if view_mode == 'list' and key == ord('T'):
            chosen_sort = prompt_sort_mode(stdscr, colors, sort_mode)
            if chosen_sort != 'cancel':
                sort_mode = chosen_sort
                app.ui.preferred_sort = chosen_sort
                app.save()
                selected = 0
                message = f'Sort mode: {sort_mode_label(chosen_sort)}.'
            else:
                message = 'Sort unchanged.'
            continue
        if key == ord('*'):
            model = active_detail_model() if view_mode != 'list' else selected_model()
            if model:
                favorite, status_text = app.toggle_favorite(model.id)
                message = f'{model.id}: {status_text}.'
            else:
                message = 'No model selected to favorite.'
            continue
        if view_mode == 'detail' and key == ord('v'):
            current_density = normalize_choice(getattr(app.ui, 'detail_density', 'simple'), tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'simple')
            app.ui.detail_density = 'advanced' if current_density == 'simple' else 'simple'
            app.save()
            message = f'Detail density: {detail_density_label(app.ui.detail_density)}.'
            continue
        if view_mode == 'benchmark' and key in (ord('W'), ord('w')):
            show_benchmark_wiki(stdscr, colors)
            message = 'Benchmark wiki closed.'
            continue
        if view_mode == 'machine_results':
            if key in (27, curses.KEY_BACKSPACE, 127, 8):
                reset_right_tabs('machine_results')
                view_mode = 'list'
                message = 'Back to model list.'
                continue
            if scroll_action:
                right_tab_scrolls[right_tab_key] = adjust_scroll_offset(right_scroll, scroll_action, right_tab_scroll_total, right_tab_scroll_rows)
                message = 'Right tab: newest lines.' if right_tab_scrolls[right_tab_key] == 0 else 'Right tab: scrolled back.'
                continue
        if view_mode == 'results':
            if key in (27, curses.KEY_BACKSPACE, 127, 8):
                reset_right_tabs('detail')
                view_mode = 'detail'
                message = 'Back to model details.'
                continue
            if scroll_action:
                right_tab_scrolls[right_tab_key] = adjust_scroll_offset(right_scroll, scroll_action, right_tab_scroll_total, right_tab_scroll_rows)
                message = 'Right tab: newest lines.' if right_tab_scrolls[right_tab_key] == 0 else 'Right tab: scrolled back.'
                continue
            if key in (curses.KEY_UP, ord('k'), curses.KEY_DOWN, ord('j')):
                model = active_detail_model()
                runs = benchmark_runs_for_model(model) if model else []
                if runs:
                    if key in (curses.KEY_UP, ord('k')):
                        results_run_index = max(0, results_run_index - 1)
                    else:
                        results_run_index = min(len(runs) - 1, results_run_index + 1)
                continue
        if active_model and view_mode in ('detail', 'benchmark') and scroll_action:
            right_tab_scrolls[right_tab_key] = adjust_scroll_offset(right_scroll, scroll_action, right_tab_scroll_total, right_tab_scroll_rows)
            message = 'Right tab: newest lines.' if right_tab_scrolls[right_tab_key] == 0 else 'Right tab: scrolled back.'
            continue
        if action_running() and key not in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE, curses.KEY_HOME, curses.KEY_END, ord('j'), ord('k'), ord('R'), ord('M'), ord('W'), ord('w'), ord('['), ord(']'), 9, getattr(curses, 'KEY_BTAB', -999), 27, curses.KEY_BACKSPACE, 127, 8):
            message = '⏳ Action is running. Watch the log window; controls unlock when it finishes.'
            continue
        if view_mode == 'benchmark' and key in (27, curses.KEY_BACKSPACE, 127, 8):
            reset_right_tabs('detail')
            view_mode = 'detail'
            message = 'Back to model details. Benchmark keeps running unless you press A.'
            continue
        if view_mode == 'detail' and key in (27, curses.KEY_BACKSPACE, 127, 8):
            view_mode = 'list'
            detail_model_id = ''
            message = 'Back to model list.'
            continue
        if key in (ord('q'), 27):
            if action_running():
                message = '⏳ Action is running. Press A to abort, then quit after cleanup finishes.'
                continue
            if should_prompt_quit_keepalive(managed_server_running(), action_running()):
                quit_policy = prompt_quit_policy(stdscr, colors)
                should_quit, quit_message = apply_quit_policy(app, quit_policy)
                message = quit_message
                if not should_quit:
                    continue
            break
        if key in (curses.KEY_UP, ord('k')) and browser_list and view_mode == 'list':
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord('j')) and browser_list and view_mode == 'list':
            selected = min(len(browser_list) - 1, selected + 1)
        elif key == ord('r'):
            count, items = app.detect_models()
            statuses = {m.id: app.health(m) for m in app.models}
            invalidate_machine_summary()
            message = items[0] if items else (f'Synced {count} model(s)' if count else 'Synced.')
        elif key == ord('S'):
            message = '; '.join(app.stop_all())[: max(20, w - 4)]
        elif key in (10, 13, curses.KEY_ENTER) and (active_detail_model() or browser_list):
            model = active_detail_model()
            if not model:
                continue
            if view_mode == 'detail':
                begin_model_launch(model)
            else:
                open_model_details(model)
        elif key == ord('l') and active_model and view_mode == 'detail':
            model = active_detail_model()
            if model:
                begin_model_launch(model)
        elif key in (ord('T'), ord('t')) and active_model and view_mode == 'detail':
            model = active_detail_model()
            if model:
                open_try_view(model)
        elif key == ord('M') and app.models:
            open_machine_results()
        elif key == ord('R') and app.models and view_mode == 'list':
            open_machine_results()
        elif key == ord('R') and active_model and view_mode in ('detail', 'benchmark', 'results'):
            model = active_detail_model()
            if model:
                open_results_view(model)
        elif key == ord('Y') and (active_detail_model() or selected_model()):
            model = active_detail_model() or selected_model()
            result = app.verify_model(model)
            message = f'{model.id}: verification {result.get("status")} - {compact_message(str(result.get("summary", "")))}'
        elif key == ord('y') and app.models:
            pending = app.benchmark_proof_model_ids(force=False)
            anchor = selected_model() or app.models[0]
            if not pending:
                message = 'All enabled models already have fresh benchmark proof.'
                continue
            invalidate_machine_summary()
            start_background_action(
                anchor,
                f'verify benchmark proof ({len(pending)} pending)',
                lambda progress, token: benchmark_all_models_deep(
                    app,
                    progress=progress,
                    cancel_token=token,
                    force=False,
                ),
                done_event='benchmark_done',
                run_kind='server_all',
            )
        elif key == ord('D') and app.models:
            choice = prompt_deep_benchmark_all(stdscr, colors)
            if choice == 'cancel':
                message = 'Deep benchmark all cancelled.'
                continue
            force = choice == 'force'
            anchor = selected_model() or app.models[0]
            label = 'deep benchmark all (force)' if force else 'deep benchmark all'
            invalidate_machine_summary()
            start_background_action(
                anchor,
                label,
                lambda progress, token, force=force: benchmark_all_models_deep(
                    app,
                    progress=progress,
                    cancel_token=token,
                    force=force,
                ),
                done_event='benchmark_done',
                run_kind='server_all',
            )
        elif key == ord('z') and app.models:
            model = active_detail_model()
            if not model:
                continue
            measured_ok, measured_msg = apply_measured_profile(model, 'auto')
            if measured_ok:
                tune_msg = f'Auto profile applied from measured benchmark: {measured_msg}'
            else:
                profile = app.hardware_profile(refresh=True)
                tier = select_best_tier(model, profile)
                tune_msg = f'Auto profile applied from estimate: {apply_best_optimization(model, tier=tier, profile=profile)}'
            app.add_or_update(model)
            sync_msg = sync_opencode_after_tuning(app)
            invalidate_machine_summary()
            message = f'{tune_msg} | {sync_msg}'
        elif key == ord('B') and app.models:
            model = active_detail_model()
            if not model:
                continue
            start_background_action(
                model,
                'smart bounded benchmark profiles',
                lambda progress, token, model=model: benchmark_best_optimization(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                ),
                done_event='benchmark_done',
                run_kind='server',
            )
        elif key == ord('F') and app.models and view_mode in ('detail', 'benchmark'):
            model = active_detail_model()
            if not model:
                continue
            start_background_action(
                model,
                'fast benchmark profiles',
                lambda progress, token, model=model: benchmark_fast_profiles(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                ),
                done_event='benchmark_done',
                run_kind='server_fast',
            )
        elif key == ord('O') and app.models and view_mode == 'detail':
            model = active_detail_model()
            if not model:
                continue
            start_background_action(
                model,
                'opencode workflow benchmark',
                lambda progress, token, model=model: benchmark_opencode_workflow(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                ),
                done_event='benchmark_done',
                run_kind='opencode',
            )
        elif key == ord('H') and app.models and view_mode == 'detail':
            model = active_detail_model()
            if not model:
                continue
            start_background_action(
                model,
                'Hermes workflow benchmark',
                lambda progress, token, model=model: benchmark_hermes_workflow(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                ),
                done_event='benchmark_done',
                run_kind='hermes',
            )
        elif key == ord('a'):
            model = prompt_model(stdscr, colors, 'Add model')
            if model:
                if not getattr(model, 'default_benchmark_status', ''):
                    model.default_benchmark_status = 'pending'
                app.add_or_update(model)
                select_model_in_browser(model.id)
                invalidate_machine_summary()
                message = f'Added {model.id} with safe defaults. Open details to start now; press B for measured settings.'
        elif key == ord('e') and app.models:
            current = active_detail_model() or selected_model()
            updated = prompt_model(stdscr, colors, f'Edit {current.id}', current) if current else None
            if updated:
                if updated.id != current.id:
                    app.delete(current.id)
                if not getattr(updated, 'default_benchmark_status', ''):
                    updated.default_benchmark_status = 'pending'
                app.add_or_update(updated)
                select_model_in_browser(updated.id)
                if view_mode == 'detail':
                    detail_model_id = updated.id
                invalidate_machine_summary()
                message = f'Updated {updated.id}.'
        elif key == ord('d') and app.models:
            delete_model = active_detail_model() or selected_model()
            if delete_model and prompt_yes_no(stdscr, colors, 'Delete Model', f'remove {delete_model.id} from llama-tui config'):
                target_id = delete_model.id
                ok, msg = app.delete(target_id)
                clamp_selected()
                if view_mode == 'detail':
                    view_mode = 'list'
                    detail_model_id = ''
                invalidate_machine_summary()
                message = f'{target_id}: {msg}'
            else:
                message = 'Delete cancelled.'
        elif key == ord('x'):
            count, items = app.detect_models()
            message = items[0] if items else (f'Detected {count} new model(s)' if count else 'No new GGUFs found.')
            if count:
                message = f'{message} | safe defaults set; start now or press B for measured settings.'
            clamp_selected()
            invalidate_machine_summary()
        elif key == ord('X'):
            count, removed = app.prune_missing_models()
            message = f'Pruned {count}: {", ".join(removed[:5])}' if count else 'No missing models to prune.'
            clamp_selected()
            invalidate_machine_summary()
        elif key == ord('g'):
            ok, msg = app.generate_opencode()
            message = msg
        elif key == ord('c'):
            ok, msg = app.generate_continue_config()
            message = msg
        elif key == ord('G') and app.models:
            model = active_detail_model() or selected_model()
            ok, msg = app.generate_hermes_config(model)
            message = msg
        elif key == ord('o'):
            if prompt_settings(stdscr, colors, app):
                message = 'Settings saved.'
            else:
                message = 'Settings unchanged.'
        elif key == ord('m') and app.models:
            model = active_detail_model() or selected_model()
            app.set_role('main', model.id)
            message = f'{model.id} set as main model.'
        elif key == ord('s') and app.models:
            model = active_detail_model() or selected_model()
            app.set_role('small', model.id)
            message = f'{model.id} set as small model.'
        elif key == ord('b') and app.models:
            model = active_detail_model() or selected_model()
            app.set_role('build', model.id)
            message = f'{model.id} set as build model.'
        elif key == ord('p') and app.models:
            model = active_detail_model() or selected_model()
            app.set_role('plan', model.id)
            message = f'{model.id} set as plan model.'
