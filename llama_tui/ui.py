import curses
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .app import AppConfig, CONTINUE_MERGE_MODES, context_per_slot
from .benchmark import (
    append_model_log,
    apply_full_suite_profile_recommendation,
    apply_full_suite_recommendations,
    apply_measured_profile,
    apply_moe_recommendation,
    benchmark_all_models_deep,
    benchmark_best_optimization,
    benchmark_detection_sources_text,
    benchmark_fast_profiles,
    benchmark_full_suite,
    benchmark_moe_placement_tuning,
    benchmark_profile_is_fresh,
    benchmark_raw_speed_profile,
    benchmark_strategy_for_app,
    deep_benchmark_model_decision,
    estimate_text_tokens,
    get_measured_profile,
    has_moe_recommendation,
    launch_hermes_stack,
    launch_opencode_stack,
    launch_with_failsafe,
    machine_best_summary,
    moe_recommendation_applied,
    record_matches_profile,
    suite_run_recommended_profile_key,
    start_model_with_progress,
    sync_opencode_after_tuning,
)
from .chat import stream_chat_events
from .constants import DEFAULT_HOST, DEFAULT_MODEL_PORT, LOGO, REFRESH_SECONDS
from .control import CancelToken, CancelledError
from .ui_action_runner import ActionRunner
from .discovery import classify_model_type, display_offload, display_runtime, extract_quant
from .gguf import architecture_detail, turboquant_detail, turboquant_short
from .hermes_benchmark import benchmark_hermes_workflow
from .hardware import HardwareProfile
from .models import ModelConfig
from .mtp import clamp_mtp_draft, mtp_label, mtp_support_label, normalize_mtp_support
from .mtp_doctor import build_mtp_doctor_report, mtp_status_for_model
from .opencode_benchmark import benchmark_opencode_workflow
from .optimize import apply_best_optimization, model_is_moe, select_best_tier
from .textutil import compact_message, ellipsize, important_log_excerpt, is_error_message, wrap_display_lines
from .ui_components import (
    draw_badge,
    draw_box,
    draw_card,
    draw_key_hint_bar,
    draw_section_title,
    draw_status_chip,
    kind_status_prefix,
    safe_addch,
    safe_addstr,
    truncate,
    wrap_card_lines,
)
from .ui_theme import health_style, kind_style, mtp_style, state_chip_style, style as theme_style
from .ui_models import (
    BENCHMARK_FRESHNESS_LABELS,
    BROWSER_HEADER,
    BROWSER_VIEW_OPTIONS,
    _measured_profile_for_recommendation,
    _profile_context,
    active_engine_binary,
    active_engine_key,
    active_engine_short,
    active_engine_warning_line,
    benchmark_freshness_display,
    benchmark_freshness_label,
    benchmark_freshness_short,
    browser_header_for_view,
    browser_model_line,
    browser_model_line_for_view,
    build_model_row_summary,
    compact_browser_header,
    compact_browser_model_line,
    format_engine_badge,
    format_model_health,
    format_model_recommendation,
    format_model_state,
    mtp_status_from_measured,
    mtp_status_short,
    status_symbol,
)
from .ui_benchmark import (
    FULL_SUITE_STAGES,
    LEADERBOARD_SORT_KEYS,
    MTP_SUITE_STAGES,
    benchmark_leaderboard_lines,
    benchmark_plan_lines,
    build_benchmark_cockpit_items,
    _status_attr_for_record,
    _table_row,
    _table_rule,
    _table_widths,
    benchmark_rank_line,
    benchmark_rank_table_items,
    benchmark_ranking_items,
    benchmark_ranking_rows,
    benchmark_record_score,
    benchmark_record_status_kind,
    benchmark_wiki_lines,
    full_suite_is_mtp,
    full_suite_stage_lines,
    full_suite_stage_map,
    full_suite_status_symbol,
    ranked_benchmark_records,
)

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
BENCHMARK_LIVE_LIMIT = 60
FLEET_BROWSER_HEADER = ' MODEL                         STATE    PICK         CTX     TOK/S   ENGINE      HEALTH'
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
    'detail': ['overview', 'launch', 'tuning', 'benchmarks', 'logs', 'command', 'exports'],
    'benchmark': ['progress', 'results', 'commands', 'logs', 'errors'],
    'try': ['profile', 'logs', 'errors', 'stats', 'command'],
    'results': ['run_summary', 'rankings', 'failures'],
    'machine_results': ['overview', 'rankings', 'failures'],
}
SIMPLE_DETAIL_TABS = ['overview', 'launch', 'benchmarks', 'logs']
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
    'launch': 'Launch',
    'tuning': 'Tuning',
    'exports': 'Exports',
}
@dataclass(frozen=True)
class SuggestedAction:
    label: str
    key: str
    reason: str
    severity: str = 'normal'


RIGHT_DEFAULT_TAB = {
    'detail': 'overview',
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

TRUTHY_VALUES = {'1', 'true', 'yes', 'y', 'on'}
FALSY_VALUES = {'0', 'false', 'no', 'n', 'off'}
VALID_RUNTIMES = ('llama.cpp',)
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
]
FILTER_SOURCE_OPTIONS = [
    ('all', 'All sources'),
    ('manual', 'Manual'),
    ('huggingface', 'Hugging Face'),
    ('hf_cache', 'HF cache'),
    ('llama_cache', 'llama.cpp cache'),
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
FILTER_COMPATIBILITY_OPTIONS = [
    ('active', 'Active engine compatible'),
    ('incompatible', 'Unsupported/uncertain for active engine'),
    ('all', 'All models'),
]


def shutdown_workers(*tokens_and_threads, join_timeout: float = 2.0) -> None:
    """Cancel cancel-tokens and join daemon worker threads on tui() exit.

    Accepts an interleaved list of ``CancelToken`` / ``threading.Thread`` /
    ``None`` values; cancels every token, then joins every live thread with
    ``join_timeout``. Exceptions during cancel/join are swallowed since this
    runs on the shutdown path and there is nowhere useful to report them.
    """
    for item in tokens_and_threads:
        if item is None:
            continue
        cancel = getattr(item, 'cancel', None)
        if callable(cancel):
            try:
                cancel('tui shutdown')
            except Exception:
                pass
    for item in tokens_and_threads:
        if item is None:
            continue
        if isinstance(item, threading.Thread) and item.is_alive():
            try:
                item.join(timeout=join_timeout)
            except Exception:
                pass


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


def browser_view_label(value: str) -> str:
    normalized = normalize_choice(value, tuple(key for key, _label in BROWSER_VIEW_OPTIONS), 'compact')
    return dict(BROWSER_VIEW_OPTIONS).get(normalized, 'Compact')




































def active_engine_kv(app: AppConfig, model: ModelConfig) -> str:
    engine = active_engine_key(app, model)
    if engine == 'llama.cpp-mtp':
        return f'MTP {mtp_label(model)} ({mtp_support_label(model)})'
    if engine != 'turboquant':
        return '-'
    try:
        runtime_profile = app.runtime_profile_from_model(
            model,
            int(getattr(model, 'ctx', 0) or 0),
            int(getattr(model, 'parallel', 1) or 1),
            int(getattr(model, 'ngl', 0) or 0),
        )
        key_mode, value_mode = str(getattr(runtime_profile, 'kv_preset', '') or '').split('/', 1)
    except (AttributeError, TypeError, ValueError):
        profile = getattr(app, 'runtime_profile', None)
        try:
            key_mode, value_mode = profile.engine_kv_pair()
        except (AttributeError, TypeError, ValueError):
            key_mode, value_mode = '', ''
    return f'key={key_mode or "-"} value={value_mode or "-"}'




def runtime_engine_source_line(app: AppConfig, model: ModelConfig) -> str:
    return (
        f'id/model runtime/active engine/source: {model.id} / {display_runtime(model)} / '
        f'{active_engine_short(app, model)} / {getattr(model, "source", "manual")}'
    )


def active_engine_detail_line(app: AppConfig, model: ModelConfig) -> str:
    binary = active_engine_binary(app, model) or '-'
    engine = active_engine_key(app, model)
    mode_label = 'mtp' if engine == 'llama.cpp-mtp' else 'kv'
    return f'active engine: {active_engine_short(app, model)}  binary: {binary}  {mode_label}: {active_engine_kv(app, model)}'


def active_engine_badge_line(app: AppConfig, model: Optional[ModelConfig] = None) -> str:
    if model is not None:
        engine = format_engine_badge(active_engine_key(app, model))
        binary = active_engine_binary(app, model)
        kv = active_engine_kv(app, model)
    else:
        profile = getattr(app, 'runtime_profile', None)
        engine = format_engine_badge(str(getattr(profile, 'engine', '') or getattr(profile, 'engine_id', '') or 'llama.cpp'))
        binary = str(getattr(profile, 'server_command', '') or getattr(app, 'llama_server', '') or '')
        try:
            kv = profile.engine_kv_pair()
            kv = f'key={kv[0]} value={kv[1]}'
        except (AttributeError, TypeError, ValueError, IndexError):
            kv = '-'
    try:
        binary_ok = bool(app.command_exists(binary))
    except (AttributeError, TypeError, OSError):
        binary_ok = bool(binary)
    binary_state = 'binary ok' if binary_ok else 'binary missing'
    return f'ENGINE: {engine} | KV {kv or "-"} | {binary_state}'


def active_engine_badge_kind(app: AppConfig, model: Optional[ModelConfig] = None) -> str:
    binary = active_engine_binary(app, model) if model is not None else str(getattr(getattr(app, 'runtime_profile', None), 'server_command', '') or getattr(app, 'llama_server', '') or '')
    try:
        if not app.command_exists(binary):
            return 'error'
    except (AttributeError, TypeError, OSError):
        pass
    if model is not None and active_engine_warning_line(app, model):
        return 'error' if 'binary warning' in active_engine_warning_line(app, model).lower() else 'warning'
    return 'engine'




def active_engine_warning_attr(message: str, colors: Dict[str, int]) -> int:
    low = (message or '').lower()
    if 'high advisory' in low or 'binary warning' in low:
        return colors['error'] | curses.A_BOLD
    return colors['warning']




def turboquant_status_kind(model: ModelConfig, turboquant_session: bool = False) -> str:
    status = (getattr(model, 'turboquant_status', '') or 'unknown').strip().lower()
    head_dim = max(
        int(getattr(model, 'turboquant_head_dim', 0) or 0),
        int(getattr(model, 'turboquant_key_dim', 0) or 0),
        int(getattr(model, 'turboquant_value_dim', 0) or 0),
    )
    if turboquant_session and head_dim == 64:
        return 'error'
    if status in ('native', 'padded'):
        return 'success'
    if status == 'incompatible':
        return 'warning'
    if turboquant_session and status in ('unknown', 'not_applicable'):
        return 'warning'
    return 'muted'


def turboquant_detail_line(model: ModelConfig) -> str:
    return f'turboquant: {turboquant_detail(model)}'


def model_engine_visibility_lines(app: AppConfig, model: ModelConfig) -> List[str]:
    try:
        features = ', '.join(app.model_runtime_features(model)) or '-'
    except (AttributeError, TypeError, OSError):
        features = '-'
    try:
        active_visibility = app.model_engine_visibility(model)
    except (AttributeError, TypeError, OSError):
        active_visibility = None
    try:
        active_compat = app.model_engine_compatibility(model)
    except (AttributeError, TypeError, OSError):
        active_compat = None
    try:
        compatible = ', '.join(format_engine_badge(engine) for engine in app.compatible_engine_ids_for_model(model)) or '-'
    except (AttributeError, TypeError, OSError):
        compatible = '-'
    try:
        hidden = app.hidden_engine_reasons_for_model(model)
    except (AttributeError, TypeError, OSError):
        hidden = {}
    hidden_text = '; '.join(
        f'{format_engine_badge(engine)}: {reason}'
        for engine, reason in hidden.items()
        if reason
    ) or '-'
    lines = [
        f'detected features: {features}',
        f'detection sources: {benchmark_detection_sources_text(model)}',
        f'compatible engines: {compatible}',
        f'hidden from: {hidden_text}',
    ]
    if active_visibility is not None:
        if not active_visibility.compatible:
            lines.insert(1, f'active engine visibility: hidden - {active_visibility.reason}')
        elif active_visibility.status == 'compatible_with_warning':
            lines.insert(1, f'active engine visibility: warning - {active_visibility.reason}')
    if active_compat is not None:
        if not active_compat.compatible:
            lines.insert(1, f'active engine launch: {active_compat.status} - {active_compat.reason}')
        elif active_compat.status == 'compatible_with_warning':
            lines.insert(1, f'active engine launch: warning - {active_compat.reason}')
    return lines


def filter_option_label(options: List[Tuple[str, str]], value: str) -> str:
    return dict(options).get(value, value or '-')


def iso_recent_key(value: str) -> float:
    if not value:
        return 0.0
    try:
        return time.mktime(time.strptime(value, '%Y-%m-%dT%H:%M:%S'))
    except (TypeError, ValueError, OverflowError):
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
    compatibility_filter: str = 'all',
) -> bool:
    runtime_filter = str(runtime_filter or 'all').strip().lower() or 'all'
    source_filter = str(source_filter or 'all').strip().lower() or 'all'
    status_filter = str(status_filter or 'all').strip() or 'all'
    tag_filter = str(tag_filter or 'all').strip().lower() or 'all'
    compatibility_filter = str(compatibility_filter or 'all').strip().lower() or 'all'
    if runtime_filter != 'all' and str(getattr(model, 'runtime', 'llama.cpp') or '').strip().lower() != runtime_filter:
        return False
    if source_filter != 'all':
        source_labels = {
            label.strip().lower()
            for label in str(getattr(model, 'source', 'manual') or 'manual').split(',')
            if label.strip()
        }
        if source_filter not in source_labels:
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
    if compatibility_filter == 'active':
        try:
            if hasattr(app, 'active_engine_model_visibility'):
                compatible, _reason = app.active_engine_model_visibility(model)
            else:
                compatible, _reason = app.active_engine_model_compatibility(model)
        except Exception:
            compatible = True
        if not compatible:
            return False
    elif compatibility_filter == 'incompatible':
        try:
            compatible, _reason = app.active_engine_model_compatibility(model)
        except Exception:
            compatible = True
        if compatible:
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
    compatibility_filter: str = 'all',
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
            compatibility_filter=compatibility_filter,
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


def right_tabs_for_view(view_mode: str, detail_density: str = 'advanced') -> List[str]:
    if view_mode == 'detail' and normalize_choice(detail_density, tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'advanced') == 'simple':
        return list(SIMPLE_DETAIL_TABS)
    return list(RIGHT_TABS.get(view_mode, []))


def default_right_tab(view_mode: str, detail_density: str = 'advanced') -> str:
    tabs = right_tabs_for_view(view_mode, detail_density)
    return RIGHT_DEFAULT_TAB.get(view_mode, tabs[0] if tabs else '')


def normalize_right_tab(view_mode: str, tab: str, detail_density: str = 'advanced') -> str:
    tabs = right_tabs_for_view(view_mode, detail_density)
    if tab in tabs:
        return tab
    return default_right_tab(view_mode, detail_density)


def cycle_right_tab(view_mode: str, current_tab: str, direction: int = 1, detail_density: str = 'advanced') -> str:
    tabs = right_tabs_for_view(view_mode, detail_density)
    if not tabs:
        return ''
    current = normalize_right_tab(view_mode, current_tab, detail_density)
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
        f'detail={detail_density_label(getattr(getattr(app, "ui", None), "detail_density", "simple"))} '
        f'browser={browser_view_label(getattr(getattr(app, "ui", None), "browser_view", "compact"))}'
    )
    roots_summary = (
        f'hf={app.hf_cache_root} | llmfit={app.llmfit_cache_root} | '
        f'local={app.llm_models_cache_root} | lm-studio={summarize_roots(app.lm_studio_roots(), body_width)}'
    )
    lines = [
        (f'config: {app.config_path}', 'muted'),
        (f'code path: {Path(__file__).resolve().parents[1]}', 'muted'),
        (f'llama-server: {app.llama_server}', 'muted'),
        (active_engine_badge_line(app), active_engine_badge_kind(app)),
        (app.runtime_indicator(), 'muted'),
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
    app: Optional[AppConfig] = None,
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
        suggested = suggested_next_action(app, active_model, status) if app is not None else SuggestedAction('Launch Model', 'T', 'Model selected.')
        pick = format_model_recommendation(app, active_model) if app is not None else 'Needs Bench'
        benchmark_state_text = benchmark_freshness_display(app, active_model) if app is not None else 'Missing'
        selected_line = (
            f'selected: {active_model.name or active_model.id}  '
            f'status: {status}  benchmark: {benchmark_state_text}  recommendation: {pick}'
        )
        next_line = f'next: [{suggested.key}] {suggested.label} - {suggested.reason}'
    else:
        active_line = 'active: none'
        selected_line = 'selected: none'
        next_line = 'next: select a model'
    engine_line = active_engine_badge_line(app, active_model) if app is not None and active_model is not None else ''

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
    ]
    if engine_line:
        lines.append((engine_line, active_engine_badge_kind(app, active_model) if app is not None and active_model is not None else 'muted'))
    lines.extend([
        (selected_line, 'status' if status == 'READY' else 'error' if status == 'ERROR' else 'muted'),
        (next_line, 'message' if active_model else 'muted'),
        (active_line, 'status' if status == 'READY' else 'error' if status == 'ERROR' else 'muted'),
        (view_line, 'muted'),
        (bench_line, 'benchmark' if benchmark_active else 'muted'),
        (f'hardware: {hardware_summary or "-"}', 'muted'),
        (f'last error: {latest_error}', 'error' if error_history else 'muted'),
    ])
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
    app: Optional[AppConfig] = None,
) -> List[Tuple[str, int]]:
    if str(state.get('run_kind') or '') == 'full_suite':
        return build_full_suite_progress_items(
            model,
            state,
            width,
            accent_attr=accent_attr,
            normal_attr=normal_attr,
        )
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
        engine_line = ''
        if app is not None:
            engine_line = f'  active engine: {active_engine_short(app, model)}'
        runtime_line = (
            f'arch: {classify_model_type(model)}  runtime: {display_runtime(model)}  '
            f'offload: {display_offload(model)}{engine_line}  ctx/slot={current_slot}  par={getattr(model, "parallel", 1)}'
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
        for line in benchmark_launch_profile_detail_lines(latest):
            items.append((line, normal_attr))
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
    cockpit = build_benchmark_cockpit_items(state, width)
    if cockpit:
        items.append(('', normal_attr))
        items.append(('live cockpit:', accent_attr))
        items.extend((text, normal_attr) for text, _kind in cockpit)
    return items


def full_suite_run_from_state(state: Dict[str, object]) -> Dict[str, object]:
    records = list(state.get('records', []) or [])
    return {
        'kind': 'full_suite',
        'status': state.get('status', ''),
        'records': records,
        'recommendations': state.get('recommendations', {}) or {},
        'warnings': state.get('warnings', []) or [],
        'summary': state.get('message', '') or '',
    }


def build_full_suite_progress_items(
    model: ModelConfig,
    state: Dict[str, object],
    width: int,
    accent_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    completed = int(state.get('completed', 0) or 0)
    total = int(state.get('total', 0) or 0)
    phase = str(state.get('phase') or '')
    records = list(state.get('records', []) or [])
    title = 'MTP Suite' if full_suite_is_mtp(records) else 'Full Suite Benchmark'
    items: List[Tuple[str, int]] = [
        (title, accent_attr),
        (f'model: {model.name or model.id}', normal_attr),
        (f'status: {state.get("status") or "idle"}   elapsed: {benchmark_elapsed_text(state)}', normal_attr),
        (f'progress: {progress_bar_text(completed, total, max(8, width - 18))} {completed}/{total if total else "?"}', accent_attr),
        ('', normal_attr),
        ('Stages', accent_attr),
    ]
    items.extend((line, normal_attr) for line in full_suite_stage_lines(records, active_phase=phase))
    message = compact_message(str(state.get('message', '') or ''))
    if message:
        items.extend([('', normal_attr), (f'Current: {message}', normal_attr)])
    feed = [compact_message(str(item)) for item in list(state.get('feed', []) or []) if compact_message(str(item))]
    if feed:
        items.extend([('', normal_attr), ('Recent', accent_attr)])
        items.extend((line, normal_attr) for line in feed[-4:])
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
        'live': {
            'tps': [],
            'vram_used': [],
            'vram_total': [],
            'gpu_temp': [],
            'thermal_throttled': False,
        },
    }


def benchmark_progress_fraction(completed: object, total: object) -> float:
    try:
        completed_value = max(0.0, float(completed or 0))
        total_value = max(0.0, float(total or 0))
    except (TypeError, ValueError):
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
    failure_category = str(record.get('failure_category', '') or '')
    if failure_category and status.lower() not in ('ok', 'probe ok', 'tests passed'):
        suffix_parts.append(f'fail={failure_category}')
    benchmark_profile = str(record.get('benchmark_profile', '') or '')
    if benchmark_profile:
        suffix_parts.append(f'profile={benchmark_profile}')
    benchmark_kind = str(record.get('benchmark_kind', '') or '')
    if benchmark_kind:
        suffix_parts.append(f'kind={benchmark_kind}')
    if bool(record.get('cpu_moe', False)):
        suffix_parts.append('placement=cmoe')
    elif int(record.get('n_cpu_moe', 0) or 0) > 0:
        suffix_parts.append(f'placement=ncmoe{int(record.get("n_cpu_moe", 0) or 0)}')
    rejection = str(record.get('rejection_reason', '') or record.get('selection_rejection_reason', '') or '')
    if rejection:
        suffix_parts.append(f'reject={rejection}')
    suffix = (' ' + ' '.join(suffix_parts)) if suffix_parts else ''
    return (
        f'{label[:18]:18} {score:7.2f} {score_label:5} {seconds:6.1f}s '
        f'ctx={ctx:<6} slot={slot:<6} par={parallel:<2} {status}{suffix}'
    )


def benchmark_launch_profile_detail_lines(record: Dict[str, object]) -> List[str]:
    profile = str(record.get('benchmark_profile', '') or record.get('benchmark_kind', '') or '')
    if not profile:
        return []
    engine = str(record.get('engine', '') or '-')
    strategy = str(record.get('benchmark_strategy_id', '') or '')
    phase = str(record.get('benchmark_phase', '') or '')
    binary = str(record.get('binary_path', '') or record.get('server_bin', '') or '-')
    ctx = int(record.get('ctx', 0) or 0)
    output = int(record.get('output', 0) or 0)
    measurement = int(record.get('measurement_output', 0) or 0)
    kv_key = str(record.get('kv_key', '') or record.get('ctk', '') or '-')
    kv_value = str(record.get('kv_value', '') or record.get('ctv', '') or '-')
    flash = str(record.get('flash_attn', '') or record.get('flash_attn_mode', '') or '-')
    fit = 'on' if bool(record.get('fit', record.get('runtime_fit', False))) else 'off'
    fit_context = int(record.get('fit_context', 0) or 0)
    fit_target = int(record.get('fit_target', 0) or 0)
    no_mmap = 'yes' if bool(record.get('no_mmap', record.get('runtime_no_mmap', False))) else 'no'
    draft_kv = ''
    if record.get('draft_ctk') or record.get('draft_ctv'):
        draft_kv = f' draft_key={record.get("draft_ctk", "-") or "-"} draft_value={record.get("draft_ctv", "-") or "-"}'
    no_shift = 'yes' if bool(record.get('no_context_shift', False)) else 'no'
    sampling = (
        f'temp={record.get("temp", "-")} top_p={record.get("top_p", "-")} '
        f'top_k={record.get("top_k", "-")} repeat={record.get("repeat_penalty", "-")} '
        f'presence={record.get("presence_penalty", "-")}'
    )
    def format_value(value: object) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        return str(value)

    kwargs = dict(record.get('chat_template_kwargs', {}) or {})
    kwargs_text = ','.join(f'{key}={format_value(value)}' for key, value in kwargs.items()) if kwargs else '-'
    unsupported = ','.join(str(item) for item in list(record.get('unsupported_launch_flags', []) or [])) or '-'
    if bool(record.get('cpu_moe', False)):
        placement = 'cmoe'
    elif int(record.get('n_cpu_moe', 0) or 0) > 0:
        placement = f'ncmoe={int(record.get("n_cpu_moe", 0) or 0)}'
    elif int(record.get('ngl', 0) or 0) > 0:
        placement = f'partial ngl={int(record.get("ngl", 0) or 0)}'
    else:
        placement = str(record.get('placement_strategy', '') or '-')
    reasoning = str(record.get('reasoning_mode', '') or '')
    if not reasoning:
        reasoning = str(record.get('reasoning', '') or '-')
        fmt = str(record.get('reasoning_format', '') or '')
        budget = record.get('reasoning_budget', None)
        if fmt and reasoning != '-':
            reasoning = f'{reasoning}/{fmt}'
        if budget is not None and reasoning != '-':
            reasoning = f'{reasoning} budget={budget}'
    rejection = str(record.get('rejection_reason', '') or record.get('selection_rejection_reason', '') or '')
    return [
        f'  profile: {profile} engine={engine} ctx={ctx} output={output} measure={measurement}',
        *( [f'  strategy: {strategy}' + (f' phase={phase}' if phase else '')] if strategy else [] ),
        f'  binary: {binary}',
        (
            f'  kv: key={kv_key} value={kv_value} flash={flash} fit={fit}'
            + (f' fit_ctx={fit_context}' if fit_context else '')
            + (f' fit_target={fit_target}' if fit_target else '')
            + f'{draft_kv}'
            + f' no_ctx_shift={no_shift}'
            + f' no_mmap={no_mmap}'
        ),
        f'  placement: {placement} reasoning={reasoning or "-"}',
        f'  sampling: {sampling}',
        f'  template: {kwargs_text} unsupported={unsupported}',
        *( [f'  rejected: {rejection}'] if rejection else [] ),
    ]


def benchmark_record_display_items(record: Dict[str, object], attr: int = 0) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = [(benchmark_row_text(record), attr)]
    for line in benchmark_launch_profile_detail_lines(record):
        items.append((line, attr))
    detail = compact_message(str(record.get('detail', '') or ''))
    if detail:
        items.append((f'  detail: {detail}', attr))
    failure_excerpt = compact_message(str(record.get('failure_excerpt', '') or ''))
    if failure_excerpt:
        items.append((f'  failure: {failure_excerpt}', attr))
    runtime_log_path = str(record.get('runtime_log_path', '') or '')
    if runtime_log_path:
        items.append((f'  log: {runtime_log_path}', attr))
    viable_ngl = int(record.get('viable_ngl', 0) or 0)
    fit_phase = str(record.get('fit_discovery_phase', '') or '')
    if viable_ngl or fit_phase:
        source = str(record.get('viable_ngl_source', '') or record.get('fit_selected_ngl_source', '') or 'unknown')
        ngl_text = str(viable_ngl) if viable_ngl else '-'
        phase_text = fit_phase or '-'
        items.append((f'  fit discovery: phase={phase_text} viable_ngl={ngl_text} source={source}', attr))
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


def latest_benchmark_run(model: ModelConfig, kind: str) -> Dict[str, object]:
    target = str(kind or '').strip()
    for run in benchmark_runs_for_model(model):
        if str(run.get('kind', '') or '') == target:
            return run if isinstance(run, dict) else {}
    return {}


def compact_bytes(value: object) -> str:
    try:
        raw = int(value or 0)
    except (TypeError, ValueError):
        raw = 0
    if raw <= 0:
        return '-'
    if raw >= 1024 ** 3:
        return f'{raw / (1024 ** 3):.1f} GiB'
    if raw >= 1024 ** 2:
        return f'{raw / (1024 ** 2):.0f} MiB'
    return f'{raw} B'


def moe_recommendation_state_text(model: ModelConfig) -> str:
    if not has_moe_recommendation(model):
        return 'none'
    return 'applied' if moe_recommendation_applied(model) else 'available, not applied'


def _safe_benchmark_freshness(app: Optional[AppConfig], model: ModelConfig) -> str:
    if app is None:
        return 'missing'
    try:
        return benchmark_freshness_label(app, model)
    except (AttributeError, TypeError, OSError, ValueError):
        status = str(getattr(model, 'default_benchmark_status', '') or '').strip().lower()
        return status if status in ('fresh', 'stale', 'missing', 'failed', 'pending', 'running') else 'missing'


def _has_any_measured_launch_profile(model: ModelConfig) -> bool:
    return any(get_measured_profile(model, key) for key in ('opencode_ready', 'auto', 'fast_chat', 'long_context'))


def suggested_next_action(
    app: Optional[AppConfig],
    model: Optional[ModelConfig],
    status: str = 'STOPPED',
) -> SuggestedAction:
    if model is None:
        return SuggestedAction('Select a model', 'Enter', 'Choose a model before launching or benchmarking.', 'normal')

    normalized_status = str(status or '').strip().upper()
    if normalized_status == 'ERROR':
        return SuggestedAction('View Logs', 'L', 'The selected model is in an error state.', 'error')

    if has_moe_recommendation(model) and not moe_recommendation_applied(model):
        return SuggestedAction(
            'Apply MoE Recommendation',
            'A',
            'Measured MoE placement exists but current launch config does not use it.',
            'warning',
        )

    freshness = _safe_benchmark_freshness(app, model)
    if freshness == 'failed' and not _has_any_measured_launch_profile(model):
        return SuggestedAction('View Results', 'R', 'The last benchmark failed and no working measured profile is saved.', 'error')

    try:
        moe_reason = _moe_menu_disabled_reason(app, model)
    except (AttributeError, TypeError, OSError, ValueError):
        moe_reason = 'not eligible'
    if not moe_reason and not has_moe_recommendation(model):
        mtp_suite = app is not None and active_engine_key(app, model) == 'llama.cpp-mtp'
        return SuggestedAction(
            'Run MTP Suite' if mtp_suite else 'Run Full Suite Benchmark',
            'B',
            'This MTP MoE model needs acceptance plus placement tuning.' if mtp_suite else 'This MoE model has not measured expert placement yet.',
            'warning',
        )

    if not _has_any_measured_launch_profile(model):
        return SuggestedAction('Run Smart Benchmark', 'B', 'No measured launch profile is saved for this model.', 'warning')

    if freshness in ('stale', 'missing'):
        return SuggestedAction('Run Smart Benchmark', 'B', f'Benchmark proof is {freshness}.', 'warning')

    if not float(getattr(model, 'last_opencode_benchmark_score', 0.0) or 0.0):
        return SuggestedAction('Run OpenCode Benchmark', 'B', 'Launch profile exists, but OpenCode workflow is not validated.', 'normal')

    if normalized_status == 'READY':
        return SuggestedAction('Try Model', 'T', 'Model is running and ready for an interactive check.', 'normal')

    return SuggestedAction('Launch Model', 'T', 'Measured profile exists; launch or try the model.', 'normal')


def overview_items(
    app: Optional[AppConfig],
    model: ModelConfig,
    status: str,
    detail: str = '',
    width: int = 120,
    success_attr: int = 0,
    warning_attr: int = 0,
    error_attr: int = 0,
    heading_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    row_summary = build_model_row_summary(app, model, status) if app is not None else {}
    action = suggested_next_action(app, model, status)
    severity_attr = {
        'error': error_attr,
        'warning': warning_attr,
        'success': success_attr,
    }.get(action.severity, normal_attr)
    health = str(row_summary.get('health', '-') or '-')
    health_reason = compact_message(str(row_summary.get('health_reason', '') or '-'))
    health_attr = success_attr if health == 'OK' else warning_attr if health in ('WARN', 'STALE') else error_attr
    model_type = classify_model_type(model)
    engine = active_engine_short(app, model) if app is not None else display_runtime(model)
    compatible = True
    visible = True
    visibility_reason = ''
    visibility_status = ''
    compatibility_reason = ''
    compatibility_status = ''
    if app is not None:
        try:
            visibility = app.model_engine_visibility(model)
            visible = visibility.compatible
            visibility_reason = visibility.reason
            visibility_status = visibility.status
        except (AttributeError, TypeError, OSError, ValueError):
            visible = True
        try:
            compatibility = app.model_engine_compatibility(model)
            compatible = compatibility.compatible
            compatibility_reason = compatibility.reason
            compatibility_status = compatibility.status
        except (AttributeError, TypeError, OSError, ValueError):
            compatible = True
    try:
        strategy = benchmark_strategy_for_app(app, model, depth='fast', objective='quick_sanity') if app is not None else None
    except (AttributeError, TypeError, OSError, ValueError):
        strategy = None
    benchmark = benchmark_freshness_display(app, model) if app is not None else 'Missing'
    moe_state = moe_recommendation_state_text(model) if model_is_moe(model) else 'not MoE'
    mtp_state = mtp_status_short(app, model) if app is not None else 'off'
    status_text = f'{status}'
    detail_text = compact_message(str(detail or ''))
    if detail_text and status != 'STOPPED':
        status_text = f'{status_text} ({detail_text})'
    measured_pick = str(row_summary.get('pick', '-') or '-')
    measured_ctx = int(row_summary.get('ctx', 0) or 0)
    measured_tps = float(row_summary.get('tokens_per_sec', 0.0) or 0.0)
    mtp_attr = (
        success_attr if mtp_state in ('ready', 'usable', 'capable')
        else warning_attr if mtp_state in ('unknown', 'risky', 'testing')
        else error_attr if mtp_state in ('blocked', 'failed')
        else normal_attr
    )
    # Card-style cockpit layout: Runtime / Performance / Health / Actions /
    # Recent benchmark. The renderer treats heading_attr rows as card titles.
    items: List[Tuple[str, int]] = [
        ('Runtime', heading_attr),
        (f'Name: {model.name or model.id}', normal_attr),
        (f'Type: {model_type}   Quant: {extract_quant(model) or "-"}', normal_attr),
        (f'Engine: {engine}', normal_attr),
        (f'Status: {status_text}', success_attr if str(status).upper() == 'READY' else error_attr if str(status).upper() == 'ERROR' else normal_attr),
    ]
    if not visible:
        items.append((f'Engine visibility: hidden - {compact_message(visibility_reason)}', warning_attr))
    if visible and not compatible:
        items.append((f'Engine launch: blocked - {compact_message(compatibility_reason)}', warning_attr))
    if visible and compatible and visibility_status == 'compatible_with_warning':
        items.append((f'Engine visibility: warning - {compact_message(visibility_reason)}', warning_attr))
    if visible and compatible and compatibility_status == 'compatible_with_warning':
        items.append((f'Engine launch: warning - {compact_message(compatibility_reason)}', warning_attr))

    items.extend([
        ('', normal_attr),
        ('Performance', heading_attr),
        (f'Profile: {measured_pick}', normal_attr),
        (f'Throughput: {measured_tps:.1f} tok/s' if measured_tps > 0 else 'Throughput: not measured', success_attr if measured_tps > 0 else warning_attr),
        (f'Context: {measured_ctx}' if measured_ctx > 0 else 'Context: -', normal_attr),
        (f'MoE placement: {moe_state}', warning_attr if moe_state == 'available, not applied' else normal_attr),
    ])
    if strategy is not None:
        items.append((
            f'Benchmark strategy: {strategy.id}'
            + (f' (blocked: {compact_message(strategy.blocked_reason)})' if getattr(strategy, 'blocked_reason', '') else ''),
            warning_attr if getattr(strategy, 'blocked_reason', '') else normal_attr,
        ))

    items.extend([
        ('', normal_attr),
        ('Health', heading_attr),
        (f'Health: {health} / {health_reason}', health_attr),
        (f'MTP: {mtp_state}', mtp_attr),
        (f'Benchmark freshness: {benchmark}', success_attr if benchmark == 'Fresh' else warning_attr),
        ('', normal_attr),
        ('Actions', heading_attr),
        (f'Suggested: {action.label}', severity_attr),
        (f'Why: {ellipsize(action.reason, max(40, int(width or 120) - 6))}', normal_attr),
        ('[B] Benchmark Menu', normal_attr),
    ])
    if action.key == 'A' or (has_moe_recommendation(model) and not moe_recommendation_applied(model)):
        items.append(('[A] Apply MoE Recommendation', warning_attr))
    items.extend([
        ('[T] Try / Launch   [R] Results   [?] Help', normal_attr),
    ])

    last_status = str(getattr(model, 'default_benchmark_status', '') or '').strip() or '-'
    last_at = str(getattr(model, 'default_benchmark_at', '') or '').strip()
    last_profile = str(getattr(model, 'last_benchmark_profile', '') or '').strip()
    last_tps = float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0)
    items.extend([
        ('', normal_attr),
        ('Recent benchmark', heading_attr),
        (
            f'Last run: {last_status}' + (f' ({last_at})' if last_at else ''),
            success_attr if last_status.lower() in ('done', 'complete', 'partial') else warning_attr if last_status != '-' else normal_attr,
        ),
        (
            f'Result: {last_tps:.1f} tok/s' + (f' / {last_profile}' if last_profile else '')
            if last_tps > 0 else 'Result: no measured throughput yet',
            normal_attr if last_tps > 0 else warning_attr,
        ),
    ])
    return items


def moe_tuning_items(
    model: ModelConfig,
    width: int = 120,
    success_attr: int = 0,
    warning_attr: int = 0,
    error_attr: int = 0,
    heading_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    profile = get_measured_profile(model, 'moe_placement')
    run = latest_benchmark_run(model, 'moe_tuning')
    items: List[Tuple[str, int]] = [('MoE Placement Recommendation', heading_attr)]
    if not profile:
        if run:
            items.append((f'latest run: {run.get("status", "-")} {run.get("summary", "")}', warning_attr))
        else:
            items.append(('No measured MoE placement recommendation yet.', warning_attr))
            items.append(('Run MoE tuning from Benchmark Menu to create a recommendation.', normal_attr))
        return items

    winner = str(profile.get('measured_candidate_name', '') or profile.get('runtime_profile', '') or '-')
    applied = moe_recommendation_applied(model)
    current_bits = []
    if bool(getattr(model, 'cpu_moe', False)):
        current_bits.append('cpu_moe_all')
    if int(getattr(model, 'n_cpu_moe', 0) or 0) > 0:
        current_bits.append(f'n_cpu_moe_{int(getattr(model, "n_cpu_moe", 0) or 0)}')
    if getattr(model, 'tensor_overrides', None):
        current_bits.append('tensor_override')
    speed = float(profile.get('tokens_per_sec', 0.0) or 0.0)
    headroom = profile.get('vram_headroom_bytes', 0)
    peak_vram = profile.get('peak_vram_used', profile.get('peak_vram_bytes', 0))
    peak_ram = profile.get('peak_ram_bytes', 0)
    ctx = int(profile.get('ctx', 0) or 0)
    reason = compact_message(str(profile.get('tuning_summary', '') or profile.get('selection_reason', '') or 'measured winner'))
    items.extend([
        (f'Winner: {winner}', success_attr | curses.A_BOLD if success_attr else normal_attr),
        (f'Applied: {"Yes" if applied else "No"}', success_attr if applied else warning_attr),
        (f'Current: {", ".join(current_bits) if current_bits else "none"}', normal_attr),
        (f'Speed: {speed:.2f} tok/s', success_attr if speed > 0 else warning_attr),
        (f'VRAM peak/headroom: {compact_bytes(peak_vram)} / {compact_bytes(headroom)}', normal_attr),
        (f'RAM peak: {compact_bytes(peak_ram)}', normal_attr),
        (f'Context: {ctx or "-"}', normal_attr),
        (f'Reason: {reason}', normal_attr),
    ])
    early_stop = str(profile.get('early_stop_reason', '') or (run.get('early_stop_reason', '') if run else ''))
    if early_stop:
        items.append((f'Early stop: {compact_message(early_stop)}', warning_attr))
    warnings = list(profile.get('warnings', []) or [])
    warnings.extend(list(run.get('warnings', []) or []) if run else [])
    deduped_warnings: List[str] = []
    for item in warnings:
        text = compact_message(str(item or ''))
        if text and text not in deduped_warnings:
            deduped_warnings.append(text)
    if deduped_warnings:
        items.append(('', normal_attr))
        items.append(('Warnings', heading_attr))
        items.extend((f'- {item}', warning_attr) for item in deduped_warnings[:5])
    applied_at = str(profile.get('applied_at', '') or '')
    if applied_at:
        items.append(('', normal_attr))
        items.append(('Traceability', heading_attr))
        items.append((f'applied_from: {profile.get("applied_from", "-")}', normal_attr))
        items.append((f'applied_at: {applied_at}', normal_attr))
        items.append((f'tuning_run_id: {profile.get("tuning_run_id", "-")}', normal_attr))
        items.append((f'measured candidate: {profile.get("measured_candidate_name", "-")}', normal_attr))
    log_path = str(profile.get('tuning_log_json', '') or (run.get('tuning_log_json', '') if run else ''))
    if log_path:
        items.append((f'log: {log_path}', normal_attr))
    args_preview = str(profile.get('effective_server_args_preview', '') or '')
    if args_preview:
        items.append((f'winner args: {ellipsize(args_preview, max(40, width - 13))}', normal_attr))
    items.append(('', normal_attr))
    items.append(('Actions', heading_attr))
    if applied:
        items.append(('MoE recommendation is already applied.', success_attr))
    else:
        items.append(('[A] Apply MoE Recommendation', warning_attr))
    items.append(('[B] Benchmark Menu', normal_attr))
    items.append(('[C] Command tab   [L] Logs tab', normal_attr))
    if run:
        items.append(('', normal_attr))
        items.append(('Candidate Rows', heading_attr))
        items.extend(benchmark_ranking_items(
            run,
            width=width,
            success_attr=success_attr,
            warning_attr=warning_attr,
            error_attr=error_attr,
            heading_attr=heading_attr,
            normal_attr=normal_attr,
        ))
    return items


def latest_full_suite_run(model: ModelConfig) -> Dict[str, object]:
    return latest_benchmark_run(model, 'full_suite')


def full_suite_results_items(
    model: ModelConfig,
    run: Dict[str, object],
    width: int = 120,
    success_attr: int = 0,
    warning_attr: int = 0,
    error_attr: int = 0,
    heading_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    if not run:
        return [('No Full Suite Benchmark run selected.', warning_attr)]
    records = list(run.get('records', []) or [])
    recommendations = run.get('recommendations', {}) if isinstance(run.get('recommendations', {}), dict) else {}
    warnings = [compact_message(str(item)) for item in list(run.get('warnings', []) or []) if compact_message(str(item))]
    status = str(run.get('status', '-') or '-')
    profile_key = suite_run_recommended_profile_key(model, run)
    moe_profile = get_measured_profile(model, 'moe_placement')
    moe_candidate = (
        str(recommendations.get('moe_placement', '') or '')
        or str(moe_profile.get('measured_candidate_name', '') or '')
        or '-'
    )
    profile_label_text = profile_key or '-'
    applied_moe = moe_recommendation_applied(model) if moe_profile else False
    summary = compact_message(str(run.get('summary', '') or ''))
    mtp_suite = full_suite_is_mtp(records)
    items: List[Tuple[str, int]] = [
        ('MTP Suite Summary' if mtp_suite else 'Full Suite Summary', heading_attr),
        (f'model: {model.name or model.id}', normal_attr),
        (f'run: {run.get("id", "-")}   status: {status}', success_attr if status == 'done' else warning_attr if status in ('running', 'aborted') else error_attr),
    ]
    if summary:
        items.append((f'summary: {ellipsize(summary, max(40, width - 9))}', normal_attr))
    items.extend([('', normal_attr), ('Stages', heading_attr)])
    for line in full_suite_stage_lines(records):
        attr = error_attr if '[!]' in line else warning_attr if '[-]' in line else success_attr if '[x]' in line else normal_attr
        items.append((line, attr))
    items.extend([
        ('', normal_attr),
        ('Recommendations', heading_attr),
        (f'MoE Placement: {moe_candidate}', success_attr if moe_candidate != '-' else warning_attr),
        (f'Default Profile: {profile_label_text}', success_attr if profile_key else warning_attr),
        (f'MoE Applied: {"Yes" if applied_moe else "No" if moe_profile else "-"}', success_attr if applied_moe else warning_attr),
    ])
    if warnings:
        items.extend([('', normal_attr), ('Warnings', heading_attr)])
        items.extend((f'- {item}', warning_attr) for item in warnings[:6])
    items.extend([
        ('', normal_attr),
        ('Actions', heading_attr),
        ('[A] Apply all recommendations', success_attr if profile_key or moe_profile else warning_attr),
        ('[M] Apply MoE only', warning_attr if moe_profile and not applied_moe else normal_attr),
        ('[P] Apply profile only', normal_attr if profile_key else warning_attr),
        ('[E] Export/sync configs', normal_attr),
    ])
    return items


def benchmark_run_line(run: Dict[str, object], index: int, selected: bool = False) -> str:
    marker = '>' if selected else ' '
    status = str(run.get('status', '-') or '-')[:8]
    run_id = str(run.get('id', f'run-{index + 1}') or f'run-{index + 1}')
    summary = str(run.get('summary', '') or 'no summary')
    return f'{marker} {index + 1:02d} {status:8} {run_id[:18]:18} {summary}'


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


def refresh_benchmark_live(state: Dict[str, object]) -> Dict[str, object]:
    """Rebuild the live cockpit ring buffers from the run's benchmark records.

    Idempotent: the telemetry series are derived straight from state['records']
    so repeated reduce calls never double-count a candidate.
    """
    records = [item for item in list(state.get('records', []) or []) if isinstance(item, dict)]
    tps: List[float] = []
    vram_used: List[int] = []
    vram_total: List[int] = []
    gpu_temp: List[int] = []
    throttled = False
    for record in records:
        try:
            value = float(record.get('tokens_per_sec', 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            tps.append(round(value, 2))
        try:
            used = int(record.get('peak_vram_used', record.get('peak_vram_bytes', 0)) or 0)
            total = int(record.get('gpu_memory_total', 0) or 0)
        except (TypeError, ValueError):
            used = total = 0
        if used > 0 and total > 0:
            vram_used.append(used)
            vram_total.append(total)
        try:
            temp = int(record.get('gpu_temp_peak', 0) or 0)
        except (TypeError, ValueError):
            temp = 0
        if temp > 0:
            gpu_temp.append(temp)
        if bool(record.get('thermal_throttled', False)):
            throttled = True
    state['live'] = {
        'tps': tps[-BENCHMARK_LIVE_LIMIT:],
        'vram_used': vram_used[-BENCHMARK_LIVE_LIMIT:],
        'vram_total': vram_total[-BENCHMARK_LIVE_LIMIT:],
        'gpu_temp': gpu_temp[-BENCHMARK_LIVE_LIMIT:],
        'thermal_throttled': throttled,
    }
    return state


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
    refresh_benchmark_live(state)
    if isinstance(payload.get('recommendations'), dict):
        state['recommendations'] = dict(payload.get('recommendations') or {})
    if isinstance(payload.get('warnings'), list):
        state['warnings'] = list(payload.get('warnings') or [])

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


# Audit #8 step 2: prompt/modal/form helpers moved to ui_modals.py.
# Re-exported here so tui() and tests that import these names from
# llama_tui.ui keep working without churn.
from .ui_modals import *  # noqa: F401,F403
# `import *` skips underscore-prefixed names, so import the private helpers
# ui.py still calls explicitly.
from .ui_modals import _moe_menu_disabled_reason  # noqa: F401



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
        elif kind == 'engine':
            attr = colors['accent'] | curses.A_BOLD
        elif kind == 'warning':
            attr = colors['warning'] | curses.A_BOLD
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
    filter_compatibility = 'active'
    sort_mode = normalize_choice(getattr(app.ui, 'preferred_sort', 'port'), tuple(key for key, _label in SORT_OPTIONS), 'port')
    view_mode = 'list'
    detail_model_id = ''
    message = 'Ready.'
    last_error_message = ''
    error_history: List[str] = []
    right_tab_by_view: Dict[str, str] = {}
    right_tab_scrolls: Dict[str, int] = {}
    leaderboard_sort = LEADERBOARD_SORT_KEYS[0]
    right_tab_scroll_total = 0
    right_tab_scroll_rows = 1
    last_refresh = 0.0
    statuses: Dict[str, Tuple[str, str]] = {}
    # Two background-worker slots (audit finding #8 step 1): the main
    # benchmark/action thread and the try-out chat thread. Each pair owns
    # one CancelToken and one Thread; ActionRunner collapses what used to
    # be four separate closure-locals into two named slots so the helpers
    # below can mutate `action.thread = ...` / `action.token = ...` without
    # needing four `nonlocal` declarations apiece.
    action = ActionRunner()
    action_queue: Queue = Queue()
    try_ = ActionRunner()
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
        return action.is_running()

    def current_browser_models() -> List[ModelConfig]:
        return browser_models(
            app,
            statuses,
            search=list_search,
            runtime_filter=filter_runtime,
            source_filter=filter_source,
            status_filter=filter_status,
            tag_filter=filter_tag,
            compatibility_filter=filter_compatibility,
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
        nonlocal message, view_mode, detail_model_id, benchmark_state
        if action_running():
            message = '⏳ Another optimization is still running. Watch the log window for progress.'
            return
        token = CancelToken()
        action.token = token
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

        action.thread = threading.Thread(target=runner, daemon=True)
        action.thread.start()
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

    def apply_moe_recommendation_for_model(model: Optional[ModelConfig]) -> str:
        nonlocal message
        if model is None:
            return 'No model selected for MoE recommendation.'
        ok, apply_msg = apply_moe_recommendation(model)
        if not ok:
            return apply_msg
        app.add_or_update(model, sync_exports=False)
        sync_msg = sync_opencode_after_tuning(app)
        invalidate_machine_summary()
        return f'{apply_msg} | {sync_msg}'

    def active_full_suite_run_for_model(model: Optional[ModelConfig]) -> Dict[str, object]:
        if model is None:
            return {}
        if view_mode == 'results':
            runs = benchmark_runs_for_model(model)
            if runs and 0 <= results_run_index < len(runs):
                run = runs[results_run_index]
                return run if isinstance(run, dict) and str(run.get('kind', '') or '') == 'full_suite' else {}
        if view_mode == 'benchmark' and str(benchmark_state.get('run_kind') or '') == 'full_suite':
            run = latest_full_suite_run(model)
            return run or full_suite_run_from_state(benchmark_state)
        return {}

    def apply_full_suite_all_for_model(model: Optional[ModelConfig]) -> str:
        if model is None:
            return 'No model selected for Full Suite recommendations.'
        run = active_full_suite_run_for_model(model)
        ok, apply_msg = apply_full_suite_recommendations(app, model, run)
        if ok:
            invalidate_machine_summary()
        return apply_msg

    def apply_full_suite_profile_for_model(model: Optional[ModelConfig]) -> str:
        if model is None:
            return 'No model selected for Full Suite profile recommendation.'
        run = active_full_suite_run_for_model(model)
        ok, apply_msg = apply_full_suite_profile_recommendation(model, run)
        if not ok:
            return apply_msg
        app.add_or_update(model, sync_exports=False)
        sync_msg = sync_opencode_after_tuning(app)
        invalidate_machine_summary()
        return f'{apply_msg} | {sync_msg}'

    def sync_suite_exports_for_model(model: Optional[ModelConfig]) -> str:
        if model is None:
            return 'No model selected for config export.'
        sync_msg = sync_opencode_after_tuning(app)
        append_model_log(app, model, f'Full Suite export sync requested: {sync_msg}')
        return sync_msg

    def start_benchmark_choice(model: Optional[ModelConfig], choice: str):
        nonlocal message
        if model is None:
            message = 'No model selected for benchmark.'
            return
        if str(choice or '').startswith('disabled:'):
            message = str(choice).split(':', 2)[-1] or 'Benchmark action unavailable.'
            return
        if choice == 'cancel':
            message = 'Benchmark cancelled.'
            return
        if choice == 'quick_benchmark':
            start_background_action(
                model,
                'quick benchmark profiles',
                lambda progress, token, model=model: benchmark_fast_profiles(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                ),
                done_event='benchmark_done',
                run_kind='server_fast',
            )
            return
        if choice == 'smart_benchmark':
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
            return
        if choice == 'moe_tuning_full':
            start_background_action(
                model,
                'MoE placement tuning (full)',
                lambda progress, token, model=model: benchmark_moe_placement_tuning(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                    depth='full',
                ),
                done_event='benchmark_done',
                run_kind='moe_tuning',
            )
            return
        if choice == 'hermes_benchmark':
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
            return
        if choice == 'opencode_benchmark':
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
            return
        if choice == 'full_suite':
            start_background_action(
                model,
                'Full Suite Benchmark',
                lambda progress, token, model=model: benchmark_full_suite(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                    depth='full',
                ),
                done_event='benchmark_done',
                run_kind='full_suite',
            )
            return
        message = 'Benchmark action unavailable.'

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
        nonlocal view_mode, detail_model_id, message, try_session, try_input_scroll, try_transcript_scroll
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
        try_.token = CancelToken()
        status, detail = app.health(model)
        if status == 'READY':
            try_status = 'ready'
            message = f'{model.id}: try-out ready. Type a prompt and press Enter.'
            return

        try_status = 'starting'
        message = f'{model.id}: starting try-out server...'
        token = try_.token
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

        try_.thread = threading.Thread(target=runner, daemon=True)
        try_.thread.start()

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
        nonlocal message, try_input, try_input_scroll, try_status, try_error, try_response_index, try_messages, try_transcript_scroll
        if view_mode != 'try':
            return
        model = active_detail_model()
        if not model:
            return
        if try_status != 'ready':
            message = f'{model.id}: wait until the try-out server is ready.'
            return
        if try_.is_running():
            message = f'{model.id}: response is still streaming.'
            return
        prompt = try_input.strip()
        if not prompt:
            return
        if try_.token is None:
            try_.token = CancelToken()
        token = try_.token
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

        try_.thread = threading.Thread(target=runner, daemon=True)
        try_.thread.start()

    def exit_try_view():
        nonlocal view_mode, message, last_refresh, try_session, try_input, try_input_scroll, try_transcript_scroll
        nonlocal try_status, try_error, try_response_index, try_launched_model_id
        model = active_detail_model()
        try_.cancel('leaving try-out')
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
        try_.reset()
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

    def drain_action_queue():
        nonlocal message, try_status, try_error, last_refresh
        nonlocal try_response_index
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
                    try_.thread = None
                    message = text or f'{detail_model_id}: try-out ready.'
                    last_refresh = 0.0
                    continue
                if event == 'try_error':
                    try_status = 'error'
                    try_error = text or 'try-out failed'
                    try_.thread = None
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
                    try_.thread = None
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
                    try_.thread = None
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
                action.reset()
                last_refresh = 0.0
                invalidate_machine_summary()

    while True:
        drain_action_queue()

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
            if key == curses.KEY_RESIZE:
                try:
                    curses.update_lines_cols()
                except Exception:
                    pass
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
                app=app,
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
        current_detail_density = normalize_choice(getattr(app.ui, 'detail_density', 'simple'), tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'simple')
        right_tabs = right_tabs_for_view(view_mode, current_detail_density)
        right_active_tab = normalize_right_tab(view_mode, right_tab_by_view.get(view_mode, ''), current_detail_density) if right_tabs else ''
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
            draw_box(stdscr, box_top, right_x, right_panel_h, right_w, 'Details / Engine / Exports', colors['accent'] | curses.A_BOLD, colors['accent'])

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
            y_cursor = box_top + 2
            if run_kind == 'full_suite':
                suite_items = build_full_suite_progress_items(
                    model,
                    benchmark_state,
                    left_w - 5,
                    accent_attr=colors['accent'] | curses.A_BOLD,
                    normal_attr=curses.A_NORMAL,
                )
                for line, attr in suite_items[:content_rows]:
                    if y_cursor > content_bottom:
                        break
                    safe_addstr(stdscr, y_cursor, 3, line[: left_w - 5], attr)
                    y_cursor += 1
            else:
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
            active_engine = app.active_engine_key_for_model(model)
            tq_kind = turboquant_status_kind(
                model,
                active_engine == 'turboquant',
            )
            tq_attr = (
                colors['success'] | curses.A_BOLD if tq_kind == 'success'
                else colors['warning'] if tq_kind == 'warning'
                else colors['muted']
            )
            cap = dict((getattr(model, 'verification_results', {}) or {}).get('cap') or {})
            cap_text = (
                f'cap: {cap.get("limiting_factor", "-")} '
                f'configured={cap.get("configured_ctx", "-")} slot={cap.get("ctx_per_slot", context_per_slot(model))} '
                f'safe={cap.get("estimated_safe_context", "-")} measured={cap.get("measured_max_context", "-")}'
            )
            if detail_density == 'advanced':
                detail_rows = [
                    ('[Esc] back   [Enter/l] actions   [T] try   [B] benchmark menu   [F] fast   [O] opencode   [H] hermes   [R] results   [z] auto   [v] simple', colors['accent'] | curses.A_BOLD),
                    ('', curses.A_NORMAL),
                    (f'name: {model.name}', curses.A_BOLD),
                    (active_engine_badge_line(app, model), colors['accent'] | curses.A_BOLD),
                    (runtime_engine_source_line(app, model), curses.A_NORMAL),
                    (active_engine_detail_line(app, model), colors['accent'] | curses.A_BOLD),
                    *[
                        (
                            line,
                            colors['warning'] if any(marker in line for marker in ('unsupported', 'unknown', 'warning')) else curses.A_NORMAL,
                        )
                        for line in model_engine_visibility_lines(app, model)
                    ],
                    (f'architecture/runtime/offload: {classify_model_type(model)} / {display_runtime(model)} / {display_offload(model)}', curses.A_NORMAL),
                    (f'architecture detail: {architecture_detail(model)}', curses.A_NORMAL),
                    (turboquant_detail_line(model), tq_attr),
                    (f'quant/source: {extract_quant(model)} / {getattr(model, "source", "manual")}', curses.A_NORMAL),
                    (f'favorite/freshness: {"yes" if getattr(model, "favorite", False) else "no"} / {freshness}', curses.A_NORMAL),
                    (f'tags: {tags_text}', curses.A_NORMAL),
                    (f'verification: {verification_status} / {verification_summary}', colors['success'] | curses.A_BOLD if verification_status == 'passed' else colors['warning'] if verification_status in ('warning', 'needs_benchmark', 'unknown') else colors['error'] | curses.A_BOLD),
                    (cap_text, curses.A_NORMAL),
                    (f'path: {model.path}', curses.A_NORMAL),
                    (f'alias/bind: {model.alias} / http://{model.host}:{model.port}', curses.A_NORMAL),
                    (f'status: {status} ({detail})', status_attr(colors, status)),
                    (f'pid/roles: {app.get_pid(model) or "-"} / {app.role_badges(model.id)}', curses.A_NORMAL),
                    (f'log: {app.logfile(model.id)}', curses.A_NORMAL),
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
                engine_warning = active_engine_warning_line(app, model)
                if engine_warning:
                    detail_rows.insert(5, (engine_warning, active_engine_warning_attr(engine_warning, colors)))
                detail_rows.extend([
                    ('', curses.A_NORMAL),
                    ('advanced details:', colors['accent'] | curses.A_BOLD),
                    ('Use the right-side Benchmarks tab for full benchmark tables.', colors['muted']),
                    (f'server rows: {len(getattr(model, "last_benchmark_results", []) or [])}', colors['muted']),
                    (f'opencode rows: {len(getattr(model, "last_opencode_benchmark_results", []) or [])}', colors['muted']),
                    (f'hermes rows: {len(getattr(model, "last_hermes_benchmark_results", []) or [])}', colors['muted']),
                ])
            else:
                detail_rows = [
                    ('[Esc] back   [Enter/l] actions   [B] benchmark menu   [T] try   [R] results   [v] advanced', colors['accent'] | curses.A_BOLD),
                    ('', curses.A_NORMAL),
                ]
                detail_rows.extend(overview_items(
                    app,
                    model,
                    status,
                    detail,
                    width=left_w - 5,
                    success_attr=colors['success'] | curses.A_BOLD,
                    warning_attr=colors['warning'],
                    error_attr=colors['error'] | curses.A_BOLD,
                    heading_attr=colors['accent'] | curses.A_BOLD,
                    normal_attr=curses.A_NORMAL,
                ))
                detail_rows.extend([
                    ('', curses.A_NORMAL),
                    ('Advanced paths, commands, and raw benchmark rows are in the right-side tabs.', colors['muted']),
                ])

            detail_items = scrollable_pane_wrapped_items(detail_rows, left_w - 5)
            for i, (line, attr) in enumerate(detail_items[:content_rows]):
                safe_addstr(stdscr, box_top + 2 + i, 3, line[: left_w - 4], attr)
        elif browser_list:
            browser_view = normalize_choice(getattr(app.ui, 'browser_view', 'compact'), tuple(key for key, _label in BROWSER_VIEW_OPTIONS), 'compact')
            header = browser_header_for_view(browser_view, left_w)
            if content_rows > 0:
                summary = (
                    f'search={list_search or "-"}  runtime={filter_option_label(FILTER_RUNTIME_OPTIONS, filter_runtime)}  '
                    f'source={filter_option_label(FILTER_SOURCE_OPTIONS, filter_source)}  '
                    f'status={filter_option_label(FILTER_STATUS_OPTIONS, filter_status)}  '
                    f'tag={filter_tag}  compat={filter_option_label(FILTER_COMPATIBILITY_OPTIONS, filter_compatibility)}  '
                    f'sort={sort_mode_label(sort_mode)}  view={browser_view_label(browser_view)}'
                )
                safe_addstr(stdscr, box_top + 2, 3, ellipsize(summary, left_w - 4), colors['muted'])
            if content_rows > 1:
                safe_addstr(stdscr, box_top + 3, 3, header, colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD)
            start_idx, end_idx = visible_selection_window(len(browser_list), selected, max(0, visible_rows - 1))
            for idx in range(start_idx, end_idx):
                model = browser_list[idx]
                status, _ = statuses.get(model.id, ('?', ''))
                line = browser_model_line_for_view(app, model, status, machine_pick_id, left_w, browser_view)
                row_y = box_top + 4 + idx - start_idx
                if idx == selected:
                    try:
                        safe_addstr(stdscr, row_y, 3, line[: left_w - 3], colors['selection'] | curses.A_BOLD)
                    except curses.error:
                        safe_addstr(stdscr, row_y, 3, line[: left_w - 3], curses.A_REVERSE)
                else:
                    safe_addstr(stdscr, row_y, 3, line[: left_w - 3])
                    if browser_view == 'advanced':
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
                safe_addstr(stdscr, box_top + 6, 3, 'Press a to add a manual llama.cpp model.', colors['muted'])
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
                row_summary = build_model_row_summary(app, model, status)
                if right_active_tab == 'overview':
                    right_items = overview_items(
                        app,
                        model,
                        status,
                        detail,
                        width=right_content_w,
                        success_attr=colors['success'] | curses.A_BOLD,
                        warning_attr=colors['warning'],
                        error_attr=colors['error'] | curses.A_BOLD,
                        heading_attr=colors['accent'] | curses.A_BOLD,
                        normal_attr=curses.A_NORMAL,
                    )
                elif right_active_tab == 'launch':
                    right_items = [
                        ('launch actions:', colors['accent'] | curses.A_BOLD),
                        ('Enter/l opens start/stop actions', curses.A_NORMAL),
                        ('T opens Try It Out', curses.A_NORMAL),
                        ('B/F/O/H run benchmark workflows', curses.A_NORMAL),
                        ('z applies automatic tuning', curses.A_NORMAL),
                        ('', curses.A_NORMAL),
                        ('saved launch settings:', colors['accent'] | curses.A_BOLD),
                        (f'alias/bind: {model.alias} / {model.host}:{model.port}', curses.A_NORMAL),
                        (f'profile: {model_profile_summary(model)}', curses.A_NORMAL),
                        (f'ctx/output: {model.ctx} / {model.output}', curses.A_NORMAL),
                        (f'last benchmark: {benchmark_score:.2f} tok/s {getattr(model, "last_benchmark_profile", "")}', curses.A_NORMAL),
                    ]
                elif right_active_tab == 'tuning':
                    right_items = [
                        ('tuning settings:', colors['accent'] | curses.A_BOLD),
                        (f'ctx/output: {model.ctx} / {model.output}', curses.A_NORMAL),
                        (f'ctx range: {getattr(model, "ctx_min", 2048)}..{getattr(model, "ctx_max", 131072)}', curses.A_NORMAL),
                        (f'ngl/threads/parallel: {model.ngl} / {model.threads} / {model.parallel}', curses.A_NORMAL),
                        (f'cache_ram/temp: {model.cache_ram} / {model.temp}', curses.A_NORMAL),
                        (f'optimize mode/tier: {model.optimize_mode} / {model.optimize_tier}', curses.A_NORMAL),
                        (f'memory reserve: {getattr(model, "memory_reserve_percent", 25)}%', curses.A_NORMAL),
                        (f'kv: {active_engine_kv(app, model)}', curses.A_NORMAL),
                        (f'extra args: {" ".join(getattr(model, "extra_args", []) or []) or "-"}', curses.A_NORMAL),
                    ]
                    right_items.extend([
                        ('', curses.A_NORMAL),
                        ('actions:', colors['accent'] | curses.A_BOLD),
                        ('B opens Benchmark Menu', curses.A_NORMAL),
                        ('', curses.A_NORMAL),
                    ])
                    if has_moe_recommendation(model) and not moe_recommendation_applied(model):
                        right_items.insert(-2, ('A applies MoE recommendation', colors['warning']))
                    elif has_moe_recommendation(model):
                        right_items.insert(-2, ('MoE recommendation is applied', colors['success']))
                    right_items.extend(moe_tuning_items(
                        model,
                        width=right_content_w,
                        success_attr=colors['success'],
                        warning_attr=colors['warning'],
                        error_attr=colors['error'],
                        heading_attr=colors['accent'] | curses.A_BOLD,
                        normal_attr=curses.A_NORMAL,
                    ))
                elif right_active_tab == 'logs':
                    right_items = [(f'log: {app.logfile(model.id)}', colors['accent'] | curses.A_BOLD)]
                    right_items.extend(build_log_items(log_lines, curses.A_NORMAL, colors['muted']))
                    if error_source_lines:
                        right_items.extend([('', curses.A_NORMAL), ('recent errors:', colors['error'] | curses.A_BOLD)])
                        right_items.extend(build_error_items(error_source_lines, colors['error'], colors['muted']))
                elif right_active_tab == 'command':
                    right_items = [
                        ('command preview:', colors['accent'] | curses.A_BOLD),
                        (active_engine_detail_line(app, model), curses.A_NORMAL),
                        (command_preview, curses.A_NORMAL),
                    ]
                    engine_warning = active_engine_warning_line(app, model)
                    if engine_warning:
                        right_items.insert(2, (engine_warning, active_engine_warning_attr(engine_warning, colors)))
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
                elif right_active_tab == 'exports':
                    continue_roles = []
                    for label, value in (
                        ('default', getattr(app.continue_settings, 'default_model_id', '')),
                        ('edit', getattr(app.continue_settings, 'edit_model_id', '')),
                        ('autocomplete', getattr(app.continue_settings, 'autocomplete_model_id', '')),
                    ):
                        if value == model.id:
                            continue_roles.append(label)
                    opencode_roles = []
                    for label, value in (
                        ('default', getattr(app.opencode, 'default_model_id', '')),
                        ('small', getattr(app.opencode, 'small_model_id', '')),
                        ('build', getattr(app.opencode, 'build_model_id', '')),
                        ('plan', getattr(app.opencode, 'plan_model_id', '')),
                    ):
                        if value == model.id:
                            opencode_roles.append(label)
                    hermes_roles = []
                    for label, value in (
                        ('default', getattr(app.hermes, 'default_model_id', '')),
                        ('code', getattr(app.hermes, 'code_model_id', '')),
                    ):
                        if value == model.id:
                            hermes_roles.append(label)
                    right_items = [
                        ('export roles:', colors['accent'] | curses.A_BOLD),
                        (f'OpenCode: {", ".join(opencode_roles) or "-"}', curses.A_NORMAL),
                        (f'Continue: {", ".join(continue_roles) or "-"}', curses.A_NORMAL),
                        (f'Hermes: {", ".join(hermes_roles) or "-"}', curses.A_NORMAL),
                        ('', curses.A_NORMAL),
                        ('config paths:', colors['accent'] | curses.A_BOLD),
                        (f'OpenCode: {getattr(app.opencode, "path", "") or "-"}', curses.A_NORMAL),
                        (f'Continue: {getattr(app.continue_settings, "path", "") or "-"}', curses.A_NORMAL),
                        (f'Hermes home: {getattr(app.hermes, "home_root", "") or "-"}', curses.A_NORMAL),
                    ]

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
                        app=app,
                    )
                elif right_active_tab == 'results':
                    if str(benchmark_state.get('run_kind') or '') == 'full_suite':
                        suite_run = latest_full_suite_run(model) or full_suite_run_from_state(benchmark_state)
                        right_items = full_suite_results_items(
                            model,
                            suite_run,
                            width=right_content_w,
                            success_attr=colors['success'] | curses.A_BOLD,
                            warning_attr=colors['warning'],
                            error_attr=colors['error'],
                            heading_attr=colors['accent'] | curses.A_BOLD,
                            normal_attr=curses.A_NORMAL,
                        )
                    else:
                        right_items = []
                        if getattr(model, 'engine_benchmark_store', {}) or {}:
                            lb_attr = {
                                'heading': colors['accent'] | curses.A_BOLD,
                                'success': colors['success'] | curses.A_BOLD,
                                'warning': colors['warning'],
                                'muted': colors['muted'],
                                'normal': curses.A_NORMAL,
                            }
                            right_items.append((
                                f'engine leaderboard (sort: {leaderboard_sort} · [s] cycle):',
                                colors['accent'] | curses.A_BOLD,
                            ))
                            right_items.extend(
                                (text, lb_attr.get(kind, curses.A_NORMAL))
                                for text, kind in benchmark_leaderboard_lines(model, leaderboard_sort)
                            )
                            right_items.append(('', curses.A_NORMAL))
                        run = {
                            'kind': str(benchmark_state.get('run_kind') or ''),
                            'records': records,
                            'winners': {},
                        }
                        right_items.extend(benchmark_ranking_items(
                            run,
                            width=right_content_w,
                            success_attr=colors['success'] | curses.A_BOLD,
                            warning_attr=colors['warning'],
                            error_attr=colors['error'],
                            heading_attr=colors['accent'] | curses.A_BOLD,
                            normal_attr=curses.A_NORMAL,
                        ))
                elif right_active_tab == 'commands':
                    right_items = [
                        (line, colors['warning'] if kind == 'current' and benchmark_state.get('active') else colors['muted'])
                        for line, kind in benchmark_command_lines(benchmark_state, right_content_w, BENCHMARK_COMMAND_LIMIT + 1)
                    ]
                elif right_active_tab == 'logs':
                    right_items = [(f'log: {app.logfile(model.id)}', colors['accent'] | curses.A_BOLD)]
                    right_items.extend(build_log_items(log_lines, curses.A_NORMAL, colors['muted']))
                elif right_active_tab == 'errors':
                    right_items = build_error_items(error_source_lines, colors['error'], colors['muted'])

            elif view_mode == 'try':
                if right_active_tab == 'profile':
                    right_items = [
                        (f'model: {model.name}', curses.A_BOLD),
                        (runtime_engine_source_line(app, model), curses.A_NORMAL),
                        (active_engine_detail_line(app, model), colors['accent'] | curses.A_BOLD),
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
                    right_items = [(f'log: {app.logfile(model.id)}', colors['accent'] | curses.A_BOLD)]
                    right_items.extend(build_log_items(log_lines, curses.A_NORMAL, colors['muted']))
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
                        (active_engine_detail_line(app, model), curses.A_NORMAL),
                        (command_preview, curses.A_NORMAL),
                    ]
                    engine_warning = active_engine_warning_line(app, model)
                    if engine_warning:
                        right_items.insert(2, (engine_warning, active_engine_warning_attr(engine_warning, colors)))

            elif view_mode == 'results':
                runs = benchmark_runs_for_model(model)
                run = runs[results_run_index] if runs and 0 <= results_run_index < len(runs) else {}
                if right_active_tab == 'run_summary':
                    if run:
                        if str(run.get('kind', '') or '') == 'full_suite':
                            right_items = full_suite_results_items(
                                model,
                                run,
                                width=right_content_w,
                                success_attr=colors['success'] | curses.A_BOLD,
                                warning_attr=colors['warning'],
                                error_attr=colors['error'],
                                heading_attr=colors['accent'] | curses.A_BOLD,
                                normal_attr=curses.A_NORMAL,
                            )
                        else:
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
                (runtime_engine_source_line(app, model), curses.A_NORMAL),
                (active_engine_detail_line(app, model), colors['accent'] | curses.A_BOLD),
                (f'favorite/freshness: {"yes" if getattr(model, "favorite", False) else "no"} / {benchmark_freshness_display(app, model)}', curses.A_NORMAL),
                (f'tags/verification: {", ".join(list(getattr(model, "tags", []) or [])) or "-"} / {getattr(model, "verification_status", "unknown")}', curses.A_NORMAL),
                (f'alias/bind: {model.alias} / {model.host}:{model.port}', curses.A_NORMAL),
                (f'quant/architecture/offload: {extract_quant(model)} / {classify_model_type(model)} / {display_offload(model)}', curses.A_NORMAL),
                (f'ctx/output: {model.ctx} / {model.output}', curses.A_NORMAL),
                (f'profile: {model_profile_summary(model)}', curses.A_NORMAL),
                (f'benchmark: {getattr(model, "last_benchmark_tokens_per_sec", 0.0):.2f} tok/s {getattr(model, "last_benchmark_profile", "")}', curses.A_NORMAL),
                (f'status: {status} ({detail})', status_attr(colors, status)),
                (f'pid/roles: {app.get_pid(model) or "-"} / {app.role_badges(model.id)}', curses.A_NORMAL),
                (f'log: {app.logfile(model.id)}', curses.A_NORMAL),
                (f'browser: search={list_search or "-"} runtime={filter_runtime} source={filter_source} status={filter_status} tag={filter_tag} compat={filter_compatibility} sort={sort_mode}', colors['muted']),
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
            model = active_detail_model()
            if str(benchmark_state.get('run_kind') or '') == 'full_suite' and not benchmark_state.get('active'):
                footer = '[Esc] details  A Apply All  M MoE  P Profile  E Export  R Results'
            else:
                footer = '[Esc] details  [F] fast  [R] results  [W] wiki  Tab/] next  [A] abort'
            if right_active_tab == 'results':
                footer += '  [s] sort leaderboard'
            footer2 = '[Up/Down/PgUp/PgDn/Home/End] scroll right tab.'
        elif view_mode == 'results':
            model = active_detail_model()
            suite_run = active_full_suite_run_for_model(model)
            if suite_run:
                footer = '[Esc] details  A Apply All  M MoE  P Profile  E Export'
            else:
                footer = '[Esc] details  [Up/Down] select run  Tab/] next tab  Shift-Tab/[ prev tab'
            footer2 = '[PgUp/PgDn/Home/End] scroll active right tab.'
        elif view_mode == 'machine_results':
            footer = '[Esc] models  [D] deep all  Tab/] next tab  Shift-Tab/[ prev tab'
            footer2 = '[PgUp/PgDn/Home/End] scroll active right tab  [M] refresh rankings.'
        elif view_mode == 'detail':
            model = active_detail_model()
            if right_active_tab == 'tuning':
                apply_hint = 'A Apply MoE  ' if model and has_moe_recommendation(model) and not moe_recommendation_applied(model) else ''
                footer = f'[Esc] Back  {apply_hint}B Benchmark  C Command  L Logs  ? Help'
                footer2 = 'Tab/] next tab  Shift-Tab/[ prev tab'
            else:
                apply_hint = 'A Apply MoE  ' if model and has_moe_recommendation(model) and not moe_recommendation_applied(model) else ''
                footer = f'[Esc] Back  B Benchmark  {apply_hint}T Try  R Results  ? Help'
                footer2 = 'Tab/] next tab  Shift-Tab/[ prev tab  [:] Actions'
        else:
            footer = '[Enter] Details  B Benchmark  R Results  / Search  ? Help'
            footer2 = '[a/e/d] models  [x/X] detect/prune  [:] palette  [q] quit'
        if action_running():
            footer = '[A] abort active action   ' + footer
        safe_addstr(stdscr, h - 2, 2, ellipsize(footer, w - 4), colors['accent'] | curses.A_BOLD)
        safe_addstr(stdscr, h - 1, 2, ellipsize(footer2, w - 4), colors['muted'] | curses.A_BOLD)
        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break

        if key == -1:
            time.sleep(0.05)
            continue
        if key == curses.KEY_RESIZE:
            # Terminal resized: refresh LINES/COLS so the next render uses
            # the new dimensions. Without this, modals and the footer can
            # draw past the new edges and curses.error gets swallowed
            # silently. See audit finding #17.
            try:
                curses.update_lines_cols()
            except Exception:
                pass
            last_refresh = 0.0
            continue
        scroll_action = right_scroll_action_for_view(view_mode, key)
        tab_direction = right_tab_key_direction(key)
        if right_tabs and tab_direction:
            current_detail_density = normalize_choice(getattr(app.ui, 'detail_density', 'simple'), tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'simple')
            right_tab_by_view[view_mode] = cycle_right_tab(view_mode, right_active_tab, tab_direction, current_detail_density)
            message = f'Right tab: {right_tab_label(right_tab_by_view[view_mode], len(error_source_lines))}.'
            continue
        if key == ord('?'):
            show_help_overlay(stdscr, colors)
            message = 'Help closed.'
            continue
        if key == ord('s') and view_mode == 'benchmark' and right_active_tab == 'results':
            idx = LEADERBOARD_SORT_KEYS.index(leaderboard_sort) if leaderboard_sort in LEADERBOARD_SORT_KEYS else 0
            leaderboard_sort = LEADERBOARD_SORT_KEYS[(idx + 1) % len(LEADERBOARD_SORT_KEYS)]
            message = f'Leaderboard sort: {leaderboard_sort}.'
            continue
        if view_mode == 'detail' and right_tabs and key in (ord('C'), ord('c'), ord('L')):
            target_tab = 'logs' if key == ord('L') else 'command'
            if target_tab in right_tabs:
                right_tab_by_view[view_mode] = target_tab
                message = f'Right tab: {right_tab_label(target_tab, len(error_source_lines))}.'
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
            action = prompt_command_palette(stdscr, colors, app, active_detail_model() or selected_model())
            palette_keys = {
                'search': ord('/'),
                'filters': ord('f'),
                'sort': ord('T'),
                'favorite': ord('*'),
                'help': ord('?'),
                'settings': ord('o'),
                'detect': ord('x'),
                'compare': ord('V'),
                'export_opencode': ord('g'),
                'export_continue': ord('c'),
                'export_hermes': ord('G'),
                'verify_selected': ord('Y'),
                'verify_all': ord('y'),
                'start_stop': ord('l'),
                'launch': ord('T'),
                'benchmark_menu': ord('B'),
                'apply_profile': ord('z'),
                'open_logs': ord('L'),
            }
            if str(action or '').startswith('disabled:'):
                message = str(action).split(':', 2)[-1] or 'Action unavailable.'
                continue
            if action == 'benchmark_plan':
                model = active_detail_model() or selected_model()
                if model is None:
                    message = 'No model selected for benchmark plan preview.'
                    continue
                show_benchmark_plan_overlay(stdscr, colors, app, model)
                message = 'Benchmark plan preview closed.'
                continue
            if action in palette_keys:
                key = palette_keys[action]
            elif action == 'density':
                current_density = normalize_choice(getattr(app.ui, 'detail_density', 'simple'), tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'simple')
                app.ui.detail_density = 'advanced' if current_density == 'simple' else 'simple'
                app.save()
                model = active_detail_model() or selected_model()
                if view_mode != 'detail' and model is not None:
                    open_model_details(model)
                message = (
                    f'Detail density: {detail_density_label(app.ui.detail_density)}. '
                    'Advanced shows extra benchmark detail in the Benchmarks tab.'
                )
                continue
            elif action == 'clear_browser':
                key = ord('C')
            elif action == 'toggle_browser_view':
                current = normalize_choice(getattr(app.ui, 'browser_view', 'compact'), tuple(key for key, _label in BROWSER_VIEW_OPTIONS), 'compact')
                app.ui.browser_view = 'advanced' if current == 'compact' else 'compact'
                app.save()
                if view_mode != 'list':
                    model = active_detail_model()
                    if view_mode == 'try':
                        exit_try_view()
                    view_mode = 'list'
                    detail_model_id = ''
                    if model is not None:
                        select_model_in_browser(model.id)
                message = f'Browser view: {browser_view_label(app.ui.browser_view)}. Returned to Models so the change is visible.'
                continue
            elif action == 'config_doctor':
                show_config_doctor_overlay(stdscr, colors, app, active_detail_model() or selected_model())
                message = 'Config Doctor closed.'
                continue
            elif action == 'mtp_doctor':
                model = active_detail_model() or selected_model()
                if model is None:
                    message = 'No model selected for MTP Doctor.'
                    continue
                show_mtp_doctor_overlay(stdscr, colors, app, model)
                message = 'MTP Doctor closed.'
                continue
            elif action == 'raw_speed_benchmark':
                model = active_detail_model() or selected_model()
                if model is None:
                    message = 'No model selected for raw speed benchmark.'
                    continue
                start_background_action(
                    model,
                    'raw speed benchmark',
                    lambda progress, token, model=model: benchmark_raw_speed_profile(
                        app,
                        model,
                        progress=progress,
                        cancel_token=token,
                    ),
                    done_event='benchmark_done',
                    run_kind='raw_speed',
                )
                continue
            elif action in ('moe_tuning_fast', 'moe_tuning_full'):
                model = active_detail_model() or selected_model()
                if model is None:
                    message = 'No model selected for MoE placement tuning.'
                    continue
                depth = 'full' if action == 'moe_tuning_full' else 'fast'
                start_background_action(
                    model,
                    f'MoE placement tuning ({depth})',
                    lambda progress, token, model=model, depth=depth: benchmark_moe_placement_tuning(
                        app,
                        model,
                        progress=progress,
                        cancel_token=token,
                        depth=depth,
                    ),
                    done_event='benchmark_done',
                    run_kind='moe_tuning',
                )
                continue
            elif action == 'apply_moe_recommendation':
                model = active_detail_model() or selected_model()
                message = apply_moe_recommendation_for_model(model)
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
            action.cancel('user requested abort')
            message = '⏳ Aborting active action and cleaning up managed processes...'
            continue
        if key == ord('A') and view_mode in ('detail', 'benchmark', 'results') and app.models:
            model = active_detail_model() or selected_model()
            if view_mode in ('benchmark', 'results'):
                run = active_full_suite_run_for_model(model)
                if run:
                    message = apply_full_suite_all_for_model(model)
                    continue
            if model and has_moe_recommendation(model) and not moe_recommendation_applied(model):
                message = apply_moe_recommendation_for_model(model)
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
            filters = prompt_browser_filters(
                stdscr,
                colors,
                filter_runtime,
                filter_source,
                filter_status,
                filter_tag,
                filter_compatibility,
            )
            if filters is not None:
                filter_runtime, filter_source, filter_status, filter_tag, filter_compatibility = filters
                selected = 0
                message = (
                    f'Filters set: {filter_option_label(FILTER_RUNTIME_OPTIONS, filter_runtime)}, '
                    f'{filter_option_label(FILTER_SOURCE_OPTIONS, filter_source)}, '
                    f'{filter_option_label(FILTER_STATUS_OPTIONS, filter_status)}, tag={filter_tag}, '
                    f'compat={filter_option_label(FILTER_COMPATIBILITY_OPTIONS, filter_compatibility)}.'
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
            filter_compatibility = 'active'
            selected = 0
            message = 'Browser search cleared; active-engine compatibility filter kept on.'
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
            count, items = app.detect_models(sync_exports=True)
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
        elif key == ord('M') and active_model and view_mode in ('benchmark', 'results'):
            model = active_detail_model()
            run = active_full_suite_run_for_model(model)
            if run:
                message = apply_moe_recommendation_for_model(model)
            else:
                open_machine_results()
        elif key == ord('P') and active_model and view_mode in ('benchmark', 'results'):
            model = active_detail_model()
            run = active_full_suite_run_for_model(model)
            if run:
                message = apply_full_suite_profile_for_model(model)
        elif key == ord('E') and active_model and view_mode in ('benchmark', 'results'):
            model = active_detail_model()
            run = active_full_suite_run_for_model(model)
            if run:
                message = sync_suite_exports_for_model(model)
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
            app.add_or_update(model, sync_exports=False)
            sync_msg = sync_opencode_after_tuning(app)
            invalidate_machine_summary()
            message = f'{tune_msg} | {sync_msg}'
        elif key == ord('B') and app.models:
            model = active_detail_model() or selected_model()
            if not model:
                continue
            choice = prompt_benchmark_menu(stdscr, colors, app, model)
            start_benchmark_choice(model, choice)
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
        elif key == ord('N') and app.models and view_mode in ('detail', 'benchmark', 'results'):
            model = active_detail_model() or selected_model()
            if not model:
                continue
            start_background_action(
                model,
                'MoE placement tuning (fast)',
                lambda progress, token, model=model: benchmark_moe_placement_tuning(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                    depth='fast',
                ),
                done_event='benchmark_done',
                run_kind='moe_tuning',
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
                app.add_or_update(model, sync_exports=True)
                select_model_in_browser(model.id)
                invalidate_machine_summary()
                message = f'Added {model.id} with safe defaults. Open details to start now; press B for measured settings.'
        elif key == ord('e') and app.models:
            current = active_detail_model() or selected_model()
            updated = prompt_model(stdscr, colors, f'Edit {current.id}', current) if current else None
            if updated:
                removed_models = []
                if updated.id != current.id:
                    removed_models.append(current)
                    app.delete(current.id, sync_exports=False)
                if not getattr(updated, 'default_benchmark_status', ''):
                    updated.default_benchmark_status = 'pending'
                app.add_or_update(updated, sync_exports=False)
                sync_msg = app.sync_generated_configs('model edit', removed_models=removed_models)
                select_model_in_browser(updated.id)
                if view_mode == 'detail':
                    detail_model_id = updated.id
                invalidate_machine_summary()
                message = f'Updated {updated.id}. {sync_msg}'
        elif key == ord('d') and app.models:
            delete_model = active_detail_model() or selected_model()
            if delete_model and prompt_yes_no(stdscr, colors, 'Delete Model', f'remove {delete_model.id} from llama-tui config'):
                target_id = delete_model.id
                ok, msg = app.delete(target_id, sync_exports=True)
                clamp_selected()
                if view_mode == 'detail':
                    view_mode = 'list'
                    detail_model_id = ''
                invalidate_machine_summary()
                message = f'{target_id}: {msg}'
            else:
                message = 'Delete cancelled.'
        elif key == ord('x'):
            count, items = app.detect_models(sync_exports=True)
            message = items[0] if items else (f'Detected {count} new model(s)' if count else 'No new GGUFs found.')
            if count:
                message = f'{message} | safe defaults set; start now or press B for measured settings.'
            clamp_selected()
            invalidate_machine_summary()
        elif key == ord('X'):
            count, removed = app.prune_missing_models(sync_exports=True)
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
                message = f'Settings saved. {app.sync_generated_configs("settings update")}'
            else:
                message = 'Settings unchanged.'
        elif key == ord('m') and app.models:
            model = active_detail_model() or selected_model()
            app.set_role('main', model.id, sync_exports=True)
            message = f'{model.id} set as main model.'
        elif key == ord('s') and app.models:
            model = active_detail_model() or selected_model()
            app.set_role('small', model.id, sync_exports=True)
            message = f'{model.id} set as small model.'
        elif key == ord('b') and app.models:
            model = active_detail_model() or selected_model()
            app.set_role('build', model.id, sync_exports=True)
            message = f'{model.id} set as build model.'
        elif key == ord('p') and app.models:
            model = active_detail_model() or selected_model()
            app.set_role('plan', model.id, sync_exports=True)
            message = f'{model.id} set as plan model.'

    # Worker thread shutdown — see audit finding #5. Cancel any in-flight
    # action/try tokens and join the daemon threads so cleanup_managed_processes
    # in main.py doesn't race with subprocess work that's still posting to the
    # action queue.
    shutdown_workers(action.token, action.thread, try_.token, try_.thread)
