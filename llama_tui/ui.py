import curses
import textwrap
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional, Tuple

from .app import AppConfig
from .benchmark import (
    append_model_log,
    apply_measured_profile,
    benchmark_best_optimization,
    estimate_text_tokens,
    launch_opencode_stack,
    launch_with_failsafe,
    start_model_with_progress,
    sync_opencode_after_tuning,
)
from .chat import stream_chat_completion
from .constants import LOGO, REFRESH_SECONDS
from .control import CancelToken, CancelledError
from .discovery import classify_model_type, display_runtime, extract_quant
from .hardware import HardwareProfile
from .models import ModelConfig
from .opencode_benchmark import benchmark_opencode_workflow
from .optimize import apply_best_optimization, select_best_tier
from .textutil import compact_message, ellipsize, important_log_excerpt, is_error_message, tail_text, wrap_display_lines

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
BENCHMARK_FEED_LIMIT = 80
BENCHMARK_RECORD_LIMIT = 120


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


def model_profile_summary(model: ModelConfig) -> str:
    mode = profile_label(getattr(model, 'optimize_mode', 'max_context_safe'))
    tier = tier_label(getattr(model, 'optimize_tier', 'moderate'))
    reserve = int(getattr(model, 'memory_reserve_percent', 25) or 25)
    measured = getattr(model, 'measured_profiles', {}) or {}
    suffix = f' / measured {len(measured)}' if measured else ''
    return f'{mode} / {tier} / reserve {reserve}%{suffix}'


def stop_try_model(app: AppConfig, model: ModelConfig) -> Tuple[bool, str]:
    return app.stop(model)


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
        'updated_at': timestamp,
        'records': [],
        'feed': [],
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


def benchmark_elapsed_text(state: Dict[str, object], now: Optional[float] = None) -> str:
    timestamp = time.monotonic() if now is None else now
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
    suffix = (' ' + ' '.join(suffix_parts)) if suffix_parts else ''
    return (
        f'{label[:18]:18} {score:7.2f} {score_label:5} {seconds:6.1f}s '
        f'ctx={ctx:<6} slot={slot:<6} par={parallel:<2} {status}{suffix}'
    )


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


def benchmark_ranking_rows(run: Dict[str, object]) -> List[str]:
    winners = run.get('winners') or {}
    records = list(run.get('records', []) or [])
    lines: List[str] = []
    for key, title in (
        ('fast_chat', 'Fast Chat'),
        ('long_context', 'Long Context'),
        ('opencode_ready', 'OpenCode-ready'),
        ('auto', 'Auto'),
    ):
        winner = winners.get(key) if isinstance(winners, dict) else {}
        if isinstance(winner, dict) and winner:
            lines.append(
                f'{title}: ctx={int(winner.get("ctx", 0) or 0)} '
                f'slot={int(winner.get("ctx_per_slot", 0) or 0)} '
                f'par={int(winner.get("parallel", 0) or 0)} '
                f'{float(winner.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s'
            )
        else:
            lines.append(f'{title}: not measured')
    failed = [row for row in records if row.get('status') != 'ok']
    if failed:
        lines.append('')
        lines.append('Failed / break points:')
        for row in failed[:8]:
            status = str(row.get('status', '-') or '-')
            marker = 'break' if row.get('break_point') else 'fail'
            detail = compact_message(str(row.get('detail', '') or ''))
            lines.append(
                f'{marker}: {row.get("objective", "-")} {row.get("variant", "-")} '
                f'ctx={row.get("ctx", 0)} par={row.get("parallel", 0)} {status} {detail}'
            )
    return lines


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
    for key in ('model_id', 'run_kind', 'phase', 'candidate', 'message'):
        value = payload.get(key)
        if value not in (None, ''):
            state[key] = value
    if 'completed' in payload:
        state['completed'] = max(0, int(payload.get('completed') or 0))
    if 'total' in payload:
        state['total'] = max(0, int(payload.get('total') or 0))

    message = compact_message(str(payload.get('message', '') or ''))
    if message:
        feed = list(state.get('feed', []) or [])
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
        if int(state.get('total', 0) or 0) <= 0:
            state['total'] = int(state.get('completed', 0) or 0)
    elif event == 'benchmark_error':
        state['active'] = False
        state['status'] = 'failed'
    elif event == 'benchmark_aborted':
        state['active'] = False
        state['status'] = 'aborted'
    elif event and event != 'benchmark_started':
        state['status'] = 'running' if state.get('active') else str(state.get('status') or 'idle')
    return state


def prompt_value(stdscr, title: str, fields: List[Tuple[str, str]]) -> Optional[Dict[str, str]]:
    curses.endwin()
    print(f'\n{title}')
    print('-' * len(title))
    print('Leave blank to keep current value. Ctrl+C to cancel.\n')
    answers = {}
    try:
        for label, default in fields:
            suffix = f' [{default}]' if default else ''
            value = input(f'{label}{suffix}: ').strip()
            answers[label] = value if value else default
    except KeyboardInterrupt:
        answers = None
    print()
    input('Press Enter to return to the TUI...')
    stdscr.clear()
    stdscr.refresh()
    return answers
def prompt_model(stdscr, title: str, initial: Optional[ModelConfig] = None) -> Optional[ModelConfig]:
    initial = initial or ModelConfig(id='', name='', path='', alias='', port=8080)
    answers = prompt_value(stdscr, title, [
        ('id', initial.id),
        ('name', initial.name),
        ('runtime (llama.cpp/vllm)', getattr(initial, 'runtime', 'llama.cpp')),
        ('optimize_mode (max_context_safe/manual)', getattr(initial, 'optimize_mode', 'max_context_safe')),
        ('optimize_tier (safe/moderate/extreme)', getattr(initial, 'optimize_tier', 'moderate')),
        ('path', initial.path),
        ('alias', initial.alias or initial.id),
        ('port', str(initial.port)),
        ('host', initial.host),
        ('ctx', str(initial.ctx)),
        ('ctx_min', str(getattr(initial, 'ctx_min', 2048))),
        ('ctx_max', str(getattr(initial, 'ctx_max', 131072))),
        ('threads', str(initial.threads)),
        ('ngl', str(initial.ngl)),
        ('temp', str(initial.temp)),
        ('parallel', str(initial.parallel)),
        ('memory_reserve_percent', str(getattr(initial, 'memory_reserve_percent', 25))),
        ('cache_ram', str(initial.cache_ram)),
        ('output', str(initial.output)),
        ('enabled true/false', str(initial.enabled).lower()),
        ('flash_attn true/false', str(initial.flash_attn).lower()),
        ('jinja true/false', str(initial.jinja).lower()),
        ('extra_args (space-separated)', ' '.join(initial.extra_args)),
    ])
    if not answers:
        return None
    try:
        return ModelConfig(
            id=answers['id'],
            name=answers['name'],
            path=answers['path'],
            alias=answers['alias'],
            port=int(answers['port']),
            host=answers['host'],
            ctx=int(answers['ctx']),
            ctx_min=int(answers['ctx_min']),
            ctx_max=int(answers['ctx_max']),
            threads=int(answers['threads']),
            ngl=int(answers['ngl']),
            temp=float(answers['temp']),
            parallel=int(answers['parallel']),
            optimize_mode=(answers['optimize_mode (max_context_safe/manual)'].strip() or 'max_context_safe'),
            optimize_tier=(answers['optimize_tier (safe/moderate/extreme)'].strip() or 'moderate'),
            memory_reserve_percent=int(answers['memory_reserve_percent']),
            cache_ram=int(answers['cache_ram']),
            output=int(answers['output']),
            enabled=answers['enabled true/false'].lower() == 'true',
            runtime=(answers['runtime (llama.cpp/vllm)'].strip().lower() or 'llama.cpp'),
            flash_attn=answers['flash_attn true/false'].lower() == 'true',
            jinja=answers['jinja true/false'].lower() == 'true',
            source=getattr(initial, 'source', 'manual'),
            extra_args=answers['extra_args (space-separated)'].split() if answers['extra_args (space-separated)'] else [],
        )
    except Exception:
        return None
def prompt_settings(stdscr, app: AppConfig) -> bool:
    o = app.opencode
    answers = prompt_value(stdscr, 'Settings', [
        ('llama_server', app.llama_server),
        ('vllm_command', app.vllm_command),
        ('hf_cache_root', app.hf_cache_root),
        ('llm_models_cache_root', app.llm_models_cache_root),
        ('llmfit_cache_root', app.llmfit_cache_root),
        ('opencode_path', o.path),
        ('opencode_backup_dir', o.backup_dir),
        ('default_model_id', o.default_model_id),
        ('small_model_id', o.small_model_id),
        ('build_model_id', o.build_model_id),
        ('plan_model_id', o.plan_model_id),
        ('instructions (comma-separated)', ', '.join(o.instructions)),
        ('build_prompt', o.build_prompt),
        ('plan_prompt', o.plan_prompt),
        ('timeout', str(o.timeout)),
        ('chunk_timeout', str(o.chunk_timeout)),
        ('terminal_command', getattr(o, 'terminal_command', '')),
        ('last_workspace_path', getattr(o, 'last_workspace_path', '')),
    ])
    if not answers:
        return False
    try:
        app.llama_server = answers['llama_server']
        app.vllm_command = answers['vllm_command']
        app.hf_cache_root = answers['hf_cache_root']
        app.llm_models_cache_root = answers['llm_models_cache_root']
        app.llmfit_cache_root = answers['llmfit_cache_root']
        o.path = answers['opencode_path']
        o.backup_dir = answers['opencode_backup_dir']
        o.default_model_id = answers['default_model_id']
        o.small_model_id = answers['small_model_id']
        o.build_model_id = answers['build_model_id']
        o.plan_model_id = answers['plan_model_id']
        o.instructions = [s.strip() for s in answers['instructions (comma-separated)'].split(',') if s.strip()]
        o.build_prompt = answers['build_prompt']
        o.plan_prompt = answers['plan_prompt']
        o.timeout = int(answers['timeout'])
        o.chunk_timeout = int(answers['chunk_timeout'])
        o.terminal_command = answers['terminal_command']
        o.last_workspace_path = answers['last_workspace_path']
        app.save()
        return True
    except Exception:
        return False
def prompt_workspace(stdscr, app: AppConfig) -> Optional[str]:
    curses.endwin()
    default = getattr(app.opencode, 'last_workspace_path', '') or str(Path.cwd())
    try:
        value = input(f'\nWorkspace path [{default}]: ').strip()
    except KeyboardInterrupt:
        value = ''
    stdscr.clear()
    stdscr.refresh()
    return value or default
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
def prompt_launch_optimization(stdscr, model: ModelConfig, colors) -> str:
    return prompt_modal_choice(stdscr, colors, f'Launch {model.id}', [
        ('1', 'Auto profile', 'auto_profile'),
        ('2', 'Balanced chat', 'balanced_chat'),
        ('3', 'Fast chat', 'fast_chat'),
        ('4', 'Long context', 'long_context'),
        ('5', 'Keep current settings', 'keep'),
        ('6', 'Advanced profiles', 'advanced'),
        ('7', 'Try it out', 'try'),
        ('8', 'Launch model + OpenCode', 'opencode'),
        ('9', 'Launch full-stack: OpenCode + VS Code', 'full_stack'),
        ('q', 'Cancel', 'cancel'),
    ])
def prompt_running_model_action(stdscr, model: ModelConfig, colors) -> str:
    return prompt_modal_choice(stdscr, colors, f'{model.id} is running', [
        ('1', 'Stop model', 'stop'),
        ('2', 'Try it out', 'try'),
        ('3', 'Launch OpenCode', 'opencode'),
        ('4', 'Launch full-stack: OpenCode + VS Code', 'full_stack'),
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
def draw_box(stdscr, y: int, x: int, h: int, w: int, title: str, title_attr: int = curses.A_BOLD, border_attr: int = 0):
    if h < 2 or w < 4:
        return
    stdscr.addstr(y, x + 2, f' {title} ', title_attr)
    for i in range(x, x + w):
        stdscr.addch(y + 1, i, curses.ACS_HLINE, border_attr)
    for i in range(y + 1, y + h):
        stdscr.addch(i, x, curses.ACS_VLINE, border_attr)
        stdscr.addch(i, x + w - 1, curses.ACS_VLINE, border_attr)
    stdscr.addch(y + 1, x, curses.ACS_ULCORNER, border_attr)
    stdscr.addch(y + 1, x + w - 1, curses.ACS_URCORNER, border_attr)
    stdscr.addch(y + h, x, curses.ACS_LLCORNER, border_attr)
    stdscr.addch(y + h, x + w - 1, curses.ACS_LRCORNER, border_attr)
    for i in range(x + 1, x + w - 1):
        stdscr.addch(y + h, i, curses.ACS_HLINE, border_attr)
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
    }
    return mapping.get(label, colors['accent'] | curses.A_BOLD)
def tui(stdscr, app: AppConfig):
    colors = init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    selected = 0
    view_mode = 'list'
    detail_model_id = ''
    message = 'Ready.'
    last_error_message = ''
    last_refresh = 0.0
    external_stack_launched = False
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
    try_live_metrics = new_try_live_metrics()
    try_input_scroll = 0
    benchmark_state = new_benchmark_run_state()
    results_run_index = 0

    def action_running() -> bool:
        return action_thread is not None and action_thread.is_alive()

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
        if not app.models:
            return None
        idx = max(0, min(selected, len(app.models) - 1))
        return app.models[idx]

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
            return f'{model.id}: safe defaults are set. Press B when you want to benchmark.'
        if status in ('failed', 'aborted'):
            return f'{model.id}: last default benchmark {status}. Press B to run it again.'
        return f'{model.id}: no benchmark yet. Press B from details to benchmark.'

    def show_benchmark_hint(model: ModelConfig):
        nonlocal message
        if action_running():
            return
        if model_is_running(model):
            message = f'{model.id}: no benchmark yet. Stop it and press B from details to benchmark.'
            return
        message = benchmark_hint(model)

    def open_model_details(model: ModelConfig):
        nonlocal view_mode, detail_model_id, message
        view_mode = 'detail'
        detail_model_id = model.id
        message = f'{model.id}: details loaded. Press Enter/l to start or Esc to return.'
        show_benchmark_hint(model)

    def open_try_view(model: ModelConfig):
        nonlocal view_mode, detail_model_id, message, try_thread, try_token, try_session, try_input_scroll
        nonlocal try_messages, try_input, try_status, try_error, try_response_index
        if action_running():
            message = '⏳ Wait for the current launch or benchmark before opening Try it out.'
            return
        view_mode = 'try'
        detail_model_id = model.id
        try_session += 1
        session = try_session
        try_messages = []
        try_input = ''
        try_input_scroll = 0
        try_error = ''
        try_response_index = None
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

        def progress(text: str):
            line = compact_message(text)
            append_model_log(app, model, line)
            action_queue.put(('try_progress', line, session))

        def runner():
            try:
                if status in ('LOADING', 'STARTING') or app.get_pid(model):
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
        view_mode = 'results'
        detail_model_id = model.id
        results_run_index = 0
        run_count = len(benchmark_runs_for_model(model))
        message = f'{model.id}: {run_count} benchmark result run(s).'

    def start_try_chat_send():
        nonlocal message, try_thread, try_token, try_input, try_input_scroll, try_status, try_error, try_response_index, try_messages
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
        request_messages = list(try_messages)
        try_messages.append({'role': 'assistant', 'content': ''})
        try_response_index = len(try_messages) - 1
        try_input = ''
        try_input_scroll = 0
        try_error = ''
        try_status = 'responding'
        message = f'{model.id}: streaming response...'

        def runner():
            chunks = 0
            try:
                for chunk in stream_chat_completion(model, request_messages, cancel_token=token):
                    chunks += 1
                    action_queue.put(('chat_chunk', chunk, session))
                action_queue.put(('chat_done', str(chunks), session))
            except CancelledError:
                action_queue.put(('chat_error', 'chat stream cancelled', session))
            except Exception as exc:
                action_queue.put(('chat_error', compact_message(str(exc)), session))

        try_thread = threading.Thread(target=runner, daemon=True)
        try_thread.start()

    def exit_try_view():
        nonlocal view_mode, message, last_refresh, try_thread, try_token, try_session, try_input, try_input_scroll
        nonlocal try_status, try_error, try_response_index
        model = active_detail_model()
        if try_token is not None:
            try_token.cancel('leaving try-out')
        try_session += 1
        stop_msg = 'no model selected'
        if model:
            _ok, stop_msg = stop_try_model(app, model)
            append_model_log(app, model, f'try-it-out exit: {stop_msg}')
            message = f'{model.id}: try-out closed; {stop_msg}'
        else:
            message = 'Try-out closed.'
        view_mode = 'detail'
        try_thread = None
        try_token = None
        try_input = ''
        try_input_scroll = 0
        try_status = 'idle'
        try_error = ''
        try_response_index = None
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
        if launch_mode in ('opencode', 'full_stack'):
            workspace = prompt_workspace(stdscr, app)
            if not workspace:
                message = 'OpenCode launch cancelled.'
                return
            label = 'full-stack launch' if launch_mode == 'full_stack' else 'OpenCode launch'
            include_vscode = launch_mode == 'full_stack'
            start_background_action(
                model,
                label,
                lambda progress, token, model=model, workspace=workspace, include_vscode=include_vscode: launch_opencode_stack(
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
            if event in ('try_progress', 'try_ready', 'try_error', 'chat_chunk', 'chat_done', 'chat_error'):
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
                    last_error_message = message
                    continue
                if event == 'chat_chunk':
                    update_try_live_metrics(try_live_metrics, text)
                    if try_response_index is not None and 0 <= try_response_index < len(try_messages):
                        try_messages[try_response_index]['content'] += text
                    message = 'streaming response...'
                    continue
                if event == 'chat_done':
                    finish_try_live_metrics(try_live_metrics)
                    if try_response_index is not None and 0 <= try_response_index < len(try_messages):
                        if not try_messages[try_response_index]['content'].strip():
                            try_messages[try_response_index]['content'] = '(no content returned)'
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
                        if try_messages[try_response_index]['content'].strip():
                            try_messages[try_response_index]['content'] += f'\n[error] {try_error}'
                        else:
                            try_messages[try_response_index]['content'] = f'[error] {try_error}'
                    try_response_index = None
                    try_thread = None
                    message = f'❌ {try_error}'
                    last_error_message = message
                    continue
            if event == 'benchmark_event':
                payload = text if isinstance(text, dict) else {}
                reduce_benchmark_event(benchmark_state, payload)
                event_message = compact_message(str(payload.get('message', '') or ''))
                if event_message:
                    message = event_message
                    if is_error_message(event_message):
                        last_error_message = event_message
                continue
            if event == 'stack_done' and text.startswith('✅'):
                external_stack_launched = True
            if is_error_message(text):
                last_error_message = text
            message = text
            if event in ('done', 'stack_done', 'benchmark_done'):
                action_thread = None
                action_token = None
                last_refresh = 0.0

        now = time.time()
        if now - last_refresh > REFRESH_SECONDS:
            statuses = {m.id: app.health(m) for m in app.models}
            last_refresh = now

        if app.models:
            selected = max(0, min(selected, len(app.models) - 1))
            if view_mode in ('detail', 'try', 'benchmark', 'results'):
                current_detail = app.get_model(detail_model_id)
                if not current_detail:
                    detail_model_id = app.models[selected].id
        else:
            selected = 0
            view_mode = 'list'
            detail_model_id = ''

        active_model = active_detail_model()
        if is_error_message(message):
            last_error_message = compact_message(message)

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 18 or w < 88:
            stdscr.addstr(1, 2, 'Window too small for llama-tui. Stretch it a bit.', colors['warning'] | curses.A_BOLD)
            stdscr.addstr(3, 2, f'Current size: {w}x{h}')
            stdscr.addstr(5, 2, '[q] quit', curses.A_BOLD)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord('q'), 27):
                break
            time.sleep(0.05)
            continue

        y = 0
        if w >= 100:
            for line in LOGO:
                stdscr.addstr(y, 2, line[:w-4], colors['banner'] | curses.A_BOLD)
                y += 1
            stdscr.addstr(1, min(w - 28, 60), 'local model control plane', colors['accent'] | curses.A_BOLD)
            header_y = y + 1
        else:
            stdscr.addstr(0, 2, 'llama-tui', colors['banner'] | curses.A_BOLD)
            stdscr.addstr(0, 14, 'local model control plane', colors['accent'] | curses.A_BOLD)
            header_y = 2

        stdscr.addstr(header_y, 2, f'config: {app.config_path}', colors['muted'])
        stdscr.addstr(header_y + 1, 2, f'llama-server: {app.llama_server}', colors['muted'])
        stdscr.addstr(header_y + 2, 2, f'vllm-command: {app.vllm_command}', colors['muted'])
        stdscr.addstr(header_y + 3, 2, f'hf-cache: {app.hf_cache_root}', colors['muted'])
        stdscr.addstr(header_y + 4, 2, f'llm-models-cache: {app.llm_models_cache_root}', colors['muted'])
        stdscr.addstr(header_y + 5, 2, f'llmfit-cache: {app.llmfit_cache_root}', colors['muted'])
        stdscr.addstr(header_y + 6, 2, f'opencode: {app.opencode.path or "<unset>"}', colors['muted'])

        counts = {'READY': 0, 'LOADING': 0, 'STARTING': 0, 'STOPPED': 0, 'ERROR': 0}
        for _mid, (st, _detail) in statuses.items():
            if st in counts:
                counts[st] += 1

        msg_attr = colors['accent'] | curses.A_BOLD
        message_is_error = is_error_message(message)
        if message_is_error:
            msg_attr = colors['warning'] | curses.A_BOLD
        elif message.startswith('✅'):
            msg_attr = colors['success'] | curses.A_BOLD
        elif message.startswith('⏳'):
            msg_attr = colors['warning'] | curses.A_BOLD
        header_message = (
            compact_message(message)
            if message_is_error and view_mode == 'try'
            else 'Error captured in the lower-right Errors box.' if message_is_error else compact_message(message)
        )
        msg_line = ellipsize(header_message, max(10, w - 4))
        stdscr.addstr(header_y + 7, 2, msg_line, msg_attr)

        chip_y = header_y + 7
        chip_x = min(max(40, len(msg_line) + 6), max(40, w - 34))
        chips = [
            ('READY', counts['READY']),
            ('LOADING', counts['LOADING'] + counts['STARTING']),
            ('STOPPED', counts['STOPPED']),
        ]
        for label, value in chips:
            text = f' {label}:{value} '
            if chip_x + len(text) < w - 2:
                stdscr.addstr(chip_y, chip_x, text, chip_attr(colors, label))
                chip_x += len(text) + 1

        box_top = header_y + 8
        left_w = max(76, min(112, (w // 2) + 8))
        right_x = left_w + 2
        right_w = max(38, w - right_x - 2)
        try_input_rows = TRY_INPUT_ROWS
        visible_rows = max(8, h - box_top - 6)
        right_total_h = max(4, h - box_top - 4)
        active_status = statuses.get(active_model.id, ('?', '')) if active_model else ('?', '')
        status_error = f'{active_model.id}: status ERROR ({active_status[1]})' if active_model and active_status[0] == 'ERROR' else ''
        try_mode = view_mode == 'try'
        benchmark_mode = view_mode == 'benchmark'
        results_mode = view_mode == 'results'
        benchmark_errors = list(benchmark_state.get('errors', []) or [])
        error_text = '\n'.join(str(item) for item in benchmark_errors[-6:]) if benchmark_mode and benchmark_errors else (last_error_message or status_error)
        error_lines = wrap_display_lines(error_text, right_w - 4) if error_text else ['No errors captured.']
        if try_mode:
            error_box_h = 0
        elif benchmark_mode:
            error_box_h = max(4, min(8, right_total_h // 4)) if right_total_h >= 14 else 0
        elif results_mode:
            error_box_h = 0
        elif right_total_h >= 14:
            desired_error_h = min(max(5, len(error_lines) + 3), 14)
            error_box_h = min(desired_error_h, max(5, right_total_h - 8))
        else:
            error_box_h = 0
        if try_mode:
            if right_total_h >= 14:
                stats_box_h = max(4, min(7, right_total_h // 4))
                profile_box_h = max(5, min(11, right_total_h // 3))
                logs_box_h = right_total_h - profile_box_h - stats_box_h - 2
                if logs_box_h < 4:
                    deficit = 4 - logs_box_h
                    profile_cut = min(deficit, max(0, profile_box_h - 4))
                    profile_box_h -= profile_cut
                    deficit -= profile_cut
                    stats_cut = min(deficit, max(0, stats_box_h - 3))
                    stats_box_h -= stats_cut
                    logs_box_h = right_total_h - profile_box_h - stats_box_h - 2
                logs_box_h = max(2, logs_box_h)
            else:
                stats_box_h = 0
                profile_box_h = max(2, min(6, right_total_h // 2))
                logs_box_h = max(2, right_total_h - profile_box_h - 1)
            logs_box_y = box_top + profile_box_h + 1
            stats_box_y = logs_box_y + logs_box_h + 1
            error_box_y = 0
        elif benchmark_mode:
            profile_box_h = max(5, min(9, right_total_h // 4 if right_total_h >= 20 else 6))
            stats_box_h = 0
            stats_box_y = 0
            logs_box_h = right_total_h - profile_box_h - (error_box_h + 2 if error_box_h else 1)
            if logs_box_h < 4:
                deficit = 4 - logs_box_h
                profile_box_h = max(4, profile_box_h - deficit)
                logs_box_h = right_total_h - profile_box_h - (error_box_h + 2 if error_box_h else 1)
            logs_box_h = max(2, logs_box_h)
            logs_box_y = box_top + profile_box_h + 1
            error_box_y = logs_box_y + logs_box_h + 1 if error_box_h else 0
        else:
            profile_box_h = 0
            stats_box_h = 0
            stats_box_y = 0
            logs_box_h = right_total_h if error_box_h == 0 else max(5, right_total_h - error_box_h - 1)
            logs_box_y = box_top
            error_box_y = box_top + logs_box_h + 1

        left_title = 'Try It Out' if try_mode else 'Benchmark' if benchmark_mode else 'Results' if results_mode else 'Model Details' if view_mode == 'detail' else 'Models'
        draw_box(stdscr, box_top, 1, h - box_top - 4, left_w, left_title, colors['accent'] | curses.A_BOLD, colors['accent'])
        if try_mode:
            draw_box(stdscr, box_top, right_x, profile_box_h, right_w, 'Active Test Profile', colors['accent'] | curses.A_BOLD, colors['accent'])
            draw_box(stdscr, logs_box_y, right_x, logs_box_h, right_w, 'Server Logs', colors['accent'] | curses.A_BOLD, colors['accent'])
            if stats_box_h:
                draw_box(stdscr, stats_box_y, right_x, stats_box_h, right_w, 'Live Stats', colors['accent'] | curses.A_BOLD, colors['accent'])
        elif benchmark_mode:
            draw_box(stdscr, box_top, right_x, profile_box_h, right_w, 'Benchmark Progress', colors['accent'] | curses.A_BOLD, colors['accent'])
            draw_box(stdscr, logs_box_y, right_x, logs_box_h, right_w, 'Benchmark Logs', colors['accent'] | curses.A_BOLD, colors['accent'])
        elif results_mode:
            draw_box(stdscr, box_top, right_x, logs_box_h, right_w, 'Run Rankings', colors['accent'] | curses.A_BOLD, colors['accent'])
        else:
            draw_box(stdscr, box_top, right_x, logs_box_h, right_w, 'Details / Logs / Roles', colors['accent'] | curses.A_BOLD, colors['accent'])
        if error_box_h:
            draw_box(stdscr, error_box_y, right_x, error_box_h, right_w, 'Errors', colors['error'] | curses.A_BOLD, colors['error'])

        if view_mode == 'results' and active_model:
            model = active_model
            runs = benchmark_runs_for_model(model)
            if runs:
                results_run_index = max(0, min(results_run_index, len(runs) - 1))
            content_h = max(1, h - box_top - 7)
            header_lines = [
                (f'model: {model.name or model.id}', curses.A_BOLD),
                (f'runs: {len(runs)} latest benchmark run(s)', colors['accent'] | curses.A_BOLD),
                ('[Up/Down] select run   [Esc] details', colors['muted']),
                ('', curses.A_NORMAL),
            ]
            y_cursor = box_top + 2
            for line, attr in header_lines[:content_h]:
                stdscr.addstr(y_cursor, 3, line[: left_w - 5], attr)
                y_cursor += 1
            if not runs:
                stdscr.addstr(y_cursor, 3, 'No benchmark history yet. Press B from details to run one.', colors['warning'])
            else:
                visible = runs[: max(1, content_h - len(header_lines))]
                for idx, run in enumerate(visible):
                    if y_cursor >= box_top + 2 + content_h:
                        break
                    line = benchmark_run_line(run, idx, selected=(idx == results_run_index))
                    attr = colors['selection'] | curses.A_BOLD if idx == results_run_index else curses.A_NORMAL
                    stdscr.addstr(y_cursor, 3, ellipsize(line, left_w - 5), attr)
                    y_cursor += 1
        elif view_mode == 'benchmark' and active_model:
            model = active_model
            run_kind = str(benchmark_state.get('run_kind') or 'server')
            status_text = str(benchmark_state.get('status') or 'idle')
            phase = str(benchmark_state.get('phase') or '-')
            candidate = str(benchmark_state.get('candidate') or '-')
            completed = int(benchmark_state.get('completed', 0) or 0)
            total = int(benchmark_state.get('total', 0) or 0)
            pct = int(round(benchmark_progress_fraction(completed, total) * 100))
            bar = progress_bar_text(completed, total, max(10, min(34, left_w - 62)))
            records = list(benchmark_state.get('records', []) or [])
            if not records:
                if run_kind == 'opencode':
                    records = list(getattr(model, 'last_opencode_benchmark_results', []) or [])
                else:
                    records = list(getattr(model, 'last_benchmark_results', []) or [])
            feed = list(benchmark_state.get('feed', []) or [])
            content_h = max(1, h - box_top - 7)
            summary_lines = [
                (f'model: {model.name or model.id}', curses.A_BOLD),
                (f'run: {run_kind}   status: {status_text}   elapsed: {benchmark_elapsed_text(benchmark_state)}', colors['accent'] | curses.A_BOLD),
                (f'phase: {phase}', curses.A_NORMAL),
                (f'candidate: {candidate}', curses.A_NORMAL),
                (f'progress: {bar} {completed}/{total if total else "?"} {pct if total else 0}%', colors['warning'] | curses.A_BOLD if benchmark_state.get('active') else colors['success'] | curses.A_BOLD),
                ('', curses.A_NORMAL),
                ('real-time results:', colors['accent'] | curses.A_BOLD),
                (f' {"LABEL":18} {"VALUE":>7} {"UNIT":5} {"TIME":>7} {"CTX/SLOT/PAR":24} STATUS', colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD),
            ]
            y_cursor = box_top + 2
            for line, attr in summary_lines[:content_h]:
                stdscr.addstr(y_cursor, 3, line[: left_w - 5], attr)
                y_cursor += 1
            rows_available = max(0, box_top + 2 + content_h - y_cursor)
            feed_target = min(7, max(0, rows_available // 3))
            result_target = max(0, rows_available - feed_target - (2 if feed_target else 0))
            visible_records = records[-result_target:] if result_target else []
            for record in visible_records:
                line = benchmark_row_text(record)
                status_value = str(record.get('status', '') or '')
                attr = (
                    colors['success'] | curses.A_BOLD
                    if status_value in ('ok', 'tests passed')
                    else colors['warning']
                    if status_value in ('time budget exhausted', 'context too small', 'tests failed', 'probe ok')
                    else colors['error']
                    if status_value not in ('', '-')
                    else curses.A_NORMAL
                )
                stdscr.addstr(y_cursor, 3, ellipsize(line, left_w - 5), attr)
                y_cursor += 1
            if not records and y_cursor < box_top + 2 + content_h:
                stdscr.addstr(y_cursor, 3, 'waiting for first measured result...', colors['muted'])
                y_cursor += 1
            if feed_target and y_cursor < box_top + 2 + content_h:
                y_cursor += 1
                if y_cursor < box_top + 2 + content_h:
                    stdscr.addstr(y_cursor, 3, 'live feed:', colors['accent'] | curses.A_BOLD)
                    y_cursor += 1
                for line in feed[-feed_target:]:
                    if y_cursor >= box_top + 2 + content_h:
                        break
                    attr = colors['error'] if is_error_message(str(line)) else colors['muted']
                    stdscr.addstr(y_cursor, 3, ellipsize(str(line), left_w - 5), attr)
                    y_cursor += 1
        elif view_mode == 'try' and active_model:
            model = active_model
            try_input_rows = min(TRY_INPUT_ROWS, max(1, h - box_top - 8))
            input_y = max(box_top + 5, h - try_input_rows - 5)
            transcript_h = max(1, input_y - box_top - 3)
            transcript_lines: List[Tuple[str, int]] = []
            if not try_messages:
                intro = (
                    'Type a prompt when the server is ready. Esc stops this model and returns to details.'
                    if try_status == 'ready'
                    else 'Starting the selected model. Input opens when /v1/models is ready.'
                )
                for line in wrap_display_lines(intro, left_w - 6):
                    transcript_lines.append((line, colors['muted']))
            for item in try_messages:
                role = item.get('role', '')
                content = item.get('content', '')
                if role == 'user':
                    prefix = 'you> '
                    attr = colors['accent'] | curses.A_BOLD
                else:
                    prefix = f'{model.alias or model.id}> '
                    attr = curses.A_NORMAL
                wrapped = wrap_display_lines(prefix + (content or '...'), left_w - 6)
                for line in wrapped:
                    transcript_lines.append((line, attr))
                transcript_lines.append(('', curses.A_NORMAL))
            visible_transcript = transcript_lines[-transcript_h:]
            for i, (line, attr) in enumerate(visible_transcript):
                stdscr.addstr(box_top + 2 + i, 3, line[: left_w - 5], attr)
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
            stdscr.addstr(input_y, 3, divider[: left_w - 5], colors['muted'])
            if try_status == 'ready':
                input_attr = colors['panel'] | curses.A_BOLD
                for row_idx, input_line in enumerate(input_lines[:try_input_rows]):
                    stdscr.addstr(input_y + 1 + row_idx, 3, input_line[: left_w - 5], input_attr)
            elif try_status == 'responding':
                input_line = 'streaming response... Esc cancels and stops the model'
                input_attr = colors['warning'] | curses.A_BOLD
                stdscr.addstr(input_y + 1, 3, input_line[: left_w - 5], input_attr)
            elif try_status == 'error':
                input_line = f'error: {try_error or "chat failed"}'
                input_attr = colors['error'] | curses.A_BOLD
                stdscr.addstr(input_y + 1, 3, input_line[: left_w - 5], input_attr)
            else:
                input_line = 'waiting for server readiness...'
                input_attr = colors['warning'] | curses.A_BOLD
                stdscr.addstr(input_y + 1, 3, input_line[: left_w - 5], input_attr)
        elif view_mode == 'detail' and active_model:
            model = active_model
            status, detail = statuses.get(model.id, ('?', ''))
            benchmark_score = float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0)
            benchmark_seconds = float(getattr(model, 'last_benchmark_seconds', 0.0) or 0.0)
            opencode_score = float(getattr(model, 'last_opencode_benchmark_score', 0.0) or 0.0)
            opencode_seconds = float(getattr(model, 'last_opencode_benchmark_seconds', 0.0) or 0.0)
            if benchmark_score > 0:
                benchmark_summary = f'{benchmark_score:.2f} tok/s in {benchmark_seconds:.2f}s'
            else:
                benchmark_summary = 'not run yet; press B when ready'
            if opencode_score > 0:
                opencode_summary = f'{opencode_score:.2f} score in {opencode_seconds:.2f}s'
            else:
                opencode_summary = 'not run yet; press O for opencode workflow'
            hardware = app.hardware_profile().short_summary()
            detail_rows = [
                ('[Esc] back   [Enter/l] actions   [T] try   [B] server bench   [O] opencode bench   [R] results   [z] auto profile', colors['accent'] | curses.A_BOLD),
                ('', curses.A_NORMAL),
                (f'name: {model.name}', curses.A_BOLD),
                (f'id/runtime/source: {model.id} / {display_runtime(model)} / {getattr(model, "source", "manual")}', curses.A_NORMAL),
                (f'quant/type: {extract_quant(model)} / {classify_model_type(model)}', curses.A_NORMAL),
                (f'path: {ellipsize(model.path, left_w - 14)}', curses.A_NORMAL),
                (f'alias/bind: {model.alias} / http://{model.host}:{model.port}', curses.A_NORMAL),
                (f'status: {status} ({detail})', status_attr(colors, status)),
                (f'pid/roles: {app.get_pid(model) or "-"} / {app.role_badges(model.id)}', curses.A_NORMAL),
                (f'ctx/output: {model.ctx} / {model.output}', curses.A_NORMAL),
                (f'threads/ngl/parallel: {model.threads} / {model.ngl} / {model.parallel}', curses.A_NORMAL),
                (f'temp/cache_ram: {model.temp} / {model.cache_ram}', curses.A_NORMAL),
                (f'profile: {model_profile_summary(model)}', curses.A_NORMAL),
                (f'ctx range: {getattr(model, "ctx_min", 2048)}..{getattr(model, "ctx_max", 131072)}', curses.A_NORMAL),
                (f'hardware: {hardware}', curses.A_NORMAL),
                (f'last benchmark: {benchmark_summary}', colors['warning'] if benchmark_score <= 0 else colors['success'] | curses.A_BOLD),
                (f'opencode benchmark: {opencode_summary}', colors['warning'] if opencode_score <= 0 else colors['success'] | curses.A_BOLD),
                ('command preview:', colors['accent'] | curses.A_BOLD),
                (ellipsize(' '.join(app.build_command(model)), left_w - 6), curses.A_NORMAL),
                ('', curses.A_NORMAL),
                ('benchmark table:', colors['accent'] | curses.A_BOLD),
                (f' {"OBJECTIVE":16} {"TOK/S":>8} {"SEC":>5} {"CTX":>6} {"SLOT":>6} {"PAR":>3} {"NGL":>4} STATUS', colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD),
            ]
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
            if benchmark_rows:
                benchmark_rows = sorted(
                    benchmark_rows,
                    key=lambda row: float(row.get('tokens_per_sec', 0.0) or 0.0),
                    reverse=True,
                )
            for row in benchmark_rows:
                row_score = float(row.get('tokens_per_sec', 0.0) or 0.0)
                row_seconds = float(row.get('seconds', 0.0) or 0.0)
                tok = f'{row_score:.2f}' if row_score > 0 else '-'
                secs = f'{row_seconds:.1f}' if row_seconds > 0 else '-'
                objective = profile_label(str(row.get('objective') or row.get('preset', '-')))[:16]
                status_text = str(row.get('status', '-'))
                ctx_value = int(row.get("ctx", 0) or 0)
                parallel_value = int(row.get("parallel", 0) or 0)
                slot_value = int(row.get("ctx_per_slot", 0) or 0) or (ctx_value // max(1, parallel_value or 1))
                line = (
                    f' {objective:16} {tok:>8} {secs:>5} '
                    f'{ctx_value:6} {slot_value:6} {parallel_value:3} '
                    f'{int(row.get("ngl", 0) or 0):4} {status_text}'
                )
                attr = colors['success'] | curses.A_BOLD if status_text == 'ok' else colors['error']
                detail_rows.append((ellipsize(line, left_w - 5), attr))
            if not benchmark_rows:
                detail_rows.append((' no benchmark rows yet; press B to run one manually', colors['warning']))

            opencode_rows = list(getattr(model, 'last_opencode_benchmark_results', []) or [])
            detail_rows.extend([
                ('', curses.A_NORMAL),
                ('opencode workflow table:', colors['accent'] | curses.A_BOLD),
                (f' {"PROFILE":16} {"TIER":9} {"SCORE":>8} {"SEC":>5} {"PASS":>5} {"CTX":>6} {"SLOT":>6} STATUS', colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD),
            ])
            if opencode_rows:
                opencode_rows = sorted(
                    opencode_rows,
                    key=lambda row: float(row.get('score', 0.0) or 0.0),
                    reverse=True,
                )
            for row in opencode_rows:
                row_score = float(row.get('score', 0.0) or 0.0)
                row_seconds = float(row.get('seconds', 0.0) or 0.0)
                preset = profile_label(str(row.get('preset', '-')))[:16]
                tier = tier_label(str(row.get('tier', '-')))[:9]
                status_text = str(row.get('status', '-'))
                pass_text = f'{int(row.get("passed", 0) or 0)}/{int(row.get("tasks", 0) or 0)}'
                ctx_value = int(row.get("ctx", 0) or 0)
                slot_value = int(row.get("ctx_per_slot", 0) or 0) or (ctx_value // max(1, int(row.get("parallel", 1) or 1)))
                line = (
                    f' {preset:16} {tier:9} {row_score:8.2f} {row_seconds:5.1f} '
                    f'{pass_text:>5} {ctx_value:6} {slot_value:6} {status_text}'
                )
                attr = (
                    colors['success'] | curses.A_BOLD
                    if status_text in ('tests passed', 'ok')
                    else colors['warning']
                    if status_text in ('tests failed', 'context too small', 'partial')
                    else colors['error']
                )
                detail_rows.append((ellipsize(line, left_w - 5), attr))
            if not opencode_rows:
                detail_rows.append((' no opencode workflow rows yet; press O to run', colors['warning']))

            for i, (line, attr) in enumerate(detail_rows[: h - box_top - 7]):
                stdscr.addstr(box_top + 2 + i, 3, line[: left_w - 4], attr)
        elif app.models:
            header = ' ID              PRT  ST        RLS  ENG        QNT      TYPE   NAME'
            stdscr.addstr(box_top + 2, 3, header, colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD)
            start_idx = max(0, selected - visible_rows + 3)
            end_idx = min(len(app.models), start_idx + visible_rows)
            for idx in range(start_idx, end_idx):
                model = app.models[idx]
                status, _ = statuses.get(model.id, ('?', ''))
                roles = app.role_badges(model.id)
                engine = display_runtime(model)[:10]
                quant = extract_quant(model)[:8]
                model_type = classify_model_type(model)[:6]
                name_col_width = max(10, left_w - 70)
                line = f' {model.id[:14]:14} {model.port:4}  {status_symbol(status)} {status[:6]:6}  {roles:3}  {engine:10} {quant:8} {model_type:6} {model.name[:name_col_width]}'
                row_y = box_top + 3 + idx - start_idx
                if idx == selected:
                    try:
                        stdscr.addstr(row_y, 3, line[: left_w - 3], colors['selection'] | curses.A_BOLD)
                    except curses.error:
                        stdscr.addstr(row_y, 3, line[: left_w - 3], curses.A_REVERSE)
                else:
                    stdscr.addstr(row_y, 3, line[: left_w - 3])
                    status_x = 3 + 1 + 14 + 1 + 4 + 2
                    stdscr.addstr(row_y, status_x, f'{status_symbol(status)} {status[:6]:6}', status_attr(colors, status))
            if len(app.models) > visible_rows:
                bar_h = max(1, visible_rows)
                track_x = left_w - 1
                for i in range(bar_h):
                    stdscr.addch(box_top + 3 + i, track_x, '│', colors['muted'])
                thumb_h = max(1, int(bar_h * (visible_rows / max(1, len(app.models)))))
                thumb_top = int((start_idx / max(1, len(app.models) - visible_rows)) * max(0, bar_h - thumb_h))
                for i in range(thumb_h):
                    stdscr.addch(box_top + 3 + thumb_top + i, track_x, '█', colors['accent'] | curses.A_BOLD)
        else:
            stdscr.addstr(box_top + 3, 3, 'No models yet. Press x to detect GGUFs or a to add a llama.cpp/vLLM model.', colors['warning'])

        if active_model and view_mode == 'results':
            model = active_model
            runs = benchmark_runs_for_model(model)
            if runs:
                results_run_index = max(0, min(results_run_index, len(runs) - 1))
                run = runs[results_run_index]
                lines = [
                    f'run: {run.get("id", "-")}',
                    f'status: {run.get("status", "-")}  kind: {run.get("kind", "-")}',
                    f'started: {run.get("started_at", "-")}',
                    f'ended: {run.get("ended_at", "-")}',
                    f'elapsed: {float(run.get("elapsed_seconds", 0.0) or 0.0):.1f}s',
                    f'summary: {run.get("summary", "no summary")}',
                    '',
                    'winners and ranking:',
                ]
                lines.extend(benchmark_ranking_rows(run))
            else:
                lines = ['No benchmark run selected.']
            for i, line in enumerate(lines[: max(1, logs_box_h - 3)]):
                attr = curses.A_NORMAL
                if line.startswith(('run:', 'winners')):
                    attr = colors['accent'] | curses.A_BOLD
                elif line.startswith(('Fast Chat:', 'Long Context:', 'OpenCode-ready:', 'Auto:')):
                    attr = colors['success'] | curses.A_BOLD if 'not measured' not in line else colors['warning']
                elif line.startswith(('break:', 'fail:')):
                    attr = colors['error']
                stdscr.addstr(box_top + 2 + i, right_x + 2, line[: right_w - 4], attr)
        elif active_model and view_mode == 'benchmark':
            model = active_model
            status, detail = statuses.get(model.id, ('?', ''))
            completed = int(benchmark_state.get('completed', 0) or 0)
            total = int(benchmark_state.get('total', 0) or 0)
            fraction = benchmark_progress_fraction(completed, total)
            progress_lines = [
                f'model: {model.id}',
                f'run: {benchmark_state.get("run_kind") or "server"}',
                f'status: {benchmark_state.get("status") or "idle"} / server {status}',
                f'phase: {benchmark_state.get("phase") or "-"}',
                f'elapsed: {benchmark_elapsed_text(benchmark_state)}',
                f'progress: {progress_bar_text(completed, total, max(8, right_w - 22))} {int(round(fraction * 100))}%',
                f'candidate: {benchmark_state.get("candidate") or "-"}',
                f'pid: {app.get_pid(model) or "-"}  {detail}',
            ]
            for i, line in enumerate(progress_lines[: max(1, profile_box_h - 3)]):
                attr = colors['accent'] | curses.A_BOLD if line.startswith(('progress:', 'status:')) else curses.A_NORMAL
                stdscr.addstr(box_top + 2 + i, right_x + 2, line[: right_w - 4], attr)
            log_lines = tail_text(app.logfile(model.id), max_lines=max(12, logs_box_h - 3))
            for i, line in enumerate(log_lines[: max(1, logs_box_h - 3)]):
                stdscr.addstr(logs_box_y + 2 + i, right_x + 2, line[: right_w - 4], curses.A_NORMAL)
        elif active_model and view_mode == 'try':
            model = active_model
            status, detail = statuses.get(model.id, ('?', ''))
            profile_lines = [
                f'model: {model.name}',
                f'id/runtime/source: {model.id} / {display_runtime(model)} / {getattr(model, "source", "manual")}',
                f'status: {status} ({detail})',
                f'pid: {app.get_pid(model) or "-"}',
                f'url: http://{model.host}:{model.port}',
                f'ctx/output: {model.ctx} / {model.output}',
                f'threads/ngl/parallel: {model.threads} / {model.ngl} / {model.parallel}',
                f'temp/cache_ram: {model.temp} / {model.cache_ram}',
                f'profile: {model_profile_summary(model)}',
                f'last bench: {getattr(model, "last_benchmark_tokens_per_sec", 0.0):.2f} tok/s {getattr(model, "last_benchmark_profile", "")}',
                f'opencode: {getattr(model, "last_opencode_benchmark_score", 0.0):.2f} score {getattr(model, "last_opencode_benchmark_profile", "")}',
                f'chat: {try_status}',
            ]
            if try_error:
                profile_lines.append(f'error: {try_error}')
            profile_lines.extend([
                'command preview:',
                ellipsize(' '.join(app.build_command(model)), right_w - 6),
            ])
            for i, line in enumerate(profile_lines[: max(1, profile_box_h - 3)]):
                attr = curses.A_NORMAL
                if line.startswith('status:'):
                    attr = status_attr(colors, status)
                elif line.startswith('error:'):
                    attr = colors['error'] | curses.A_BOLD
                elif line in ('command preview:',):
                    attr = colors['accent'] | curses.A_BOLD
                stdscr.addstr(box_top + 2 + i, right_x + 2, line[: right_w - 4], attr)
            log_lines = tail_text(app.logfile(model.id), max_lines=max(12, logs_box_h - 3))
            for i, line in enumerate(log_lines[: max(1, logs_box_h - 3)]):
                stdscr.addstr(logs_box_y + 2 + i, right_x + 2, line[: right_w - 4], curses.A_NORMAL)
            if stats_box_h:
                stats_lines = build_try_live_stat_lines(
                    model,
                    try_status,
                    app.get_pid(model),
                    try_live_metrics,
                )
                for i, line in enumerate(stats_lines[: max(1, stats_box_h - 3)]):
                    attr = colors['accent'] | curses.A_BOLD if line.startswith(('benchmark:', 'live:', 'last:')) else curses.A_NORMAL
                    stdscr.addstr(stats_box_y + 2 + i, right_x + 2, line[: right_w - 4], attr)
        elif active_model:
            model = active_model
            status, detail = statuses.get(model.id, ('?', ''))
            lines = [
                f'name: {model.name}',
                f'id/runtime/source: {model.id} / {display_runtime(model)} / {getattr(model, "source", "manual")}',
                f'path: {ellipsize(model.path, right_w - 12)}',
                f'alias/bind: {model.alias} / {model.host}:{model.port}',
                f'quant/type: {extract_quant(model)} / {classify_model_type(model)}',
                f'ctx/output: {model.ctx} / {model.output}  threads/ngl/par: {model.threads}/{model.ngl}/{model.parallel}',
                f'temp/cache_ram: {model.temp} / {model.cache_ram}',
                f'profile={model_profile_summary(model)} ctx_range={getattr(model, "ctx_min", 2048)}..{getattr(model, "ctx_max", 131072)}',
                f'benchmark={getattr(model, "last_benchmark_tokens_per_sec", 0.0):.2f} tok/s {getattr(model, "last_benchmark_profile", "")}',
                f'opencode_bench={getattr(model, "last_opencode_benchmark_score", 0.0):.2f} score {getattr(model, "last_opencode_benchmark_profile", "")}',
                f'flags: enabled={model.enabled} flash_attn={model.flash_attn} jinja={model.jinja}',
                f'status: {status} ({detail})',
                f'pid/roles: {app.get_pid(model) or "-"} / {app.role_badges(model.id)}  [m main] [s small] [b build] [p plan]',
                f'log: {app.logfile(model.id)}',
                'command preview:',
                ellipsize(' '.join(app.build_command(model)), right_w - 6),
                '',
                'last important log lines:' if status in ('ERROR', 'STOPPED') and error_text else 'last log lines:',
            ]
            if status in ('ERROR', 'STOPPED') and error_text:
                log_lines = important_log_excerpt(app.logfile(model.id), max_lines=max(12, logs_box_h), after_last_launch=True)
            else:
                log_lines = tail_text(app.logfile(model.id), max_lines=max(12, logs_box_h))
            lines.extend(log_lines)
            for i, line in enumerate(lines[: max(1, logs_box_h - 3)]):
                attr = curses.A_NORMAL
                if line.startswith('status:'):
                    attr = status_attr(colors, status)
                elif line in ('command preview:', 'last log lines:', 'last important log lines:'):
                    attr = colors['accent'] | curses.A_BOLD
                stdscr.addstr(box_top + 2 + i, right_x + 2, line[: right_w - 4], attr)

        if error_box_h:
            error_attr = colors['error'] if error_text else colors['muted']
            for i, line in enumerate(error_lines[: max(1, error_box_h - 3)]):
                stdscr.addstr(error_box_y + 2 + i, right_x + 2, line[: right_w - 4], error_attr)

        if view_mode == 'try':
            footer = '[Enter] send  [Up/Down] scroll input  [Ctrl+U] clear  [Esc] stop model + exit'
            footer2 = 'Prompt editor shows 5 wrapped rows. Transcript is temporary.'
        elif view_mode == 'benchmark':
            footer = '[Esc] details  [R] results  [A] abort active benchmark'
            footer2 = 'Dashboard shows measured tradeoffs: possible, fastest, ideal, highest context, and OpenCode-ready.'
        elif view_mode == 'results':
            footer = '[Esc] details  [Up/Down] select benchmark run'
            footer2 = 'Results keep the latest 10 runs with winners, runner-ups, failures, and break points.'
        elif view_mode == 'detail':
            footer = '[Esc] models  [Enter/l] actions  [T] try  [B] server bench  [O] opencode bench  [R] results  [z] auto'
            footer2 = '[m/s/b/p] roles  [g] gen opencode  [S] stop-all  [q] quit'
        else:
            footer = '[Enter] details  [z] auto profile  [B] benchmark best  [x] detect  [X] prune'
            footer2 = '[a/e/d] models  [m/s/b/p] set roles  [r] sync inventory  [S] stop-all  [q] quit'
        if action_running():
            footer = '[A] abort active action   ' + footer
        stdscr.addstr(h - 2, 2, footer[: w - 4], colors['accent'] | curses.A_BOLD)
        stdscr.addstr(h - 1, 2, footer2[: w - 4], colors['muted'] | curses.A_BOLD)
        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break

        if key == -1:
            time.sleep(0.05)
            continue
        if view_mode == 'try':
            if key == 27:
                exit_try_view()
                continue
            if key in (curses.KEY_UP, curses.KEY_DOWN):
                if try_status == 'ready':
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
        if view_mode == 'results':
            if key in (27, curses.KEY_BACKSPACE, 127, 8):
                view_mode = 'detail'
                message = 'Back to model details.'
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
        if action_running() and key not in (curses.KEY_UP, curses.KEY_DOWN, ord('j'), ord('k'), ord('R'), 27, curses.KEY_BACKSPACE, 127, 8):
            message = '⏳ Action is running. Watch the log window; controls unlock when it finishes.'
            continue
        if view_mode == 'benchmark' and key in (27, curses.KEY_BACKSPACE, 127, 8):
            view_mode = 'detail'
            message = 'Back to model details. Benchmark keeps running unless you press A.'
            continue
        if view_mode == 'detail' and key in (27, curses.KEY_BACKSPACE, 127, 8):
            view_mode = 'list'
            detail_model_id = ''
            message = 'Back to model list.'
            continue
        if key in (ord('q'), 27):
            if external_stack_launched and managed_server_running():
                quit_policy = prompt_quit_policy(stdscr, colors)
                if quit_policy == 'cancel':
                    message = 'Quit cancelled.'
                    continue
                if quit_policy == 'leave':
                    app.leave_managed_processes_running()
                    message = 'Leaving managed model servers running.'
            break
        if key in (curses.KEY_UP, ord('k')) and app.models and view_mode == 'list':
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord('j')) and app.models and view_mode == 'list':
            selected = min(len(app.models) - 1, selected + 1)
        elif key == ord('r'):
            count, items = app.detect_models()
            statuses = {m.id: app.health(m) for m in app.models}
            message = items[0] if items else (f'Synced {count} model(s)' if count else 'Synced.')
        elif key == ord('S'):
            message = '; '.join(app.stop_all())[: max(20, w - 4)]
        elif key in (10, 13, curses.KEY_ENTER) and app.models:
            model = active_detail_model()
            if not model:
                continue
            if view_mode == 'detail':
                begin_model_launch(model)
            else:
                open_model_details(model)
        elif key == ord('l') and app.models and view_mode == 'detail':
            model = active_detail_model()
            if model:
                begin_model_launch(model)
        elif key in (ord('T'), ord('t')) and app.models and view_mode == 'detail':
            model = active_detail_model()
            if model:
                open_try_view(model)
        elif key == ord('R') and app.models and view_mode in ('detail', 'benchmark', 'results'):
            model = active_detail_model()
            if model:
                open_results_view(model)
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
            message = f'{tune_msg} | {sync_msg}'
        elif key == ord('B') and app.models:
            model = active_detail_model()
            if not model:
                continue
            start_background_action(
                model,
                'coarse-to-fine benchmark profiles',
                lambda progress, token, model=model: benchmark_best_optimization(
                    app,
                    model,
                    progress=progress,
                    cancel_token=token,
                ),
                done_event='benchmark_done',
                run_kind='server',
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
        elif key == ord('a'):
            model = prompt_model(stdscr, 'Add model')
            if model:
                if not getattr(model, 'default_benchmark_status', ''):
                    model.default_benchmark_status = 'pending'
                app.add_or_update(model)
                selected = len(app.models) - 1
                message = f'Added {model.id} with safe defaults. Open details and press B to benchmark.'
        elif key == ord('e') and app.models:
            current = active_detail_model() or app.models[selected]
            updated = prompt_model(stdscr, f'Edit {current.id}', current)
            if updated:
                if updated.id != current.id:
                    app.delete(current.id)
                if not getattr(updated, 'default_benchmark_status', ''):
                    updated.default_benchmark_status = 'pending'
                app.add_or_update(updated)
                selected = min(selected, len(app.models) - 1)
                if view_mode == 'detail':
                    detail_model_id = updated.id
                message = f'Updated {updated.id}.'
        elif key == ord('d') and app.models:
            delete_model = active_detail_model() or app.models[selected]
            curses.endwin()
            ans = input(f'Delete {delete_model.id} from llama-tui config? [y/N]: ').strip().lower()
            stdscr.clear(); stdscr.refresh()
            if ans == 'y':
                target_id = delete_model.id
                ok, msg = app.delete(target_id)
                selected = max(0, min(selected, len(app.models) - 1))
                if view_mode == 'detail':
                    view_mode = 'list'
                    detail_model_id = ''
                message = f'{target_id}: {msg}'
            else:
                message = 'Delete cancelled.'
        elif key == ord('x'):
            count, items = app.detect_models()
            message = items[0] if items else (f'Detected {count} new model(s)' if count else 'No new GGUFs found.')
            if count:
                message = f'{message} | safe defaults set; press B from details to benchmark.'
            selected = min(selected, len(app.models) - 1 if app.models else 0)
        elif key == ord('X'):
            count, removed = app.prune_missing_models()
            message = f'Pruned {count}: {", ".join(removed[:5])}' if count else 'No missing models to prune.'
            selected = max(0, min(selected, len(app.models) - 1))
        elif key == ord('g'):
            ok, msg = app.generate_opencode()
            message = msg
        elif key == ord('o'):
            if prompt_settings(stdscr, app):
                message = 'Settings saved.'
            else:
                message = 'Settings unchanged.'
        elif key == ord('m') and app.models:
            model = active_detail_model() or app.models[selected]
            app.set_role('main', model.id)
            message = f'{model.id} set as main model.'
        elif key == ord('s') and app.models:
            model = active_detail_model() or app.models[selected]
            app.set_role('small', model.id)
            message = f'{model.id} set as small model.'
        elif key == ord('b') and app.models:
            model = active_detail_model() or app.models[selected]
            app.set_role('build', model.id)
            message = f'{model.id} set as build model.'
        elif key == ord('p') and app.models:
            model = active_detail_model() or app.models[selected]
            app.set_role('plan', model.id)
            message = f'{model.id} set as plan model.'
