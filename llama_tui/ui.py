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
    benchmark_best_optimization,
    launch_opencode_stack,
    launch_with_failsafe,
    start_model_with_progress,
    sync_opencode_after_tuning,
)
from .constants import LOGO, REFRESH_SECONDS
from .discovery import classify_model_type, display_runtime, extract_quant
from .hardware import HardwareProfile
from .models import ModelConfig
from .opencode_benchmark import benchmark_opencode_workflow
from .optimize import apply_best_optimization, select_best_tier
from .textutil import compact_message, ellipsize, important_log_excerpt, is_error_message, tail_text, wrap_display_lines


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
        ('1', 'Best optimization for this PC', 'best'),
        ('2', 'Optimize for max context', 'max_context'),
        ('3', 'Optimize for tokens/sec', 'tokens_per_sec'),
        ('4', 'Keep current settings', 'keep'),
        ('5', 'Launch model + OpenCode', 'opencode'),
        ('6', 'Launch full-stack: OpenCode + VS Code', 'full_stack'),
        ('q', 'Cancel', 'cancel'),
    ])
def prompt_running_model_action(stdscr, model: ModelConfig, colors) -> str:
    return prompt_modal_choice(stdscr, colors, f'{model.id} is running', [
        ('1', 'Stop model', 'stop'),
        ('2', 'Launch OpenCode', 'opencode'),
        ('3', 'Launch full-stack: OpenCode + VS Code', 'full_stack'),
        ('q', 'Cancel', 'cancel'),
    ])
def prompt_optimization_tier(stdscr, colors) -> str:
    return prompt_modal_choice(stdscr, colors, 'Optimization tier', [
        ('1', 'Safe (most stable)', 'safe'),
        ('2', 'Moderate (balanced)', 'moderate'),
        ('3', 'Extreme (aggressive)', 'extreme'),
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
    auto_benchmark_started: set[str] = set()
    message = 'Ready.'
    last_error_message = ''
    last_refresh = 0.0
    external_stack_launched = False
    statuses: Dict[str, Tuple[str, str]] = {}
    action_thread: Optional[threading.Thread] = None
    action_queue: Queue = Queue()

    def action_running() -> bool:
        return action_thread is not None and action_thread.is_alive()

    def start_background_action(
        model: ModelConfig,
        label: str,
        worker: Callable[[Callable[[str], None]], Tuple[bool, str]],
        done_event: str = 'done',
    ):
        nonlocal action_thread, message
        if action_running():
            message = '⏳ Another optimization is still running. Watch the log window for progress.'
            return

        def progress(text: str):
            line = compact_message(text)
            append_model_log(app, model, line)
            action_queue.put(('progress', line))

        def runner():
            try:
                progress(f'{label} started for {model.id}')
                _ok, result = worker(progress)
            except Exception as exc:
                result = f'❌ {label} failed: {exc}'
                progress(result)
            action_queue.put((done_event, compact_message(result)))

        action_thread = threading.Thread(target=runner, daemon=True)
        action_thread.start()
        message = f'⏳ {label} started for {model.id}. Progress is in the log window.'

    def selected_model() -> Optional[ModelConfig]:
        if not app.models:
            return None
        idx = max(0, min(selected, len(app.models) - 1))
        return app.models[idx]

    def active_detail_model() -> Optional[ModelConfig]:
        if view_mode == 'detail' and detail_model_id:
            return app.get_model(detail_model_id) or selected_model()
        return selected_model()

    def model_is_running(model: ModelConfig) -> bool:
        status, _detail = app.health(model)
        return status in ('READY', 'LOADING', 'STARTING') or bool(app.get_pid(model))

    def managed_server_running() -> bool:
        return any(app.get_pid(model, discover=False, managed_only=True) for model in app.models)

    def has_benchmark(model: ModelConfig) -> bool:
        return float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0) > 0.0

    def maybe_auto_benchmark(model: ModelConfig):
        nonlocal message
        if has_benchmark(model) or model.id in auto_benchmark_started or action_running():
            return
        if model_is_running(model):
            message = f'{model.id}: no benchmark yet. Stop it and press B from details to benchmark.'
            return
        auto_benchmark_started.add(model.id)
        start_background_action(
            model,
            'auto benchmark optimization',
            lambda progress, model=model: benchmark_best_optimization(app, model, progress=progress),
        )
        message = f'⏳ No benchmark found for {model.id}; auto benchmark started. Watch the log window.'

    def open_model_details(model: ModelConfig):
        nonlocal view_mode, detail_model_id, message
        view_mode = 'detail'
        detail_model_id = model.id
        message = f'{model.id}: details loaded. Press Enter/l to start or Esc to return.'
        maybe_auto_benchmark(model)

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
                lambda progress, model=model, workspace=workspace, include_vscode=include_vscode: launch_opencode_stack(
                    app,
                    model,
                    workspace,
                    include_vscode=include_vscode,
                    progress=progress,
                ),
                done_event='stack_done',
            )
            return
        if launch_mode == 'best':
            start_background_action(
                model,
                'best launch optimization',
                lambda progress, model=model, launch_mode=launch_mode: launch_with_failsafe(app, model, launch_mode, 'auto', progress=progress),
            )
        elif launch_mode in ('max_context', 'tokens_per_sec'):
            tier = prompt_optimization_tier(stdscr, colors)
            if tier == 'cancel':
                message = 'Launch cancelled.'
                return
            start_background_action(
                model,
                f'{launch_mode}/{tier} launch optimization',
                lambda progress, model=model, launch_mode=launch_mode, tier=tier: launch_with_failsafe(app, model, launch_mode, tier, progress=progress),
            )
        else:
            start_background_action(
                model,
                'model launch',
                lambda progress, model=model: start_model_with_progress(app, model, progress=progress),
            )

    while True:
        while True:
            try:
                event, text = action_queue.get_nowait()
            except Empty:
                break
            if event == 'stack_done' and text.startswith('✅'):
                external_stack_launched = True
            if is_error_message(text):
                last_error_message = text
            message = text
            if event in ('done', 'stack_done'):
                action_thread = None
                last_refresh = 0.0

        now = time.time()
        if now - last_refresh > REFRESH_SECONDS:
            statuses = {m.id: app.health(m) for m in app.models}
            last_refresh = now

        if app.models:
            selected = max(0, min(selected, len(app.models) - 1))
            if view_mode == 'detail':
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
        header_message = 'Error captured in the lower-right Errors box.' if message_is_error else compact_message(message)
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
        visible_rows = max(8, h - box_top - 6)
        right_total_h = max(4, h - box_top - 4)
        active_status = statuses.get(active_model.id, ('?', '')) if active_model else ('?', '')
        status_error = f'{active_model.id}: status ERROR ({active_status[1]})' if active_model and active_status[0] == 'ERROR' else ''
        error_text = last_error_message or status_error
        error_lines = wrap_display_lines(error_text, right_w - 4) if error_text else ['No errors captured.']
        if right_total_h >= 14:
            desired_error_h = min(max(5, len(error_lines) + 3), 14)
            error_box_h = min(desired_error_h, max(5, right_total_h - 8))
        else:
            error_box_h = 0
        logs_box_h = right_total_h if error_box_h == 0 else max(5, right_total_h - error_box_h - 1)
        error_box_y = box_top + logs_box_h + 1

        left_title = 'Model Details' if view_mode == 'detail' else 'Models'
        draw_box(stdscr, box_top, 1, h - box_top - 4, left_w, left_title, colors['accent'] | curses.A_BOLD, colors['accent'])
        draw_box(stdscr, box_top, right_x, logs_box_h, right_w, 'Details / Logs / Roles', colors['accent'] | curses.A_BOLD, colors['accent'])
        if error_box_h:
            draw_box(stdscr, error_box_y, right_x, error_box_h, right_w, 'Errors', colors['error'] | curses.A_BOLD, colors['error'])

        if view_mode == 'detail' and active_model:
            model = active_model
            status, detail = statuses.get(model.id, ('?', ''))
            benchmark_score = float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0)
            benchmark_seconds = float(getattr(model, 'last_benchmark_seconds', 0.0) or 0.0)
            opencode_score = float(getattr(model, 'last_opencode_benchmark_score', 0.0) or 0.0)
            opencode_seconds = float(getattr(model, 'last_opencode_benchmark_seconds', 0.0) or 0.0)
            if benchmark_score > 0:
                benchmark_summary = f'{benchmark_score:.2f} tok/s in {benchmark_seconds:.2f}s'
            else:
                benchmark_summary = 'not run yet; auto-benchmark starts here when safe'
            if opencode_score > 0:
                opencode_summary = f'{opencode_score:.2f} score in {opencode_seconds:.2f}s'
            else:
                opencode_summary = 'not run yet; press O for opencode workflow'
            hardware = app.hardware_profile().short_summary()
            detail_rows = [
                ('[Esc] back   [Enter/l] actions   [B] server bench   [O] opencode bench   [z] optimize', colors['accent'] | curses.A_BOLD),
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
                (f'optimization: {getattr(model, "optimize_mode", "max_context_safe")}/{getattr(model, "optimize_tier", "moderate")} ram_reserve={getattr(model, "memory_reserve_percent", 25)}%', curses.A_NORMAL),
                (f'ctx range: {getattr(model, "ctx_min", 2048)}..{getattr(model, "ctx_max", 131072)}', curses.A_NORMAL),
                (f'hardware: {hardware}', curses.A_NORMAL),
                (f'last benchmark: {benchmark_summary}', colors['warning'] if benchmark_score <= 0 else colors['success'] | curses.A_BOLD),
                (f'opencode benchmark: {opencode_summary}', colors['warning'] if opencode_score <= 0 else colors['success'] | curses.A_BOLD),
                ('command preview:', colors['accent'] | curses.A_BOLD),
                (ellipsize(' '.join(app.build_command(model)), left_w - 6), curses.A_NORMAL),
                ('', curses.A_NORMAL),
                ('benchmark table:', colors['accent'] | curses.A_BOLD),
                (f' {"OPTIMIZATION":18} {"TIER":8} {"TOK/S":>8} {"SEC":>6} {"CTX":>7} {"PAR":>3} {"THR":>3} {"NGL":>4} STATUS', colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD),
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
                preset = str(row.get('preset', '-'))[:18]
                tier = str(row.get('tier', '-'))[:8]
                status_text = str(row.get('status', '-'))
                line = (
                    f' {preset:18} {tier:8} {tok:>8} {secs:>6} '
                    f'{int(row.get("ctx", 0) or 0):7} {int(row.get("parallel", 0) or 0):3} '
                    f'{int(row.get("threads", 0) or 0):3} {int(row.get("ngl", 0) or 0):4} {status_text}'
                )
                attr = colors['success'] | curses.A_BOLD if status_text == 'ok' else colors['error']
                detail_rows.append((ellipsize(line, left_w - 5), attr))
            if not benchmark_rows:
                detail_rows.append((' no benchmark rows yet; auto benchmark will fill this table', colors['warning']))

            opencode_rows = list(getattr(model, 'last_opencode_benchmark_results', []) or [])
            detail_rows.extend([
                ('', curses.A_NORMAL),
                ('opencode workflow table:', colors['accent'] | curses.A_BOLD),
                (f' {"OPTIMIZATION":18} {"TIER":8} {"SCORE":>8} {"SEC":>6} {"PASS":>5} {"CTX":>7} STATUS', colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD),
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
                preset = str(row.get('preset', '-'))[:18]
                tier = str(row.get('tier', '-'))[:8]
                status_text = str(row.get('status', '-'))
                pass_text = f'{int(row.get("passed", 0) or 0)}/{int(row.get("tasks", 0) or 0)}'
                line = (
                    f' {preset:18} {tier:8} {row_score:8.2f} {row_seconds:6.1f} '
                    f'{pass_text:>5} {int(row.get("ctx", 0) or 0):7} {status_text}'
                )
                attr = colors['success'] | curses.A_BOLD if status_text == 'ok' else colors['warning'] if status_text == 'partial' else colors['error']
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

        if active_model:
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
                f'optimize={getattr(model, "optimize_mode", "max_context_safe")}/{getattr(model, "optimize_tier", "moderate")} ctx_range={getattr(model, "ctx_min", 2048)}..{getattr(model, "ctx_max", 131072)} ram_reserve={getattr(model, "memory_reserve_percent", 25)}%',
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

        if view_mode == 'detail':
            footer = '[Esc] models  [Enter/l] actions  [B] server bench  [O] opencode bench  [z] optimize'
            footer2 = '[m/s/b/p] roles  [g] gen opencode  [S] stop-all  [q] quit'
        else:
            footer = '[Enter] details  [z] best optimization  [B] benchmark best  [x] detect  [X] prune'
            footer2 = '[a/e/d] models  [m/s/b/p] set roles  [r] sync inventory  [S] stop-all  [q] quit'
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
        if action_running() and key not in (curses.KEY_UP, curses.KEY_DOWN, ord('j'), ord('k'), 27, curses.KEY_BACKSPACE, 127, 8):
            message = '⏳ Action is running. Watch the log window; controls unlock when it finishes.'
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
        elif key == ord('z') and app.models:
            model = active_detail_model()
            if not model:
                continue
            profile = app.hardware_profile(refresh=True)
            tier = select_best_tier(model, profile)
            tune_msg = apply_best_optimization(model, tier=tier, profile=profile)
            app.add_or_update(model)
            sync_msg = sync_opencode_after_tuning(app)
            message = f'{tune_msg} | {sync_msg}'
        elif key == ord('B') and app.models:
            model = active_detail_model()
            if not model:
                continue
            start_background_action(
                model,
                'benchmark optimization',
                lambda progress, model=model: benchmark_best_optimization(app, model, progress=progress),
            )
        elif key == ord('O') and app.models and view_mode == 'detail':
            model = active_detail_model()
            if not model:
                continue
            start_background_action(
                model,
                'opencode workflow benchmark',
                lambda progress, model=model: benchmark_opencode_workflow(app, model, progress=progress),
            )
        elif key == ord('a'):
            model = prompt_model(stdscr, 'Add model')
            if model:
                app.add_or_update(model)
                selected = len(app.models) - 1
                message = f'Added {model.id}.'
        elif key == ord('e') and app.models:
            current = active_detail_model() or app.models[selected]
            updated = prompt_model(stdscr, f'Edit {current.id}', current)
            if updated:
                if updated.id != current.id:
                    app.delete(current.id)
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
