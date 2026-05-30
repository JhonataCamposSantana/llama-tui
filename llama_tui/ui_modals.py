# Audit #8 step 2: cohesive block of modal/prompt helpers extracted from
# ui.py. Re-exported via ``from .ui_modals import *`` from ui.py so
# tui() and existing test imports (``from llama_tui.ui import
# prompt_*, *_form_*, config_doctor_items, ...``) keep working.
import curses
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .app import AppConfig, CONTINUE_MERGE_MODES
from .benchmark import (
    apply_moe_recommendation,
    benchmark_strategy_for_app,
    has_moe_recommendation,
    moe_recommendation_applied,
)
from .constants import DEFAULT_HOST, DEFAULT_MODEL_PORT
from .discovery import classify_model_type, display_runtime
from .models import ModelConfig
from .mtp import clamp_mtp_draft, normalize_mtp_support
from .mtp_doctor import build_mtp_doctor_report
from .optimize import model_is_moe
from .textutil import compact_message, ellipsize
from .ui_benchmark import benchmark_plan_lines
from .ui_components import draw_box, kind_status_prefix, safe_addstr
from .ui_modal_session import open_modal
from .ui_models import (
    BROWSER_VIEW_OPTIONS,
    active_engine_binary,
    active_engine_short,
    benchmark_freshness_display,
)
from .ui_theme import kind_style

# Helpers defined earlier in ui.py. Imported lazily because ui.py
# pulls us in via ``from .ui_modals import *`` *after* it has finished
# defining everything above its modal section, so by the time this
# import runs the names below are already bound in the (partially
# loaded) ui module.
from .ui import (  # noqa: E402  -- intentional bottom-of-ui dependency
    DETAIL_DENSITY_OPTIONS,
    FILTER_COMPATIBILITY_OPTIONS,
    FILTER_RUNTIME_OPTIONS,
    FILTER_SOURCE_OPTIONS,
    FILTER_STATUS_OPTIONS,
    RIGHT_PANE_SCROLL_KEYS,
    SORT_OPTIONS,
    VALID_OPTIMIZE_MODES,
    VALID_OPTIMIZE_TIERS,
    VALID_RUNTIMES,
    active_engine_badge_kind,
    active_engine_badge_line,
    adjust_scroll_offset,
    browser_view_label,
    clamp_scroll,
    detail_density_label,
    normalize_choice,
    parse_bool_text,
    scrollable_pane_item_view,
    sort_mode_label,
    turboquant_detail_line,
    turboquant_status_kind,
    wrap_display_item_lines,
)


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


def form_is_single_field(field_count: int) -> bool:
    return int(field_count or 0) <= 1


def form_key_submits(key: int, field_count: int) -> bool:
    if key in (19, getattr(curses, 'KEY_F2', -1)):
        return True
    if form_is_single_field(field_count) and key in (10, 13, getattr(curses, 'KEY_ENTER', -1)):
        return True
    return False


def form_key_advances(key: int, field_count: int) -> bool:
    if key in (curses.KEY_DOWN, 9):
        return True
    if not form_is_single_field(field_count) and key in (10, 13, getattr(curses, 'KEY_ENTER', -1)):
        return True
    return False


def form_status_text(field_count: int) -> str:
    if form_is_single_field(field_count):
        return 'Edit value in place. Enter/F2 saves. Esc cancels.'
    return 'Edit values in place. F2 saves. Enter moves to next field. Esc cancels.'


def form_footer_text(field_count: int) -> str:
    if form_is_single_field(field_count):
        return '[Enter/F2] save  [Esc] cancel  [Ctrl+U] clear field'
    return '[F2] save  [Enter/Tab] next  [Esc] cancel  [Ctrl+U] clear field'


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
    field_count = len(fields)
    status = form_status_text(field_count)
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
            safe_addstr(modal, box_h - 2, 2, ellipsize(form_footer_text(field_count), box_w - 4), colors['accent'] | curses.A_BOLD)
            modal.refresh()

            key = modal.getch()
            if key in (27,):
                return None
            if form_key_submits(key, field_count):
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
            if form_key_advances(key, field_count):
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
        form_field('path', 'path', initial.path, 'GGUF path'),
        form_field('alias', 'alias', initial.alias or initial.id, 'OpenAI-style model name served on the port'),
        form_field('runtime', 'runtime', getattr(initial, 'runtime', 'llama.cpp'), 'llama.cpp'),
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
        form_field('supports_mtp', 'supports_mtp', normalize_mtp_support(getattr(initial, 'supports_mtp', 'auto')), 'auto/yes/no'),
        form_field('tags', 'tags', ', '.join(list(getattr(initial, 'tags', []) or [])), 'comma-separated: coding/autocomplete/long-context/fast-chat/custom'),
        form_field('extra_args', 'extra_args', ' '.join(initial.extra_args), 'space-separated extra runtime flags'),
    ]


def parse_model_form_answers(answers: Dict[str, str], initial: Optional[ModelConfig] = None) -> Tuple[Optional[ModelConfig], Dict[str, str]]:
    initial = initial or ModelConfig(id='', name='', path='', alias='', port=DEFAULT_MODEL_PORT)
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
        errors['runtime'] = 'runtime must be llama.cpp'
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
        supports_mtp=normalize_mtp_support(cleaned.get('supports_mtp', 'auto')),
        mtp_enabled=bool(getattr(initial, 'mtp_enabled', False)),
        mtp_draft_n_max=clamp_mtp_draft(getattr(initial, 'mtp_draft_n_max', 3), default=3),
        source=getattr(initial, 'source', 'manual'),
        source_path=str(getattr(initial, 'source_path', '') or ''),
        source_root=str(getattr(initial, 'source_root', '') or ''),
        source_repo_id=str(getattr(initial, 'source_repo_id', '') or ''),
        source_snapshot=str(getattr(initial, 'source_snapshot', '') or ''),
        source_labels=list(getattr(initial, 'source_labels', []) or []),
        last_used_at=str(getattr(initial, 'last_used_at', '') or ''),
        sort_rank=int(getattr(initial, 'sort_rank', 0) or 0),
        tags=[item.strip() for item in cleaned.get('tags', '').split(',') if item.strip()],
        verification_status=str(getattr(initial, 'verification_status', 'unknown') or 'unknown'),
        verification_at=str(getattr(initial, 'verification_at', '') or ''),
        verification_fingerprint=str(getattr(initial, 'verification_fingerprint', '') or ''),
        verification_summary=str(getattr(initial, 'verification_summary', '') or ''),
        verification_results=dict(getattr(initial, 'verification_results', {}) or {}),
        turboquant_status=str(getattr(initial, 'turboquant_status', 'unknown') or 'unknown'),
        turboquant_head_dim=int(getattr(initial, 'turboquant_head_dim', 0) or 0),
        turboquant_key_dim=int(getattr(initial, 'turboquant_key_dim', 0) or 0),
        turboquant_value_dim=int(getattr(initial, 'turboquant_value_dim', 0) or 0),
        turboquant_source=str(getattr(initial, 'turboquant_source', '') or ''),
        turboquant_reason=str(getattr(initial, 'turboquant_reason', '') or ''),
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
        form_field('browser_view', 'browser_view', getattr(app.ui, 'browser_view', 'compact'), 'compact or advanced'),
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
    browser_view = normalize_choice(cleaned.get('browser_view', ''), tuple(key for key, _label in BROWSER_VIEW_OPTIONS), 'compact')
    if browser_view != (cleaned.get('browser_view', '').lower() or 'compact'):
        errors['browser_view'] = 'browser_view must be compact or advanced'
    continue_merge_mode = cleaned.get('continue_merge_mode', 'preserve_sections') or 'preserve_sections'
    if continue_merge_mode not in CONTINUE_MERGE_MODES:
        errors['continue_merge_mode'] = 'must be preserve_sections or managed_file'
    payload = {
        'llama_server': cleaned.get('llama_server', ''),
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
            'browser_view': browser_view,
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
    initial = initial or ModelConfig(id='', name='', path='', alias='', port=DEFAULT_MODEL_PORT)
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


def browser_filter_fields(
    runtime_filter: str,
    source_filter: str,
    status_filter: str,
    tag_filter: str = 'all',
    compatibility_filter: str = 'all',
) -> List[Dict[str, str]]:
    return [
        form_field('runtime_filter', 'runtime_filter', runtime_filter, 'all/llama.cpp'),
        form_field('source_filter', 'source_filter', source_filter, 'all/manual/huggingface/hf_cache/llama_cache/llmfit/llm-models/lm-studio'),
        form_field('status_filter', 'status_filter', status_filter, 'all/READY/LOADING/STARTING/STOPPED/ERROR/fresh/stale/missing/failed/pending/running'),
        form_field('tag_filter', 'tag_filter', tag_filter, 'all/coding/autocomplete/long-context/fast-chat/custom tag'),
        form_field('compatibility_filter', 'compatibility_filter', compatibility_filter, 'active/incompatible/all'),
    ]


def parse_browser_filter_answers(answers: Dict[str, str]) -> Tuple[Optional[Tuple[str, str, str, str, str]], Dict[str, str]]:
    cleaned = {key: str(value or '').strip() for key, value in answers.items()}
    runtime = cleaned.get('runtime_filter', '').lower() or 'all'
    source = cleaned.get('source_filter', '').lower() or 'all'
    status = cleaned.get('status_filter', '') or 'all'
    tag = cleaned.get('tag_filter', '').lower() or 'all'
    compatibility = cleaned.get('compatibility_filter', '').lower() or 'all'
    errors: Dict[str, str] = {}
    if runtime not in dict(FILTER_RUNTIME_OPTIONS):
        errors['runtime_filter'] = 'unsupported runtime filter'
    if source not in dict(FILTER_SOURCE_OPTIONS):
        errors['source_filter'] = 'unsupported source filter'
    if status not in dict(FILTER_STATUS_OPTIONS):
        errors['status_filter'] = 'unsupported status filter'
    if compatibility not in dict(FILTER_COMPATIBILITY_OPTIONS):
        errors['compatibility_filter'] = 'unsupported compatibility filter'
    if errors:
        return None, errors
    return (runtime, source, status, tag, compatibility), {}


def prompt_browser_filters(
    stdscr,
    colors,
    runtime_filter: str,
    source_filter: str,
    status_filter: str,
    tag_filter: str = 'all',
    compatibility_filter: str = 'all',
) -> Optional[Tuple[str, str, str, str, str]]:
    return prompt_form(
        stdscr,
        colors,
        'Model Filters',
        browser_filter_fields(runtime_filter, source_filter, status_filter, tag_filter, compatibility_filter),
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
        'B opens the Benchmark Menu. F/N/O/H remain advanced benchmark shortcuts.',
        '',
        'Power tools',
        '? opens this help. : opens the command palette (start/stop, benchmark, MTP Doctor,',
        'benchmark plan preview, apply measured profile, open logs, export OpenCode).',
        'g/c/G export OpenCode, Continue, and Hermes configs.',
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
    with open_modal(stdscr, box_h=box_h, box_w=box_w, min_h=12, min_w=56) as session:
        if session is None:
            return
        modal = session.window
        content_h = max(1, session.box_h - 4)
        lines = [line for raw in help_overlay_lines() for line in wrap_display_item_lines(raw, session.box_w - 4)]
        scroll = 0
        while True:
            scroll = clamp_scroll(scroll, len(lines), content_h)
            modal.erase()
            draw_box(modal, 0, 0, session.box_h - 1, session.box_w, 'Help', colors['accent'] | curses.A_BOLD, colors['accent'])
            visible = lines[scroll: scroll + content_h]
            for idx, line in enumerate(visible):
                attr = colors['accent'] | curses.A_BOLD if line in ('Quick Help', 'List view', 'Detail view', 'Power tools') else curses.A_NORMAL
                safe_addstr(modal, 2 + idx, 2, line[: session.box_w - 4], attr)
            safe_addstr(modal, session.box_h - 2, 2, '[Up/Down] scroll  [Esc/q] close'[: session.box_w - 4], colors['muted'])
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


def config_doctor_items(app: AppConfig, active_model: Optional[ModelConfig] = None) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = [('Config Doctor', 'heading')]
    llama_ok = app.command_exists(app.llama_server)
    if active_model is not None:
        active_engine = active_engine_short(app, active_model)
        active_binary = active_engine_binary(app, active_model) or app.llama_server
    else:
        runtime_profile = getattr(app, 'runtime_profile', None)
        active_engine = getattr(runtime_profile, 'engine', 'llama.cpp') or 'llama.cpp'
        active_binary = getattr(runtime_profile, 'server_command', app.llama_server) or app.llama_server
    active_ok = app.command_exists(active_binary)
    hermes_ok = app.command_exists(getattr(app.hermes, 'command', 'hermes') or 'hermes')
    continue_path = getattr(app.continue_settings, 'path', '') or ''
    items.extend([
        (f'code path: {Path(__file__).resolve().parents[1]}', 'muted'),
        (f'llama-server: {"ok" if llama_ok else "missing"}  {app.llama_server}', 'success' if llama_ok else 'error'),
        (active_engine_badge_line(app, active_model), active_engine_badge_kind(app, active_model)),
        (f'active engine path: {"ok" if active_ok else "missing"}  {active_engine}  {active_binary}', 'success' if active_ok else 'error'),
        (f'Hermes: {"ok" if hermes_ok else "missing"}  {getattr(app.hermes, "command", "hermes") or "hermes"}', 'success' if hermes_ok else 'warning'),
        (f'OpenCode config: {app.opencode.path or "<unset>"}', 'success' if app.opencode.path else 'warning'),
        (f'Continue config: {continue_path or "<unset>"} mode={getattr(app.continue_settings, "merge_mode", "preserve_sections")}', 'success' if continue_path else 'warning'),
        (f'Hermes home: {getattr(app.hermes, "home_root", "") or "<unset>"}', 'success' if getattr(app.hermes, 'home_root', '') else 'warning'),
    ])
    enabled_continue_models = [
        model
        for model in list(getattr(app, 'models', []) or [])
        if bool(getattr(model, 'enabled', True)) and continue_path
    ]
    items.append((
        f'Continue Agent tools: tool_use exported for {len(enabled_continue_models)} model(s); MCP requires Agent Mode',
        'success' if enabled_continue_models else 'muted',
    ))
    if active_model is not None and str(active_engine).lower() in ('llama.cpp', 'llama.cpp-mtp', 'turboquant'):
        try:
            tool_jinja_ready = bool(app.continue_tool_use_launch_required(active_model)) or bool(getattr(active_model, 'jinja', True))
        except (AttributeError, TypeError, OSError, ValueError):
            tool_jinja_ready = bool(getattr(active_model, 'jinja', True))
        items.append((
            'Continue llama.cpp tools: --jinja ready; use extra_args --chat-template-file when needed'
            if tool_jinja_ready
            else 'Continue llama.cpp tools: --jinja disabled; Agent tools may fail',
            'success' if tool_jinja_ready else 'warning',
        ))
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

    def model_endpoint(model: ModelConfig) -> str:
        host = str(getattr(model, 'host', DEFAULT_HOST) or DEFAULT_HOST).strip() or DEFAULT_HOST
        try:
            port = int(getattr(model, 'port', DEFAULT_MODEL_PORT) or DEFAULT_MODEL_PORT)
        except (TypeError, ValueError):
            port = DEFAULT_MODEL_PORT
        return f'{host}:{port}'

    enabled_endpoints = sorted({
        model_endpoint(model)
        for model in list(getattr(app, 'models', []) or [])
        if bool(getattr(model, 'enabled', True))
    })
    if not enabled_endpoints:
        items.append((f'server endpoint: default {DEFAULT_HOST}:{DEFAULT_MODEL_PORT}', 'muted'))
    elif len(enabled_endpoints) == 1:
        kind = 'success' if enabled_endpoints[0] == f'{DEFAULT_HOST}:{DEFAULT_MODEL_PORT}' else 'warning'
        items.append((f'server endpoint: {enabled_endpoints[0]}', kind))
    else:
        preview = ', '.join(enabled_endpoints[:3])
        if len(enabled_endpoints) > 3:
            preview += ' ...'
        items.append((
            f'server endpoints split: {len(enabled_endpoints)} endpoints ({preview}); single-server workflow prefers {DEFAULT_HOST}:{DEFAULT_MODEL_PORT}',
            'warning',
        ))
    pending = app.benchmark_proof_model_ids(force=False)
    items.append((f'benchmark proof needed: {len(pending)} model(s)', 'warning' if pending else 'success'))
    if active_model:
        result = getattr(active_model, 'verification_results', {}) or {}
        cap = result.get('cap', {}) if isinstance(result, dict) else {}
        active_engine = getattr(app, 'active_engine_key_for_model', lambda _model: '')(active_model)
        items.extend([
            ('', 'normal'),
            (f'active model: {active_model.id}', 'heading'),
            (f'verification: {getattr(active_model, "verification_status", "unknown")} {getattr(active_model, "verification_summary", "")}', 'normal'),
            (f'cap: factor={cap.get("limiting_factor", "-")} configured={cap.get("configured_ctx", "-")} slot={cap.get("ctx_per_slot", "-")} safe={cap.get("estimated_safe_context", "-")} measured={cap.get("measured_max_context", "-")}', 'normal'),
            (
                turboquant_detail_line(active_model),
                turboquant_status_kind(active_model, active_engine == 'turboquant'),
            ),
        ])
    return items


def mtp_doctor_items(app: AppConfig, active_model: Optional[ModelConfig] = None) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = [('MTP Doctor', 'heading')]
    if active_model is None:
        items.append(('selected model: none', 'warning'))
        items.append(('next action: select an MTP-capable GGUF before opening MTP Doctor', 'muted'))
        return items
    report = build_mtp_doctor_report(app, active_model)

    def yn(value: bool) -> str:
        return 'yes' if bool(value) else 'no'

    def kind_for_status(status: str) -> str:
        normalized = str(status or '').lower()
        if normalized in ('ready', 'usable', 'preferred', 'compatible'):
            return 'success'
        if normalized in ('blocked', 'failed', 'unsupported', 'fit blocked', 'memory-bound'):
            return 'error'
        if normalized in ('unknown', 'risky', 'risky acceptance', 'compatible_with_warning'):
            return 'warning'
        return 'muted'

    values = ', '.join(report.spec_type_values) if report.spec_type_values else 'none'
    checked = ', '.join(report.checked_paths[:4])
    if len(report.checked_paths) > 4:
        checked += ' ...'
    features = ', '.join(report.detected_features) if report.detected_features else 'none'
    items.extend([
        (f'final status: {report.launch_status}', kind_for_status(report.launch_status)),
        (f'reason: {report.reason}', kind_for_status(report.launch_status)),
        (f'next action: {report.next_action}', 'normal'),
        ('', 'normal'),
        ('Binary', 'heading'),
        (f'engine id: {report.engine_id}', 'normal'),
        (f'command: {report.binary_command or "-"}', 'normal'),
        (f'source: {report.binary_source or "unknown"}', 'normal'),
        (f'exists: {yn(report.binary_exists)}  executable: {yn(report.binary_executable)}', 'success' if report.binary_exists and report.binary_executable else 'error'),
        (f'resolved path: {report.binary_resolved_path or "-"}', 'muted'),
    ])
    if checked:
        items.append((f'checked paths: {checked}', 'muted'))
    items.extend([
        ('', 'normal'),
        ('Capabilities', 'heading'),
        (f'help inspected: {yn(report.help_inspected)}', 'success' if report.help_inspected else 'warning'),
        (f'--spec-type present: {yn(report.supports_spec_type)}', 'success' if report.supports_spec_type else 'error'),
        (f'advertised spec values: {values}', 'normal' if report.spec_type_values else 'warning'),
        (f'selected spec value: {report.selected_spec_type or "none"}', 'success' if report.selected_spec_type else 'error'),
        (f'--spec-draft-n-max present: {yn(report.supports_spec_draft_n_max)}', 'success' if report.supports_spec_draft_n_max else 'error'),
        (f'supports MTP: {yn(report.supports_mtp)}', 'success' if report.supports_mtp else 'error'),
        (f'supports draft KV (--spec-draft-type-k/v): {yn(report.supports_draft_kv)}', 'success' if report.supports_draft_kv else 'muted'),
        (f'supports fit: {yn(report.supports_fit)}', 'success' if report.supports_fit else 'muted'),
        (f'supports no-mmap: {yn(report.supports_no_mmap)}', 'success' if report.supports_no_mmap else 'muted'),
        (f'supports cache-ram: {yn(report.supports_cache_ram)}', 'success' if report.supports_cache_ram else 'muted'),
        (f'supports no-warmup: {yn(report.supports_no_warmup)}', 'success' if report.supports_no_warmup else 'muted'),
        (f'supports parallel: {yn(report.supports_parallel)}', 'success' if report.supports_parallel else 'warning'),
        (f'supports cache flags: {yn(report.supports_cache_flags)}', 'success' if report.supports_cache_flags else 'muted'),
        ('', 'normal'),
        ('Selected Model', 'heading'),
        (f'model: {report.model_id}  {report.model_name}', 'normal'),
        (f'path: {report.model_path}', 'muted'),
        (f'detected features: {features}', 'normal'),
        (f'supports_mtp: {report.supports_mtp_setting}', 'normal'),
        (f'mtp_enabled: {yn(report.mtp_enabled)}  mtp_draft_n_max: {report.mtp_draft_n_max}', 'success' if report.mtp_enabled else 'muted'),
        (f'mmproj/vision detected: {yn(report.mmproj_detected)}', 'error' if report.mmproj_detected else 'success'),
        (f'model allowed for MTP: {yn(report.model_allowed)}', 'success' if report.model_allowed else 'warning'),
        (f'compatibility: {report.compatibility_status} - {report.compatibility_reason}', kind_for_status(report.compatibility_status)),
        ('', 'normal'),
        ('Final Command Preview', 'heading'),
        (f'--spec-type included: {yn(report.launch.includes_spec_type)}', 'success' if report.launch.includes_spec_type else 'muted'),
        (f'--spec-draft-n-max included: {yn(report.launch.includes_spec_draft_n_max)}', 'success' if report.launch.includes_spec_draft_n_max else 'muted'),
        (f'--parallel/-np included: {yn(report.launch.includes_parallel)}', 'success' if report.launch.includes_parallel else 'warning'),
        (f'--no-warmup included: {yn(report.launch.includes_no_warmup)}', 'success' if report.launch.includes_no_warmup else 'muted'),
        (f'--no-mmap included: {yn(report.launch.includes_no_mmap)}', 'success' if report.launch.includes_no_mmap else 'muted'),
        (f'cache flags included: {yn(report.launch.includes_cache_flags)}', 'success' if report.launch.includes_cache_flags else 'muted'),
        (f'command: {report.launch.command_preview or "-"}', 'normal' if report.launch.command_preview else 'warning'),
    ])
    for warning in report.launch.warnings:
        items.append((warning, 'warning'))
    return items


def show_mtp_doctor_overlay(stdscr, colors, app: AppConfig, active_model: Optional[ModelConfig] = None):
    h, w = stdscr.getmaxyx()
    box_w = min(112, max(64, w - 8))
    box_h = min(max(12, h - 6), 26)
    if h < 12 or w < 66:
        return
    box_x = max(2, (w - box_w) // 2)
    box_y = max(2, (h - box_h) // 2)
    modal = curses.newwin(box_h, box_w, box_y, box_x)
    modal.keypad(True)
    content_h = max(1, box_h - 4)
    rows = mtp_doctor_items(app, active_model=active_model)
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
        display_text = kind_status_prefix(text, kind)
        items.extend((line, attr) for line in wrap_display_item_lines(display_text, box_w - 4))
    scroll = 0
    stdscr.nodelay(False)
    try:
        while True:
            visible, scroll, _older, _newer, _total = scrollable_pane_item_view(items, box_w - 4, content_h, scroll)
            modal.erase()
            draw_box(modal, 0, 0, box_h - 1, box_w, 'MTP Doctor', colors['accent'] | curses.A_BOLD, colors['accent'])
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


def show_benchmark_plan_overlay(stdscr, colors, app: AppConfig, active_model: Optional[ModelConfig] = None):
    if active_model is None:
        return
    h, w = stdscr.getmaxyx()
    box_w = min(112, max(64, w - 8))
    box_h = min(max(12, h - 6), 26)
    if h < 12 or w < 66:
        return
    box_x = max(2, (w - box_w) // 2)
    box_y = max(2, (h - box_h) // 2)
    modal = curses.newwin(box_h, box_w, box_y, box_x)
    modal.keypad(True)
    content_h = max(1, box_h - 4)
    items = []
    for text, kind in benchmark_plan_lines(app, active_model, depth='fast'):
        attr = kind_style(colors, kind)
        display_text = kind_status_prefix(text, kind)
        items.extend((line, attr) for line in wrap_display_item_lines(display_text, box_w - 4))
    scroll = 0
    stdscr.nodelay(False)
    try:
        while True:
            visible, scroll, _older, _newer, _total = scrollable_pane_item_view(items, box_w - 4, content_h, scroll)
            modal.erase()
            draw_box(modal, 0, 0, box_h - 1, box_w, 'Benchmark Plan Preview', colors['accent'] | curses.A_BOLD, colors['accent'])
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


LLAMA_CPP_FAMILY_ENGINES = ('llama.cpp', 'llama.cpp-mtp', 'turboquant')


def _active_engine_for_menu(app: Optional[AppConfig], model: Optional[ModelConfig]) -> str:
    if app is not None and model is not None and hasattr(app, 'active_engine_key_for_model'):
        try:
            return str(app.active_engine_key_for_model(model) or '')
        except (AttributeError, TypeError, OSError, ValueError):
            return ''
    return str(getattr(model, 'runtime', '') or '')


def _moe_menu_disabled_reason(app: Optional[AppConfig], model: Optional[ModelConfig]) -> str:
    if model is None:
        return 'no model selected'
    if not str(getattr(model, 'path', '') or '').lower().endswith('.gguf'):
        return 'GGUF model required'
    if not model_is_moe(model):
        return 'model is not detected as MoE'
    engine = _active_engine_for_menu(app, model).strip().lower()
    if engine not in LLAMA_CPP_FAMILY_ENGINES:
        return f'active engine {engine or "-"} is not eligible'
    return ''


def benchmark_menu_recommendation(app: Optional[AppConfig], model: Optional[ModelConfig]) -> Tuple[str, str]:
    if model is None:
        return 'quick_benchmark', 'select a model first'
    if has_moe_recommendation(model) and not moe_recommendation_applied(model):
        return 'apply_moe_recommendation', 'measured MoE placement exists; press A in the Tuning tab before routine benchmarks'
    status = str(getattr(model, 'default_benchmark_status', '') or '').strip().lower()
    moe_disabled = _moe_menu_disabled_reason(app, model)
    if not moe_disabled and not has_moe_recommendation(model):
        if _active_engine_for_menu(app, model).strip().lower() == 'llama.cpp-mtp':
            return 'full_suite', 'MTP acceptance and MoE placement have not both been measured'
        return 'full_suite', 'MoE placement has not been measured; the suite can tune it before downstream benchmarks'
    if status in ('failed', 'aborted'):
        return 'smart_benchmark', f'last benchmark status is {status}'
    if not float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0):
        return 'smart_benchmark', 'no measured launch profile is saved'
    if not float(getattr(model, 'last_opencode_benchmark_score', 0.0) or 0.0):
        return 'opencode_benchmark', 'benchmark exists, but OpenCode has not been validated'
    return 'quick_benchmark', 'current benchmark proof exists; quick check is enough'


def benchmark_menu_intro_lines(app: Optional[AppConfig], model: Optional[ModelConfig]) -> List[str]:
    if model is None:
        return ['Selected model: none']
    recommended, reason = benchmark_menu_recommendation(app, model)
    try:
        strategy = benchmark_strategy_for_app(app, model, depth='fast', objective='quick_sanity')
        if getattr(strategy, 'blocked_reason', ''):
            strategy_line = f'Strategy: {strategy.id} blocked - {compact_message(strategy.blocked_reason)}'
        else:
            strategy_line = f'Strategy: {strategy.id} ({strategy.hard_budget_seconds // 60}m, {len(strategy.phases)} phases)'
    except Exception:
        strategy_line = 'Strategy: unavailable'
    label_by_value = {
        value: label
        for _key, label, value in benchmark_menu_options(app, model, include_recommendation=False)
        if not value.startswith('disabled:')
    }
    label = label_by_value.get(recommended, recommended.replace('_', ' '))
    return [
        f'Selected: {getattr(model, "name", "") or getattr(model, "id", "-")}',
        f'Type: {classify_model_type(model)}   Engine: {_active_engine_for_menu(app, model) or "-"}',
        strategy_line,
        f'Recommended: {label}',
        f'Reason: {reason}',
    ]


def benchmark_menu_options(
    app: Optional[AppConfig],
    model: Optional[ModelConfig],
    include_recommendation: bool = True,
) -> List[Tuple[str, str, str]]:
    recommended, _reason = benchmark_menu_recommendation(app, model) if include_recommendation else ('', '')

    def label(value: str, text: str) -> str:
        return f'{text}  (Recommended)' if value == recommended else text

    moe_reason = _moe_menu_disabled_reason(app, model)
    moe_value = f'disabled:moe:{moe_reason}' if moe_reason else 'moe_tuning_full'
    moe_label = 'MoE Placement Tuning - expert CPU/GPU placement'
    if moe_reason:
        moe_label = f'MoE Placement Tuning - unavailable: {moe_reason}'
    mtp_suite = model is not None and _active_engine_for_menu(app, model).strip().lower() == 'llama.cpp-mtp'
    full_suite_label = (
        'MTP Suite - Acceptance + MoE Tuning'
        if mtp_suite
        else 'Full Suite Benchmark - MoE -> Smart -> Hermes -> OpenCode'
    )
    return [
        ('1', label('quick_benchmark', 'Quick Benchmark - fast sanity check'), 'quick_benchmark'),
        ('2', label('smart_benchmark', 'Smart Benchmark - speed/context profiles'), 'smart_benchmark'),
        ('3', label('moe_tuning_full', moe_label), moe_value),
        ('4', label('hermes_benchmark', 'Hermes Benchmark - workflow validation'), 'hermes_benchmark'),
        ('5', label('opencode_benchmark', 'OpenCode Benchmark - workflow validation'), 'opencode_benchmark'),
        ('6', label('full_suite', full_suite_label), 'full_suite'),
        ('q', 'Cancel', 'cancel'),
    ]


def prompt_benchmark_menu(stdscr, colors, app: AppConfig, model: ModelConfig) -> str:
    return prompt_modal_choice(
        stdscr,
        colors,
        'Benchmark Menu',
        benchmark_menu_options(app, model),
        intro_lines=benchmark_menu_intro_lines(app, model),
        footer='[Up/Down] Select   Enter Run   Esc Cancel',
    )


def command_palette_options(app: Optional[AppConfig] = None, model: Optional[ModelConfig] = None) -> List[Tuple[str, str, str]]:
    current_density = normalize_choice(getattr(getattr(app, 'ui', None), 'detail_density', 'simple'), tuple(key for key, _label in DETAIL_DENSITY_OPTIONS), 'simple')
    next_density = 'advanced' if current_density == 'simple' else 'simple'
    current_browser = normalize_choice(getattr(getattr(app, 'ui', None), 'browser_view', 'compact'), tuple(key for key, _label in BROWSER_VIEW_OPTIONS), 'compact')
    next_browser = 'advanced' if current_browser == 'compact' else 'compact'
    apply_moe_value = 'apply_moe_recommendation' if model is not None and has_moe_recommendation(model) else 'disabled:apply_moe:Run MoE placement tuning first'
    apply_moe_label = 'Apply MoE Recommendation' if apply_moe_value == 'apply_moe_recommendation' else 'Apply MoE Recommendation - run MoE tuning first'
    return [
        ('1', 'Search models', 'search'),
        ('2', 'Set filters', 'filters'),
        ('3', 'Sort models', 'sort'),
        ('4', 'Toggle favorite', 'favorite'),
        ('5', f'Detail Density: {detail_density_label(current_density)} -> {detail_density_label(next_density)}', 'density'),
        ('6', f'Browser View: {browser_view_label(current_browser)} -> {browser_view_label(next_browser)}', 'toggle_browser_view'),
        ('7', 'Open help', 'help'),
        ('8', 'Settings...', 'settings'),
        ('9', 'Detect models', 'detect'),
        ('p', 'Raw Speed Benchmark...', 'raw_speed_benchmark'),
        ('n', 'Tune MoE Placement (fast)', 'moe_tuning_fast'),
        ('u', 'Tune MoE Placement (full)', 'moe_tuning_full'),
        ('r', apply_moe_label, apply_moe_value),
        ('m', 'Compare with machine pick', 'compare'),
        ('o', 'Export OpenCode config', 'export_opencode'),
        ('c', 'Export Continue config', 'export_continue'),
        ('g', 'Export Hermes config', 'export_hermes'),
        ('v', 'Verify selected model', 'verify_selected'),
        ('a', 'Verify all benchmark proof', 'verify_all'),
        ('!', 'Config Doctor...', 'config_doctor'),
        ('t', 'MTP Doctor...', 'mtp_doctor'),
        ('s', 'Start / stop selected model', 'start_stop'),
        ('j', 'Launch / Try selected model', 'launch'),
        ('b', 'Benchmark selected model', 'benchmark_menu'),
        ('e', 'Benchmark plan preview...', 'benchmark_plan'),
        ('z', 'Apply measured profile', 'apply_profile'),
        ('l', 'Open logs', 'open_logs'),
        ('q', 'Cancel', 'cancel'),
    ]


def prompt_command_palette(stdscr, colors, app: Optional[AppConfig] = None, model: Optional[ModelConfig] = None) -> str:
    return prompt_modal_choice(stdscr, colors, 'Actions', command_palette_options(app, model))


def prompt_modal_choice(
    stdscr,
    colors,
    title: str,
    options: List[Tuple[str, str, str]],
    intro_lines: Optional[List[str]] = None,
    footer: str = 'Press key to run. Up/Down scroll. Esc cancels.',
) -> str:
    h, w = stdscr.getmaxyx()
    box_w = min(68, max(48, w - 8))
    intro = [str(line) for line in list(intro_lines or []) if str(line).strip()]
    box_h = min(max(8, len(options) + 6 + len(intro)), max(8, h - 4))
    with open_modal(stdscr, box_h=box_h + 1, box_w=box_w, min_h=12, min_w=box_w + 4) as session:
        if session is None:
            return 'cancel'
        modal = session.window
        scroll = 0
        selected_idx = 0
        visible_rows = max(1, box_h - 5 - len(intro))
        while True:
            selected_idx = max(0, min(selected_idx, max(0, len(options) - 1)))
            if selected_idx < scroll:
                scroll = selected_idx
            if selected_idx >= scroll + visible_rows:
                scroll = selected_idx - visible_rows + 1
            scroll = clamp_scroll(scroll, len(options), visible_rows)
            modal.erase()
            draw_box(modal, 0, 0, box_h - 1, box_w, title, colors['accent'] | curses.A_BOLD, colors['accent'])
            y = 2
            for line in intro:
                safe_addstr(modal, y, 2, ellipsize(line, box_w - 4), colors['panel'])
                y += 1
            safe_addstr(modal, y, 2, ellipsize('Choose an action:', box_w - 4), colors['panel'] | curses.A_BOLD)
            y += 1
            for row, (option_key, label, _val) in enumerate(options[scroll: scroll + visible_rows]):
                absolute = scroll + row
                marker = ''
                if row == 0 and scroll > 0:
                    marker = '^ '
                elif row == visible_rows - 1 and scroll + visible_rows < len(options):
                    marker = 'v '
                selected_marker = '> ' if absolute == selected_idx else '  '
                value = str(_val or '')
                attr = colors['muted'] if value.startswith('disabled:') else colors['selection'] if absolute == selected_idx else colors['panel']
                safe_addstr(modal, y + row, 2, ellipsize(f'{selected_marker}{marker}[{option_key}] {label}', box_w - 4), attr)
            safe_addstr(modal, box_h - 1, 2, ellipsize(footer, box_w - 6), colors['muted'])
            modal.refresh()
            key = modal.getch()
            if key == -1:
                continue
            if key in (27, ord('q')):
                return 'cancel'
            if key in (curses.KEY_UP, ord('k')):
                selected_idx -= 1
                continue
            if key in (curses.KEY_DOWN, ord('j')):
                selected_idx += 1
                continue
            if key in (curses.KEY_ENTER, 10, 13):
                return options[selected_idx][2] if options else 'cancel'
            key_str = chr(key).lower() if 0 <= key <= 255 else ''
            for option_key, _label, value in options:
                if key_str == option_key:
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

