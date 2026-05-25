"""Model-list and engine display helpers for the curses UI.

Extracted from ui.py so the model browser / overview rendering lives in one
focused module. Pure-ish: depends only on lower-level modules (never on
ui.py), which keeps imports acyclic.
"""

from typing import Dict, Optional, Tuple

from .app import AppConfig, context_per_slot
from .benchmark import benchmark_profile_is_fresh, get_measured_profile
from .discovery import classify_model_type, display_runtime, extract_quant
from .engines import resolve_runtime_engine_context
from .gguf import turboquant_short
from .models import ModelConfig
from .mtp import mtp_support_auto_hint, normalize_mtp_support
from .textutil import ellipsize


BROWSER_HEADER = ' ID              PRT  ST        BN    RLS  ENGINE     QNT      TQ  ARCH   NAME'
BROWSER_VIEW_OPTIONS = [('compact', 'Compact'), ('advanced', 'Advanced')]
BENCHMARK_FRESHNESS_LABELS = {
    'fresh': 'Fresh',
    'stale': 'Stale',
    'missing': 'Missing',
    'failed': 'Failed',
    'pending': 'Pending',
    'running': 'Running',
}


def _is_advanced_view(browser_view: str) -> bool:
    return str(browser_view or '').strip().lower() == 'advanced'


def format_model_state(status: str) -> str:
    normalized = str(status or '').strip().upper()
    if normalized == 'READY':
        return 'running'
    if normalized in ('LOADING', 'STARTING'):
        return 'starting'
    if normalized == 'ERROR':
        return 'error'
    return 'stopped'


def status_symbol(status: str) -> str:
    symbols = {
        'READY': '●',
        'LOADING': '◐',
        'STARTING': '◔',
        'STOPPED': '○',
        'ERROR': '✖',
    }
    return symbols.get(status, '·')


def format_engine_badge(engine_id: str, narrow: bool = False) -> str:
    normalized = str(engine_id or '').strip().lower()
    if normalized == 'turboquant':
        return 'TQ' if narrow else 'TurboQuant+'
    if normalized == 'llama.cpp-mtp':
        return 'MTP' if narrow else 'llama.cpp MTP'
    if normalized == 'llama.cpp':
        return 'llama' if narrow else 'llama.cpp'
    return '?' if narrow else 'Unknown'


def active_engine_key(app: AppConfig, model: ModelConfig) -> str:
    try:
        return str(app.active_engine_key_for_model(model) or '')
    except Exception:
        return str(getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp')


def active_engine_short(app: AppConfig, model: ModelConfig) -> str:
    engine = active_engine_key(app, model)
    if engine == 'turboquant':
        return 'turboquant'
    if engine == 'llama.cpp-mtp':
        return 'llama.cpp-mtp'
    return engine or display_runtime(model)


def active_engine_binary(app: AppConfig, model: ModelConfig) -> str:
    try:
        return str(app.active_runtime_binary_for_model(model) or '')
    except Exception:
        runtime = getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp'
        try:
            return str(app.runtime_server_command(runtime) or '')
        except Exception:
            return ''


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


def active_engine_warning_line(app: AppConfig, model: ModelConfig) -> str:
    messages = []
    for name in (
        'turboquant_session_advisory',
        'turboquant_binary_warning',
        'mtp_session_advisory',
        'mtp_binary_warning',
    ):
        try:
            value = str(getattr(app, name)(model) or '')
        except Exception:
            value = ''
        if value:
            messages.append(value)
    return ' | '.join(messages)


def _measured_profile_for_recommendation(model: ModelConfig) -> Tuple[str, Dict[str, object]]:
    for key in ('opencode_ready', 'fast_chat', 'long_context', 'auto'):
        profile = get_measured_profile(model, key)
        if profile:
            return key, profile
    return '', {}


def _profile_context(profile: Dict[str, object]) -> int:
    for key in ('ctx_per_slot', 'ctx'):
        try:
            value = int(profile.get(key, 0) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def mtp_status_from_measured(measured: Dict[str, object]) -> str:
    """Map a measured ``mtp_acceptance`` profile to a short status word.

    Pure helper (no app/binary access) so it is unit testable.
    """
    if not measured:
        return ''
    status = str(measured.get('status', '') or '').strip().lower()
    risk = str(measured.get('mtp_risk_level', '') or '').strip().lower()
    if status in ('ok', 'complete', 'partial'):
        if risk in ('excellent', 'good', 'usable'):
            return 'usable'
        if risk == 'risky':
            return 'risky'
        return 'usable'
    if status:
        return 'blocked'
    return ''


def mtp_status_short(app: Optional[AppConfig], model: ModelConfig) -> str:
    """Capability-driven MTP status for a model row.

    Reports off / unsupported / capable / ready / usable / risky / blocked
    based on the resolved binary's MTP capability and any measured results --
    never on the engine name being ``llama.cpp-mtp``.
    """
    if app is None:
        return 'off'
    support = normalize_mtp_support(getattr(model, 'supports_mtp', 'auto'))
    if support == 'no':
        return 'unsupported'
    measured = dict((getattr(model, 'measured_profiles', {}) or {}).get('mtp_acceptance') or {})
    mtp_relevant = (
        support == 'yes'
        or mtp_support_auto_hint(model)
        or bool(getattr(model, 'mtp_enabled', False))
        or bool(measured)
    )
    if not mtp_relevant:
        return 'off'
    measured_status = mtp_status_from_measured(measured)
    if measured_status:
        return measured_status
    try:
        context = resolve_runtime_engine_context(app, model=model)
        binary_supports_mtp = bool(getattr(context, 'supports_mtp', False))
    except Exception:
        binary_supports_mtp = False
    if binary_supports_mtp:
        return 'ready' if bool(getattr(model, 'mtp_enabled', False)) else 'capable'
    return 'unsupported'


def format_model_recommendation(app: AppConfig, model: ModelConfig) -> str:
    if benchmark_profile_is_fresh(app, model):
        key, _profile = _measured_profile_for_recommendation(model)
        labels = {
            'opencode_ready': 'OpenCode',
            'fast_chat': 'Fast Chat',
            'long_context': 'Long Ctx',
            'auto': 'Auto',
        }
        if key:
            return labels.get(key, key.replace('_', ' ').title())
    if (getattr(model, 'optimize_mode', '') or '').strip().lower() == 'manual':
        return 'Manual'
    return 'Needs Bench'


def format_model_health(app: AppConfig, model: ModelConfig, status: str = 'STOPPED') -> Tuple[str, str]:
    try:
        valid_target, target_reason = app.validate_model_target(model)
    except Exception:
        valid_target, target_reason = True, ''
    if not valid_target:
        return 'FAIL', target_reason

    try:
        engine_command = app.active_runtime_binary_for_model(model)
        engine_exists = app.command_exists(engine_command)
    except Exception:
        engine_command, engine_exists = '', True
    if not engine_exists:
        return 'FAIL', f'engine missing: {engine_command or "-"}'

    verification = (getattr(model, 'verification_status', '') or '').strip().lower()
    if verification == 'failed':
        return 'FAIL', getattr(model, 'verification_summary', '') or 'verification failed'

    freshness = benchmark_freshness_label(app, model)
    if freshness == 'failed' and not _measured_profile_for_recommendation(model)[1]:
        return 'FAIL', 'last benchmark failed'
    if freshness == 'stale':
        return 'STALE', 'benchmark stale'

    try:
        engine = active_engine_key(app, model)
        capabilities = app.engine_capabilities()
        if not str(getattr(capabilities, 'help_text', '') or '').strip():
            return 'WARN', 'engine capabilities unknown'
    except Exception:
        pass

    warning = active_engine_warning_line(app, model)
    if warning:
        return 'WARN', warning

    arch = (getattr(model, 'architecture_type', '') or '').strip().lower()
    if arch in ('', 'unknown'):
        return 'WARN', 'metadata unknown'
    if not _measured_profile_for_recommendation(model)[1]:
        return 'WARN', 'no measured profile'
    return 'OK', 'ready'


def build_model_row_summary(app: AppConfig, model: ModelConfig, status: str = 'STOPPED') -> Dict[str, object]:
    recommendation = format_model_recommendation(app, model)
    _key, measured = _measured_profile_for_recommendation(model)
    measured_ctx = _profile_context(measured)
    ctx = measured_ctx or context_per_slot(model)
    try:
        tok_s = float(measured.get('tokens_per_sec', getattr(model, 'last_benchmark_tokens_per_sec', 0.0)) or 0.0)
    except Exception:
        tok_s = 0.0
    engine_id = active_engine_key(app, model)
    health, health_reason = format_model_health(app, model, status)
    return {
        'display_name': getattr(model, 'name', '') or getattr(model, 'id', '') or '-',
        'state': format_model_state(status),
        'pick': recommendation,
        'ctx': ctx,
        'tokens_per_sec': tok_s,
        'engine': format_engine_badge(engine_id),
        'health': health,
        'health_reason': health_reason,
        'mtp': mtp_status_short(app, model),
    }


def compact_browser_header(left_w: int) -> str:
    return compact_browser_model_line(None, None, '', '', left_w, header=True)


def compact_browser_model_line(
    app: Optional[AppConfig],
    model: Optional[ModelConfig],
    status: str,
    machine_pick_id: str,
    left_w: int,
    header: bool = False,
) -> str:
    name_w = max(12, int(left_w or 80) - 62)
    if header:
        return (
            f' {"MODEL":{name_w}} {"STATE":8} {"PICK":11} '
            f'{"CTX":>7} {"TOK/S":>7} {"ENGINE":11} {"MTP":7} {"HEALTH":6}'
        )
    assert app is not None and model is not None
    summary = build_model_row_summary(app, model, status)
    display_name = str(summary['display_name'])
    if getattr(model, 'favorite', False) and name_w >= 14:
        display_name = '* ' + display_name
    if model.id == machine_pick_id and name_w >= 15:
        display_name = f'{display_name} BEST'
    ctx = int(summary.get('ctx', 0) or 0)
    ctx_text = str(ctx) if ctx > 0 else '-'
    tok_s = float(summary.get('tokens_per_sec', 0.0) or 0.0)
    tok_text = f'{tok_s:.1f}' if tok_s > 0 else '-'
    return (
        f' {ellipsize(display_name, name_w):{name_w}} '
        f'{str(summary["state"])[:8]:8} '
        f'{str(summary["pick"])[:11]:11} '
        f'{ctx_text:>7} {tok_text:>7} '
        f'{str(summary["engine"])[:11]:11} '
        f'{str(summary.get("mtp", "off"))[:7]:7} '
        f'{str(summary["health"])[:6]:6}'
    )


def browser_model_line(
    app: AppConfig,
    model: ModelConfig,
    status: str,
    machine_pick_id: str,
    left_w: int,
) -> str:
    roles = app.role_badges(model.id)
    engine = active_engine_short(app, model)[:10]
    quant = extract_quant(model)[:8]
    tq = turboquant_short(model)[:3]
    model_type = classify_model_type(model)[:6]
    freshness = benchmark_freshness_short(app, model)
    name_col_width = max(10, left_w - 79)
    favorite_prefix = '★ ' if getattr(model, 'favorite', False) and name_col_width >= 14 else ''
    best_badge = ' BEST' if model.id == machine_pick_id and name_col_width >= 15 else ''
    display_name = favorite_prefix + (model.name or model.id)
    display_name = display_name[: max(1, name_col_width - len(best_badge))] + best_badge
    return (
        f' {model.id[:14]:14} {model.port:4}  {status_symbol(status)} {status[:6]:6}  '
        f'{freshness:4}  {roles:3}  {engine:10} {quant:8} {tq:3} {model_type:6} {display_name}'
    )


def browser_header_for_view(browser_view: str, left_w: int) -> str:
    if _is_advanced_view(browser_view):
        return BROWSER_HEADER
    return compact_browser_header(left_w)


def browser_model_line_for_view(
    app: AppConfig,
    model: ModelConfig,
    status: str,
    machine_pick_id: str,
    left_w: int,
    browser_view: str,
) -> str:
    if _is_advanced_view(browser_view):
        return browser_model_line(app, model, status, machine_pick_id, left_w)
    return compact_browser_model_line(app, model, status, machine_pick_id, left_w)
