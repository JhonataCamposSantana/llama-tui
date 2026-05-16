import shlex
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from .engines import (
    ENGINE_VLLM,
    engine_display_name,
    resolve_runtime_engine_context,
)
from .model_compat import detect_model_runtime_features, engine_supports_model
from .mtp import (
    clamp_mtp_draft,
    model_has_mmproj_config,
    model_mtp_allowed,
    model_mtp_enabled,
    model_mtp_draft_n_max,
    normalize_mtp_support,
)
from .runtime_profiles import EngineCapabilities, build_mtp_args, mtp_spec_type_value


@dataclass(frozen=True)
class MtpStatusSummary:
    status: str
    reason: str
    next_action: str
    risk_level: str = 'info'


@dataclass(frozen=True)
class MtpLaunchDiagnostics:
    selected_spec_type: str = ''
    draft_n_max: int = 0
    added_flags: Tuple[str, ...] = ()
    skipped_flags: Tuple[str, ...] = ()
    command_preview: str = ''
    includes_spec_type: bool = False
    includes_spec_draft_n_max: bool = False
    includes_parallel: bool = False
    includes_no_warmup: bool = False
    includes_no_mmap: bool = False
    includes_cache_flags: bool = False
    warnings: Tuple[str, ...] = ()
    blocked_reason: str = ''


@dataclass(frozen=True)
class MtpDoctorReport:
    engine_id: str
    binary_command: str
    binary_source: str
    binary_exists: bool
    binary_executable: bool
    binary_resolved_path: str
    checked_paths: Tuple[str, ...]
    help_inspected: bool
    supports_spec_type: bool
    spec_type_values: Tuple[str, ...]
    selected_spec_type: str
    supports_spec_draft_n_max: bool
    supports_mtp: bool
    supports_no_warmup: bool
    supports_parallel: bool
    supports_cache_flags: bool
    supports_draft_kv: bool
    supports_fit: bool
    supports_no_mmap: bool
    supports_cache_ram: bool
    model_id: str
    model_name: str
    model_path: str
    detected_features: Tuple[str, ...]
    supports_mtp_setting: str
    mtp_enabled: bool
    mtp_draft_n_max: int
    mmproj_detected: bool
    model_allowed: bool
    compatibility_status: str
    compatibility_reason: str
    launch_status: str
    reason: str
    next_action: str
    risk_level: str
    launch: MtpLaunchDiagnostics


def _command_contains(tokens: Sequence[str], *flags: str) -> bool:
    lowered = tuple(str(item or '').strip().lower() for item in tokens)
    for flag in flags:
        low = str(flag or '').strip().lower()
        if low in lowered:
            return True
        if any(item.startswith(f'{low}=') for item in lowered):
            return True
    return False


def _command_value(tokens: Sequence[str], *flags: str) -> str:
    lowered = [str(item or '').strip() for item in tokens]
    flag_set = {str(flag or '').strip() for flag in flags if str(flag or '').strip()}
    for idx, token in enumerate(lowered):
        if token in flag_set and idx + 1 < len(lowered):
            return lowered[idx + 1]
        for flag in flag_set:
            prefix = f'{flag}='
            if token.startswith(prefix):
                return token[len(prefix):]
    return ''


def _add_unique(items: list, text: str):
    if text and text not in items:
        items.append(text)


def _mtp_profile_warnings(model: Any, runtime_profile: Any, command_tokens: Sequence[str]) -> Tuple[str, ...]:
    warnings = []
    cmd_has_fit = _command_contains(command_tokens, '-fit', '--fit')
    cmd_has_ngl = _command_contains(command_tokens, '-ngl', '--n-gpu-layers')
    if cmd_has_fit and cmd_has_ngl:
        _add_unique(warnings, 'MTP fit was blocked because --n-gpu-layers was fixed. Use fit-assisted MTP profile.')
    ctx_value = _command_value(command_tokens, '--ctx-size', '-c')
    try:
        ctx = int(ctx_value or getattr(runtime_profile, 'ctx_size', 0) or getattr(model, 'ctx', 0) or 0)
    except Exception:
        ctx = 0
    key_kv = (_command_value(command_tokens, '-ctk', '--cache-type-k') or str(getattr(runtime_profile, 'kv_preset', '') or '')).strip().lower()
    value_kv = (_command_value(command_tokens, '-ctv', '--cache-type-v') or key_kv).strip().lower()
    if ctx >= 131072 and (key_kv in ('', 'default', 'f16') or value_kv in ('', 'default', 'f16')):
        _add_unique(warnings, 'MTP 128k is using default/f16 KV; q8_0 target and draft KV may be required for this hardware.')
    has_cpu_override = (
        _command_contains(command_tokens, '-ot', '--override-tensor', '--override-tensors', '-cmoe', '--cpu-moe', '-ncmoe', '--n-cpu-moe')
        or bool(getattr(runtime_profile, 'cpu_moe', False))
        or int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0) > 0
        or bool(tuple(getattr(runtime_profile, 'tensor_overrides', ()) or ()))
    )
    if has_cpu_override and not _command_contains(command_tokens, '--no-mmap'):
        _add_unique(warnings, 'MTP model uses CPU tensor overrides with mmap enabled; --no-mmap may improve performance.')

    profiles = dict(getattr(model, 'measured_profiles', {}) or {})
    records = [dict(item) for item in list(getattr(model, 'last_benchmark_results', []) or []) if isinstance(item, dict)]
    records.extend(dict(item) for item in profiles.values() if isinstance(item, dict))
    for record in records:
        category = str(record.get('failure_category', '') or '')
        if category in ('FIXED_GPU_LAYERS_BLOCKED_FIT', 'FIXED_GPU_LAYERS_FIT_FAILED'):
            _add_unique(warnings, 'MTP fit was blocked because --n-gpu-layers was fixed. Use fit-assisted MTP profile.')
        warning = str(record.get('warning', '') or '')
        runtime_warnings = {str(item) for item in list(record.get('runtime_warnings', []) or [])}
        if warning == 'MMAP_CPU_OVERRIDE_SLOWPATH' or 'MMAP_CPU_OVERRIDE_SLOWPATH' in runtime_warnings:
            _add_unique(warnings, 'MTP model uses CPU tensor overrides with mmap enabled; --no-mmap may improve performance.')
        try:
            acceptance = float(record.get('accept_rate', record.get('acceptance_rate', -1.0)) or -1.0)
        except Exception:
            acceptance = -1.0
        if bool(record.get('mtp_enabled')) and 0.0 <= acceptance < 0.70:
            _add_unique(warnings, 'Draft acceptance below 70%; this draft_n may be too aggressive.')
        kv = str(record.get('kv_preset', '') or '').lower()
        draft_kv = str(record.get('mtp_draft_kv_preset', '') or '').lower()
        if (
            bool(record.get('mtp_enabled'))
            and str(record.get('status', '') or '') == 'ok'
            and float(record.get('tokens_per_sec', 0.0) or 0.0) > 0.0
            and acceptance >= 0.70
            and bool(record.get('no_mmap', record.get('runtime_no_mmap', False)))
            and (kv == 'q8_0/q8_0' or (str(record.get('ctk', '') or '').lower() == 'q8_0' and str(record.get('ctv', '') or '').lower() == 'q8_0'))
            and (draft_kv == 'q8_0/q8_0' or (str(record.get('draft_ctk', '') or '').lower() == 'q8_0' and str(record.get('draft_ctv', '') or '').lower() == 'q8_0'))
        ):
            _add_unique(
                warnings,
                f'MTP fit-assisted q8/no-mmap profile is usable: {float(record.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s, {acceptance:.0%} acceptance.',
            )
    return tuple(warnings)


def _build_launch_diagnostics(
    app: Any,
    model: Any,
    runtime_profile: Any,
    capabilities: EngineCapabilities,
    blocked_reason: str = '',
) -> MtpLaunchDiagnostics:
    warnings = []
    command_tokens = []
    command_preview = ''
    effective_profile = runtime_profile
    if effective_profile is None:
        try:
            effective_profile = app.runtime_profile_from_model(
                model,
                int(getattr(model, 'ctx', 0) or 0),
                int(getattr(model, 'parallel', 1) or 1),
                int(getattr(model, 'ngl', 0) or 0),
            )
        except Exception:
            effective_profile = None
    _mtp_args, mtp_arg_diagnostics = build_mtp_args(model, effective_profile, capabilities)
    try:
        command_tokens = list(app.build_command(model, runtime_profile=runtime_profile))
        command_preview = shlex.join(str(item) for item in command_tokens)
    except Exception as exc:
        warnings.append(f'command preview unavailable: {str(exc).splitlines()[0]}')
    warnings.extend(mtp_arg_diagnostics.warnings)
    warnings.extend(_mtp_profile_warnings(model, effective_profile, command_tokens))
    selected = mtp_arg_diagnostics.selected_spec_type or mtp_spec_type_value(capabilities)
    return MtpLaunchDiagnostics(
        selected_spec_type=selected,
        draft_n_max=mtp_arg_diagnostics.draft_n_max,
        added_flags=mtp_arg_diagnostics.added_flags,
        skipped_flags=mtp_arg_diagnostics.skipped_flags,
        command_preview=command_preview,
        includes_spec_type=_command_contains(command_tokens, '--spec-type'),
        includes_spec_draft_n_max=_command_contains(command_tokens, '--spec-draft-n-max'),
        includes_parallel=_command_contains(command_tokens, '--parallel', '-np'),
        includes_no_warmup=_command_contains(command_tokens, '--no-warmup'),
        includes_no_mmap=_command_contains(command_tokens, '--no-mmap'),
        includes_cache_flags=_command_contains(command_tokens, '--cache-type-k', '--cache-type-v', '-ctk', '-ctv'),
        warnings=tuple(warnings),
        blocked_reason=mtp_arg_diagnostics.blocked_reason or blocked_reason,
    )


def _capability_block_reason(install: Any, capabilities: EngineCapabilities) -> str:
    if not bool(getattr(install, 'exists', False)):
        return 'binary not found'
    if not bool(getattr(install, 'executable', False)):
        return 'binary not executable'
    if not str(getattr(capabilities, 'help_text', '') or '').strip():
        return 'help output unavailable'
    if not bool(getattr(capabilities, 'supports_spec_type', False)):
        return 'missing --spec-type'
    if not mtp_spec_type_value(capabilities):
        return 'missing mtp/draft-mtp value'
    if not bool(getattr(capabilities, 'supports_spec_draft_n_max', False)):
        return 'missing --spec-draft-n-max'
    return ''


def _measured_mtp_status(model: Any) -> Tuple[str, str, str]:
    profiles: Dict[str, Any] = dict(getattr(model, 'measured_profiles', {}) or {})
    profile = dict(profiles.get('mtp_acceptance') or {})
    records = [
        dict(item) for item in list(getattr(model, 'last_benchmark_results', []) or [])
        if isinstance(item, dict)
    ]
    records.extend(dict(item) for item in profiles.values() if isinstance(item, dict))
    for item in records:
        category = str(item.get('failure_category', '') or '').strip()
        if category in ('FIXED_GPU_LAYERS_BLOCKED_FIT', 'FIXED_GPU_LAYERS_FIT_FAILED'):
            return 'fit blocked', 'MTP fit was blocked because --n-gpu-layers was fixed.', 'Use fit-assisted MTP profile.'
    for item in records:
        category = str(item.get('failure_category', '') or '').strip()
        if category in ('MEMORY_FIT_FAILED', 'CUDA_OOM_KV', 'CUDA_OOM_WEIGHTS', 'MEMORY_GUARDRAIL'):
            return 'memory-bound', f'MTP profile hit {category}.', 'Try q8 target/draft KV, fit-assisted placement, and --no-mmap.'
    if not profile:
        return '', '', ''
    status = str(profile.get('status', '') or '').strip().lower()
    if status not in ('ok', 'complete', 'partial'):
        return 'failed', f'latest MTP acceptance status is {status or "unknown"}', 'Run MTP Optimizer again after fixing the failure.'
    accept_rate = profile.get('accept_rate', profile.get('acceptance_rate', None))
    try:
        acceptance = float(accept_rate)
    except (TypeError, ValueError):
        acceptance = -1.0
    draft = int(profile.get('mtp_draft_n_max', profile.get('draft_n', 0)) or 0)
    if acceptance >= 0.75:
        return 'usable', f'MTP acceptance is usable at draft_n={draft or "-"} accept={acceptance:.0%}', 'Use the measured MTP profile or run MTP Optimizer for more workloads.'
    if acceptance >= 0.0:
        return 'risky acceptance', f'MTP acceptance is low at draft_n={draft or "-"} accept={acceptance:.0%}', 'Keep MTP opt-in and run the MTP Optimizer before saving a profile.'
    return 'usable', f'MTP acceptance has a usable {status} record', 'Review MTP Optimizer details before promoting the profile.'


def build_mtp_doctor_report(app: Any, model: Any, runtime_profile: Any = None) -> MtpDoctorReport:
    # MTP is a capability of whichever llama.cpp-compatible binary is resolved
    # for this launch, not a property of one dedicated engine. Resolve the
    # actual engine/binary/capabilities instead of assuming llama.cpp-mtp.
    engine_context = resolve_runtime_engine_context(app, model=model, runtime_profile=runtime_profile)
    engine_id = engine_context.engine_id
    install = engine_context.install
    command = engine_context.command
    capabilities = engine_context.capabilities
    selected_spec = engine_context.selected_mtp_spec_type
    features = tuple(sorted(detect_model_runtime_features(model)))
    support_setting = normalize_mtp_support(getattr(model, 'supports_mtp', 'auto'))
    mtp_enabled = model_mtp_enabled(model, runtime_profile)
    draft_n = model_mtp_draft_n_max(model, runtime_profile) if mtp_enabled else clamp_mtp_draft(getattr(model, 'mtp_draft_n_max', 3), default=3)
    mmproj = model_has_mmproj_config(model) or 'mmproj' in features or 'vision' in features
    model_allowed = (
        model_mtp_allowed(model)
        and not mmproj
        and (support_setting == 'yes' or 'mtp_native' in features or 'nextn_native' in features)
    )
    compatibility = engine_supports_model(engine_id, model, capabilities)
    model_runtime = str(getattr(model, 'runtime', '') or '').strip().lower()
    cap_reason = _capability_block_reason(install, capabilities)

    launch_status = 'ready'
    reason = 'MTP launch prerequisites are ready.'
    next_action = 'Run MTP Optimizer before promoting an MTP profile.'
    risk_level = 'info'
    if engine_id == ENGINE_VLLM or model_runtime == ENGINE_VLLM:
        launch_status = 'off'
        reason = f'active engine is {engine_display_name(engine_id)}; MTP requires a llama.cpp-compatible binary'
        next_action = 'Select a llama.cpp-compatible engine to test MTP.'
        risk_level = 'muted'
    elif support_setting == 'no':
        launch_status = 'blocked'
        reason = 'supports_mtp=no disables MTP for this model.'
        next_action = 'Set supports_mtp=auto/yes only if this GGUF is MTP-capable.'
        risk_level = 'block'
    elif mmproj:
        launch_status = 'blocked'
        reason = 'MTP + mmproj/vision is currently unsupported.'
        next_action = 'Disable MTP or remove the mmproj/vision entry.'
        risk_level = 'block'
    elif not model_allowed:
        launch_status = 'blocked' if mtp_enabled else 'unknown'
        reason = 'model MTP capability is unknown.'
        next_action = (
            'Set supports_mtp=yes only if this GGUF is MTP-capable, or disable MTP.'
            if mtp_enabled else
            'Set supports_mtp=yes only if this GGUF is MTP-capable, or run baseline validation first.'
        )
        risk_level = 'block' if mtp_enabled else 'warn'
    elif cap_reason:
        launch_status = 'failed' if cap_reason in ('binary not found', 'binary not executable', 'help output unavailable') else 'blocked'
        reason = cap_reason
        next_action = 'Build or select a llama.cpp-compatible binary that advertises --spec-type draft-mtp and --spec-draft-n-max.'
        risk_level = 'block'
    elif not mtp_enabled:
        measured_status, measured_reason, measured_action = _measured_mtp_status(model)
        if measured_status:
            launch_status = measured_status
            reason = measured_reason
            next_action = measured_action
            risk_level = 'success' if measured_status == 'usable' else 'warn'
        else:
            launch_status = 'ready'
            reason = 'MTP is available but disabled for the current launch profile.'
            next_action = 'Run MTP Optimizer or enable mtp_enabled for a measured profile.'
            risk_level = 'info'
    else:
        measured_status, measured_reason, measured_action = _measured_mtp_status(model)
        if measured_status in ('usable', 'risky'):
            launch_status = measured_status
            reason = measured_reason
            next_action = measured_action
            risk_level = 'success' if measured_status == 'usable' else 'warn'
        else:
            launch_status = 'ready'
            reason = f'MTP launch enabled with spec_type={selected_spec or "none"} draft_n={draft_n}.'
            next_action = 'Launch or run MTP Optimizer to measure baseline-vs-MTP behavior.'
            risk_level = 'info'

    launch = _build_launch_diagnostics(app, model, runtime_profile, capabilities, cap_reason)
    return MtpDoctorReport(
        engine_id=engine_id,
        binary_command=command,
        binary_source=str(getattr(install, 'source', '') or ''),
        binary_exists=bool(getattr(install, 'exists', False)),
        binary_executable=bool(getattr(install, 'executable', False)),
        binary_resolved_path=str(getattr(install, 'resolved_path', '') or ''),
        checked_paths=tuple(str(item) for item in (getattr(install, 'checked_paths', []) or [])),
        help_inspected=bool(str(getattr(capabilities, 'help_text', '') or '').strip()),
        supports_spec_type=bool(getattr(capabilities, 'supports_spec_type', False)),
        spec_type_values=tuple(str(item) for item in (getattr(capabilities, 'spec_type_values', ()) or ())),
        selected_spec_type=selected_spec,
        supports_spec_draft_n_max=bool(getattr(capabilities, 'supports_spec_draft_n_max', False)),
        supports_mtp=bool(getattr(capabilities, 'supports_mtp', False)) and bool(selected_spec),
        supports_no_warmup=bool(getattr(capabilities, 'supports_no_warmup', False)),
        supports_parallel=bool(getattr(capabilities, 'supports_parallel', False)),
        supports_cache_flags=bool(getattr(capabilities, 'supports_cache_type_kv', False) or getattr(capabilities, 'supports_ctk_ctv', False)),
        supports_draft_kv=bool(getattr(capabilities, 'supports_spec_draft_type_kv', False)),
        supports_fit=bool(getattr(capabilities, 'supports_fit', False)),
        supports_no_mmap=bool(getattr(capabilities, 'supports_no_mmap', False)),
        supports_cache_ram=bool(getattr(capabilities, 'supports_cache_ram', False)),
        model_id=str(getattr(model, 'id', '') or ''),
        model_name=str(getattr(model, 'name', '') or ''),
        model_path=str(getattr(model, 'path', '') or ''),
        detected_features=features,
        supports_mtp_setting=support_setting,
        mtp_enabled=mtp_enabled,
        mtp_draft_n_max=draft_n,
        mmproj_detected=mmproj,
        model_allowed=model_allowed,
        compatibility_status=str(getattr(compatibility, 'status', '') or ''),
        compatibility_reason=str(getattr(compatibility, 'reason', '') or ''),
        launch_status=launch_status,
        reason=reason,
        next_action=next_action,
        risk_level=risk_level,
        launch=launch,
    )


def mtp_status_for_model(app: Any, model: Any, runtime_profile: Any = None) -> MtpStatusSummary:
    report = build_mtp_doctor_report(app, model, runtime_profile=runtime_profile)
    return MtpStatusSummary(
        status=report.launch_status,
        reason=report.reason,
        next_action=report.next_action,
        risk_level=report.risk_level,
    )
