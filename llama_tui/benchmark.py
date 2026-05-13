import json
import os
import re
import signal
import shlex
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib import request

from .constants import CACHE_DIR
from .control import CancelToken, CancelledError, check_cancelled, sleep_with_cancel
from .discovery import extract_quant
from .engines import ENGINE_LLAMA_CPP_MTP, ENGINE_TQ3, resolve_engine_install
from .gguf import architecture_label, extra_arg_value, gguf_layer_count, model_file_size, read_gguf_metadata, set_model_extra_arg, strip_extra_args
from .hardware import HardwareProfile, ProcessPressureSnapshot, benchmark_current_process_pressure, process_pressure_label
from .launch_profiles import (
    BenchmarkLaunchProfile,
    benchmark_launch_metadata,
    benchmark_profile_request_fields,
    benchmark_profile_server_args,
    build_benchmark_launch_profile,
)
from .memory_guardrail import (
    MemoryGuardrailDecision,
    MemoryGuardrailState,
    memory_guardrail_record_fields,
    start_memory_guardrail_watchdog,
)
from .moe_placement import MoePlacementCandidate, generate_moe_placement_candidates
from .models import ModelConfig
from .mtp import MTP_DRAFT_VALUES, clamp_mtp_draft, model_mtp_allowed, model_mtp_draft_n_max, model_mtp_enabled, mtp_support_auto_hint
from .optimize import (
    apply_hardware_baseline,
    apply_best_optimization,
    apply_optimization_preset,
    estimate_safe_context_for_profile,
    choose_best_preset,
    model_is_moe,
    select_best_tier,
)
from .runtime_profiles import (
    RuntimeProfile,
    TurboKvProfile,
    default_engine_capabilities,
    kv_modes_from_preset,
    supported_turbo_kv_profiles,
    turbo_kv_profile_for_preset,
)
from .textutil import compact_message
from .tuning import (
    TuningCandidate,
    TuningObjective,
    early_stop_reason,
    generate_moe_tuning_candidates,
    generate_refinement_candidates,
    moe_tuning_eligibility_reason,
    score_tuning_record,
    select_measured_tuning_winner,
    target_vram_headroom_bytes,
    tuning_objective_for_model,
)

BENCHMARK_MAX_CANDIDATES = 6
BENCHMARK_WARMUP_TOKENS = 16
BENCHMARK_SAMPLE_TOKENS = 96
BENCHMARK_WARMUP_TIMEOUT = 120
BENCHMARK_SAMPLE_TIMEOUT = 240
BENCHMARK_READY_TIMEOUT = 180
BENCHMARK_MIN_CANDIDATE_BUDGET_SECONDS = 10.0
TQ3_MIN_CANDIDATE_BUDGET_SECONDS = 30.0
TQ3_RAW_PRESEARCH_FAST_BUDGET_SECONDS = 30.0
TQ3_RAW_PRESEARCH_FULL_BUDGET_SECONDS = 60.0
TQ3_RAW_PRESEARCH_FAST_CASE_TIMEOUT = 12.0
TQ3_RAW_PRESEARCH_FULL_CASE_TIMEOUT = 20.0
TQ3_RAW_PRESEARCH_FAILURE_LIMIT = 2
SAFE_BOOTSTRAP_PRESETS = (
    ('max_context', 'safe'),
    ('tokens_per_sec', 'safe'),
)
SAFE_BOOTSTRAP_Q8_TARGET_CTX = 4096
ADAPTIVE_BENCHMARK_TIME_BUDGET_SECONDS = 20 * 60
ALL_MODELS_ADAPTIVE_TIME_BUDGET_SECONDS = 6 * 60
ADAPTIVE_CONTEXT_ROUNDING = 256
ADAPTIVE_BINARY_STEPS = 4
ADAPTIVE_MAX_CONTEXT_PROBES = 12
ADAPTIVE_MAX_MEASUREMENTS = 20
EXHAUSTIVE_CONTEXT_STEP = 2048
COARSE_CONTEXT_LOW_LIMIT = 16_384
COARSE_CONTEXT_MID_LIMIT = 65_536
COARSE_CONTEXT_LOW_STEP = 2_048
COARSE_CONTEXT_MID_STEP = 4_096
COARSE_CONTEXT_HIGH_STEP = 8_192
CONTEXT_REFINE_STEP = 2_048
CONTEXT_KNEE_ROUNDING = 1_024
BENCHMARK_HISTORY_LIMIT = 10
FAST_BENCHMARK_CONTEXT_TARGETS = (8_192, 16_384)
FAST_BENCHMARK_PARALLEL_TARGETS = (1, 2, 4)
FAST_RUNTIME_PROFILE_BUDGET_SECONDS = 30 * 60
SMART_BENCHMARK_SOFT_BUDGET_SECONDS = 45 * 60
FULL_RUNTIME_PROFILE_BUDGET_SECONDS = 120 * 60
CONTEXT_HEALTH_PRESSURE_CAP = 0.45
CONTEXT_HEALTH_HIGH_PRESSURE = 0.80
RUNTIME_CONTEXT_MILESTONES = (8_192, 16_384, 24_576, 32_768, 49_152, 65_536, 98_304, 131_072)
FAST_RUNTIME_CONTEXT_TARGET_LIMIT = 4
SMART_FRONTIER_MAX_PROBES = 10
SMART_PARALLEL_IMPROVEMENT_THRESHOLD = 0.08
SMART_PARALLEL_NON_IMPROVING_LIMIT = 2
SMART_Q8_CONTEXT_GAIN_THRESHOLD = 0.15
MAX_CONTEXT_PROBE_TARGETS = (131_072, 160_000, 196_608, 224_000, 262_144)
MAX_CONTEXT_PROBE_MAX_CANDIDATES = 15
WORKFLOW_CACHE_RAM_VARIANTS = (0, 512, 1024, 2048)
WORKFLOW_CACHE_RAM_BASE_CANDIDATES = 1
WORKFLOW_CACHE_RAM_RESERVE_MIB = 1024
TURBOQUANT_VALIDATED_SYMMETRIC_FAMILIES = (
    ('llama', 'q4_k_m'),
    ('mistral', 'q4_k_m'),
    ('command-r', 'q4_k_m'),
    ('command-r+', 'q4_k_m'),
    ('cohere', 'q4_k_m'),
)
SMART_MAX_FULL_CONTEXTS_PER_VARIANT = 5
ADAPTIVE_PROFILE_KEYS = ('fast_chat', 'long_context', 'opencode_ready', 'auto')
ADAPTIVE_RESERVE_BY_OBJECTIVE = {
    'fast_chat': 25,
    'long_context': 35,
    'opencode_ready': 30,
    'auto': 30,
}
ADAPTIVE_TIER_BY_OBJECTIVE = {
    'fast_chat': 'extreme',
    'long_context': 'safe',
    'opencode_ready': 'moderate',
    'auto': 'moderate',
}

SPECTRUM_LABELS = {
    'possible': 'Possible',
    'fastest': 'Fastest',
    'ideal': 'Ideal',
    'longest': 'Highest Context',
    'opencode': 'OpenCode-ready',
    'winner': 'Winner',
    'runner_up': 'Runner-up',
    'failed': 'Failed',
    'break_point': 'Break Point',
}


class BenchmarkDeadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + max(0.0, float(seconds or 0.0))

    @classmethod
    def from_end(cls, end: float) -> 'BenchmarkDeadline':
        item = cls(0)
        item.end = float(end)
        return item

    def remaining(self) -> float:
        return max(0.0, self.end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def cap_timeout(self, requested: float, floor: float = 1.0) -> float:
        remaining = self.remaining()
        if remaining <= 0.0:
            return 0.0
        requested_value = max(0.0, float(requested or 0.0))
        if requested_value <= 0.0:
            requested_value = remaining
        capped = min(requested_value, remaining)
        return max(0.0, capped if remaining < floor else max(float(floor or 0.0), capped))


def sync_opencode_after_tuning(app: AppConfig) -> str:
    if hasattr(app, 'sync_generated_configs'):
        return app.sync_generated_configs('profile sync')
    messages: List[str] = []
    if app.opencode.path:
        ok, msg = app.generate_opencode()
        messages.append(msg if ok else f'opencode sync failed: {msg}')
    if getattr(app, 'continue_settings', None) and getattr(app.continue_settings, 'path', ''):
        ok, msg = app.generate_continue_config()
        messages.append(msg if ok else f'continue sync failed: {msg}')
    if messages:
        return ' | '.join(messages)
    return 'opencode.path unset; continue.path unset; skipped config sync'
def append_model_log(app: AppConfig, model: ModelConfig, text: str):
    app.append_log(model.id, text)


def benchmark_command_preview(
    app: AppConfig,
    model: ModelConfig,
    runtime_profile: Optional[RuntimeProfile] = None,
    benchmark_profile: Optional[BenchmarkLaunchProfile] = None,
) -> str:
    try:
        return shlex.join([str(item) for item in app.build_command(
            model,
            runtime_profile=runtime_profile,
            benchmark_profile=benchmark_profile,
        )])
    except Exception:
        return ''


def benchmark_effective_server_args(
    app: AppConfig,
    model: ModelConfig,
    runtime_profile: Optional[RuntimeProfile] = None,
    benchmark_profile: Optional[BenchmarkLaunchProfile] = None,
) -> List[str]:
    try:
        return [
            str(item)
            for item in app.build_command(
                model,
                runtime_profile=runtime_profile,
                benchmark_profile=benchmark_profile,
            )
        ]
    except Exception:
        return []


FAILURE_CATEGORIES = (
    'ENGINE_BINARY_MISSING',
    'CLI_INVALID',
    'MEMORY_GUARDRAIL',
    'MEMORY_FIT_FAILED',
    'FIXED_GPU_LAYERS_FIT_FAILED',
    'CUDA_OOM_WEIGHTS',
    'CUDA_OOM_KV',
    'KV_MODE_INCOMPATIBLE',
    'BUUN_FIT_FAILED',
    'BUUN_CPU_WARMUP_ABORT',
    'MODEL_LOAD_FAILED',
    'RAW_ENGINE_TIMEOUT',
    'SERVER_TIMEOUT',
    'API_TIMEOUT',
    'PORT_UNREACHABLE',
    'CHAT_TEMPLATE_ERROR',
)

FAILURE_EXCERPT_MARKERS = (
    'failed to fit params',
    'cannot meet free memory target',
    'n_gpu_layers already set by user',
    'gpu layers already set by user',
    'cudamalloc failed',
    'failed to allocate cuda',
    'unable to allocate cuda',
    'alloc_tensor_range',
    'failed to allocate buffer for kv cache',
    'failed to create context',
    'llama_model_load',
    'failed to load model',
)


def benchmark_failure_excerpt(text: str, limit: int = 320) -> str:
    lines = [line.strip() for line in str(text or '').splitlines() if line.strip()]
    if not lines:
        return ''
    selected: List[str] = []
    for marker in FAILURE_EXCERPT_MARKERS:
        for line in lines:
            if marker in line.lower() and line not in selected:
                selected.append(line)
                break
        if len(selected) >= 3:
            break
    if not selected:
        selected = lines[-3:]
    return concise_failure(' | '.join(selected), limit=limit)


def infer_fit_selected_ngl(text: str) -> Tuple[int, str, str]:
    excerpt = benchmark_failure_excerpt(text, limit=320)
    patterns = (
        (r'offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?', 'offloaded_layers'),
        (r'offloading\s+(\d+)\s+repeating layers?', 'offloading_layers'),
        (r'n_gpu_layers\s*=\s*(\d+)', 'n_gpu_layers_log'),
        (r'gpu layers?\s*[:=]\s*(\d+)', 'gpu_layers_log'),
    )
    for pattern, source in patterns:
        match = re.search(pattern, str(text or ''), re.IGNORECASE)
        if not match:
            continue
        try:
            return max(0, int(match.group(1))), source, excerpt
        except Exception:
            continue
    return 0, 'unknown', excerpt


def runtime_log_text_for_record(app: AppConfig, candidate: ModelConfig, max_lines: int = 400) -> str:
    try:
        lines = app._runtime_log_after_last_launch(candidate, max_lines=max_lines)
        if lines:
            return '\n'.join(str(line) for line in lines)
    except Exception:
        pass
    try:
        path = app.logfile(candidate.id)
        if path.exists():
            return '\n'.join(path.read_text(errors='replace').splitlines()[-max_lines:])
    except Exception:
        pass
    return ''


def enrich_fit_discovery_metadata(
    record: Dict[str, object],
    app: AppConfig,
    candidate: ModelConfig,
    runtime_profile: Optional[RuntimeProfile],
    success: bool,
) -> Dict[str, object]:
    if hasattr(app, 'logfile'):
        try:
            record['runtime_log_path'] = str(app.logfile(candidate.id))
        except Exception:
            pass
    if runtime_profile is None:
        if str(record.get('status', '') or '') not in ('ok', 'probe ok'):
            record['failure_excerpt'] = benchmark_failure_excerpt(str(record.get('detail', '') or ''))
        return record
    phase = str(getattr(runtime_profile, 'fit_discovery_phase', '') or '')
    if phase:
        record['fit_discovery_phase'] = phase
    text = '\n'.join([
        str(record.get('detail', '') or ''),
        runtime_log_text_for_record(app, candidate),
    ])
    if str(record.get('status', '') or '') not in ('ok', 'probe ok'):
        record['failure_excerpt'] = benchmark_failure_excerpt(text)
    if not (phase or bool(getattr(runtime_profile, 'fit', False))):
        return record
    selected_ngl, source, excerpt = infer_fit_selected_ngl(text)
    record['fit_selected_ngl'] = selected_ngl
    record['fit_selected_ngl_source'] = source
    record['fit_log_excerpt'] = excerpt
    if success and selected_ngl > 0:
        record['viable_ngl'] = selected_ngl
        record['viable_ngl_source'] = source
    elif success:
        record['viable_ngl'] = int(getattr(runtime_profile, 'viable_ngl', 0) or 0)
        record['viable_ngl_source'] = str(getattr(runtime_profile, 'viable_ngl_source', '') or 'unknown')
    return record


def classify_benchmark_failure(text: str, default_category: str = 'SERVER_TIMEOUT') -> Dict[str, str]:
    excerpt = benchmark_failure_excerpt(text)
    detail = compact_message(text or '')
    low = detail.lower()
    category = default_category if default_category in FAILURE_CATEGORIES else 'SERVER_TIMEOUT'
    reason = excerpt or detail or category
    suggested = ''
    terminal = False
    if 'engine_binary_missing' in low or ('llama.cpp-tq3 server not found' in low and 'tq3_llama_server_bin' in low):
        category = 'ENGINE_BINARY_MISSING'
        reason = detail or 'The active TQ3 server binary is missing.'
        suggested = 'Set TQ3_LLAMA_SERVER_BIN=/path/to/llama-server'
        terminal = True
    if re.search(r'(unknown|invalid|unrecognized).{0,80}(argument|option|value|flag)', low) or 'requires an argument' in low:
        category = 'CLI_INVALID'
        suggested = 'Check the generated command and use syntax supported by this server binary.'
        terminal = True
        if 'flash-attn' in low or '-fa' in low:
            reason = 'The binary rejected the generated flash-attn syntax.'
            suggested = 'Use "--flash-attn on" or "-fa on" for builds that require a flash-attn value.'
    if 'unknown value for --flash-attn' in low:
        category = 'CLI_INVALID'
        reason = 'The binary requires --flash-attn to receive a valid value.'
        suggested = 'Use "--flash-attn on" or "-fa on"; do not emit bare "-fa".'
        terminal = True
    if 'unsupported cache type' in low and ('cache' in low or 'turbo' in low):
        category = 'KV_MODE_INCOMPATIBLE'
        reason = detail or 'The selected KV mode is not supported by this server/model combination.'
        suggested = 'Try a different TurboKV mode or benchmark default/q8 KV cache.'
        terminal = True
    if 'chat template' in low or 'jinja' in low and 'template' in low:
        category = 'CHAT_TEMPLATE_ERROR'
        suggested = 'Disable Jinja or adjust the model chat template before benchmarking.'
    if 'does not divide' in low and ('cache' in low or 'turbo' in low):
        category = 'KV_MODE_INCOMPATIBLE'
        reason = detail or 'The selected KV mode is incompatible with this model/head dimension.'
        suggested = 'Try a different TurboKV mode or benchmark default/q8 KV cache.'
        terminal = True
    fit_memory_failure = (
        'failed to fit params to free device memory' in low
        or 'cannot meet free memory target' in low
        or ('projected to use' in low and 'device memory' in low and 'free device memory' in low)
        or ('llama_params_fit_impl' in low and 'free device memory' in low)
    )
    fixed_gpu_layer_failure = fit_memory_failure and (
        'n_gpu_layers already set by user' in low
        or 'gpu layers already set by user' in low
        or re.search(r'n_gpu_layers.{0,80}set by user', low) is not None
    )
    if fixed_gpu_layer_failure:
        category = 'FIXED_GPU_LAYERS_FIT_FAILED'
        reason = detail or 'The fixed GPU-layer candidate could not fit current free device memory.'
        suggested = 'Stop retrying fixed GPU-layer probes for this run; use fit/default/q8 fallbacks instead.'
        terminal = True
    elif fit_memory_failure:
        category = 'MEMORY_FIT_FAILED'
        reason = detail or 'The runtime fit planner could not meet the current free memory target.'
        suggested = 'Reduce context/offload for this run or retry after freeing RAM/VRAM.'
        terminal = True
    if (
        ('ggml-cpu/ops.cpp' in low or 'ggml_compute_forward_scale' in low)
        and ('fatal error' in low or 'abort' in low or 'aborted' in low)
    ):
        category = 'BUUN_CPU_WARMUP_ABORT'
        reason = detail or 'buun CPU/default warmup aborted before serving.'
        suggested = 'Skip the CPU/default probe and try a GPU fit profile with --no-warmup.'
        terminal = True
    memory_allocation_failure = (
        'cudamalloc failed' in low
        or ('cuda error' in low and 'out of memory' in low)
        or 'out of memory' in low
        or 'failed to allocate cuda' in low
        or 'unable to allocate cuda' in low
        or 'alloc_tensor_range' in low
        or 'failed to allocate buffer for kv cache' in low
        or ('failed to create context' in low and ('memory' in low or 'kv' in low or 'cache' in low))
    )
    if not terminal and memory_allocation_failure:
        weight_allocation_failure = (
            'failed to allocate cuda' in low
            or 'unable to allocate cuda' in low
            or 'alloc_tensor_range' in low
            or 'loading model tensors' in low
            or 'llama_model_load' in low
        )
        if not weight_allocation_failure and ('kv' in low or 'cache' in low or 'context' in low):
            category = 'CUDA_OOM_KV'
            reason = excerpt or detail or 'CUDA memory allocation failed for KV/context.'
            suggested = 'Reduce context, parallelism, or KV cache size; for OpenCode prefer parallel=1.'
        else:
            category = 'CUDA_OOM_WEIGHTS'
            reason = excerpt or detail or 'CUDA memory allocation failed while loading model weights.'
            suggested = 'Reduce GPU layers; for heavy MoE models try partial offload such as -ngl 20 and --parallel 1.'
        terminal = True
    if not terminal and ('failed to load model' in low or ('llama_model_load' in low and 'failed' in low) or 'model load failed' in low):
        category = 'MODEL_LOAD_FAILED'
        suggested = suggested or 'Verify the GGUF file and model path, then retry with a smaller launch profile.'
    if not terminal and ('connection refused' in low or 'failed to establish a new connection' in low or 'port unreachable' in low):
        category = 'PORT_UNREACHABLE'
        suggested = 'The server process did not expose the expected API port.'
    if not terminal and ('timed out' in low or 'timeout' in low):
        if default_category == 'RAW_ENGINE_TIMEOUT':
            category = 'RAW_ENGINE_TIMEOUT'
            suggested = suggested or 'Keep raw engine probes bounded; promote only candidates that pass server validation.'
        else:
            category = 'API_TIMEOUT' if default_category == 'API_TIMEOUT' else 'SERVER_TIMEOUT'
        suggested = suggested or 'Retry with a smaller context or lower GPU layer count.'
    return {
        'failure_category': category,
        'failure_reason': concise_failure(reason, limit=500),
        'suggested_fix': suggested,
        'failure_excerpt': excerpt,
    }


def benchmark_failure_summary(records: List[Dict[str, object]], fallback: str) -> str:
    for record in records:
        category = str(record.get('failure_category', '') or '')
        if not category:
            continue
        profile = str(record.get('runtime_profile', '') or record.get('objective', '') or record.get('preset', '') or 'candidate')
        reason = str(record.get('failure_reason', '') or record.get('detail', '') or category)
        fix = str(record.get('suggested_fix', '') or '')
        summary = f'{profile} failed before benchmark: {category} - {concise_failure(reason)}'
        if fix:
            summary += f' Suggested fix: {fix}'
        return summary
    return fallback


def _good_measured_profiles(profiles: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    if not isinstance(profiles, dict):
        return {}
    good = {}
    for key, value in profiles.items():
        if isinstance(value, dict) and str(value.get('status', 'ok') or 'ok') == 'ok':
            good[str(key)] = dict(value)
    return good


def previous_usable_measured_profiles(app, model: ModelConfig) -> Dict[str, Dict[str, object]]:
    sources = []
    if hasattr(app, 'get_model'):
        try:
            current = app.get_model(model.id)
        except Exception:
            current = None
        if current is not None:
            sources.append(current)
    sources.append(model)
    for source in sources:
        profiles = _good_measured_profiles(getattr(source, 'measured_profiles', {}) or {})
        if profiles:
            return dict(getattr(source, 'measured_profiles', {}) or profiles)
    return {}


def failed_benchmark_model_state(
    app,
    model: ModelConfig,
    records: List[Dict[str, object]],
    ended_at: str,
) -> Tuple[ModelConfig, bool]:
    source = model
    if hasattr(app, 'get_model'):
        try:
            source = app.get_model(model.id) or model
        except Exception:
            source = model
    saved = ModelConfig(**asdict(source))
    previous = previous_usable_measured_profiles(app, model)
    saved.last_benchmark_results = records
    saved.default_benchmark_at = ended_at
    if previous:
        saved.measured_profiles = previous
        saved.default_benchmark_status = 'done'
        return saved, True
    saved.measured_profiles = {}
    saved.default_benchmark_status = 'failed'
    return saved, False


def preserved_profiles_message(prefix: str, records: List[Dict[str, object]]) -> str:
    summary = benchmark_failure_summary(records, 'no new measured candidates completed')
    return f'⚠ {prefix}: {summary}; kept previous working measured profiles'


def _call_model_health(app, model: ModelConfig) -> Tuple[str, str]:
    try:
        return app.health(model)
    except Exception as exc:
        return 'UNKNOWN', str(exc)


def _stop_managed_model(app, model: ModelConfig) -> Tuple[bool, str]:
    if not hasattr(app, 'stop'):
        return False, 'app cannot stop managed model processes'
    try:
        return app.stop(model, managed_only=True)
    except TypeError:
        return app.stop(model)


def _known_llama_pressure(payload: Dict[str, object]) -> str:
    known = payload.get('process_known', {}) if isinstance(payload, dict) else {}
    count = 0
    if isinstance(known, dict):
        try:
            count = int(known.get('llama', 0) or 0)
        except Exception:
            count = 0
    if count <= 0:
        return ''
    detail = str(payload.get('process_pressure_detail', '') or '')
    return f'unmanaged llama-family process(es) still visible: {count}' + (f' | {detail}' if detail else '')


def benchmark_preflight_cleanup(
    app,
    model: ModelConfig,
    run_kind: str,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    check_cancelled(cancel_token)
    try:
        active_engine = app.active_engine_key_for_model(model) if hasattr(app, 'active_engine_key_for_model') else ''
    except Exception:
        active_engine = ''
    if active_engine == ENGINE_TQ3:
        install = resolve_engine_install(app, ENGINE_TQ3)
        if not install.exists:
            command = str(install.resolved_command or 'tq3-llama-server')
            return False, (
                f'ENGINE_BINARY_MISSING: llama.cpp-tq3 server not found: {command}. '
                'Set TQ3_LLAMA_SERVER_BIN=/path/to/llama-server'
            )
    if hasattr(app, 'active_engine_model_compatibility'):
        try:
            compatible, compatibility_msg = app.active_engine_model_compatibility(model)
        except Exception:
            compatible, compatibility_msg = True, ''
        if not compatible:
            return False, f'❌ benchmark preflight blocked: {compatibility_msg}'
    stopped: List[str] = []
    models = list(getattr(app, 'models', []) or [])
    if model.id not in {getattr(item, 'id', '') for item in models}:
        models.insert(0, model)
    for item in models:
        try:
            managed_pid = _get_model_pid(app, item, discover=False, managed_only=True)
        except Exception:
            managed_pid = None
        if not managed_pid:
            continue
        ok, msg = _stop_managed_model(app, item)
        if not ok:
            return False, f'❌ benchmark preflight could not stop managed {item.id}: {msg}'
        stopped.append(f'{item.id}:{managed_pid}')
    if stopped:
        sleep_with_cancel(0.3, cancel_token)

    status, detail = _call_model_health(app, model)
    try:
        managed_pid = _get_model_pid(app, model, discover=False, managed_only=True)
    except Exception:
        managed_pid = None
    try:
        any_pid = managed_pid or _get_model_pid(app, model)
    except Exception:
        any_pid = managed_pid
    running = status in ('READY', 'LOADING', 'STARTING') or bool(any_pid)
    if running and not managed_pid:
        return False, (
            f'❌ benchmark preflight blocked: unmanaged server/process is already running for {model.id}. '
            f'Stop the manual llama-cli/llama-server process first; status={status} detail={compact_message(detail)}'
        )

    pressure = current_process_pressure_payload()
    pieces = [f'benchmark preflight ({run_kind}): stopped {len(stopped)} managed process(es)']
    pressure_detail = str(pressure.get('process_pressure_detail', '') or '')
    if pressure_detail:
        pieces.append(pressure_detail)
    llama_note = _known_llama_pressure(pressure)
    if llama_note:
        pieces.append(llama_note)
    msg = '; '.join(pieces)
    if progress:
        progress(msg)
    return True, msg


def runtime_profile_kv_disable_key(record: Dict[str, object], runtime_profile: RuntimeProfile) -> Tuple[str, str]:
    category = str(record.get('failure_category', '') or '')
    if category != 'KV_MODE_INCOMPATIBLE':
        return '', ''
    kv_preset = str(getattr(runtime_profile, 'kv_preset', '') or '')
    text = ' '.join([
        str(record.get('failure_reason', '') or ''),
        str(record.get('detail', '') or ''),
        kv_preset,
    ]).lower()
    kv_family = str(getattr(runtime_profile, 'kv_family', '') or '')
    if 'does not divide' in text or 'block size' in text:
        return 'family', kv_family or ('turbo' if 'turbo' in kv_preset else 'cache')
    return 'preset', kv_preset


def runtime_profile_skip_reason(runtime_profile: RuntimeProfile, disabled: set[Tuple[str, str]]) -> str:
    kv_preset = str(getattr(runtime_profile, 'kv_preset', '') or '')
    kv_family = str(getattr(runtime_profile, 'kv_family', '') or '')
    if ('preset', kv_preset) in disabled:
        return f'KV preset {kv_preset} was already rejected for this run'
    if kv_family and ('family', kv_family) in disabled:
        return f'KV family {kv_family} was already rejected for this run'
    if not kv_family and 'turbo' in kv_preset and ('family', 'turbo') in disabled:
        return 'TurboKV was already rejected for this run'
    return ''


def _runtime_profile_placement_shape(runtime_profile: RuntimeProfile) -> str:
    placement = str(getattr(runtime_profile, 'placement_strategy', '') or '')
    cpu_moe = 'cmoe' if bool(getattr(runtime_profile, 'cpu_moe', False)) else ''
    n_cpu_moe = str(int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0))
    tensor_overrides = ','.join(str(item) for item in tuple(getattr(runtime_profile, 'tensor_overrides', ()) or ()))
    return '|'.join((placement, cpu_moe, n_cpu_moe, tensor_overrides))


def _runtime_profile_memory_shape(runtime_profile: RuntimeProfile) -> Tuple[str, str, str, str, str, str]:
    engine = str(getattr(runtime_profile, 'engine_id', '') or '')
    kv_preset = str(getattr(runtime_profile, 'kv_preset', '') or '')
    if getattr(runtime_profile, 'gpu_layers', None) is None and bool(getattr(runtime_profile, 'fit', False)):
        mode = 'fit'
        layers = 'fit'
    elif getattr(runtime_profile, 'gpu_layers', None) is not None:
        mode = 'fixed'
        layers = str(int(getattr(runtime_profile, 'gpu_layers') or 0))
    else:
        mode = 'default'
        layers = ''
    parallel = str(max(1, int(getattr(runtime_profile, 'parallel', 1) or 1)))
    placement = _runtime_profile_placement_shape(runtime_profile)
    return engine, kv_preset, mode, layers, parallel, placement


def runtime_profile_memory_disable_key(record: Dict[str, object], runtime_profile: RuntimeProfile) -> Tuple[str, ...]:
    category = str(record.get('failure_category', '') or '')
    if category in ('FIXED_GPU_LAYERS_FIT_FAILED',) or (
        category == 'CUDA_OOM_WEIGHTS' and getattr(runtime_profile, 'gpu_layers', None) is not None
    ):
        return ('fixed_ngl', str(getattr(runtime_profile, 'engine_id', '') or ''), _runtime_profile_placement_shape(runtime_profile))
    if category in ('MEMORY_FIT_FAILED', 'CUDA_OOM_WEIGHTS') and bool(getattr(runtime_profile, 'fit', False)):
        return ('fit_engine', str(getattr(runtime_profile, 'engine_id', '') or ''))
    if category in ('CUDA_OOM_KV', 'MEMORY_GUARDRAIL'):
        return ('context_shape', *_runtime_profile_memory_shape(runtime_profile), str(int(getattr(runtime_profile, 'ctx_size', 0) or 0)))
    return ()


def runtime_profile_memory_skip_reason(runtime_profile: RuntimeProfile, disabled: set[Tuple[str, ...]]) -> str:
    engine = str(getattr(runtime_profile, 'engine_id', '') or '')
    shape = _runtime_profile_memory_shape(runtime_profile)
    ctx = int(getattr(runtime_profile, 'ctx_size', 0) or 0)
    for key in disabled:
        if not key:
            continue
        if key[0] == 'fixed_ngl' and len(key) >= 2:
            bad_placement = key[2] if len(key) >= 3 else ''
            if (
                key[1] == engine
                and getattr(runtime_profile, 'gpu_layers', None) is not None
                and bad_placement == _runtime_profile_placement_shape(runtime_profile)
            ):
                return 'fixed GPU-layer profiles were already rejected by memory fit/OOM for this run'
        if key[0] == 'fit_engine' and len(key) >= 2:
            if key[1] == engine and bool(getattr(runtime_profile, 'fit', False)):
                return 'fit discovery already failed for this engine in this run'
        if key[0] == 'context_shape' and len(key) >= 3:
            bad_shape = tuple(key[1:-1])
            try:
                bad_ctx = int(key[-1] or 0)
            except Exception:
                bad_ctx = 0
            if shape == bad_shape and bad_ctx > 0 and ctx >= bad_ctx:
                return f'same runtime/KV shape already hit memory guardrail/OOM at ctx={bad_ctx}'
    return ''


def emit_benchmark_event(
    progress: Optional[Callable[[object], None]],
    event: str,
    model: ModelConfig,
    run_kind: str,
    message: str = '',
    phase: str = '',
    completed: Optional[int] = None,
    total: Optional[int] = None,
    candidate: str = '',
    command: str = '',
    record: Optional[Dict[str, object]] = None,
    records: Optional[List[Dict[str, object]]] = None,
):
    if not progress:
        return
    payload: Dict[str, object] = {
        'event': event,
        'run_kind': run_kind,
        'model_id': model.id,
        'message': compact_message(message or phase or event),
    }
    if phase:
        payload['phase'] = phase
    if completed is not None:
        payload['completed'] = int(completed)
    if total is not None:
        payload['total'] = int(total)
    if candidate:
        payload['candidate'] = candidate
    if command:
        payload['command'] = command
    if record is not None:
        payload['record'] = dict(record)
    if records is not None:
        payload['records'] = [dict(item) for item in records]
    progress(payload)


def architecture_payload(model: ModelConfig) -> Dict[str, object]:
    payload = {
        'architecture_type': getattr(model, 'architecture_type', 'unknown') or 'unknown',
        'architecture': getattr(model, 'architecture', '') or '',
        'architecture_label': architecture_label(model),
        'model_family': getattr(model, 'model_family', '') or '',
        'expert_count': int(getattr(model, 'expert_count', 0) or 0),
        'expert_used_count': int(getattr(model, 'expert_used_count', 0) or 0),
        'expert_shared_count': int(getattr(model, 'expert_shared_count', 0) or 0),
        'active_expert_ratio': float(getattr(model, 'active_expert_ratio', 0.0) or 0.0),
        'classification_confidence': float(getattr(model, 'classification_confidence', 0.0) or 0.0),
        'classification_source': getattr(model, 'classification_source', '') or '',
    }
    try:
        metadata = read_gguf_metadata(getattr(model, 'path', '') or '')
        arch = str(metadata.get('general.architecture') or getattr(model, 'architecture', '') or '')
        prefix = f'{arch}.' if arch else ''

        def metadata_int(suffix: str) -> int:
            keys = [f'{prefix}{suffix}'] if prefix else []
            keys.extend(key for key in metadata if key.endswith(f'.{suffix}') and key not in keys)
            for key in keys:
                try:
                    value = int(metadata.get(key) or 0)
                except Exception:
                    value = 0
                if value > 0:
                    return value
            return 0

        payload.update({
            'model_file_size': int(model_file_size(model) or 0),
            'native_context_length': int(metadata.get('general.context_length') or metadata_int('context_length') or 0),
            'attention_key_length': metadata_int('attention.key_length'),
            'attention_value_length': metadata_int('attention.value_length'),
            'attention_head_count': metadata_int('attention.head_count'),
            'attention_head_count_kv': metadata_int('attention.head_count_kv'),
        })
    except Exception:
        payload.update({
            'model_file_size': int(model_file_size(model) or 0),
            'native_context_length': 0,
            'attention_key_length': 0,
            'attention_value_length': 0,
            'attention_head_count': 0,
            'attention_head_count_kv': 0,
        })
    return payload


def turboquant_head_dim(model: ModelConfig) -> int:
    values = (
        getattr(model, 'turboquant_head_dim', 0),
        getattr(model, 'turboquant_key_dim', 0),
        getattr(model, 'turboquant_value_dim', 0),
    )
    parsed: List[int] = []
    for value in values:
        try:
            parsed.append(int(value or 0))
        except Exception:
            parsed.append(0)
    return max(parsed or [0])


def normalized_model_quant(model: ModelConfig) -> str:
    return str(extract_quant(model) or '').strip().lower()


def _normalized_family_tokens(model: ModelConfig) -> Tuple[str, ...]:
    text = ' '.join([
        str(getattr(model, 'model_family', '') or ''),
        str(getattr(model, 'architecture', '') or ''),
        str(getattr(model, 'name', '') or ''),
    ]).lower()
    cleaned = re.sub(r'[^a-z0-9+.-]+', ' ', text)
    return tuple(token for token in cleaned.split() if token)


def turboquant_symmetric_auto_allowed(model: ModelConfig, model_quant: str = '') -> bool:
    quant = (model_quant or normalized_model_quant(model)).strip().lower()
    if quant in ('q8_0', 'f16', 'fp16', 'f32', 'fp32', 'bf16') or quant.startswith('fp'):
        return True
    families = _normalized_family_tokens(model)
    for family, validated_quant in TURBOQUANT_VALIDATED_SYMMETRIC_FAMILIES:
        if quant == validated_quant and any(token == family or family in token for token in families):
            return True
    return False


def is_turboquant_symmetric_profile(kv_preset: str) -> bool:
    key_mode, value_mode = kv_modes_from_preset(kv_preset)
    return bool(key_mode and value_mode and key_mode == value_mode and key_mode.startswith('turbo'))


def turboquant_auto_profiles(
    model: ModelConfig,
    capabilities,
    depth: str,
) -> List[object]:
    profiles = supported_turbo_kv_profiles(capabilities, depth, engine_id='turboquant')
    head_dim = turboquant_head_dim(model)
    if head_dim == 64 or (0 < head_dim < 128):
        return [item for item in profiles if item.kv_preset == 'q8_0/q8_0']
    if head_dim <= 0:
        return [item for item in profiles if item.kv_preset == 'q8_0/q8_0']
    allow_symmetric = turboquant_symmetric_auto_allowed(model)
    selected = []
    for profile in profiles:
        if is_turboquant_symmetric_profile(profile.kv_preset) and not allow_symmetric:
            continue
        selected.append(profile)
    return selected


def process_pressure_payload(snapshot: Optional[ProcessPressureSnapshot]) -> Dict[str, object]:
    if snapshot is None:
        return {}
    return {
        'process_pressure_level': snapshot.pressure_level,
        'process_pressure_score': float(snapshot.pressure_score or 0.0),
        'process_pressure_detail': snapshot.detail or process_pressure_label(snapshot),
        'process_load_1m': float(snapshot.load_1m or 0.0),
        'process_load_ratio': float(snapshot.load_ratio or 0.0),
        'process_count': int(snapshot.process_count or 0),
        'process_known': dict(snapshot.known_processes or {}),
        'process_known_memory': dict(snapshot.known_memory or {}),
        'process_top_memory': [dict(item) for item in list(snapshot.top_memory or [])[:3]],
        'process_top_cpu': [dict(item) for item in list(snapshot.top_cpu or [])[:3]],
    }


def current_process_pressure_payload() -> Dict[str, object]:
    try:
        return process_pressure_payload(benchmark_current_process_pressure())
    except Exception:
        return {}


def _pressure_score_from_payload(payload: Optional[Dict[str, object]]) -> float:
    try:
        return max(0.0, min(1.0, float((payload or {}).get('process_pressure_score', 0.0) or 0.0)))
    except Exception:
        return 0.0


def _guardrail_profile(app: AppConfig) -> HardwareProfile:
    try:
        return app.hardware_profile(refresh=True)
    except Exception:
        return HardwareProfile()


def _candidate_required_for_opencode_floor(candidate: ModelConfig, observed_floor: int) -> bool:
    return int(observed_floor or 0) > 0 and ctx_per_slot(candidate) >= int(observed_floor or 0)


def memory_guardrail_admission(
    profile: HardwareProfile,
    candidate: ModelConfig,
    estimated_safe_ctx: int,
    pressure_payload: Optional[Dict[str, object]] = None,
    observed_floor: int = 0,
    state: Optional[MemoryGuardrailState] = None,
) -> MemoryGuardrailDecision:
    guardrail_state = state or MemoryGuardrailState()
    return guardrail_state.observe(
        profile,
        phase='admission',
        candidate_ctx=int(getattr(candidate, 'ctx', 0) or 0),
        safe_ctx=int(estimated_safe_ctx or 0),
        observed_floor=int(observed_floor or 0),
        required_for_floor=_candidate_required_for_opencode_floor(candidate, observed_floor),
        pressure_score=_pressure_score_from_payload(pressure_payload),
    )


def apply_memory_guardrail_record(
    record: Dict[str, object],
    decision: Optional[MemoryGuardrailDecision] = None,
    state: Optional[MemoryGuardrailState] = None,
) -> Dict[str, object]:
    if decision is not None:
        record.update(memory_guardrail_record_fields(decision))
    if state is not None:
        record.update(state.record_fields())
        decision = state.stop_decision or state.skip_decision or decision
    if decision is not None and decision.action in ('skip', 'stop'):
        record['failure_category'] = 'MEMORY_GUARDRAIL'
        record['failure_reason'] = concise_failure(decision.reason, limit=500)
        record['suggested_fix'] = 'Candidate was stopped or skipped to protect system memory; try safer fit/default fallbacks or free RAM/VRAM.'
        record['detail'] = concise_failure(
            f'candidate {"skipped" if decision.action == "skip" else "stopped"} by memory guardrail: {decision.reason}',
            limit=500,
        )
        record['startup_result'] = 'FAILED'
    return record


def memory_guardrail_skip_record(
    candidate: ModelConfig,
    objective: str,
    decision: MemoryGuardrailDecision,
    runtime_context: Dict[str, object],
    process_snapshots: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    snapshot = dict(decision.snapshot or {})
    record = adaptive_record_from_candidate(
        candidate,
        objective,
        'skipped',
        detail=f'candidate skipped by memory guardrail: {decision.reason}',
        ram_available=int(snapshot.get('ram_available', 0) or 0),
        gpu_memory_free=int(snapshot.get('gpu_memory_free', 0) or 0),
        gpu_memory_total=int(snapshot.get('gpu_memory_total', 0) or 0),
        process_snapshots=process_snapshots,
        **runtime_context,
    )
    return apply_memory_guardrail_record(record, decision)


def _record_headroom_score(record: Dict[str, object]) -> float:
    ram = int(record.get('ram_available', 0) or 0)
    vram = int(record.get('gpu_memory_free', 0) or 0)
    score = min(1.0, (ram / 1024**3) / 8.0) if ram else 0.35
    if vram:
        score = max(score, min(1.0, (vram / 1024**3) / 2.0))
    pressure = float(record.get('process_pressure_score', 0.0) or 0.0)
    if pressure:
        score *= max(0.45, 1.0 - pressure * 0.35)
    return max(0.0, min(1.0, score))


def _record_stability_score(record: Dict[str, object]) -> float:
    score = 1.0
    if int(record.get('retry_attempt', 1) or 1) > 1:
        score -= 0.15
    status = str(record.get('status', '') or '').lower()
    if status not in ('ok', 'probe ok', 'tests passed'):
        score -= 0.35
    detail = compact_message(str(record.get('detail', '') or '')).lower()
    if detail not in ('', '1 samples', '2 samples', '3 samples', 'all tasks passed'):
        score -= 0.05
    ready = float(record.get('ready_seconds', 0.0) or 0.0)
    if ready > 90:
        score -= 0.10
    return max(0.0, min(1.0, score))


def _tps_curve(record: Dict[str, object]) -> float:
    tps = float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0)
    return max(0.0, min(1.0, tps / (tps + 30.0))) if tps > 0 else 0.0


def _ctx_curve(record: Dict[str, object], cap: int) -> float:
    ctx = int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0)
    return max(0.0, min(1.0, ctx / max(1, cap)))


def _record_kv_quality_penalty(record: Dict[str, object]) -> float:
    try:
        explicit = float(record.get('kv_score_penalty', 0.0) or 0.0)
    except Exception:
        explicit = 0.0
    if explicit:
        return max(0.0, explicit)
    profile = turbo_kv_profile_for_preset(str(record.get('kv_preset', '') or ''))
    return max(0.0, float(profile.score_penalty if profile else 0.0))


LOW_SPEED_PROMOTION_TOKENS_PER_SEC = 3.0


def low_speed_guardrail_reason(record: Dict[str, object]) -> str:
    engine = str(record.get('engine', '') or '').strip().lower()
    if engine != 'tq3':
        return ''
    try:
        tps = float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0)
    except Exception:
        tps = 0.0
    if 0.0 < tps < LOW_SPEED_PROMOTION_TOKENS_PER_SEC:
        return (
            f'TQ3 decode speed {tps:.2f} tok/s is below the '
            f'{LOW_SPEED_PROMOTION_TOKENS_PER_SEC:.1f} tok/s promotion floor'
        )
    return ''


def runtime_profile_is_tq3(runtime_profile: Optional[RuntimeProfile]) -> bool:
    return str(getattr(runtime_profile, 'engine_id', '') or '').strip().lower() == ENGINE_TQ3


def record_engine_is_tq3(record: Dict[str, object], runtime_profile: Optional[RuntimeProfile] = None) -> bool:
    return (
        runtime_profile_is_tq3(runtime_profile)
        or str(record.get('engine', '') or '').strip().lower() == ENGINE_TQ3
    )


def tq3_terminal_candidate_failure(record: Dict[str, object], runtime_profile: Optional[RuntimeProfile] = None) -> bool:
    if not record_engine_is_tq3(record, runtime_profile):
        return False
    category = str(record.get('failure_category', '') or '').strip().upper()
    if category in ('API_TIMEOUT', 'RAW_ENGINE_TIMEOUT'):
        return True
    return bool(low_speed_guardrail_reason(record))


def candidate_min_budget_seconds(runtime_profile: Optional[RuntimeProfile] = None, engine: str = '') -> float:
    engine_key = str(engine or getattr(runtime_profile, 'engine_id', '') or '').strip().lower()
    if engine_key == ENGINE_TQ3:
        return TQ3_MIN_CANDIDATE_BUDGET_SECONDS
    return BENCHMARK_MIN_CANDIDATE_BUDGET_SECONDS


def deadline_has_candidate_budget(
    deadline: Optional[BenchmarkDeadline],
    runtime_profile: Optional[RuntimeProfile] = None,
    engine: str = '',
) -> bool:
    if deadline is None:
        return True
    return deadline.remaining() >= candidate_min_budget_seconds(runtime_profile, engine)


def budget_exhausted_record(
    candidate: ModelConfig,
    objective: str,
    detail: str,
    runtime_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    record = adaptive_record_from_candidate(
        candidate,
        objective,
        'skipped_budget',
        detail=detail,
        **(runtime_context or {}),
    )
    record['failure_category'] = 'BUDGET_EXHAUSTED'
    record['failure_reason'] = detail
    record['suggested_fix'] = 'Run a shorter fast benchmark or increase the benchmark budget.'
    return record


def _record_decode_tps(record: Dict[str, object]) -> float:
    try:
        return float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0)
    except Exception:
        return 0.0


def _is_tq3_cpu_moe_placement(record: Dict[str, object]) -> bool:
    return (
        str(record.get('engine', '') or '').strip().lower() == 'tq3'
        and (bool(record.get('cpu_moe', False)) or int(record.get('n_cpu_moe', 0) or 0) > 0)
    )


def _is_tq3_partial_fixed_offload(record: Dict[str, object]) -> bool:
    if str(record.get('engine', '') or '').strip().lower() != 'tq3':
        return False
    if _is_tq3_cpu_moe_placement(record):
        return False
    if str(record.get('gpu_layers_mode', '') or '').strip().lower() not in ('', 'fixed'):
        return False
    try:
        ngl = int(record.get('ngl', 0) or 0)
    except Exception:
        ngl = 0
    return 0 < ngl < 999


def reject_slow_tq3_partial_offload_rows(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    faster = [
        item for item in records
        if item.get('status') == 'ok'
        and str(item.get('measurement_type', 'full') or 'full') == 'full'
        and _is_tq3_cpu_moe_placement(item)
        and _record_decode_tps(item) > 0
    ]
    if not faster:
        return records
    fastest = max(faster, key=_record_decode_tps)
    fastest_tps = _record_decode_tps(fastest)
    kept = []
    for item in records:
        tps = _record_decode_tps(item)
        if _is_tq3_partial_fixed_offload(item) and tps > 0 and fastest_tps > tps * 3.0:
            item['rejection_reason'] = 'rejected_slow_partial_offload'
            item['selection_rejection_reason'] = 'rejected_slow_partial_offload'
            item['rejected_by_profile'] = str(fastest.get('runtime_profile', '') or fastest.get('placement_strategy', '') or 'tq3_moe_placement')
            item['rejected_by_tokens_per_sec'] = round(fastest_tps, 2)
            continue
        kept.append(item)
    return kept


def propagate_rejection_reasons(records: List[Dict[str, object]], measured: List[Dict[str, object]]) -> None:
    rejected = [
        item for item in measured
        if str(item.get('rejection_reason', '') or item.get('selection_rejection_reason', '') or '')
    ]
    if not rejected:
        return
    for rejected_item in rejected:
        reason = str(rejected_item.get('rejection_reason', '') or rejected_item.get('selection_rejection_reason', '') or '')
        fingerprint = str(rejected_item.get('config_fingerprint', '') or '')
        shape = (
            str(rejected_item.get('runtime_profile', '') or ''),
            str(rejected_item.get('kv_preset', '') or ''),
            str(rejected_item.get('placement_strategy', '') or ''),
            int(rejected_item.get('ngl', 0) or 0),
            int(rejected_item.get('ctx', 0) or 0),
        )
        for record in records:
            if fingerprint and str(record.get('config_fingerprint', '') or '') == fingerprint:
                record['rejection_reason'] = reason
                record['selection_rejection_reason'] = reason
                continue
            record_shape = (
                str(record.get('runtime_profile', '') or ''),
                str(record.get('kv_preset', '') or ''),
                str(record.get('placement_strategy', '') or ''),
                int(record.get('ngl', 0) or 0),
                int(record.get('ctx', 0) or 0),
            )
            if record_shape == shape:
                record['rejection_reason'] = reason
                record['selection_rejection_reason'] = reason


def score_fast_chat(record: Dict[str, object], model: ModelConfig) -> float:
    cap = 16384 if model_is_moe(model) else 8192
    tps = float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0)
    return (
        tps
        + 4.0 * _record_headroom_score(record)
        + 2.0 * _record_stability_score(record)
        + 3.0 * _ctx_curve(record, cap)
    )


def score_long_context(record: Dict[str, object], model: ModelConfig) -> float:
    cap = max(32768, int(getattr(model, 'ctx_max', 32768) or 32768))
    return (
        0.55 * _ctx_curve(record, cap)
        + 0.20 * _record_headroom_score(record)
        + 0.15 * _tps_curve(record)
        + 0.10 * _record_stability_score(record)
        - _record_kv_quality_penalty(record)
    )


def score_opencode_ready(record: Dict[str, object], model: ModelConfig) -> float:
    ctx = int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0)
    target = 32768 if model_is_moe(model) else 16384
    score = (
        0.40 * _ctx_curve(record, target)
        + 0.30 * _tps_curve(record)
        + 0.20 * _record_headroom_score(record)
        + 0.10 * _record_stability_score(record)
    )
    if model_is_moe(model) and ctx < 16384:
        score *= 0.35
    if model_is_moe(model) and ctx >= 32768:
        score += 0.15
    return score - _record_kv_quality_penalty(record)


def score_auto(record: Dict[str, object], model: ModelConfig) -> float:
    if model_is_moe(model) and int(record.get('ngl', 0) or 0) != 999:
        return (
            0.35 * score_opencode_ready(record, model)
            + 0.35 * score_long_context(record, model)
            + 0.20 * score_fast_chat(record, model)
            + 0.10 * _record_headroom_score(record)
        )
    return (
        0.50 * _tps_curve(record)
        + 0.30 * _ctx_curve(record, 32768)
        + 0.12 * _record_headroom_score(record)
        + 0.08 * _record_stability_score(record)
        - _record_kv_quality_penalty(record)
    )


def concise_failure(text: str, limit: int = 320) -> str:
    message = compact_message(text)
    if len(message) <= limit:
        return message
    return message[: max(0, limit - 3)] + '...'


def round_context(value: int, step: int = ADAPTIVE_CONTEXT_ROUNDING) -> int:
    value = max(1, int(value or 1))
    step = max(1, int(step or 1))
    return max(step, int(round(value / step) * step))


def round_context_down(value: int, step: int = ADAPTIVE_CONTEXT_ROUNDING) -> int:
    value = max(1, int(value or 1))
    step = max(1, int(step or 1))
    return max(step, int(value // step) * step)


def round_context_up(value: int, step: int = ADAPTIVE_CONTEXT_ROUNDING) -> int:
    value = max(1, int(value or 1))
    step = max(1, int(step or 1))
    return max(step, int(((value + step - 1) // step) * step))


def ctx_per_slot(model: ModelConfig) -> int:
    return int(getattr(model, 'ctx', 0) or 0) // max(1, int(getattr(model, 'parallel', 1) or 1))


def measured_profile_key_for_launch(mode: str, tier: str = '') -> str:
    normalized = (mode or '').strip().lower()
    tier = (tier or '').strip().lower()
    if normalized in ('tokens_per_sec', 'fast_chat', 'balanced_chat'):
        return 'fast_chat'
    if normalized in ('max_context', 'long_context'):
        return 'long_context'
    if normalized == 'opencode_ready':
        return 'opencode_ready'
    if normalized in ('best', 'auto', 'auto_profile') or tier == 'auto':
        return 'auto'
    return ''


def get_measured_profile(model: ModelConfig, key: str) -> Dict[str, object]:
    profiles = getattr(model, 'measured_profiles', {}) or {}
    profile = profiles.get(key) or {}
    return profile if isinstance(profile, dict) and profile.get('status', 'ok') == 'ok' else {}


def get_moe_recommendation(model: ModelConfig) -> Dict[str, object]:
    return get_measured_profile(model, 'moe_placement')


def has_moe_recommendation(model: ModelConfig) -> bool:
    return bool(get_moe_recommendation(model))


def _normalized_tensor_overrides(value: object) -> List[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def infer_moe_placement_strategy(cpu_moe: bool, n_cpu_moe: int, tensor_overrides: Sequence[str] = ()) -> str:
    if bool(cpu_moe):
        return 'cpu_moe_all'
    value = max(0, int(n_cpu_moe or 0))
    if value > 0:
        return f'n_cpu_moe_{value}'
    if tuple(tensor_overrides or ()):
        return 'tensor_override'
    return ''


def moe_placement_payload(
    source: Dict[str, object],
    strategy: str = '',
    source_name: str = '',
) -> Dict[str, object]:
    tensor_overrides = _normalized_tensor_overrides(source.get('tensor_overrides', []))
    cpu_moe = _profile_bool(source, 'cpu_moe')
    n_cpu_moe = max(0, _profile_int(source, 'n_cpu_moe', 0))
    strategy = str(
        strategy
        or source.get('strategy')
        or source.get('placement_strategy')
        or source.get('moe_placement_strategy')
        or infer_moe_placement_strategy(cpu_moe, n_cpu_moe, tensor_overrides)
    ).strip()
    payload = {
        'strategy': strategy,
        'cpu_moe': bool(cpu_moe),
        'n_cpu_moe': int(n_cpu_moe),
        'tensor_overrides': tensor_overrides,
    }
    if source_name:
        payload['source'] = source_name
    for key in (
        'measured_candidate_name',
        'tokens_per_sec',
        'ctx',
        'ctx_per_slot',
        'runtime_profile',
        'moe_placement_mode',
        'fit_assisted_moe_placement',
        'benchmark_depth',
    ):
        if key in source:
            payload[key] = source.get(key)
    return payload


def profile_moe_placement(profile: Dict[str, object]) -> Dict[str, object]:
    nested = profile.get('moe_placement') if isinstance(profile, dict) else {}
    if isinstance(nested, dict) and (
        nested.get('strategy')
        or nested.get('placement_strategy')
        or nested.get('cpu_moe')
        or _profile_int(nested, 'n_cpu_moe', 0) > 0
        or _normalized_tensor_overrides(nested.get('tensor_overrides', []))
    ):
        return moe_placement_payload(nested, source_name=str(nested.get('source', '') or 'profile'))
    return moe_placement_payload(profile, source_name='flat_profile')


def moe_profile_key_for_context(ctx: int) -> str:
    ctx = max(0, int(ctx or 0))
    if ctx <= 16_384:
        return 'fast_chat'
    if ctx <= 49_152:
        return 'auto'
    if ctx <= 98_304:
        return 'hermes_ready'
    return 'long_context'


def profile_moe_placements_from_model(model: ModelConfig) -> Dict[str, Dict[str, object]]:
    moe_profile = get_moe_recommendation(model)
    source = moe_profile.get('profile_moe_placements') if moe_profile else {}
    if not isinstance(source, dict):
        return {}
    result: Dict[str, Dict[str, object]] = {}
    for key, value in source.items():
        if isinstance(value, dict):
            result[str(key)] = moe_placement_payload(value, source_name=str(value.get('source', '') or 'context_validation'))
    return result


def moe_recommendation_applied(model: ModelConfig) -> bool:
    profile = get_moe_recommendation(model)
    if not profile:
        return False
    if bool(getattr(model, 'cpu_moe', False)) != bool(profile.get('cpu_moe', False)):
        return False
    if max(0, int(getattr(model, 'n_cpu_moe', 0) or 0)) != max(0, int(profile.get('n_cpu_moe', 0) or 0)):
        return False
    if _normalized_tensor_overrides(getattr(model, 'tensor_overrides', [])) != _normalized_tensor_overrides(profile.get('tensor_overrides', [])):
        return False
    if bool(profile.get('ngl_required_for_moe_tuning', False)):
        try:
            return int(getattr(model, 'ngl', 0) or 0) == int(profile.get('ngl', 0) or 0)
        except Exception:
            return False
    return True


def _profile_int(profile: Dict[str, object], key: str, default: int = 0) -> int:
    try:
        return int(profile.get(key, default) or default)
    except Exception:
        return int(default or 0)


def _profile_bool(profile: Dict[str, object], key: str, default: bool = False) -> bool:
    value = profile.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _measured_profile_config(profile: Dict[str, object]) -> Dict[str, object]:
    raw = profile.get('config_fingerprint')
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _command_tokens_from_profile(profile: Dict[str, object]) -> List[str]:
    command = profile.get('command')
    if isinstance(command, list):
        return [str(item) for item in command]
    if not isinstance(command, str) or not command.strip():
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _command_flag_value(tokens: List[str], *flags: str) -> str:
    flag_set = {flag for flag in flags if flag}
    for idx, token in enumerate(tokens):
        if token in flag_set and idx + 1 < len(tokens):
            return str(tokens[idx + 1])
        for flag in flag_set:
            prefix = f'{flag}='
            if token.startswith(prefix):
                return token[len(prefix):]
    return ''


def _command_flag_values(tokens: List[str], *flags: str) -> List[str]:
    values: List[str] = []
    flag_set = {flag for flag in flags if flag}
    idx = 0
    while idx < len(tokens):
        token = str(tokens[idx])
        matched = ''
        for flag in flag_set:
            if token == flag:
                matched = flag
                break
            prefix = f'{flag}='
            if token.startswith(prefix):
                value = token[len(prefix):].strip()
                if value:
                    values.append(value)
                matched = flag
                break
        if matched and token == matched and idx + 1 < len(tokens):
            value = str(tokens[idx + 1]).strip()
            if value:
                values.append(value)
            idx += 2
            continue
        idx += 1
    return values


def _command_has_flag(tokens: List[str], *flags: str) -> bool:
    flag_set = {flag for flag in flags if flag}
    return any(token in flag_set for token in tokens)


def measured_profile_runtime_profile(
    model: ModelConfig,
    key: str,
) -> Optional[RuntimeProfile]:
    profile = get_measured_profile(model, key)
    if not profile:
        return None

    fingerprint = _measured_profile_config(profile)
    command_tokens = _command_tokens_from_profile(profile)
    replay_fields = (
        'runtime_fit',
        'fit_context',
        'runtime_no_warmup',
        'gpu_layers_mode',
        'batch_size',
        'ubatch_size',
        'runtime_profile',
        'placement_strategy',
        'cpu_moe',
        'n_cpu_moe',
        'tensor_overrides',
        'reasoning',
        'reasoning_budget',
        'reasoning_format',
        'mtp_enabled',
        'mtp_draft_n_max',
        'spec_type',
    )
    has_replay_data = (
        any(bool(profile.get(field)) for field in replay_fields)
        or any(field in fingerprint for field in (
            'engine_id',
            'runtime_profile',
            'fit',
            'gpu_layers',
            'kv_preset',
            'placement_strategy',
            'cpu_moe',
            'n_cpu_moe',
            'tensor_overrides',
            'reasoning',
            'reasoning_budget',
            'reasoning_format',
            'mtp_enabled',
            'mtp_draft_n_max',
        ))
        or any(token in command_tokens for token in (
            '-fit',
            '--fit',
            '-fitc',
            '--fit-ctx',
            '-ctk',
            '-ctv',
            '--cache-type-k',
            '--cache-type-v',
            '-cmoe',
            '--cpu-moe',
            '-ncmoe',
            '--n-cpu-moe',
            '-ot',
            '--override-tensor',
            '--override-tensors',
            '--reasoning',
            '--reasoning-budget',
            '--reasoning-format',
            '--spec-type',
            '--spec-draft-n-max',
        ))
    )
    if not has_replay_data:
        return None

    engine = str(profile.get('engine') or fingerprint.get('engine_id') or getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp')
    if 'mtp' in engine.lower():
        engine = ENGINE_LLAMA_CPP_MTP
    elif 'turboquant' in engine.lower():
        engine = 'turboquant'
    elif 'tq3' in engine.lower():
        engine = 'tq3'
    elif 'buun' in engine.lower():
        engine = 'buun'
    elif engine.strip().lower() == 'vllm':
        return None
    else:
        engine = 'llama.cpp'

    runtime_name = str(profile.get('runtime_profile') or fingerprint.get('runtime_profile') or '')
    ctx = _profile_int(profile, 'ctx', int(getattr(model, 'ctx', 0) or 0))
    parallel = max(1, _profile_int(profile, 'parallel', int(getattr(model, 'parallel', 1) or 1)))

    kv_preset = str(profile.get('kv_preset') or fingerprint.get('kv_preset') or '').strip()
    if not kv_preset:
        key_mode = _command_flag_value(command_tokens, '-ctk', '--cache-type-k')
        value_mode = _command_flag_value(command_tokens, '-ctv', '--cache-type-v')
        kv_preset = f'{key_mode}/{value_mode}' if key_mode and value_mode else 'default'

    flash_attn = str(profile.get('flash_attn_mode') or fingerprint.get('flash_attn') or '').strip()
    if not flash_attn and 'flash_attn' in profile:
        flash_attn = 'on' if bool(profile.get('flash_attn')) else 'off'
    if not flash_attn:
        flash_attn = _command_flag_value(command_tokens, '--flash-attn', '-fa') or 'on'

    fit = (
        _profile_bool(profile, 'runtime_fit')
        or bool(fingerprint.get('fit'))
        or (_command_flag_value(command_tokens, '-fit', '--fit').strip().lower() in ('on', 'true', '1', 'yes'))
    )
    fit_context = _profile_int(profile, 'fit_context', int(fingerprint.get('fit_context', 0) or 0))
    if fit_context <= 0:
        fit_context = int(_command_flag_value(command_tokens, '-fitc', '--fit-ctx') or 0)

    no_warmup = (
        _profile_bool(profile, 'runtime_no_warmup')
        or bool(fingerprint.get('no_warmup'))
        or _command_has_flag(command_tokens, '--no-warmup')
    )

    batch_size = _profile_int(profile, 'batch_size', int(fingerprint.get('batch_size', 0) or 0))
    if batch_size <= 0:
        batch_size = int(_command_flag_value(command_tokens, '--batch-size', '-b') or 0)
    ubatch_size = _profile_int(profile, 'ubatch_size', int(fingerprint.get('ubatch_size', 0) or 0))
    if ubatch_size <= 0:
        ubatch_size = int(_command_flag_value(command_tokens, '--ubatch-size', '-ub') or 0)

    gpu_layers_mode = str(profile.get('gpu_layers_mode') or '').strip().lower()
    fingerprint_gpu_layers = fingerprint.get('gpu_layers')
    if fit and (gpu_layers_mode == 'fit' or fingerprint_gpu_layers is None):
        gpu_layers = None
    else:
        command_ngl = _command_flag_value(command_tokens, '-ngl', '--n-gpu-layers')
        if command_ngl:
            gpu_layers = int(command_ngl)
        elif fingerprint_gpu_layers is not None:
            gpu_layers = int(fingerprint_gpu_layers)
        else:
            gpu_layers = _profile_int(profile, 'ngl', int(getattr(model, 'ngl', 0) or 0))

    turbo_profile = turbo_kv_profile_for_preset(kv_preset)
    family = str(profile.get('kv_family') or '').strip()
    if not family:
        family = 'turbo' if turbo_profile is not None else 'cache' if kv_preset and kv_preset != 'default' else 'default'
    extra_args = fingerprint.get('extra_args') if isinstance(fingerprint.get('extra_args'), list) else ()
    tensor_overrides = profile.get('tensor_overrides')
    if not isinstance(tensor_overrides, list):
        tensor_overrides = fingerprint.get('tensor_overrides') if isinstance(fingerprint.get('tensor_overrides'), list) else []
    tensor_values = [str(item).strip() for item in tensor_overrides if str(item).strip()]
    for value in _command_flag_values(command_tokens, '-ot', '--override-tensor', '--override-tensors'):
        if value not in tensor_values:
            tensor_values.append(value)
    cpu_moe = (
        _profile_bool(profile, 'cpu_moe')
        or bool(fingerprint.get('cpu_moe'))
        or _command_has_flag(command_tokens, '-cmoe', '--cpu-moe')
    )
    n_cpu_moe = _profile_int(profile, 'n_cpu_moe', int(fingerprint.get('n_cpu_moe', 0) or 0))
    if n_cpu_moe <= 0:
        n_cpu_moe = int(_command_flag_value(command_tokens, '-ncmoe', '--n-cpu-moe') or 0)
    placement_strategy = str(profile.get('placement_strategy') or fingerprint.get('placement_strategy') or '').strip()
    if not placement_strategy:
        if cpu_moe:
            placement_strategy = 'cpu_moe_all'
        elif n_cpu_moe > 0:
            placement_strategy = f'n_cpu_moe_{n_cpu_moe}'
        elif tensor_values:
            placement_strategy = 'tensor_override'
    placement_payload = profile_moe_placement(profile)
    if (
        placement_payload.get('strategy')
        or placement_payload.get('cpu_moe')
        or int(placement_payload.get('n_cpu_moe', 0) or 0) > 0
        or placement_payload.get('tensor_overrides')
    ):
        placement_strategy = str(placement_payload.get('strategy') or '')
        cpu_moe = bool(placement_payload.get('cpu_moe', False))
        n_cpu_moe = max(0, int(placement_payload.get('n_cpu_moe', 0) or 0))
        tensor_values = _normalized_tensor_overrides(placement_payload.get('tensor_overrides', []))
    reasoning = str(profile.get('reasoning') or fingerprint.get('reasoning') or '').strip().lower()
    if not reasoning:
        reasoning = _command_flag_value(command_tokens, '--reasoning', '-rea').strip().lower()
    if reasoning not in ('', 'on', 'off', 'auto'):
        reasoning = ''
    reasoning_budget = _profile_int(profile, 'reasoning_budget', int(fingerprint.get('reasoning_budget', -1) or -1))
    if reasoning_budget < 0:
        budget_text = _command_flag_value(command_tokens, '--reasoning-budget')
        reasoning_budget = int(budget_text) if budget_text else -1
    reasoning_format = str(profile.get('reasoning_format') or fingerprint.get('reasoning_format') or '').strip().lower()
    if not reasoning_format:
        reasoning_format = _command_flag_value(command_tokens, '--reasoning-format').strip().lower()
    mtp_enabled = (
        _profile_bool(profile, 'mtp_enabled')
        or bool(fingerprint.get('mtp_enabled'))
        or _command_flag_value(command_tokens, '--spec-type').strip().lower() == 'mtp'
    )
    mtp_draft_n_max = _profile_int(profile, 'mtp_draft_n_max', int(fingerprint.get('mtp_draft_n_max', 0) or 0))
    if mtp_draft_n_max <= 0:
        mtp_draft_n_max = int(_command_flag_value(command_tokens, '--spec-draft-n-max') or 0)
    return RuntimeProfile(
        engine_id=engine,
        name=runtime_name or f'measured_{key}',
        ctx_size=max(1, int(ctx or getattr(model, 'ctx', 1) or 1)),
        gpu_layers=gpu_layers,
        parallel=parallel,
        kv_preset=kv_preset or 'default',
        flash_attn=flash_attn or 'on',
        batch_size=max(0, int(batch_size or 0)),
        ubatch_size=max(0, int(ubatch_size or 0)),
        fit=bool(fit),
        fit_context=max(0, int(fit_context or 0)),
        no_warmup=bool(no_warmup),
        extra_args=tuple(str(item) for item in extra_args),
        kv_family=family,
        kv_quality_tier=str(profile.get('kv_quality_tier') or getattr(turbo_profile, 'quality_tier', '') or ''),
        kv_compression_tier=str(profile.get('kv_compression_tier') or getattr(turbo_profile, 'compression_tier', '') or ''),
        kv_score_penalty=float(profile.get('kv_score_penalty', getattr(turbo_profile, 'score_penalty', 0.0) or 0.0) or 0.0),
        benchmark_depth=str(profile.get('benchmark_depth') or ''),
        fit_discovery_phase=str(profile.get('fit_discovery_phase') or ''),
        viable_ngl=max(0, int(profile.get('viable_ngl', 0) or 0)),
        viable_ngl_source=str(profile.get('viable_ngl_source') or ''),
        fit_selected_ngl=max(0, int(profile.get('fit_selected_ngl', 0) or 0)),
        fit_selected_ngl_source=str(profile.get('fit_selected_ngl_source') or ''),
        fit_log_excerpt=str(profile.get('fit_log_excerpt') or ''),
        placement_strategy=placement_strategy,
        cpu_moe=bool(cpu_moe),
        n_cpu_moe=max(0, int(n_cpu_moe or 0)),
        tensor_overrides=tuple(tensor_values),
        reasoning=reasoning,
        reasoning_budget=reasoning_budget,
        reasoning_format=reasoning_format,
        mtp_enabled=bool(mtp_enabled),
        mtp_draft_n_max=clamp_mtp_draft(mtp_draft_n_max, default=3) if mtp_enabled else 0,
    )


def apply_measured_profile(model: ModelConfig, key: str) -> Tuple[bool, str]:
    profile = get_measured_profile(model, key)
    if not profile:
        return False, f'no measured {key} profile'
    int_fields = ('ctx', 'parallel', 'threads', 'ngl', 'output', 'cache_ram', 'memory_reserve_percent')
    for field in int_fields:
        if field in profile:
            try:
                setattr(model, field, int(profile[field]))
            except Exception:
                pass
    if 'temp' in profile:
        try:
            model.temp = float(profile['temp'])
        except Exception:
            pass
    if 'flash_attn' in profile:
        model.flash_attn = bool(profile['flash_attn'])
    if 'jinja' in profile:
        model.jinja = bool(profile['jinja'])
    if 'mtp_enabled' in profile:
        model.mtp_enabled = bool(profile.get('mtp_enabled'))
    if 'mtp_draft_n_max' in profile:
        model.mtp_draft_n_max = clamp_mtp_draft(profile.get('mtp_draft_n_max'), default=3)
    if isinstance(profile.get('extra_args'), list):
        model.extra_args = [str(item) for item in profile.get('extra_args', [])]
    placement = profile_moe_placement(profile)
    if (
        placement.get('strategy')
        or placement.get('cpu_moe')
        or int(placement.get('n_cpu_moe', 0) or 0) > 0
        or placement.get('tensor_overrides')
    ):
        model.moe_placement_strategy = str(placement.get('strategy') or '')
        model.cpu_moe = bool(placement.get('cpu_moe', False))
        model.n_cpu_moe = max(0, int(placement.get('n_cpu_moe', 0) or 0))
        model.tensor_overrides = _normalized_tensor_overrides(placement.get('tensor_overrides', []))
    model.optimize_mode = f'measured_{key}'
    model.optimize_tier = 'measured'
    return True, (
        f'measured {key}: ctx={model.ctx} parallel={model.parallel} '
        f'threads={model.threads} ngl={model.ngl} '
        f'{float(profile.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s'
    )


def apply_moe_recommendation(model: ModelConfig) -> Tuple[bool, str]:
    profile = get_measured_profile(model, 'moe_placement')
    if not profile:
        return False, 'no measured MoE placement recommendation'
    if str(profile.get('status', '') or '').lower() != 'ok':
        return False, 'measured MoE placement recommendation is not successful'

    candidate_name = str(profile.get('measured_candidate_name') or profile.get('runtime_profile') or 'unknown').strip()
    model.moe_placement_strategy = f'measured:moe_tuning:{candidate_name or "unknown"}'
    model.cpu_moe = bool(profile.get('cpu_moe', False))
    model.n_cpu_moe = max(0, int(profile.get('n_cpu_moe', 0) or 0))
    tensor_overrides = profile.get('tensor_overrides', [])
    if isinstance(tensor_overrides, (list, tuple)):
        model.tensor_overrides = [str(item).strip() for item in tensor_overrides if str(item).strip()]
    elif str(tensor_overrides or '').strip():
        model.tensor_overrides = [str(tensor_overrides).strip()]
    else:
        model.tensor_overrides = []
    if bool(profile.get('ngl_required_for_moe_tuning', False)) and 'ngl' in profile:
        try:
            model.ngl = int(profile.get('ngl') or model.ngl)
        except Exception:
            pass

    measured_profiles = dict(getattr(model, 'measured_profiles', {}) or {})
    updated = dict(profile)
    updated['applied_from'] = 'moe_placement'
    updated['applied_at'] = datetime.now().isoformat(timespec='seconds')
    updated['tuning_run_id'] = str(profile.get('tuning_run_id', '') or '')
    updated['measured_candidate_name'] = candidate_name
    updated['placement_strategy'] = model.moe_placement_strategy
    measured_profiles['moe_placement'] = updated
    model.measured_profiles = measured_profiles
    return True, (
        f'MoE recommendation applied: {model.moe_placement_strategy} '
        f'ngl={model.ngl} {float(profile.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s'
    )


def moe_overlay_expected_flags(overlay: Dict[str, object]) -> List[str]:
    flags: List[str] = []
    if bool(overlay.get('cpu_moe', False)):
        flags.append('-cmoe')
    else:
        n_cpu_moe = max(0, int(overlay.get('n_cpu_moe', 0) or 0))
        if n_cpu_moe > 0:
            flags += ['-ncmoe', str(n_cpu_moe)]
    for override in _normalized_tensor_overrides(overlay.get('tensor_overrides', [])):
        flags += ['-ot', override]
    return flags


def moe_overlay_flags_preview(overlay: Dict[str, object]) -> str:
    return shlex.join(moe_overlay_expected_flags(overlay))


def moe_overlay_log_text(overlay: Dict[str, object], prefix: str = 'Full Suite MoE overlay') -> str:
    flags = moe_overlay_flags_preview(overlay)
    tensor_overrides = _normalized_tensor_overrides(overlay.get('tensor_overrides', []))
    return (
        f'{prefix}: source={overlay.get("source") or "unknown"} '
        f'winner={overlay.get("candidate_name") or overlay.get("moe_placement_strategy") or "unknown"} '
        f'cpu_moe={bool(overlay.get("cpu_moe", False))} '
        f'n_cpu_moe={max(0, int(overlay.get("n_cpu_moe", 0) or 0))} '
        f'tensor_overrides={tensor_overrides} flags="{flags}"'
    )


def build_runtime_overlay_from_moe_recommendation(
    model: ModelConfig,
    source: str = 'measured_profiles.moe_placement',
) -> Dict[str, object]:
    profile = get_moe_recommendation(model)
    if not profile:
        return {}
    candidate_name = str(profile.get('measured_candidate_name') or profile.get('runtime_profile') or '').strip()
    overlay: Dict[str, object] = {
        'source': source or 'measured_profiles.moe_placement',
        'candidate_name': candidate_name,
        'moe_placement_strategy': str(
            profile.get('placement_strategy')
            or f'measured:moe_tuning:{candidate_name or "unknown"}'
        ),
        'cpu_moe': bool(profile.get('cpu_moe', False)),
        'n_cpu_moe': max(0, int(profile.get('n_cpu_moe', 0) or 0)),
        'tensor_overrides': _normalized_tensor_overrides(profile.get('tensor_overrides', [])),
        'ngl_required_for_moe_tuning': bool(profile.get('ngl_required_for_moe_tuning', False)),
        'tuning_run_id': str(profile.get('tuning_run_id', '') or ''),
        'tokens_per_sec': float(profile.get('tokens_per_sec', 0.0) or 0.0),
    }
    if overlay['ngl_required_for_moe_tuning'] and 'ngl' in profile:
        try:
            overlay['ngl'] = int(profile.get('ngl') or 0)
        except Exception:
            pass
    overlay['expected_flags'] = moe_overlay_expected_flags(overlay)
    overlay['expected_flags_preview'] = moe_overlay_flags_preview(overlay)
    return overlay


def model_with_moe_overlay(model: ModelConfig, overlay: Dict[str, object]) -> ModelConfig:
    candidate = ModelConfig(**asdict(model))
    if not overlay:
        return candidate
    candidate.moe_placement_strategy = str(overlay.get('moe_placement_strategy') or '')
    candidate.cpu_moe = bool(overlay.get('cpu_moe', False))
    candidate.n_cpu_moe = max(0, int(overlay.get('n_cpu_moe', 0) or 0))
    candidate.tensor_overrides = _normalized_tensor_overrides(overlay.get('tensor_overrides', []))
    if bool(overlay.get('ngl_required_for_moe_tuning', False)) and 'ngl' in overlay:
        try:
            candidate.ngl = int(overlay.get('ngl') or candidate.ngl)
        except Exception:
            pass
    return candidate


def runtime_profile_with_overlay(runtime_profile: RuntimeProfile, overlay: Dict[str, object]) -> RuntimeProfile:
    if not overlay:
        return runtime_profile
    gpu_layers = runtime_profile.gpu_layers
    if bool(overlay.get('ngl_required_for_moe_tuning', False)) and 'ngl' in overlay:
        try:
            gpu_layers = int(overlay.get('ngl') or 0)
        except Exception:
            gpu_layers = runtime_profile.gpu_layers
    placement = str(overlay.get('moe_placement_strategy') or '')
    if placement.startswith('measured:moe_tuning:'):
        placement = placement.rsplit(':', 1)[-1]
    return replace(
        runtime_profile,
        gpu_layers=gpu_layers,
        placement_strategy=placement,
        cpu_moe=bool(overlay.get('cpu_moe', False)),
        n_cpu_moe=max(0, int(overlay.get('n_cpu_moe', 0) or 0)),
        tensor_overrides=tuple(_normalized_tensor_overrides(overlay.get('tensor_overrides', []))),
    )


def full_suite_recommended_profile_key(model: ModelConfig) -> str:
    for key in ('opencode_ready', 'auto', 'fast_chat'):
        if get_measured_profile(model, key):
            return key
    return ''


SUITE_BENCHMARK_STATE_FIELDS = {
    'last_benchmark_tokens_per_sec',
    'last_benchmark_seconds',
    'last_benchmark_profile',
    'last_benchmark_results',
    'measured_profiles',
    'benchmark_runs',
    'benchmark_fingerprint',
    'default_benchmark_status',
    'default_benchmark_at',
    'engine_benchmark_store',
    'last_opencode_benchmark_score',
    'last_opencode_benchmark_seconds',
    'last_opencode_benchmark_profile',
    'last_opencode_benchmark_results',
    'last_hermes_benchmark_score',
    'last_hermes_benchmark_seconds',
    'last_hermes_benchmark_profile',
    'last_hermes_benchmark_results',
}


def _clone_model(model: ModelConfig) -> ModelConfig:
    return ModelConfig(**asdict(model))


def _current_model_for_suite(app, fallback: ModelConfig) -> ModelConfig:
    if hasattr(app, 'get_model'):
        try:
            current = app.get_model(fallback.id)
            if current is not None:
                return current
        except Exception:
            pass
    return fallback


def _suite_restore_config_fields(
    current: ModelConfig,
    original: ModelConfig,
    prior_profiles: Optional[Dict[str, Dict[str, object]]] = None,
) -> ModelConfig:
    restored = _clone_model(current)
    for field in getattr(ModelConfig, '__dataclass_fields__', {}):
        if field in SUITE_BENCHMARK_STATE_FIELDS:
            continue
        setattr(restored, field, getattr(original, field))
    if prior_profiles:
        merged = {
            str(key): (dict(value) if isinstance(value, dict) else value)
            for key, value in prior_profiles.items()
        }
        for key, value in (getattr(current, 'measured_profiles', {}) or {}).items():
            merged[str(key)] = dict(value) if isinstance(value, dict) else value
        restored.measured_profiles = merged
    return restored

def _annotate_latest_stage_run(
    model: ModelConfig,
    kind: str,
    suite_run_id: str,
    stage: str,
    overlay: Optional[Dict[str, object]] = None,
    profile_used: str = '',
) -> bool:
    overlay = dict(overlay or {})
    changed = False
    for run in list(getattr(model, 'benchmark_runs', []) or []):
        if str(run.get('kind', '') or '') != kind:
            continue
        if str(run.get('suite_run_id', '') or '') and str(run.get('suite_run_id')) != suite_run_id:
            continue
        run['suite_run_id'] = suite_run_id
        run['suite_stage'] = stage
        if overlay:
            run['uses_suite_overlay'] = True
            run['moe_overlay_source'] = str(overlay.get('source') or 'measured_profiles.moe_placement')
            run['moe_overlay_candidate'] = str(overlay.get('candidate_name') or '')
            run['moe_overlay_cpu_moe'] = bool(overlay.get('cpu_moe', False))
            run['moe_overlay_n_cpu_moe'] = max(0, int(overlay.get('n_cpu_moe', 0) or 0))
            run['moe_overlay_tensor_overrides'] = _normalized_tensor_overrides(overlay.get('tensor_overrides', []))
            run['moe_overlay_expected_flags'] = list(overlay.get('expected_flags') or moe_overlay_expected_flags(overlay))
            run['moe_overlay_flags'] = str(overlay.get('expected_flags_preview') or moe_overlay_flags_preview(overlay))
        if profile_used:
            run['profile_used'] = profile_used
        for record in list(run.get('records', []) or []):
            if isinstance(record, dict):
                record['suite_run_id'] = suite_run_id
                record['suite_stage'] = stage
                if overlay:
                    record['uses_suite_overlay'] = True
                    record['moe_overlay_source'] = str(overlay.get('source') or 'measured_profiles.moe_placement')
                    record['moe_overlay_candidate'] = str(overlay.get('candidate_name') or '')
                    record['moe_overlay_expected_flags'] = list(overlay.get('expected_flags') or moe_overlay_expected_flags(overlay))
                    record['moe_overlay_flags'] = str(overlay.get('expected_flags_preview') or moe_overlay_flags_preview(overlay))
        for winner in (run.get('winners', {}) or {}).values():
            if isinstance(winner, dict):
                winner['suite_run_id'] = suite_run_id
                winner['suite_stage'] = stage
                if overlay:
                    winner['uses_suite_overlay'] = True
                    winner['moe_overlay_source'] = str(overlay.get('source') or 'measured_profiles.moe_placement')
                    winner['moe_overlay_candidate'] = str(overlay.get('candidate_name') or '')
                    winner['moe_overlay_expected_flags'] = list(overlay.get('expected_flags') or moe_overlay_expected_flags(overlay))
                    winner['moe_overlay_flags'] = str(overlay.get('expected_flags_preview') or moe_overlay_flags_preview(overlay))
        changed = True
        break
    return changed


def _write_full_suite_log(
    app,
    model: ModelConfig,
    suite_run_id: str,
    engine: str,
    run: Dict[str, object],
    stage_records: Sequence[Dict[str, object]],
    overlay: Dict[str, object],
) -> str:
    root = (
        CACHE_DIR
        / 'suites'
        / _safe_tuning_path_part(getattr(model, 'id', 'model'))
        / _safe_tuning_path_part(engine or 'engine')
        / _safe_tuning_path_part(suite_run_id)
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'suite.json'
    try:
        fingerprint = app.model_fingerprint(model)
    except Exception:
        fingerprint = benchmark_config_fingerprint(model)
    payload = {
        'suite_run_id': suite_run_id,
        'model_id': getattr(model, 'id', ''),
        'model_path': getattr(model, 'path', ''),
        'model_fingerprint': fingerprint,
        'engine_id': engine,
        'run': dict(run),
        'stages': [dict(item) for item in stage_records],
        'moe_overlay': dict(overlay or {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')
    return str(path)


def _moe_tuning_skip_reason_for_app(app, model: ModelConfig, depth: str = 'full') -> str:
    try:
        hardware = app.hardware_profile(refresh=True)
    except Exception:
        hardware = HardwareProfile()
    try:
        baseline = _moe_tuning_baseline_profile(app, model, depth)
    except Exception:
        baseline = RuntimeProfile(
            engine_id=getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp',
            name='baseline_current',
            ctx_size=max(1, int(getattr(model, 'ctx', 1) or 1)),
            gpu_layers=int(getattr(model, 'ngl', 0) or 0),
            parallel=max(1, int(getattr(model, 'parallel', 1) or 1)),
        )
    capabilities = _moe_tuning_capabilities(app, baseline)
    try:
        engine = app.active_engine_key_for_model(model)
    except Exception:
        engine = baseline.engine_id
    return moe_tuning_eligibility_reason(model, hardware, capabilities, engine)


def benchmark_full_suite(
    app,
    model: ModelConfig,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
    depth: str = 'full',
    moe_runner: Optional[Callable[..., Tuple[bool, str]]] = None,
    smart_runner: Optional[Callable[..., Tuple[bool, str]]] = None,
    hermes_runner: Optional[Callable[..., Tuple[bool, str]]] = None,
    opencode_runner: Optional[Callable[..., Tuple[bool, str]]] = None,
) -> Tuple[bool, str]:
    depth_key = 'fast' if str(depth or '').strip().lower() == 'fast' else 'full'
    original = _clone_model(model)
    started_at = datetime.now().isoformat(timespec='seconds')
    suite_run_id = f'full-suite-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    try:
        engine = app.active_engine_key_for_model(model)
    except Exception:
        engine = getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp'
    try:
        hardware = app.hardware_profile(refresh=True).short_summary()
    except Exception:
        hardware = ''
    stage_records: List[Dict[str, object]] = []
    warnings: List[str] = []
    recommendations: Dict[str, object] = {}
    overlay: Dict[str, object] = {}
    current_suite_moe_winner = False

    def emit(message: str, event: str = 'benchmark_phase', stage: str = ''):
        if progress:
            progress(message)
        emit_benchmark_event(
            progress,
            event,
            model,
            'full_suite',
            message=message,
            phase=stage or compact_message(message),
            completed=len(stage_records),
            total=5,
            records=stage_records,
        )

    def add_stage(stage: str, status: str, detail: str = '', **extra: object):
        row = {
            'objective': 'full_suite',
            'stage': stage,
            'status': status,
            'detail': concise_failure(detail, limit=500),
            'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
        }
        row.update(extra)
        stage_records.append(row)
        return row

    def persist_suite(status: str, ended_at: str = ''):
        current = _current_model_for_suite(app, model)
        saved = _suite_restore_config_fields(current, original)
        winners: Dict[str, Dict[str, object]] = {}
        moe = get_moe_recommendation(saved)
        if moe:
            winners['moe_placement'] = dict(moe)
        profile_key = str(recommendations.get('default_profile', '') or '')
        if profile_key and get_measured_profile(saved, profile_key):
            winners[profile_key] = get_measured_profile(saved, profile_key)
        run = build_benchmark_run(
            suite_run_id,
            'full_suite',
            status,
            stage_records,
            winners,
            started_at,
            ended_at,
            hardware,
        )
        run['suite_run_id'] = suite_run_id
        run['depth'] = depth_key
        run['stages'] = {
            str(row.get('stage', '')): {
                key: value
                for key, value in row.items()
                if key not in ('objective', 'benchmarked_at')
            }
            for row in stage_records
        }
        run['warnings'] = list(warnings)
        run['recommendations'] = dict(recommendations)
        if overlay:
            run['uses_suite_overlay'] = True
            run['moe_overlay'] = dict(overlay)
        try:
            run['suite_log_json'] = _write_full_suite_log(app, saved, suite_run_id, engine, run, stage_records, overlay)
        except Exception as exc:
            warnings.append(f'full suite log write failed: {compact_message(str(exc))}')
            run['warnings'] = list(warnings)
        run['summary'] = full_suite_summary_text(stage_records, recommendations, warnings)
        upsert_benchmark_run(saved, run)
        app.add_or_update(saved)
        return saved, run

    moe_runner = moe_runner or benchmark_moe_placement_tuning
    smart_runner = smart_runner or benchmark_best_optimization
    if hermes_runner is None:
        from .hermes_benchmark import benchmark_hermes_workflow as hermes_runner
    if opencode_runner is None:
        from .opencode_benchmark import benchmark_opencode_workflow as opencode_runner

    try:
        emit(f'Full Suite Benchmark started: model={model.name or model.id} engine={engine}', 'benchmark_started', 'preflight')
        preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'full_suite', progress, cancel_token)
        add_stage('preflight', 'ok' if preflight_ok else 'failed', preflight_msg)
        if not preflight_ok:
            ended_at = datetime.now().isoformat(timespec='seconds')
            persist_suite('failed', ended_at)
            return False, preflight_msg
        persist_suite('running')

        current = _current_model_for_suite(app, model)
        skip_reason = _moe_tuning_skip_reason_for_app(app, current, depth_key)
        if skip_reason:
            add_stage('moe_placement', 'skipped', skip_reason)
            warnings.append(f'MoE placement skipped: {skip_reason}')
        else:
            prior_profiles = dict(getattr(current, 'measured_profiles', {}) or {})
            emit('Full Suite stage: MoE placement tuning', stage='moe placement')
            ok, msg = moe_runner(app, current, progress=progress, cancel_token=cancel_token, depth=depth_key)
            current = _current_model_for_suite(app, current)
            current = _suite_restore_config_fields(current, original, prior_profiles)
            _annotate_latest_stage_run(current, 'moe_tuning', suite_run_id, 'moe_placement')
            app.add_or_update(current)
            add_stage('moe_placement', 'done' if ok else 'failed', msg)
            if ok:
                overlay = build_runtime_overlay_from_moe_recommendation(current, source='current_suite')
                if overlay:
                    overlay['suite_run_id'] = suite_run_id
                    current_suite_moe_winner = True
                    recommendations['moe_placement'] = overlay.get('candidate_name') or overlay.get('moe_placement_strategy')
                    emit(moe_overlay_log_text(overlay, prefix='Full Suite MoE winner'), stage='moe placement')
                    if stage_records and stage_records[-1].get('stage') == 'moe_placement':
                        stage_records[-1]['moe_overlay'] = dict(overlay)
                        stage_records[-1]['moe_overlay_source'] = overlay.get('source')
                        stage_records[-1]['moe_overlay_candidate'] = overlay.get('candidate_name')
                        stage_records[-1]['moe_overlay_expected_flags'] = list(overlay.get('expected_flags') or [])
                        stage_records[-1]['moe_overlay_flags'] = overlay.get('expected_flags_preview', '')
            else:
                warnings.append(f'MoE placement failed: {compact_message(msg)}')
        persist_suite('running')

        current = _current_model_for_suite(app, model)
        if not overlay and not current_suite_moe_winner:
            overlay = build_runtime_overlay_from_moe_recommendation(current, source='stored_recommendation')
        if overlay:
            emit(moe_overlay_log_text(overlay, prefix='Full Suite MoE overlay'), stage='smart benchmark')
        smart_model = model_with_moe_overlay(current, overlay) if overlay else _clone_model(current)
        prior_profiles = dict(getattr(current, 'measured_profiles', {}) or {})
        emit('Full Suite stage: smart benchmark', stage='smart benchmark')
        ok, msg = smart_runner(app, smart_model, progress=progress, cancel_token=cancel_token)
        current = _current_model_for_suite(app, smart_model)
        current = _suite_restore_config_fields(current, original, prior_profiles)
        _annotate_latest_stage_run(current, 'server', suite_run_id, 'model_benchmark', overlay)
        profile_key = full_suite_recommended_profile_key(current)
        if profile_key:
            recommendations['default_profile'] = profile_key
        app.add_or_update(current)
        add_stage(
            'model_benchmark',
            'done' if ok else 'failed',
            msg,
            uses_suite_overlay=bool(overlay),
            profile_used=profile_key,
            moe_overlay_source=str(overlay.get('source', '') or '') if overlay else '',
            moe_overlay_candidate=str(overlay.get('candidate_name', '') or '') if overlay else '',
            moe_overlay_expected_flags=list(overlay.get('expected_flags') or []) if overlay else [],
            moe_overlay_flags=str(overlay.get('expected_flags_preview', '') or '') if overlay else '',
        )
        if not ok:
            warnings.append(f'Smart benchmark failed: {compact_message(msg)}')
        persist_suite('running')

        current = _current_model_for_suite(app, model)
        workflow_profile = full_suite_recommended_profile_key(current)
        workflow_overlay = overlay or build_runtime_overlay_from_moe_recommendation(current)
        workflow_model = model_with_moe_overlay(current, workflow_overlay) if workflow_overlay else _clone_model(current)
        prior_profiles = dict(getattr(current, 'measured_profiles', {}) or {})
        emit('Full Suite stage: Hermes workflow benchmark', stage='hermes benchmark')
        ok, msg = hermes_runner(app, workflow_model, progress=progress, cancel_token=cancel_token)
        current = _current_model_for_suite(app, workflow_model)
        current = _suite_restore_config_fields(current, original, prior_profiles)
        _annotate_latest_stage_run(current, 'hermes', suite_run_id, 'hermes', workflow_overlay, workflow_profile)
        app.add_or_update(current)
        add_stage('hermes', 'passed' if ok else 'failed', msg, uses_suite_overlay=bool(workflow_overlay), profile_used=workflow_profile)
        if not ok:
            warnings.append(f'Hermes benchmark failed: {compact_message(msg)}')
        persist_suite('running')

        current = _current_model_for_suite(app, model)
        workflow_profile = full_suite_recommended_profile_key(current)
        workflow_overlay = overlay or build_runtime_overlay_from_moe_recommendation(current)
        workflow_model = model_with_moe_overlay(current, workflow_overlay) if workflow_overlay else _clone_model(current)
        prior_profiles = dict(getattr(current, 'measured_profiles', {}) or {})
        emit('Full Suite stage: OpenCode workflow benchmark', stage='opencode benchmark')
        ok, msg = opencode_runner(app, workflow_model, progress=progress, cancel_token=cancel_token)
        current = _current_model_for_suite(app, workflow_model)
        current = _suite_restore_config_fields(current, original, prior_profiles)
        _annotate_latest_stage_run(current, 'opencode', suite_run_id, 'opencode', workflow_overlay, workflow_profile)
        app.add_or_update(current)
        add_stage('opencode', 'passed' if ok else 'failed', msg, uses_suite_overlay=bool(workflow_overlay), profile_used=workflow_profile)
        if not ok:
            warnings.append(f'OpenCode benchmark failed: {compact_message(msg)}')

    except CancelledError:
        ended_at = datetime.now().isoformat(timespec='seconds')
        add_stage('aborted', 'aborted', 'user requested abort')
        saved, _run = persist_suite('aborted', ended_at)
        msg = 'Full Suite Benchmark aborted; managed processes stopped'
        emit(msg, 'benchmark_aborted', 'aborted')
        return False, msg

    ended_at = datetime.now().isoformat(timespec='seconds')
    failed_stages = [
        row for row in stage_records
        if str(row.get('status', '') or '').lower() in ('failed', 'aborted')
        and str(row.get('stage', '') or '') not in ('moe_placement',)
    ]
    status = 'done' if not failed_stages else 'failed'
    saved, run = persist_suite(status, ended_at)
    msg = run.get('summary', '') or full_suite_summary_text(stage_records, recommendations, warnings)
    if progress:
        progress(msg)
    emit_benchmark_event(
        progress,
        'benchmark_done' if status == 'done' else 'benchmark_error',
        saved,
        'full_suite',
        message=msg,
        phase='summary',
        completed=len(stage_records),
        total=len(stage_records),
        records=stage_records,
    )
    return status == 'done', msg


def full_suite_summary_text(
    stage_records: Sequence[Dict[str, object]],
    recommendations: Optional[Dict[str, object]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> str:
    recommendations = dict(recommendations or {})
    warnings = list(warnings or [])
    parts = []
    for name in ('moe_placement', 'model_benchmark', 'hermes', 'opencode'):
        row = next((item for item in stage_records if str(item.get('stage', '') or '') == name), None)
        if not row:
            continue
        status = str(row.get('status', '') or '')
        if name == 'moe_placement' and recommendations.get('moe_placement'):
            parts.append(f'moe={recommendations["moe_placement"]}')
        elif name == 'model_benchmark' and recommendations.get('default_profile'):
            parts.append(f'profile={recommendations["default_profile"]}')
        else:
            parts.append(f'{name}={status}')
    if warnings:
        parts.append(f'{len(warnings)} warning(s)')
    return 'Full Suite Benchmark complete: ' + ', '.join(parts) if parts else 'Full Suite Benchmark complete'


def suite_run_recommended_profile_key(model: ModelConfig, run: Optional[Dict[str, object]] = None) -> str:
    recommendations = (run or {}).get('recommendations') if isinstance(run, dict) else {}
    if isinstance(recommendations, dict):
        key = str(recommendations.get('default_profile', '') or '').strip()
        if key and get_measured_profile(model, key):
            return key
    return full_suite_recommended_profile_key(model)


def apply_full_suite_profile_recommendation(
    model: ModelConfig,
    run: Optional[Dict[str, object]] = None,
) -> Tuple[bool, str]:
    key = suite_run_recommended_profile_key(model, run)
    if not key:
        return False, 'no measured model profile recommendation to apply'
    moe_snapshot = {
        'moe_placement_strategy': getattr(model, 'moe_placement_strategy', ''),
        'cpu_moe': bool(getattr(model, 'cpu_moe', False)),
        'n_cpu_moe': max(0, int(getattr(model, 'n_cpu_moe', 0) or 0)),
        'tensor_overrides': list(getattr(model, 'tensor_overrides', []) or []),
    }
    ok, msg = apply_measured_profile(model, key)
    model.moe_placement_strategy = str(moe_snapshot['moe_placement_strategy'])
    model.cpu_moe = bool(moe_snapshot['cpu_moe'])
    model.n_cpu_moe = int(moe_snapshot['n_cpu_moe'])
    model.tensor_overrides = list(moe_snapshot['tensor_overrides'])
    if not ok:
        return False, msg
    return True, f'Full Suite profile applied ({key}): {msg}'


def _add_or_update_without_export(app, model: ModelConfig):
    try:
        app.add_or_update(model, sync_exports=False)
    except TypeError:
        app.add_or_update(model)


def apply_full_suite_recommendations(
    app,
    model: ModelConfig,
    run: Optional[Dict[str, object]] = None,
) -> Tuple[bool, str]:
    messages: List[str] = []
    profile_ok, profile_msg = apply_full_suite_profile_recommendation(model, run)
    if profile_ok:
        messages.append(profile_msg)

    moe_ok = False
    moe_msg = ''
    if has_moe_recommendation(model):
        moe_ok, moe_msg = apply_moe_recommendation(model)
        if moe_ok:
            messages.append(moe_msg)

    if not messages:
        return False, profile_msg or moe_msg or 'no Full Suite recommendations to apply'

    _add_or_update_without_export(app, model)
    sync_msg = sync_opencode_after_tuning(app)
    profile_key = suite_run_recommended_profile_key(model, run)
    moe_name = str(get_moe_recommendation(model).get('measured_candidate_name', '') or '')
    if hasattr(app, 'append_log'):
        try:
            app.append_log(
                model.id,
                f'Applied full suite recommendations: moe={moe_name or "-"} profile={profile_key or "-"}',
            )
        except Exception:
            pass
    return True, f'Applied Full Suite recommendations: {" | ".join(messages)} | {sync_msg}'


def model_from_measured_profile(model: ModelConfig, key: str) -> Optional[ModelConfig]:
    candidate = ModelConfig(**asdict(model))
    ok, _msg = apply_measured_profile(candidate, key)
    return candidate if ok else None


def model_and_runtime_profile_from_measured_profile(
    model: ModelConfig,
    key: str,
) -> Tuple[Optional[ModelConfig], Optional[RuntimeProfile]]:
    candidate = model_from_measured_profile(model, key)
    if candidate is None:
        return None, None
    runtime_profile = measured_profile_runtime_profile(model, key)
    if runtime_profile is not None:
        candidate.ctx = int(runtime_profile.ctx_size or candidate.ctx)
        candidate.parallel = max(1, int(runtime_profile.parallel or candidate.parallel or 1))
        if runtime_profile.gpu_layers is not None:
            candidate.ngl = int(runtime_profile.gpu_layers)
    return candidate, runtime_profile


def measured_profile_ctx_per_slot(model: ModelConfig, key: str) -> int:
    profile = get_measured_profile(model, key)
    if not profile:
        return 0
    ctx = int(profile.get('ctx', 0) or 0)
    parallel = max(1, int(profile.get('parallel', 1) or 1))
    return ctx // parallel


def _set_extra_arg(args: List[str], flag: str, value: str) -> List[str]:
    cleaned = strip_extra_args(args, flag)
    return cleaned + [flag, value]


def _strip_runtime_tuning_args(args: List[str]) -> List[str]:
    return strip_extra_args(
        list(args or []),
        '--batch-size',
        '--ubatch-size',
        '--cache-type-k',
        '--cache-type-v',
        '--gpu-memory-utilization',
        '--max-num-seqs',
        '--max-num-batched-tokens',
    )


def _power_of_two_at_most(value: int, floor: int, ceiling: int) -> int:
    value = max(floor, min(ceiling, int(value or floor)))
    power = floor
    while power * 2 <= value and power * 2 <= ceiling:
        power *= 2
    return power


def adaptive_batch_sizes(ctx: int, parallel: int, objective: str, moe: bool = False) -> Tuple[int, int]:
    ctx = max(1, int(ctx or 1))
    parallel = max(1, int(parallel or 1))
    if moe and objective == 'fast_chat':
        batch = _power_of_two_at_most(max(256, ctx // max(1, parallel)), 256, 1024)
        ubatch = _power_of_two_at_most(max(128, batch // 2), 128, min(batch, 512))
        return batch, ubatch
    if moe:
        batch = _power_of_two_at_most(max(128, ctx // 16), 128, 256)
        ubatch = _power_of_two_at_most(max(64, batch // 2), 64, min(batch, 128))
        return batch, ubatch
    if objective == 'fast_chat':
        batch = _power_of_two_at_most(max(256, ctx // max(1, parallel)), 256, 2048)
    else:
        batch = _power_of_two_at_most(max(128, ctx // 12), 128, 1024)
    ubatch = _power_of_two_at_most(max(64, batch // 2), 64, batch)
    return batch, ubatch


def configure_adaptive_candidate(
    model: ModelConfig,
    profile: HardwareProfile,
    objective: str,
    ctx: int,
    parallel: int,
    variant: str = 'default',
) -> ModelConfig:
    candidate = ModelConfig(**asdict(model))
    tier = ADAPTIVE_TIER_BY_OBJECTIVE.get(objective, 'moderate')
    candidate.optimize_tier = tier
    apply_hardware_baseline(candidate, profile, tier)
    candidate.optimize_mode = f'measured_{objective}'
    candidate.parallel = max(1, int(parallel or 1))
    candidate.ctx = max(1, int(ctx or 1))
    candidate.memory_reserve_percent = max(
        ADAPTIVE_RESERVE_BY_OBJECTIVE.get(objective, 30),
        int(getattr(model, 'memory_reserve_percent', 25) or 25),
    )
    if objective == 'fast_chat':
        candidate.output = max(256, min(int(getattr(candidate, 'output', 2048) or 2048), 2048))
    else:
        candidate.output = max(1024, min(max(int(getattr(candidate, 'output', 4096) or 4096), 2048), 4096))

    extra_args = _strip_runtime_tuning_args(list(getattr(candidate, 'extra_args', []) or []))
    runtime = getattr(candidate, 'runtime', 'llama.cpp')
    if runtime == 'llama.cpp':
        batch, ubatch = adaptive_batch_sizes(candidate.ctx, candidate.parallel, objective, moe=model_is_moe(candidate))
        extra_args = _set_extra_arg(extra_args, '--batch-size', str(batch))
        extra_args = _set_extra_arg(extra_args, '--ubatch-size', str(ubatch))
        if variant == 'q8_kv':
            extra_args = _set_extra_arg(extra_args, '--cache-type-k', 'q8_0')
            extra_args = _set_extra_arg(extra_args, '--cache-type-v', 'q8_0')
    elif runtime == 'vllm':
        utilization = max(0.65, min(0.94, (100 - candidate.memory_reserve_percent) / 100.0))
        extra_args = _set_extra_arg(extra_args, '--gpu-memory-utilization', f'{utilization:.2f}')
        extra_args = _set_extra_arg(extra_args, '--max-num-seqs', str(candidate.parallel))
        batched = max(1024, min(65536, candidate.ctx * candidate.parallel))
        extra_args = _set_extra_arg(extra_args, '--max-num-batched-tokens', str(round_context(batched, 512)))
    candidate.extra_args = extra_args
    return candidate


def adaptive_context_upper_bound(
    model: ModelConfig,
    profile: HardwareProfile,
    objective: str,
    parallel: int = 1,
    variant: str = 'default',
) -> int:
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072) or 131072))
    seed = configure_adaptive_candidate(model, profile, objective, ctx_min, parallel, variant)
    safe_ctx = estimate_safe_context_for_profile(
        seed,
        profile,
        int(getattr(seed, 'memory_reserve_percent', 30) or 30),
        max(1, parallel),
        ctx_min,
        ctx_max,
    )
    if safe_ctx <= 0:
        return ctx_min
    return max(ctx_min, min(ctx_max, round_context_down(safe_ctx)))


def adaptive_context_search(
    ctx_min: int,
    ctx_upper: int,
    probe: Callable[[int], bool],
    max_probes: int = ADAPTIVE_MAX_CONTEXT_PROBES,
) -> Tuple[List[int], List[int]]:
    ctx_min = round_context(max(256, ctx_min))
    ctx_upper = max(ctx_min, round_context_down(ctx_upper))
    successes: List[int] = []
    failures: List[int] = []
    seen = set()

    def run_probe(value: int) -> bool:
        value = max(ctx_min, min(ctx_upper, round_context(value)))
        if value in seen or len(seen) >= max_probes:
            return value in successes
        seen.add(value)
        ok = bool(probe(value))
        (successes if ok else failures).append(value)
        return ok

    current = ctx_min
    last_success = 0
    first_failure = 0
    while len(seen) < max_probes:
        ok = run_probe(current)
        if ok:
            last_success = current
            if current >= ctx_upper:
                break
            current = min(ctx_upper, max(current + ADAPTIVE_CONTEXT_ROUNDING, current * 2))
            continue
        first_failure = current
        break

    if last_success and not first_failure and last_success < ctx_upper and len(seen) < max_probes:
        if run_probe(ctx_upper):
            last_success = ctx_upper
        else:
            first_failure = ctx_upper

    if last_success and first_failure:
        low = min(last_success, first_failure)
        high = max(last_success, first_failure)
        for _ in range(ADAPTIVE_BINARY_STEPS):
            if len(seen) >= max_probes:
                break
            midpoint = round_context((low + high) // 2)
            if midpoint <= low or midpoint >= high:
                break
            if run_probe(midpoint):
                low = midpoint
            else:
                high = midpoint

    while len(seen) < max_probes and len(successes) >= 2:
        ordered = sorted(set(successes))
        gaps = [(ordered[idx + 1] - ordered[idx], ordered[idx], ordered[idx + 1]) for idx in range(len(ordered) - 1)]
        gaps = [gap for gap in sorted(gaps, reverse=True) if gap[0] > ADAPTIVE_CONTEXT_ROUNDING * 2]
        if not gaps:
            break
        _gap, low, high = gaps[0]
        midpoint = round_context((low + high) // 2)
        if midpoint in seen:
            break
        run_probe(midpoint)

    return sorted(set(successes)), sorted(set(failures))


def coarse_context_step(ctx: int) -> int:
    ctx = max(1, int(ctx or 1))
    if ctx < COARSE_CONTEXT_LOW_LIMIT:
        return COARSE_CONTEXT_LOW_STEP
    if ctx < COARSE_CONTEXT_MID_LIMIT:
        return COARSE_CONTEXT_MID_STEP
    return COARSE_CONTEXT_HIGH_STEP


def exhaustive_context_ladder(ctx_min: int, ctx_max: int, step: int = EXHAUSTIVE_CONTEXT_STEP) -> List[int]:
    ctx_min = max(1, int(ctx_min or 1))
    ctx_max = max(ctx_min, int(ctx_max or ctx_min))
    values = [ctx_min]
    current = ctx_min
    while current < ctx_max:
        current = min(ctx_max, current + coarse_context_step(current))
        if current != values[-1]:
            values.append(current)
    if values[-1] != ctx_max:
        values.append(ctx_max)
    return values


def break_refinement_contexts(last_success_ctx: int, break_ctx: int, tested: set) -> List[int]:
    last_success_ctx = int(last_success_ctx or 0)
    break_ctx = int(break_ctx or 0)
    if last_success_ctx <= 0 or break_ctx <= last_success_ctx + CONTEXT_REFINE_STEP:
        return []
    values = []
    current = last_success_ctx + CONTEXT_REFINE_STEP
    while current < break_ctx:
        if current not in tested:
            values.append(current)
        current += CONTEXT_REFINE_STEP
    return values


def context_knee_refinement_contexts(
    records: List[Dict[str, object]],
    tested: set,
    ctx_max: int,
) -> List[int]:
    successful = sorted(
        [record for record in records if record.get('status') == 'ok'],
        key=lambda record: int(record.get('ctx', 0) or 0),
    )
    if len(successful) < 2:
        return []
    ctx_max = max(1, int(ctx_max or 1))
    max_tps = max(float(record.get('tokens_per_sec', 0.0) or 0.0) for record in successful) or 1.0
    max_ctx = max(int(record.get('ctx_per_slot', 0) or record.get('ctx', 0) or 0) for record in successful) or 1
    candidates = set()
    scored = []
    for idx in range(len(successful) - 1):
        left = successful[idx]
        right = successful[idx + 1]
        left_ctx = int(left.get('ctx', 0) or 0)
        right_ctx = int(right.get('ctx', 0) or 0)
        gap = right_ctx - left_ctx
        if gap <= CONTEXT_REFINE_STEP:
            continue
        left_tps = float(left.get('tokens_per_sec', 0.0) or 0.0)
        right_tps = float(right.get('tokens_per_sec', 0.0) or 0.0)
        drop = max(0.0, left_tps - right_tps) / max(left_tps, 1.0)
        ctx_gain = gap / max(ctx_max, 1)
        midpoint = round_context((left_ctx + right_ctx) // 2, CONTEXT_KNEE_ROUNDING)
        if left_ctx < midpoint < right_ctx and midpoint not in tested:
            if drop >= 0.18 or (drop >= 0.05 and ctx_gain >= 0.20):
                candidates.add(midpoint)
        left_score = 0.55 * (left_tps / max_tps) + 0.45 * (int(left.get('ctx_per_slot', left_ctx) or left_ctx) / max_ctx)
        scored.append((left_score, idx))
    last = successful[-1]
    last_score = 0.55 * (float(last.get('tokens_per_sec', 0.0) or 0.0) / max_tps) + 0.45 * (
        int(last.get('ctx_per_slot', last.get('ctx', 0)) or 0) / max_ctx
    )
    scored.append((last_score, len(successful) - 1))
    if scored:
        _score, best_idx = max(scored)
        for neighbor_idx in (best_idx - 1, best_idx):
            if 0 <= neighbor_idx < len(successful) - 1:
                left_ctx = int(successful[neighbor_idx].get('ctx', 0) or 0)
                right_ctx = int(successful[neighbor_idx + 1].get('ctx', 0) or 0)
                if right_ctx - left_ctx > CONTEXT_REFINE_STEP:
                    midpoint = round_context((left_ctx + right_ctx) // 2, CONTEXT_KNEE_ROUNDING)
                    if left_ctx < midpoint < right_ctx and midpoint not in tested:
                        candidates.add(midpoint)
    return sorted(candidates)


def smart_break_refinement_contexts(last_success_ctx: int, break_ctx: int, tested: set) -> List[int]:
    last_success_ctx = int(last_success_ctx or 0)
    break_ctx = int(break_ctx or 0)
    if last_success_ctx <= 0 or break_ctx <= last_success_ctx:
        return []
    candidates = {
        last_success_ctx + CONTEXT_REFINE_STEP,
        round_context((last_success_ctx + break_ctx) // 2, CONTEXT_KNEE_ROUNDING),
        break_ctx - CONTEXT_REFINE_STEP,
    }
    return sorted(
        value for value in candidates
        if last_success_ctx < value < break_ctx and value not in tested
    )


def _nearest_context_at_or_above(contexts: List[int], target: int) -> int:
    if not contexts:
        return 0
    target = int(target or 0)
    above = [ctx for ctx in contexts if ctx >= target]
    if above:
        return min(above, key=lambda ctx: (ctx - target, ctx))
    return max(contexts)


def smart_measurement_contexts(
    successes: List[int],
    failures: List[int],
    ctx_min: int,
    ctx_max: int,
    chat_floor: int,
    opencode_floor: int = 0,
) -> List[int]:
    ordered = sorted(set(int(ctx) for ctx in successes if int(ctx) > 0))
    if not ordered:
        return []
    selected = {ordered[0], ordered[-1]}
    if len(ordered) >= 2:
        selected.add(ordered[-2])
    span = max(0, ordered[-1] - ordered[0])
    if span > CONTEXT_REFINE_STEP:
        midpoint = round_context((ordered[0] + ordered[-1]) // 2, CONTEXT_KNEE_ROUNDING)
        selected.add(min(ordered, key=lambda ctx: (abs(ctx - midpoint), ctx)))
    for target in (chat_floor, opencode_floor, ctx_min, ctx_max):
        if int(target or 0) > 0:
            selected.add(_nearest_context_at_or_above(ordered, int(target)))
    positive_failures = [int(ctx) for ctx in failures if int(ctx) > 0]
    if positive_failures:
        first_break = min(positive_failures)
        below_break = [ctx for ctx in ordered if ctx < first_break]
        if below_break:
            selected.add(max(below_break))
    return sorted(ctx for ctx in selected if ctx in ordered)[:SMART_MAX_FULL_CONTEXTS_PER_VARIANT]


def smart_fast_contexts(successes: List[int], chat_floor: int) -> List[int]:
    ordered = sorted(set(int(ctx) for ctx in successes if int(ctx) > 0))
    if not ordered:
        return []
    primary = _nearest_context_at_or_above(ordered, chat_floor)
    selected = {primary}
    above = [ctx for ctx in ordered if ctx > primary]
    if above and primary <= max(chat_floor, 1) * 2:
        selected.add(above[0])
    return sorted(selected)


def adaptive_parallel_values(model: ModelConfig, profile: HardwareProfile, objective: str, ctx: int, variant: str) -> List[int]:
    if objective != 'fast_chat':
        return [1]
    max_cpu = max(1, min(4 if model_is_moe(model) else 16, int(getattr(profile, 'cpu_logical', 0) or 8)))
    values = []
    parallel = 1
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072) or 131072))
    while parallel <= max_cpu:
        candidate = configure_adaptive_candidate(model, profile, objective, ctx, parallel, variant)
        safe_ctx = estimate_safe_context_for_profile(
            candidate,
            profile,
            int(getattr(candidate, 'memory_reserve_percent', 30) or 30),
            parallel,
            ctx_min,
            ctx_max,
        )
        if safe_ctx >= min(ctx, ctx_max):
            values.append(parallel)
            parallel *= 2
            continue
        break
    return values or [1]

def fallback_tiers(selected_tier: str) -> List[str]:
    order = ['extreme', 'moderate', 'safe']
    selected = (selected_tier or 'moderate').strip().lower()
    if selected == 'auto':
        selected = 'moderate'
    if selected not in order:
        selected = 'moderate'
    return order[order.index(selected):]
def launch_with_failsafe(
    app: AppConfig,
    model: ModelConfig,
    mode: str,
    tier: str,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    attempts = []
    profile = app.hardware_profile(refresh=True)
    measured_key = measured_profile_key_for_launch(mode, tier)
    if measured_key:
        measured_model, runtime_profile = model_and_runtime_profile_from_measured_profile(model, measured_key)
        if measured_model is not None:
            model.ctx = measured_model.ctx
            model.parallel = measured_model.parallel
            model.threads = measured_model.threads
            model.ngl = measured_model.ngl
            model.output = measured_model.output
            model.cache_ram = measured_model.cache_ram
            model.temp = measured_model.temp
            model.flash_attn = measured_model.flash_attn
            model.jinja = measured_model.jinja
            model.memory_reserve_percent = measured_model.memory_reserve_percent
            model.extra_args = list(getattr(measured_model, 'extra_args', []) or [])
            model.optimize_mode = measured_model.optimize_mode
            model.optimize_tier = measured_model.optimize_tier
            _ok, measured_msg = apply_measured_profile(measured_model, measured_key)
            runtime_note = ''
            if runtime_profile is not None:
                runtime_note = f' runtime={runtime_profile.name or runtime_profile.engine_id}'
                if runtime_profile.fit:
                    runtime_note += f' fit=on fitc={runtime_profile.fit_context or "-"}'
            if progress:
                progress(f'trying measured {measured_key} profile: {measured_msg}{runtime_note}')
            app.add_or_update(model)
            sync_msg = sync_opencode_after_tuning(app)
            ok, msg = app.start(model, runtime_profile=runtime_profile)
            if ok:
                try:
                    ready_ok, ready_msg = app.wait_until_ready(model, timeout=120, cancel_token=cancel_token)
                except CancelledError:
                    app.stop(model, managed_only=True)
                    raise
                if ready_ok:
                    if progress:
                        progress(f'measured {measured_key} ready: {ready_msg}')
                    return True, f'{ready_msg} [measured {measured_key}] | {measured_msg} | {sync_msg}'
                app.stop(model, managed_only=True)
                attempts.append(f'measured {measured_key}: not ready ({concise_failure(ready_msg)})')
                if progress:
                    progress(f'measured {measured_key} was not ready; falling back to estimated profiles.')
            else:
                attempts.append(f'measured {measured_key}: start failed ({concise_failure(msg)})')
                if progress:
                    progress(f'measured {measured_key} failed to start: {concise_failure(msg)}')
        elif progress:
            progress(f'{model.id}: no measured {measured_key} profile; using estimated launch profile.')
    if mode == 'opencode_ready':
        mode = 'best'
    if tier == 'auto':
        tier = select_best_tier(model, profile)
    if progress:
        progress(f'launch optimization started: mode={mode} tier={tier} {profile.short_summary()}')
    for current_tier in fallback_tiers(tier):
        check_cancelled(cancel_token)
        if mode == 'best':
            tune_msg = apply_best_optimization(model, tier=current_tier, profile=profile)
        else:
            tune_msg = apply_optimization_preset(model, mode, tier=current_tier, profile=profile)
        if progress:
            progress(f'trying launch profile {mode}/{current_tier}: {tune_msg}')
        app.add_or_update(model)
        sync_msg = sync_opencode_after_tuning(app)
        ok, msg = app.start(model)
        if not ok:
            if progress:
                progress(f'launch profile {mode}/{current_tier} failed to start: {concise_failure(msg)}')
            attempts.append(f'{current_tier}: start failed ({concise_failure(msg)})')
            continue
        if progress:
            progress(f'launch profile {mode}/{current_tier} started; waiting for readiness...')
        try:
            ready_ok, ready_msg = app.wait_until_ready(model, timeout=120, cancel_token=cancel_token)
        except CancelledError:
            app.stop(model, managed_only=True)
            raise
        if ready_ok:
            if progress:
                progress(f'launch profile {mode}/{current_tier} ready: {ready_msg}')
            return True, f'{ready_msg} [{current_tier}] | {tune_msg} | {sync_msg}'
        app.stop(model, managed_only=True)
        if progress:
            progress(f'launch profile {mode}/{current_tier} was not ready; stopped and trying fallback.')
        attempts.append(f'{current_tier}: not ready ({concise_failure(ready_msg)})')
    msg = '❌ optimization failed; fallback exhausted -> ' + '; '.join(attempts[:3])
    if progress:
        progress(msg)
    return False, msg
def start_model_with_progress(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    check_cancelled(cancel_token)
    if progress:
        progress(f'starting {model.id} with current settings...')
    ok, msg = app.start(model)
    if not ok:
        if progress:
            progress(f'{model.id} failed to start: {msg}')
        return False, msg
    if progress:
        progress(f'{model.id} started; waiting for readiness...')
    try:
        ready_ok, ready_msg = app.wait_until_ready(model, timeout=120, cancel_token=cancel_token)
    except CancelledError:
        app.stop(model, managed_only=True)
        raise
    if progress:
        progress(ready_msg if ready_ok else concise_failure(ready_msg))
    return ready_ok, ready_msg


def ensure_agent_stack_model_ready(
    app: AppConfig,
    model: ModelConfig,
    runtime_label: str,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str, bool]:
    check_cancelled(cancel_token)
    status, _detail = app.health(model)
    if status == 'READY':
        if progress:
            progress(f'{model.id} already ready; using current server for {runtime_label}.')
        return True, f'{model.id} already ready', False
    if status in ('LOADING', 'STARTING') or app.get_pid(model):
        if progress:
            progress(f'{model.id} is starting; waiting for readiness before {runtime_label} launch...')
        ready_ok, ready_msg = app.wait_until_ready(model, timeout=180, cancel_token=cancel_token)
        return ready_ok, concise_failure(ready_msg) if not ready_ok else ready_msg, False
    if progress:
        progress(f'{model.id} is stopped; launching agent-ready profile before {runtime_label}...')
    ready_ok, ready_msg = launch_with_failsafe(app, model, 'opencode_ready', 'auto', progress=progress, cancel_token=cancel_token)
    return ready_ok, concise_failure(ready_msg) if not ready_ok else ready_msg, bool(ready_ok)


def launch_agent_stack(
    app: AppConfig,
    model: ModelConfig,
    workspace: str,
    runtime_label: str,
    command_name: str,
    remember_workspace: Callable[[str], None],
    sync_config: Callable[[], Tuple[bool, str]],
    launch_terminal: Callable[[ModelConfig, Path], Tuple[bool, str]],
    include_vscode: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    check_cancelled(cancel_token)
    valid, workspace_path, reason = app.validate_workspace_path(workspace)
    if not valid or workspace_path is None:
        return False, f'❌ {reason}'
    if not getattr(model, 'enabled', True):
        return False, f'❌ {model.id} is disabled; enable it before launching {runtime_label}.'

    remember_workspace(str(workspace_path))
    app.save()

    ready_ok, ready_msg, started_for_stack = ensure_agent_stack_model_ready(
        app,
        model,
        runtime_label,
        progress=progress,
        cancel_token=cancel_token,
    )
    if not ready_ok:
        return False, ready_msg

    if cancel_token is not None and cancel_token.is_cancelled():
        if started_for_stack:
            app.stop(model, managed_only=True)
        check_cancelled(cancel_token)
    sync_ok, sync_msg = sync_config()
    if not sync_ok:
        if started_for_stack:
            app.stop(model, managed_only=True)
        return False, f'❌ {sync_msg}'
    if progress:
        progress(sync_msg)

    if not app.command_exists(command_name):
        if started_for_stack:
            app.stop(model, managed_only=True)
        return False, f'❌ {runtime_label} command not found: {command_name}'

    warnings = []
    if cancel_token is not None and cancel_token.is_cancelled():
        if started_for_stack:
            app.stop(model, managed_only=True)
        check_cancelled(cancel_token)
    if include_vscode:
        code_ok, code_msg = app.launch_vscode_workspace(workspace_path)
        if progress:
            progress(code_msg if code_ok else f'VS Code warning: {code_msg}')
        if not code_ok:
            warnings.append(code_msg)

    if cancel_token is not None and cancel_token.is_cancelled():
        if started_for_stack:
            app.stop(model, managed_only=True)
        check_cancelled(cancel_token)
    open_ok, open_msg = launch_terminal(model, workspace_path)
    if not open_ok:
        if started_for_stack:
            app.stop(model, managed_only=True)
        detail = f'❌ {open_msg}'
        if warnings:
            detail += ' | warnings: ' + '; '.join(warnings)
        return False, detail
    if progress:
        progress(open_msg)

    stack_label = f'{runtime_label} full-stack' if include_vscode else runtime_label
    detail = f'✅ launched {stack_label} for {model.id} in {workspace_path}'
    if warnings:
        detail += ' | warnings: ' + '; '.join(warnings)
    return True, detail


def launch_opencode_stack(
    app: AppConfig,
    model: ModelConfig,
    workspace: str,
    include_vscode: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    if not (getattr(app.opencode, 'path', '') or '').strip():
        return False, '❌ Set opencode.path first in settings.'
    return launch_agent_stack(
        app,
        model,
        workspace,
        'OpenCode',
        'opencode',
        lambda value: setattr(app.opencode, 'last_workspace_path', value),
        app.generate_opencode,
        app.launch_opencode_terminal,
        include_vscode=include_vscode,
        progress=progress,
        cancel_token=cancel_token,
    )


def launch_hermes_stack(
    app: AppConfig,
    model: ModelConfig,
    workspace: str,
    include_vscode: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    command_name = app.hermes_command_prefix()[0]
    return launch_agent_stack(
        app,
        model,
        workspace,
        'Hermes',
        command_name,
        lambda value: setattr(app.hermes, 'last_workspace_path', value),
        lambda: app.generate_hermes_config(model),
        app.launch_hermes_terminal,
        include_vscode=include_vscode,
        progress=progress,
        cancel_token=cancel_token,
    )


def clone_model_config(model: ModelConfig) -> ModelConfig:
    return ModelConfig(**asdict(model))


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\s\w]", text)))
def completion_text_from_response(data: Dict) -> str:
    choices = data.get('choices') or []
    if not choices:
        return ''
    first = choices[0] or {}
    message = first.get('message') or {}
    content = message.get('content')
    if isinstance(content, list):
        return ' '.join(str(item.get('text', item)) if isinstance(item, dict) else str(item) for item in content)
    if content is not None:
        return str(content)
    return str(first.get('text', ''))
def post_json(url: str, payload: Dict, timeout: int) -> Dict:
    body = json.dumps(payload).encode('utf-8')
    req = request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8', errors='replace'))

BENCHMARK_PROMPTS = [
    (
        'Write a concise technical checklist for keeping a local language model '
        'server fast and stable. Use short bullet points.'
    ),
    (
        'Explain how to diagnose a CUDA out-of-memory error in a local inference '
        'server. Include practical steps and keep the answer compact.'
    ),
]

def benchmark_completion(
    model: ModelConfig,
    max_tokens: Optional[int] = None,
    timeout: int = 180,
    prompt: Optional[str] = None,
    cancel_token: Optional[CancelToken] = None,
    launch_profile: Optional[BenchmarkLaunchProfile] = None,
    deadline: Optional[BenchmarkDeadline] = None,
) -> Tuple[bool, Dict]:
    check_cancelled(cancel_token)
    if deadline is not None:
        timeout = deadline.cap_timeout(timeout)
        if timeout <= 0.0:
            return False, {'error': 'benchmark budget expired before API request'}
    prompt = prompt or BENCHMARK_PROMPTS[0]
    if launch_profile is not None:
        request_fields = benchmark_profile_request_fields(launch_profile, max_tokens=max_tokens)
    else:
        request_fields = {
            'max_tokens': max(1, int(max_tokens or 64)),
            'temperature': 0,
        }
    payload = {
        'model': model.alias,
        'messages': [
            {'role': 'system', 'content': 'You are a concise local model benchmark assistant.'},
            {'role': 'user', 'content': prompt},
        ],
        'stream': False,
    }
    payload.update(request_fields)
    url = f'http://{model.host}:{model.port}/v1/chat/completions'
    started = time.time()
    try:
        data = post_json(url, payload, timeout=timeout)
    except Exception as exc:
        return False, {'error': str(exc)}
    check_cancelled(cancel_token)
    elapsed = max(0.001, time.time() - started)
    usage = data.get('usage') or {}
    text = completion_text_from_response(data)
    completion_tokens = int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
    prompt_tokens = int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
    if completion_tokens <= 0:
        completion_tokens = estimate_text_tokens(text)
    if prompt_tokens <= 0:
        prompt_tokens = estimate_text_tokens(prompt)
    return True, {
        'elapsed': elapsed,
        'completion_tokens': completion_tokens,
        'prompt_tokens': prompt_tokens,
        'tokens_per_sec': completion_tokens / elapsed,
        'text': text,
    }
def benchmark_completion_suite(
    model: ModelConfig,
    max_tokens: int = BENCHMARK_SAMPLE_TOKENS,
    timeout: int = BENCHMARK_SAMPLE_TIMEOUT,
    cancel_token: Optional[CancelToken] = None,
    launch_profile: Optional[BenchmarkLaunchProfile] = None,
    deadline: Optional[BenchmarkDeadline] = None,
) -> Tuple[bool, Dict]:
    samples = []
    failures = []
    for prompt in BENCHMARK_PROMPTS:
        check_cancelled(cancel_token)
        ok, bench = benchmark_completion(
            model,
            max_tokens=max_tokens,
            timeout=timeout,
            prompt=prompt,
            cancel_token=cancel_token,
            launch_profile=launch_profile,
            deadline=deadline,
        )
        if ok:
            samples.append(bench)
        else:
            failures.append(str(bench.get('error', 'unknown error')))
    if not samples:
        return False, {'error': '; '.join(failures) if failures else 'no benchmark samples completed'}

    scores = [float(sample['tokens_per_sec']) for sample in samples]
    elapsed = sum(float(sample['elapsed']) for sample in samples)
    completion_tokens = sum(int(sample['completion_tokens']) for sample in samples)
    prompt_tokens = sum(int(sample['prompt_tokens']) for sample in samples)
    return True, {
        'elapsed': elapsed,
        'completion_tokens': completion_tokens,
        'prompt_tokens': prompt_tokens,
        'tokens_per_sec': statistics.median(scores),
        'sample_tokens_per_sec': scores,
        'sample_count': len(samples),
        'error': '; '.join(failures),
    }
def benchmark_candidate_models(model: ModelConfig, profile: HardwareProfile) -> List[Tuple[str, str, ModelConfig, str]]:
    selected_tier = select_best_tier(model, profile)
    selected_preset = choose_best_preset(model, profile)
    alternate_preset = 'max_context' if selected_preset == 'tokens_per_sec' else 'tokens_per_sec'
    tier_order = ['safe', 'moderate', 'extreme']
    selected_idx = tier_order.index(selected_tier)
    neighbor_tiers = [selected_tier]
    if selected_idx > 0:
        neighbor_tiers.append(tier_order[selected_idx - 1])
    if selected_idx < len(tier_order) - 1:
        neighbor_tiers.append(tier_order[selected_idx + 1])

    requested: List[Tuple[str, str]] = []
    for tier in neighbor_tiers:
        requested.append((selected_preset, tier))
    requested.append((alternate_preset, selected_tier))
    if selected_tier != 'safe':
        requested.append((alternate_preset, 'safe'))

    candidates = []
    seen = set()
    for preset, tier in requested:
        variants = ['default']
        if (
            getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp'
            and profile.has_usable_gpu()
            and preset == 'tokens_per_sec'
        ):
            variants.append('q8_kv')
        for variant in variants:
            label = preset if variant == 'default' else f'{preset}_{variant}'
            key = (label, tier)
            if key in seen:
                continue
            seen.add(key)
            candidate = clone_model_config(model)
            tune_msg = apply_optimization_preset(candidate, preset, tier=tier, profile=profile)
            if variant == 'q8_kv':
                set_model_extra_arg(candidate, '--cache-type-k', 'q8_0')
                set_model_extra_arg(candidate, '--cache-type-v', 'q8_0')
                ctx_min = max(256, int(getattr(candidate, 'ctx_min', 2048)))
                ctx_max = max(ctx_min, int(getattr(candidate, 'ctx_max', 131072)))
                target_ctx = {
                    'safe': 4096,
                    'moderate': 8192,
                    'extreme': 12288,
                }.get(tier, 8192)
                safe_ctx = estimate_safe_context_for_profile(
                    candidate,
                    profile,
                    int(getattr(candidate, 'memory_reserve_percent', 30) or 30),
                    int(getattr(candidate, 'parallel', 1) or 1),
                    ctx_min,
                    ctx_max,
                )
                if safe_ctx >= ctx_min:
                    candidate.ctx = max(ctx_min, min(target_ctx, ctx_max, safe_ctx))
                tune_msg += ' kv=q8_0'
            candidates.append((label, tier, candidate, tune_msg))
            if len(candidates) >= BENCHMARK_MAX_CANDIDATES:
                break
        if len(candidates) >= BENCHMARK_MAX_CANDIDATES:
            break
    return candidates
def safe_bootstrap_candidate_models(model: ModelConfig, profile: HardwareProfile) -> List[Tuple[str, str, ModelConfig, str]]:
    candidates: List[Tuple[str, str, ModelConfig, str]] = []
    for preset, tier in SAFE_BOOTSTRAP_PRESETS:
        candidate = clone_model_config(model)
        tune_msg = apply_optimization_preset(candidate, preset, tier=tier, profile=profile)
        candidates.append((preset, tier, candidate, tune_msg))
        if (
            preset == 'tokens_per_sec'
            and getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp'
            and profile.has_usable_gpu()
        ):
            q8_candidate = clone_model_config(model)
            q8_msg = apply_optimization_preset(q8_candidate, preset, tier=tier, profile=profile)
            set_model_extra_arg(q8_candidate, '--cache-type-k', 'q8_0')
            set_model_extra_arg(q8_candidate, '--cache-type-v', 'q8_0')
            ctx_min = max(256, int(getattr(q8_candidate, 'ctx_min', 2048)))
            ctx_max = max(ctx_min, int(getattr(q8_candidate, 'ctx_max', 131072)))
            safe_ctx = estimate_safe_context_for_profile(
                q8_candidate,
                profile,
                int(getattr(q8_candidate, 'memory_reserve_percent', 40) or 40),
                int(getattr(q8_candidate, 'parallel', 1) or 1),
                ctx_min,
                ctx_max,
            )
            if safe_ctx >= ctx_min:
                q8_candidate.ctx = max(ctx_min, min(SAFE_BOOTSTRAP_Q8_TARGET_CTX, ctx_max, safe_ctx))
                candidates.append((f'{preset}_q8_kv', tier, q8_candidate, f'{q8_msg} kv=q8_0'))
    return candidates[:3]


def _run_server_benchmark_candidates(
    app: AppConfig,
    model: ModelConfig,
    candidates: List[Tuple[str, str, ModelConfig, str]],
    profile: HardwareProfile,
    label: str,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
    update_default_status: bool = False,
) -> Tuple[bool, str]:
    status, _detail = app.health(model)
    if status in ('READY', 'LOADING', 'STARTING') or app.get_pid(model):
        return False, f'❌ Stop the model before running {label}.'

    results = []
    failures = []
    benchmark_records: List[Dict[str, object]] = []
    starting_pressure = current_process_pressure_payload()
    if progress:
        progress(f'{label} started: {len(candidates)} candidate(s), {profile.short_summary()} {starting_pressure.get("process_pressure_detail", "")}')
    if update_default_status:
        running_model = clone_model_config(model)
        running_model.default_benchmark_status = 'running'
        app.add_or_update(running_model)

    def add_benchmark_record(
        preset: str,
        tier: str,
        candidate: ModelConfig,
        status: str,
        score: float = 0.0,
        elapsed: float = 0.0,
        detail: str = '',
        launch_profile: Optional[BenchmarkLaunchProfile] = None,
    ):
        runtime_context = runtime_record_context(app, candidate, benchmark_profile=launch_profile)
        record = {
            'preset': preset,
            'tier': tier,
            'status': status,
            'tokens_per_sec': round(float(score), 2),
            'decode_tokens_per_sec': round(float(score), 2),
            'generation_tokens_per_sec': round(float(score), 2),
            'prompt_tokens_per_sec': 0.0,
            'seconds': round(float(elapsed), 2),
            'ctx': int(getattr(candidate, 'ctx', 0) or 0),
            'parallel': int(getattr(candidate, 'parallel', 0) or 0),
            'threads': int(getattr(candidate, 'threads', 0) or 0),
            'ngl': int(getattr(candidate, 'ngl', 0) or 0),
            'startup_result': 'READY' if status == 'ok' else 'FAILED',
            'failure_category': '',
            'failure_reason': '',
            'suggested_fix': '',
            'detail': concise_failure(detail, limit=500),
            'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
        }
        record.update(runtime_context)
        if status != 'ok':
            apply_failure_context(record, detail or status, default_category='SERVER_TIMEOUT')
        record.update(architecture_payload(candidate))
        record.update(current_process_pressure_payload())
        benchmark_records.append(record)

    current: Optional[Tuple[str, str, ModelConfig, Optional[BenchmarkLaunchProfile]]] = None
    try:
        for attempt, (preset, tier, candidate, tune_msg) in enumerate(candidates, start=1):
            check_cancelled(cancel_token)
            try:
                capabilities = app.engine_capabilities()
            except Exception:
                capabilities = None
            launch_profile = build_benchmark_launch_profile(
                candidate,
                None,
                capabilities,
                purpose='serve_default',
                depth='full',
            )
            current = (preset, tier, candidate, launch_profile)
            if progress:
                progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier}: {tune_msg}')
            ok, msg = app.start(candidate, benchmark_profile=launch_profile)
            if not ok:
                if progress:
                    progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} failed to start: {concise_failure(msg)}')
                add_benchmark_record(preset, tier, candidate, 'start failed', detail=msg, launch_profile=launch_profile)
                failures.append(f'{preset}/{tier}: start failed ({concise_failure(msg)})')
                continue

            try:
                if progress:
                    progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} started; waiting for readiness...')
                ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=BENCHMARK_READY_TIMEOUT, cancel_token=cancel_token)
                if not ready_ok:
                    if progress:
                        progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} not ready: {concise_failure(ready_msg)}')
                    add_benchmark_record(preset, tier, candidate, 'not ready', detail=ready_msg, launch_profile=launch_profile)
                    failures.append(f'{preset}/{tier}: {concise_failure(ready_msg)}')
                    continue

                if progress:
                    progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} ready; warming benchmark prompt...')
                benchmark_completion(
                    candidate,
                    max_tokens=min(BENCHMARK_WARMUP_TOKENS, max(1, launch_profile.measurement_output)),
                    timeout=BENCHMARK_WARMUP_TIMEOUT,
                    cancel_token=cancel_token,
                    launch_profile=launch_profile,
                )
                if progress:
                    progress(
                        f'candidate {attempt}/{len(candidates)} {preset}/{tier} '
                        f'measuring {len(BENCHMARK_PROMPTS)}x{launch_profile.measurement_output}-token serve_default suite...'
                    )
                bench_ok, bench = benchmark_completion_suite(
                    candidate,
                    max_tokens=max(1, launch_profile.measurement_output),
                    timeout=BENCHMARK_SAMPLE_TIMEOUT,
                    cancel_token=cancel_token,
                    launch_profile=launch_profile,
                )
                if not bench_ok:
                    if progress:
                        progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} benchmark failed: {bench.get("error", "unknown error")}')
                    add_benchmark_record(
                        preset,
                        tier,
                        candidate,
                        'benchmark failed',
                        detail=str(bench.get('error', 'unknown error')),
                        launch_profile=launch_profile,
                    )
                    failures.append(f'{preset}/{tier}: benchmark failed ({bench.get("error", "unknown error")})')
                    continue

                score = float(bench['tokens_per_sec'])
                elapsed = float(bench['elapsed'])
                if progress:
                    progress(
                        f'candidate {attempt}/{len(candidates)} {preset}/{tier} '
                        f'scored median {score:.2f} tok/s across {int(bench.get("sample_count", 1))} sample(s)'
                    )
                add_benchmark_record(preset, tier, candidate, 'ok', score=score, elapsed=elapsed, launch_profile=launch_profile)
                results.append({
                    'score': score,
                    'preset': preset,
                    'tier': tier,
                    'model': candidate,
                    'elapsed': elapsed,
                    'completion_tokens': int(bench['completion_tokens']),
                    'prompt_tokens': int(bench['prompt_tokens']),
                    'tune_msg': tune_msg,
                })
            finally:
                app.stop(candidate, managed_only=True)
                if progress:
                    progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} stopped.')
                sleep_with_cancel(0.5, cancel_token)
    except CancelledError:
        if current is not None:
            preset, tier, candidate, launch_profile = current
            add_benchmark_record(
                preset,
                tier,
                candidate,
                'aborted',
                detail='user requested abort',
                launch_profile=launch_profile,
            )
            app.stop(candidate, managed_only=True)
        recorded_model = clone_model_config(model)
        recorded_model.last_benchmark_results = benchmark_records
        if update_default_status:
            recorded_model.benchmark_fingerprint = app.model_fingerprint(recorded_model)
            recorded_model.default_benchmark_status = 'aborted'
            recorded_model.default_benchmark_at = datetime.now().isoformat(timespec='seconds')
        app.add_or_update(recorded_model)
        msg = '⚠ aborted; managed processes stopped'
        if progress:
            progress(msg)
        return False, msg

    if not results:
        details = '; '.join(failures[:3]) if failures else 'no candidates completed'
        ended_at = datetime.now().isoformat(timespec='seconds')
        recorded_model, preserved = failed_benchmark_model_state(app, model, benchmark_records, ended_at)
        if update_default_status:
            recorded_model.benchmark_fingerprint = app.model_fingerprint(recorded_model)
            if not preserved:
                recorded_model.default_benchmark_status = 'failed'
            recorded_model.default_benchmark_at = ended_at
        app.add_or_update(recorded_model)
        msg = (
            preserved_profiles_message(f'{label} found no better working candidate', benchmark_records)
            if preserved
            else f'❌ {label} failed: {benchmark_failure_summary(benchmark_records, details)}'
        )
        if progress:
            progress(msg)
        return False, msg

    best = max(results, key=lambda item: item['score'])
    best_model = best['model']
    best_model.last_benchmark_tokens_per_sec = round(best['score'], 2)
    best_model.last_benchmark_seconds = round(best['elapsed'], 2)
    best_model.last_benchmark_profile = (
        f'{best["preset"]}/{best["tier"]} '
        f'{best["score"]:.2f} tok/s '
        f'{profile.short_summary()}'
    )
    best_model.last_benchmark_results = benchmark_records
    if update_default_status:
        best_model.benchmark_fingerprint = app.model_fingerprint(best_model)
        best_model.default_benchmark_status = 'done'
        best_model.default_benchmark_at = datetime.now().isoformat(timespec='seconds')
    app.add_or_update(best_model)
    sync_msg = sync_opencode_after_tuning(app)
    msg = (
        f'✅ {label} winner: {best_model.id} {best["preset"]}/{best["tier"]} '
        f'{best["score"]:.2f} tok/s ctx={best_model.ctx} parallel={best_model.parallel} '
        f'threads={best_model.threads} ngl={best_model.ngl} | {sync_msg}'
    )
    if progress:
        progress(msg)
    return True, msg


def adaptive_profile_dict(
    key: str,
    candidate: ModelConfig,
    record: Dict[str, object],
    profile: HardwareProfile,
) -> Dict[str, object]:
    profile_dict = {
        'status': 'ok',
        'objective': key,
        'ctx': int(getattr(candidate, 'ctx', 0) or 0),
        'ctx_per_slot': ctx_per_slot(candidate),
        'parallel': int(getattr(candidate, 'parallel', 1) or 1),
        'threads': int(getattr(candidate, 'threads', 0) or 0),
        'ngl': int(getattr(candidate, 'ngl', 0) or 0),
        'output': int(getattr(candidate, 'output', 0) or 0),
        'cache_ram': int(getattr(candidate, 'cache_ram', 0) or 0),
        'mtp_enabled': bool(getattr(candidate, 'mtp_enabled', False)),
        'mtp_draft_n_max': clamp_mtp_draft(getattr(candidate, 'mtp_draft_n_max', 3), default=3),
        'temp': float(getattr(candidate, 'temp', 0.7) or 0.7),
        'flash_attn': bool(getattr(candidate, 'flash_attn', True)),
        'jinja': bool(getattr(candidate, 'jinja', True)),
        'memory_reserve_percent': int(getattr(candidate, 'memory_reserve_percent', 25) or 25),
        'extra_args': list(getattr(candidate, 'extra_args', []) or []),
        'variant': str(record.get('variant', '') or 'default'),
        'tokens_per_sec': round(float(record.get('tokens_per_sec', 0.0) or 0.0), 2),
        'seconds': round(float(record.get('seconds', 0.0) or 0.0), 2),
        'ram_available': int(record.get('ram_available', 0) or 0),
        'gpu_memory_free': int(record.get('gpu_memory_free', 0) or 0),
        'detail': str(record.get('detail', '')),
        'selection_score': round(float(record.get('selection_score', 0.0) or 0.0), 4),
        'selection_reason': str(record.get('selection_reason', '') or ''),
        'benchmarked_at': str(record.get('benchmarked_at') or datetime.now().isoformat(timespec='seconds')),
        'hardware': profile.short_summary(),
    }
    profile_dict.update(architecture_payload(candidate))
    for key in (
        'process_pressure_level',
        'process_pressure_score',
        'process_pressure_detail',
        'process_load_1m',
        'process_load_ratio',
        'process_count',
        'process_known',
        'process_known_memory',
        'engine',
        'server_bin',
        'runtime_profile',
        'benchmark_profile',
        'benchmark_purpose',
        'measurement_output',
        'top_p',
        'top_k',
        'repeat_penalty',
        'presence_penalty',
        'min_p',
        'seed',
        'samplers',
        'kv_preset',
        'kv_key',
        'kv_value',
        'flash_attn_mode',
        'fit',
        'fit_target',
        'no_context_shift',
        'cache_prompt',
        'cache_reuse',
        'reasoning',
        'reasoning_budget',
        'reasoning_format',
        'reasoning_mode',
        'preserve_thinking',
        'preserve_thinking_source',
        'chat_template_kwargs',
        'unsupported_launch_flags',
        'kv_family',
        'kv_quality_tier',
        'kv_compression_tier',
        'kv_score_penalty',
        'benchmark_depth',
        'runtime_fit',
        'fit_context',
        'runtime_no_warmup',
        'gpu_layers_mode',
        'batch_size',
        'ubatch_size',
        'ctk',
        'ctv',
        'detected_head_dim',
        'model_quant',
        'model_family',
        'binary_path',
        'help_supported_cache_types',
        'runtime_log_path',
        'failure_excerpt',
        'fit_discovery_phase',
        'viable_ngl',
        'viable_ngl_source',
        'fit_selected_ngl',
        'fit_selected_ngl_source',
        'fit_log_excerpt',
        'placement_strategy',
        'cpu_moe',
        'n_cpu_moe',
        'tensor_overrides',
        'mtp_enabled',
        'mtp_draft_n_max',
        'spec_type',
        'rejection_reason',
        'selection_rejection_reason',
        'rejected_by_profile',
        'rejected_by_tokens_per_sec',
        'config_fingerprint',
        'command',
    ):
        if key in record:
            profile_dict[key] = record.get(key)
    placement = moe_placement_payload(profile_dict, source_name='measured_profile')
    if (
        placement.get('strategy')
        or placement.get('cpu_moe')
        or int(placement.get('n_cpu_moe', 0) or 0) > 0
        or placement.get('tensor_overrides')
    ):
        profile_dict['moe_placement'] = placement
    return profile_dict


def chat_min_ctx_per_slot(model: ModelConfig) -> int:
    prompt_tokens = max(estimate_text_tokens(prompt) for prompt in BENCHMARK_PROMPTS)
    output = max(256, min(2048, int(getattr(model, 'output', 2048) or 2048)))
    return max(int(getattr(model, 'ctx_min', 2048) or 2048), prompt_tokens + output + 512)


def parse_context_requirement(text: str) -> int:
    patterns = (
        r'request\s*\((\d+)\s*tokens?\)\s*exceeds',
        r'(\d+)\s*tokens?\s*exceeds',
        r'needs?\s+(?:about\s+)?(\d+)\s*(?:ctx|context|tokens?)',
    )
    for pattern in patterns:
        match = re.search(pattern, str(text or ''), re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return 0


def observed_opencode_context_floor(model: ModelConfig) -> int:
    floor = 0
    for row in getattr(model, 'last_opencode_benchmark_results', []) or []:
        try:
            floor = max(floor, int(row.get('context_required', 0) or 0))
        except Exception:
            pass
        floor = max(floor, parse_context_requirement(str(row.get('detail', ''))))
        for task in row.get('task_details', []) or []:
            if isinstance(task, dict):
                try:
                    floor = max(floor, int(task.get('context_required', 0) or 0))
                except Exception:
                    pass
                floor = max(floor, parse_context_requirement(str(task.get('detail', ''))))
                floor = max(floor, parse_context_requirement(' '.join(str(x) for x in task.get('stderr_tail', []) or [])))
                floor = max(floor, parse_context_requirement(' '.join(str(x) for x in task.get('stdout_tail', []) or [])))
    return floor


def measured_profile_meets_opencode_floor(profile: Dict[str, object], floor: int) -> bool:
    if not profile or profile.get('status', 'ok') != 'ok':
        return False
    floor = int(floor or 0)
    if floor <= 0:
        return True
    return int(profile.get('ctx_per_slot', profile.get('ctx', 0)) or 0) >= floor


def select_measured_profiles(
    model: ModelConfig,
    measured: List[Dict[str, object]],
    profile: HardwareProfile,
) -> Dict[str, Dict[str, object]]:
    successful = [
        item for item in measured
        if item.get('status') == 'ok' and str(item.get('measurement_type', 'full') or 'full') == 'full'
    ]
    if not successful:
        return {}
    promotion_pool = reject_slow_tq3_partial_offload_rows(successful)
    if not promotion_pool:
        promotion_pool = successful
    max_tps = max(float(item.get('tokens_per_sec', 0.0) or 0.0) for item in successful) or 1.0
    max_ctx = max(int(item.get('ctx_per_slot', 0) or 0) for item in successful) or 1
    fast_floor = chat_min_ctx_per_slot(model)
    opencode_floor = max(observed_opencode_context_floor(model), 16384 if model_is_moe(model) else 0)

    fast_pool = [item for item in promotion_pool if int(item.get('ctx_per_slot', 0) or 0) >= fast_floor] or promotion_pool
    long_candidates = [item for item in promotion_pool if int(item.get('parallel', 1) or 1) == 1]
    long_pool = [item for item in long_candidates if not low_speed_guardrail_reason(item)]
    opencode_single_slot_pool = [item for item in promotion_pool if int(item.get('parallel', 1) or 1) == 1] or promotion_pool
    opencode_pool = (
        [item for item in opencode_single_slot_pool if int(item.get('ctx_per_slot', 0) or 0) >= opencode_floor]
        if opencode_floor
        else opencode_single_slot_pool
    )
    opencode_pool = [item for item in opencode_pool if not low_speed_guardrail_reason(item)]

    fast = max(fast_pool, key=lambda item: (score_fast_chat(item, model), float(item.get('tokens_per_sec', 0.0) or 0.0)))
    auto = max(promotion_pool, key=lambda item: score_auto(item, model))
    winner_specs = {
        'fast_chat': (
            fast,
            float(fast.get('tokens_per_sec', 0.0) or 0.0),
            f'fastest full measurement with ctx/slot >= {fast_floor}',
        ),
        'auto': (
            auto,
            score_auto(auto, model),
            (
                'quality score: architecture-aware tok/s, ctx/slot, process headroom, and stability'
            ),
        ),
    }
    if long_pool:
        long = max(long_pool, key=lambda item: (score_long_context(item, model), int(item.get('ctx_per_slot', 0) or 0)))
        winner_specs['long_context'] = (
            long,
            float(long.get('ctx_per_slot', 0) or 0),
            'largest full single-slot context, tok/s as tie-breaker',
        )
    if opencode_pool:
        opencode = max(opencode_pool, key=lambda item: (score_opencode_ready(item, model), int(item.get('ctx_per_slot', 0) or 0)))
        winner_specs['opencode_ready'] = (
            opencode,
            float(opencode.get('ctx_per_slot', 0) or 0),
            (
                f'largest full single-slot context meeting OpenCode floor {opencode_floor}'
                if opencode_floor
                else 'best full single-slot fallback; no OpenCode context floor was observed'
            ),
        )
    profiles = {}
    for key, (item, selection_score, selection_reason) in winner_specs.items():
        selected = dict(item)
        selected['selection_score'] = round(float(selection_score or 0.0), 4)
        selected['selection_reason'] = selection_reason
        profile_dict = adaptive_profile_dict(key, item['model'], selected, profile)
        source_objective = str(item.get('objective', '') or '')
        if source_objective and source_objective != key:
            profile_dict['reused_from'] = source_objective
        profiles[key] = profile_dict
    if 'opencode_ready' not in profiles:
        previous_opencode = get_measured_profile(model, 'opencode_ready')
        if measured_profile_meets_opencode_floor(previous_opencode, opencode_floor):
            preserved = dict(previous_opencode)
            preserved['selection_reason'] = f'preserved previous OpenCode-ready profile meeting OpenCode floor {opencode_floor}'
            preserved['preserved_from_previous'] = True
            profiles['opencode_ready'] = preserved
        else:
            fallback = max(
                opencode_single_slot_pool or successful,
                key=lambda item: (score_opencode_ready(item, model), int(item.get('ctx_per_slot', 0) or 0)),
            )
            selected = dict(fallback)
            selected['selection_score'] = round(float(selected.get('ctx_per_slot', 0) or 0), 4)
            selected['selection_reason'] = f'not OpenCode-ready: no measured row met observed OpenCode floor {opencode_floor}'
            profile_dict = adaptive_profile_dict('opencode_ready', fallback['model'], selected, profile)
            source_objective = str(fallback.get('objective', '') or '')
            if source_objective and source_objective != 'opencode_ready':
                profile_dict['reused_from'] = source_objective
            profile_dict['status'] = 'not_ready'
            profile_dict['context_required'] = int(opencode_floor or 0)
            profile_dict['opencode_floor'] = int(opencode_floor or 0)
            profiles['opencode_ready'] = profile_dict
    return profiles


def profile_frontier_fit_mode(record: Dict[str, object]) -> str:
    if bool(record.get('runtime_fit', False) or record.get('fit', False) or record.get('fit_assisted_moe_placement', False)):
        return 'fit_assisted'
    placement = profile_moe_placement(record)
    if (
        placement.get('strategy')
        or placement.get('cpu_moe')
        or int(placement.get('n_cpu_moe', 0) or 0) > 0
        or placement.get('tensor_overrides')
    ):
        return 'locked_moe'
    return 'standard'


def _profile_frontier_slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower()).strip('_')
    return slug or 'profile'


def _profile_frontier_moe_slug(record: Dict[str, object]) -> str:
    placement = profile_moe_placement(record)
    if bool(placement.get('cpu_moe', False)):
        return 'moe_cpu'
    n_cpu_moe = max(0, int(placement.get('n_cpu_moe', 0) or 0))
    if n_cpu_moe > 0:
        return f'moe_ncpu{n_cpu_moe}'
    if placement.get('tensor_overrides'):
        return 'moe_ot'
    return 'no_moe'


def profile_frontier_profile_id(
    category: str,
    record: Dict[str, object],
    source_profile_key: str = '',
) -> str:
    base = (
        source_profile_key
        or str(record.get('objective', '') or '')
        or category
    )
    fit_mode = profile_frontier_fit_mode(record)
    placement_slug = _profile_frontier_moe_slug(record)
    if fit_mode == 'locked_moe' and placement_slug.startswith('moe_'):
        placement_slug = placement_slug[len('moe_'):]
    return '_'.join(
        item for item in (
            _profile_frontier_slug(base),
            fit_mode,
            placement_slug,
        )
        if item
    )


def profile_frontier_stability_rating(record: Dict[str, object]) -> str:
    status = str(record.get('status', '') or '').lower()
    if status not in ('ok', 'probe ok', 'tests passed'):
        return 'failed'
    if low_speed_guardrail_reason(record):
        return 'fragile'
    score = _record_stability_score(record)
    vram_headroom = _record_vram_headroom(record)
    if 0 < vram_headroom < 512 * 1024**2:
        return 'experimental'
    if score >= 0.85:
        return 'stable'
    if score >= 0.65:
        return 'usable'
    return 'fragile'


def profile_frontier_entry(
    category: str,
    label: str,
    record: Dict[str, object],
    reason: str,
    source_profile_key: str = '',
    status: str = '',
    default_eligible: bool = True,
) -> Dict[str, object]:
    placement = profile_moe_placement(record)
    placement_payload: Dict[str, object] = {}
    if (
        placement.get('strategy')
        or placement.get('cpu_moe')
        or int(placement.get('n_cpu_moe', 0) or 0) > 0
        or placement.get('tensor_overrides')
    ):
        placement_payload = dict(placement)
    vram_headroom = _record_vram_headroom(record)
    entry_status = status or str(record.get('status', 'ok') or 'ok')
    fit_mode = profile_frontier_fit_mode(record)
    entry = {
        'category': category,
        'label': label,
        'profile_id': profile_frontier_profile_id(category, record, source_profile_key=source_profile_key),
        'profile_class': 'adaptive' if fit_mode == 'fit_assisted' else 'stable',
        'status': entry_status,
        'source_profile_key': source_profile_key,
        'source_objective': str(record.get('objective', '') or ''),
        'runtime_profile': str(record.get('runtime_profile', '') or ''),
        'tokens_per_sec': round(float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0), 2),
        'ctx': int(record.get('ctx', 0) or 0),
        'ctx_per_slot': int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0),
        'parallel': int(record.get('parallel', 1) or 1),
        'kv_preset': str(record.get('kv_preset', '') or ''),
        'ctk': str(record.get('ctk', '') or ''),
        'ctv': str(record.get('ctv', '') or ''),
        'fit_mode': fit_mode,
        'runtime_fit': bool(record.get('runtime_fit', False) or record.get('fit', False)),
        'fit_context': int(record.get('fit_context', 0) or 0),
        'vram_headroom_bytes': int(vram_headroom or 0),
        'ram_available_bytes': int(record.get('ram_available', 0) or 0),
        'ram_pressure_level': str(record.get('process_pressure_level', '') or ''),
        'ram_pressure_score': round(float(record.get('process_pressure_score', 0.0) or 0.0), 4),
        'stability_score': round(float(_record_stability_score(record)), 4),
        'stability_rating': profile_frontier_stability_rating(record),
        'default_eligible': bool(default_eligible),
        'selection_reason': reason,
        'workflow_result': str(record.get('workflow_result', '') or record.get('objective', '') or category),
    }
    if placement_payload:
        entry['moe_placement'] = placement_payload
    if record.get('reused_from'):
        entry['reused_from'] = record.get('reused_from')
    if record.get('context_required') or record.get('opencode_floor'):
        entry['context_required'] = int(record.get('context_required', record.get('opencode_floor', 0)) or 0)
    if record.get('failure_category'):
        entry['failure_category'] = str(record.get('failure_category') or '')
    if record.get('failure_reason'):
        entry['failure_reason'] = str(record.get('failure_reason') or '')
    if record.get('selection_rejection_reason') or record.get('rejection_reason'):
        entry['selection_rejection_reason'] = str(record.get('selection_rejection_reason') or record.get('rejection_reason') or '')
    return entry


def _frontier_pick_stable_profile(
    records: Sequence[Dict[str, object]],
    model: ModelConfig,
) -> Dict[str, object]:
    pool = [
        item for item in records
        if profile_frontier_fit_mode(item) != 'fit_assisted'
        and profile_frontier_stability_rating(item) in ('stable', 'usable')
    ]
    if not pool:
        return {}
    return max(
        pool,
        key=lambda item: (
            score_auto(item, model),
            profile_frontier_fit_mode(item) == 'locked_moe',
            int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0),
            float(item.get('decode_tokens_per_sec', item.get('tokens_per_sec', 0.0)) or 0.0),
        ),
    )


def _frontier_pick_adaptive_profile(
    records: Sequence[Dict[str, object]],
    model: ModelConfig,
) -> Dict[str, object]:
    pool = [
        item for item in records
        if profile_frontier_fit_mode(item) == 'fit_assisted'
        and str(item.get('status', '') or '').lower() == 'ok'
    ]
    if not pool:
        return {}
    return max(
        pool,
        key=lambda item: (
            int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0),
            score_auto(item, model),
            float(item.get('decode_tokens_per_sec', item.get('tokens_per_sec', 0.0)) or 0.0),
        ),
    )


def _frontier_metric_tuple(record: Dict[str, object]) -> Tuple[float, int, float, float]:
    return (
        float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0),
        int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0),
        float(_record_headroom_score(record)),
        float(_record_stability_score(record)),
    )


def _frontier_dominates(left: Dict[str, object], right: Dict[str, object]) -> bool:
    left_metrics = _frontier_metric_tuple(left)
    right_metrics = _frontier_metric_tuple(right)
    return all(a >= b for a, b in zip(left_metrics, right_metrics)) and any(a > b for a, b in zip(left_metrics, right_metrics))


def pareto_frontier_records(records: Sequence[Dict[str, object]], limit: int = 8) -> List[Dict[str, object]]:
    unique: List[Dict[str, object]] = []
    seen = set()
    for record in records:
        key = (
            int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0),
            int(record.get('parallel', 1) or 1),
            str(record.get('runtime_profile', '') or ''),
            str(record.get('kv_preset', '') or ''),
            str(record.get('placement_strategy', '') or ''),
            bool(record.get('cpu_moe', False)),
            int(record.get('n_cpu_moe', 0) or 0),
            bool(record.get('runtime_fit', False) or record.get('fit', False)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    frontier = [
        record for record in unique
        if not any(other is not record and _frontier_dominates(other, record) for other in unique)
    ]
    frontier.sort(
        key=lambda item: (
            -int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0),
            -float(item.get('decode_tokens_per_sec', item.get('tokens_per_sec', 0.0)) or 0.0),
            str(item.get('runtime_profile', '') or ''),
        )
    )
    return frontier[: max(1, int(limit or 1))]


def _matching_frontier_source(
    records: Sequence[Dict[str, object]],
    profile: Dict[str, object],
) -> Dict[str, object]:
    for record in records:
        if record_matches_profile(record, profile):
            merged = dict(record)
            for key, value in profile.items():
                if key not in merged or merged.get(key) in ('', 0, None, [], {}):
                    merged[key] = value
            return merged
    return dict(profile)


def build_profile_frontier(
    model: ModelConfig,
    measured: Sequence[Dict[str, object]],
    winners: Dict[str, Dict[str, object]],
    profile: HardwareProfile,
    generated_at: str = '',
) -> Dict[str, object]:
    successful = [
        dict(item) for item in list(measured or [])
        if item.get('status') == 'ok' and str(item.get('measurement_type', 'full') or 'full') == 'full'
    ]
    if not successful and not winners:
        return {}
    usable = [
        item for item in successful
        if not low_speed_guardrail_reason(item)
        and not str(item.get('selection_rejection_reason', '') or item.get('rejection_reason', '') or '')
    ] or successful
    single_slot = [item for item in usable if int(item.get('parallel', 1) or 1) == 1] or usable
    stable_single_slot = [
        item for item in single_slot
        if profile_frontier_stability_rating(item) in ('stable', 'usable')
    ] or single_slot
    categories: Dict[str, Dict[str, object]] = {}
    profile_classes: Dict[str, Dict[str, object]] = {}

    stable_profile = _frontier_pick_stable_profile(usable, model)
    if stable_profile:
        profile_classes['stable_profile'] = profile_frontier_entry(
            'stable_profile',
            'Stable Profile',
            stable_profile,
            'predictable measured profile without fit-assisted placement',
            source_profile_key=str(stable_profile.get('objective', '') or 'stable_profile'),
            default_eligible=True,
        )
    adaptive_profile = _frontier_pick_adaptive_profile(usable, model)
    if adaptive_profile:
        profile_classes['adaptive_profile'] = profile_frontier_entry(
            'adaptive_profile',
            'Adaptive Profile',
            adaptive_profile,
            'fit-assisted measured profile; useful when current free memory can change placement',
            source_profile_key=str(adaptive_profile.get('objective', '') or 'adaptive_profile'),
            default_eligible=False,
        )

    if usable:
        fastest = max(usable, key=lambda item: (float(item.get('decode_tokens_per_sec', item.get('tokens_per_sec', 0.0)) or 0.0), int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0)))
        categories['fastest_usable'] = profile_frontier_entry(
            'fastest_usable',
            'Fastest Usable',
            fastest,
            'highest measured decode throughput among usable profiles',
        )

    auto_profile = winners.get('auto') or {}
    if auto_profile:
        auto_source = _matching_frontier_source(successful, auto_profile)
        categories['best_balanced'] = profile_frontier_entry(
            'best_balanced',
            'Best Balanced',
            auto_source,
            str(auto_profile.get('selection_reason') or 'selected auto profile'),
            source_profile_key='auto',
        )
    elif usable:
        balanced = max(usable, key=lambda item: score_auto(item, model))
        categories['best_balanced'] = profile_frontier_entry(
            'best_balanced',
            'Best Balanced',
            balanced,
            'highest balanced score among usable profiles',
        )

    if stable_single_slot:
        highest = max(
            stable_single_slot,
            key=lambda item: (
                int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0),
                float(item.get('decode_tokens_per_sec', item.get('tokens_per_sec', 0.0)) or 0.0),
            ),
        )
        categories['highest_stable_context'] = profile_frontier_entry(
            'highest_stable_context',
            'Highest Stable Context',
            highest,
            'largest single-slot context with stable or usable rating',
        )

    opencode_profile = winners.get('opencode_ready') or {}
    if opencode_profile:
        opencode_source = _matching_frontier_source(successful, opencode_profile)
        categories['opencode_ready'] = profile_frontier_entry(
            'opencode_ready',
            'OpenCode Ready',
            opencode_source,
            str(opencode_profile.get('selection_reason') or 'selected OpenCode-ready profile'),
            source_profile_key='opencode_ready',
            status=str(opencode_profile.get('status', opencode_source.get('status', 'ok')) or 'ok'),
            default_eligible=str(opencode_profile.get('status', 'ok') or 'ok') == 'ok',
        )

    hermes_target = 65_536
    hermes_pool = [
        item for item in single_slot
        if int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0) >= hermes_target
        and profile_frontier_stability_rating(item) in ('stable', 'usable')
    ]
    if hermes_pool:
        hermes = max(hermes_pool, key=lambda item: (score_long_context(item, model), float(item.get('decode_tokens_per_sec', item.get('tokens_per_sec', 0.0)) or 0.0)))
        categories['hermes_ready'] = profile_frontier_entry(
            'hermes_ready',
            'Hermes Ready',
            hermes,
            f'single-slot context meets Hermes floor {hermes_target}',
            source_profile_key='hermes_ready',
        )
    elif stable_single_slot:
        fallback = max(stable_single_slot, key=lambda item: int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0))
        not_ready = dict(fallback)
        not_ready['context_required'] = hermes_target
        categories['hermes_ready'] = profile_frontier_entry(
            'hermes_ready',
            'Hermes Ready',
            not_ready,
            f'not Hermes-ready: no stable single-slot profile met floor {hermes_target}',
            source_profile_key='hermes_ready',
            status='not_ready',
            default_eligible=False,
        )

    if single_slot:
        max_context = max(
            single_slot,
            key=lambda item: (
                int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0),
                float(item.get('decode_tokens_per_sec', item.get('tokens_per_sec', 0.0)) or 0.0),
            ),
        )
        categories['max_context_experimental'] = profile_frontier_entry(
            'max_context_experimental',
            'Max Context Experimental',
            max_context,
            'largest measured single-slot context; stored separately from defaults',
            source_profile_key='max_context_experimental',
            status='experimental',
            default_eligible=False,
        )

    pareto = [
        profile_frontier_entry(
            f'pareto_{index + 1}',
            f'Pareto {index + 1}',
            record,
            'non-dominated by measured throughput, context, headroom, and stability',
            default_eligible=profile_frontier_stability_rating(record) in ('stable', 'usable'),
        )
        for index, record in enumerate(pareto_frontier_records(usable))
    ]
    if not categories and not pareto:
        return {}
    result = {
        'status': 'ok',
        'kind': 'profile_frontier',
        'schema_version': 1,
        'generated_at': generated_at or datetime.now().isoformat(timespec='seconds'),
        'hardware': profile.short_summary() if hasattr(profile, 'short_summary') else '',
        'categories': categories,
        'mode_explanations': {
            'stable': 'Predictable repeatable launch profile; does not rely on fit-assisted runtime placement.',
            'adaptive': 'Fit-assisted launch profile; may fit larger contexts but can vary with currently free memory.',
            'locked_moe': 'Measured manual MoE placement flags are fixed in the launch command.',
            'fit_assisted': 'Measured run used fit-assisted launch behavior in addition to any explicit placement flags.',
        },
        'pareto': pareto,
    }
    if profile_classes:
        result['profile_classes'] = profile_classes
        stable_entry = profile_classes.get('stable_profile') or {}
        adaptive_entry = profile_classes.get('adaptive_profile') or {}
        if stable_entry:
            result['stable_profile'] = str(stable_entry.get('profile_id', '') or '')
        if adaptive_entry:
            result['adaptive_profile'] = str(adaptive_entry.get('profile_id', '') or '')
    return result


def attach_profile_frontier(
    model: ModelConfig,
    measured: Sequence[Dict[str, object]],
    winners: Dict[str, Dict[str, object]],
    profile: HardwareProfile,
    generated_at: str = '',
) -> Dict[str, Dict[str, object]]:
    updated = dict(winners or {})
    frontier = build_profile_frontier(model, measured, updated, profile, generated_at=generated_at)
    if frontier:
        updated['profile_frontier'] = frontier
    return updated


def max_context_probe_targets(model: ModelConfig) -> List[int]:
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', getattr(model, 'ctx', 0) or ctx_min) or ctx_min))
    targets = [value for value in MAX_CONTEXT_PROBE_TARGETS if ctx_min <= value <= ctx_max]
    if ctx_max >= ctx_min and ctx_max not in targets:
        targets.append(ctx_max)
    return sorted(set(max(ctx_min, min(ctx_max, int(value))) for value in targets))


def _kv_profile_sort_key(profile: TurboKvProfile, preferred: Sequence[str]) -> Tuple[int, str]:
    try:
        index = tuple(preferred).index(profile.kv_preset)
    except ValueError:
        index = len(tuple(preferred))
    return index, profile.kv_preset


def max_context_probe_kv_profiles(
    model: ModelConfig,
    capabilities,
    engine: str,
) -> List[TurboKvProfile]:
    engine_key = str(engine or '').strip().lower()
    if engine_key == 'turboquant':
        preferred = ('q8_0/turbo4', 'q8_0/turbo3', 'q8_0/turbo2', 'q8_0/q8_0')
        profiles = [
            item for item in turboquant_auto_profiles(model, capabilities, 'full')
            if item.kv_preset in preferred
        ]
        return sorted(profiles, key=lambda item: _kv_profile_sort_key(item, preferred))
    if engine_key == 'tq3':
        profiles = tq3_kv_profiles_for_model(model, capabilities, 'full')
        preferred = ('q8_0/q8_0', 'tq3_0/tq3_0', 'q4_0/tq3_0')
        return sorted(profiles, key=lambda item: _kv_profile_sort_key(item, preferred))
    if engine_key == 'buun':
        preferred = ('turbo4/turbo4', 'turbo3_tcq/turbo3_tcq', 'turbo3_tcq/turbo2_tcq', 'turbo2_tcq/turbo2_tcq')
        profiles = [
            item for item in supported_turbo_kv_profiles(capabilities, 'full', engine_id='buun')
            if item.kv_preset in preferred
        ]
        return sorted(profiles, key=lambda item: _kv_profile_sort_key(item, preferred))
    allowed = {str(item).strip().lower() for item in tuple(getattr(capabilities, 'supported_kv_modes', ()) or ())}
    if bool(getattr(capabilities, 'supports_cache_type_kv', False)) and {'q8_0'} <= allowed:
        return [TurboKvProfile('q8_0/q8_0', 'q8 baseline', 'baseline', '8.5bpv', 'fast', 0.0, False, 'baseline')]
    return [TurboKvProfile('default', 'default KV', 'baseline', 'default', 'fast', 0.0, False, 'baseline')]


def _max_context_probe_moe_overlay(model: ModelConfig) -> Dict[str, object]:
    saved_placement = current_model_moe_placement(model)
    if saved_placement is None:
        return {}
    payload = (
        profile_moe_placements_from_model(model).get('long_context')
        or profile_moe_placements_from_model(model).get('hermes_ready')
        or profile_moe_placements_from_model(model).get('auto')
        or {}
    )
    placement = moe_placement_candidate_from_payload(payload) or saved_placement
    return {
        'source': str(payload.get('source') or 'current_model_moe_placement'),
        'candidate_name': placement.name,
        'moe_placement_strategy': placement.name,
        'cpu_moe': bool(placement.cpu_moe),
        'n_cpu_moe': max(0, int(placement.n_cpu_moe or 0)),
        'tensor_overrides': list(placement.tensor_overrides or ()),
    }


def max_context_probe_runtime_profiles(
    app: AppConfig,
    model: ModelConfig,
    profile: HardwareProfile,
) -> List[RuntimeProfile]:
    try:
        engine = app.active_engine_key_for_model(model)
    except Exception:
        engine = str(getattr(model, 'runtime', '') or 'llama.cpp')
    try:
        capabilities = app.engine_capabilities()
    except Exception:
        capabilities = default_engine_capabilities(engine)
    targets = max_context_probe_targets(model)
    kv_profiles = max_context_probe_kv_profiles(model, capabilities, engine)
    if not targets or not kv_profiles:
        return []
    try:
        base = app.runtime_profile_from_model(
            model,
            int(getattr(model, 'ctx', 0) or targets[0]),
            1,
            int(getattr(model, 'ngl', 0) or 0),
        )
    except Exception:
        base = RuntimeProfile(
            engine_id=engine,
            ctx_size=int(getattr(model, 'ctx', 0) or targets[0]),
            gpu_layers=int(getattr(model, 'ngl', 0) or 0),
            parallel=1,
            kv_preset='default',
            name='max_context_probe_base',
        )
    overlay = _max_context_probe_moe_overlay(model)
    profiles: List[RuntimeProfile] = []
    for ctx in targets:
        for kv_profile in kv_profiles:
            name = f'max_context_probe_{ctx}_{kv_profile.name_slug}'
            candidate = replace(
                base,
                name=name,
                ctx_size=int(ctx),
                parallel=1,
                kv_preset=kv_profile.kv_preset,
                batch_size=128,
                ubatch_size=64,
                fit=False,
                fit_context=0,
                no_warmup=bool(getattr(capabilities, 'supports_no_warmup', False)),
                kv_family='turbo' if any(mode.startswith('turbo') for mode in kv_modes_from_preset(kv_profile.kv_preset)) else 'cache' if kv_profile.kv_preset != 'default' else 'default',
                kv_quality_tier=kv_profile.quality_tier,
                kv_compression_tier=kv_profile.compression_tier,
                kv_score_penalty=float(kv_profile.score_penalty or 0.0),
                benchmark_depth='full',
                fit_discovery_phase='max_context_probe',
            )
            profiles.append(runtime_profile_with_overlay(candidate, overlay))
            if len(profiles) >= MAX_CONTEXT_PROBE_MAX_CANDIDATES:
                return profiles
    return profiles


def max_context_record_is_safe(record: Dict[str, object], profile: HardwareProfile, target_headroom: int) -> bool:
    if int(getattr(profile, 'gpu_memory_total', 0) or 0) <= 0:
        return True
    headroom = _record_vram_headroom(record)
    return bool(headroom and headroom >= int(target_headroom or 0))


def _max_context_record_sort_key(record: Dict[str, object]) -> Tuple[int, float, float, str]:
    return (
        int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0),
        -float(record.get('kv_score_penalty', 0.0) or 0.0),
        float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0),
        str(record.get('runtime_profile', '') or ''),
    )


def max_context_profile_from_record(
    key: str,
    record: Dict[str, object],
    profile: HardwareProfile,
    target_headroom: int,
    safety_class: str,
    default_eligible: bool,
    selection_reason: str,
    status: str = 'ok',
) -> Dict[str, object]:
    candidate = record.get('model') if isinstance(record.get('model'), ModelConfig) else None
    if candidate is None:
        candidate = ModelConfig(
            id=str(record.get('model_id', '') or 'max-context'),
            name=str(record.get('model_name', '') or record.get('model_id', '') or 'Max Context'),
            path=str(record.get('path', '') or ''),
            alias=str(record.get('alias', '') or 'max-context'),
            port=int(record.get('port', 0) or 0),
            ctx=int(record.get('ctx', 0) or 0),
            parallel=int(record.get('parallel', 1) or 1),
            ngl=int(record.get('ngl', 0) or 0),
        )
    selected = dict(record)
    selected['selection_score'] = int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0)
    selected['selection_reason'] = selection_reason
    profile_dict = adaptive_profile_dict(key, candidate, selected, profile)
    profile_dict['status'] = status
    profile_dict['kind'] = 'max_context_probe'
    profile_dict['safety_class'] = safety_class
    profile_dict['default_eligible'] = bool(default_eligible)
    profile_dict['target_vram_headroom_bytes'] = int(target_headroom or 0)
    profile_dict['vram_headroom_bytes'] = _record_vram_headroom(record)
    profile_dict['profile_class'] = 'adaptive' if profile_frontier_fit_mode(record) == 'fit_assisted' else 'stable'
    profile_dict['fit_mode'] = profile_frontier_fit_mode(record)
    profile_dict['selection_reason'] = selection_reason
    return profile_dict


def select_max_context_probe_profiles(
    model: ModelConfig,
    measured: Sequence[Dict[str, object]],
    profile: HardwareProfile,
) -> Dict[str, Dict[str, object]]:
    successful = [
        dict(item) for item in list(measured or [])
        if str(item.get('status', '') or '').lower() == 'ok'
        and str(item.get('measurement_type', 'full') or 'full') == 'full'
    ]
    if not successful:
        return {}
    target_headroom = target_vram_headroom_bytes(profile)
    safe_pool = [
        item for item in successful
        if max_context_record_is_safe(item, profile, target_headroom)
    ]
    largest_success = max(successful, key=_max_context_record_sort_key)
    winners: Dict[str, Dict[str, object]] = {}
    if safe_pool:
        safe = max(safe_pool, key=_max_context_record_sort_key)
        winners['max_context_safe'] = max_context_profile_from_record(
            'max_context_safe',
            safe,
            profile,
            target_headroom,
            'safe',
            True,
            f'largest measured context with VRAM headroom >= {target_headroom // (1024 ** 2)} MiB',
        )
    else:
        winners['max_context_safe'] = max_context_profile_from_record(
            'max_context_safe',
            largest_success,
            profile,
            target_headroom,
            'unsafe',
            False,
            f'not safe: no measured max-context row kept VRAM headroom >= {target_headroom // (1024 ** 2)} MiB',
            status='not_ready',
        )
    experimental_safe = max_context_record_is_safe(largest_success, profile, target_headroom)
    winners['max_context_experimental'] = max_context_profile_from_record(
        'max_context_experimental',
        largest_success,
        profile,
        target_headroom,
        'safe' if experimental_safe else 'experimental',
        False,
        'largest measured successful max-context row; stored separately from defaults',
    )
    return winners


def _cache_ram_mib(model: ModelConfig) -> int:
    return max(0, int(getattr(model, 'cache_ram', 0) or 0))


def workflow_cache_ram_values(model: ModelConfig, profile: Optional[HardwareProfile] = None) -> List[int]:
    current = _cache_ram_mib(model)
    values: List[int] = [current]
    values.extend(int(item) for item in WORKFLOW_CACHE_RAM_VARIANTS)
    available_mib = 0
    if profile is not None:
        try:
            available_mib = max(0, int(getattr(profile, 'memory_available', 0) or 0) // (1024 ** 2))
        except (TypeError, ValueError):
            available_mib = 0
    soft_limit = max(0, available_mib - WORKFLOW_CACHE_RAM_RESERVE_MIB) if available_mib else 0
    unique: List[int] = []
    for value in values:
        cache_ram = max(0, int(value or 0))
        if cache_ram in unique:
            continue
        if soft_limit and cache_ram > soft_limit and cache_ram not in (0, current):
            continue
        unique.append(cache_ram)
    return unique


def model_with_cache_ram_variant(model: ModelConfig, cache_ram_mib: int) -> ModelConfig:
    candidate = ModelConfig(**asdict(model))
    candidate.cache_ram = max(0, int(cache_ram_mib or 0))
    return candidate


def _cache_ram_variant_name(value: int) -> str:
    return f'cache_ram_{max(0, int(value or 0))}'


def expand_workflow_cache_ram_candidates(
    candidates: Sequence[Tuple[object, ...]],
    profile: Optional[HardwareProfile] = None,
    *,
    max_base_candidates: int = WORKFLOW_CACHE_RAM_BASE_CANDIDATES,
) -> List[Tuple[object, ...]]:
    expanded: List[Tuple[object, ...]] = []
    seen = set()
    for index, item in enumerate(list(candidates or [])):
        if len(item) < 4 or not isinstance(item[2], ModelConfig):
            if item not in expanded:
                expanded.append(item)
            continue
        label, tier, model, detail, *rest = item
        values = workflow_cache_ram_values(model, profile) if index < max_base_candidates else [_cache_ram_mib(model)]
        current = _cache_ram_mib(model)
        for value in values:
            variant_model = model_with_cache_ram_variant(model, value)
            if value == current:
                variant_label = str(label)
                variant_tier = str(tier)
            else:
                suffix = _cache_ram_variant_name(value)
                variant_label = f'{label}_{suffix}'
                variant_tier = f'{tier}+{suffix}'
            variant_detail = f'{detail} cache_ram={value} MiB'
            key = (
                variant_label,
                variant_tier,
                int(getattr(variant_model, 'ctx', 0) or 0),
                int(getattr(variant_model, 'parallel', 1) or 1),
                int(getattr(variant_model, 'cache_ram', 0) or 0),
                tuple(getattr(variant_model, 'extra_args', []) or []),
                tuple(str(part) for part in rest),
            )
            if key in seen:
                continue
            seen.add(key)
            expanded.append((variant_label, variant_tier, variant_model, variant_detail, *rest))
    return expanded


def workflow_cache_ram_record_fields(model: ModelConfig) -> Dict[str, object]:
    cache_ram = _cache_ram_mib(model)
    return {
        'cache_ram': cache_ram,
        'cache_ram_mib': cache_ram,
        'cache_ram_variant': _cache_ram_variant_name(cache_ram),
    }


def workflow_cache_ram_selection_key(record: Dict[str, object]) -> Tuple[float, float, int, float, float, float]:
    score = float(record.get('score', 0.0) or 0.0)
    tasks = max(1, int(record.get('tasks', 0) or 0))
    passed = max(0, int(record.get('passed', 0) or 0))
    pass_ratio = min(1.0, passed / tasks)
    status = str(record.get('status', '') or '').lower()
    timeout_type = str(record.get('timeout_type', '') or '').strip().lower()
    stable = 0 if 'timeout' in status or timeout_type else 1
    seconds = max(0.0, float(record.get('seconds', 0.0) or 0.0))
    cache_ram = float(record.get('cache_ram_mib', record.get('cache_ram', 0)) or 0.0)
    pressure = float(record.get('process_pressure_score', 0.0) or 0.0)
    return (score, pass_ratio, stable, -seconds, -cache_ram, -pressure)


def workflow_cache_ram_score(record: Dict[str, object]) -> float:
    score, pass_ratio, stable, neg_seconds, neg_cache_ram, neg_pressure = workflow_cache_ram_selection_key(record)
    return (
        score * 100.0
        + pass_ratio * 20.0
        + float(stable) * 5.0
        + neg_seconds * 0.05
        + neg_cache_ram * 0.001
        + neg_pressure
    )


def workflow_cache_ram_profile_from_record(workflow: str, record: Dict[str, object]) -> Dict[str, object]:
    cache_ram = int(record.get('cache_ram_mib', record.get('cache_ram', 0)) or 0)
    passed = int(record.get('passed', 0) or 0)
    tasks = int(record.get('tasks', 0) or 0)
    return {
        'status': 'ok' if tasks > 0 and passed >= tasks else 'not_ready',
        'kind': 'workflow_cache_ram',
        'workflow': str(workflow or ''),
        'cache_ram': cache_ram,
        'cache_ram_mib': cache_ram,
        'cache_ram_variant': _cache_ram_variant_name(cache_ram),
        'workflow_cache_ram_score': round(workflow_cache_ram_score(record), 4),
        'score': float(record.get('score', 0.0) or 0.0),
        'seconds': float(record.get('seconds', 0.0) or 0.0),
        'passed': passed,
        'tasks': tasks,
        'ctx': int(record.get('ctx', 0) or 0),
        'ctx_per_slot': int(record.get('ctx_per_slot', 0) or 0),
        'preset': str(record.get('preset', '') or ''),
        'tier': str(record.get('tier', '') or ''),
        'selection_reason': 'best workflow result by task score, completion time, stability, and cache-RAM overhead',
        'benchmarked_at': str(record.get('benchmarked_at', '') or datetime.now().isoformat(timespec='seconds')),
    }


def benchmark_profile_is_fresh(app: AppConfig, model: ModelConfig) -> bool:
    if not getattr(model, 'enabled', True):
        return False
    if (getattr(model, 'default_benchmark_status', '') or '').strip().lower() != 'done':
        return False
    saved_fingerprint = str(getattr(model, 'benchmark_fingerprint', '') or '')
    if not saved_fingerprint:
        return False
    try:
        if saved_fingerprint != app.model_fingerprint(model):
            return False
    except Exception:
        return False
    auto_profile = get_measured_profile(model, 'auto')
    if not auto_profile:
        return False
    return (
        float(auto_profile.get('tokens_per_sec', 0.0) or 0.0) > 0.0
        and int(auto_profile.get('ctx_per_slot', auto_profile.get('ctx', 0)) or 0) > 0
    )


def deep_benchmark_model_decision(app: AppConfig, model: ModelConfig, force: bool = False) -> Tuple[bool, str]:
    if not getattr(model, 'enabled', True):
        return False, 'disabled'
    if force:
        return True, 'force refresh'
    if benchmark_profile_is_fresh(app, model):
        return False, 'fresh benchmark'

    status = (getattr(model, 'default_benchmark_status', '') or '').strip().lower()
    saved_fingerprint = str(getattr(model, 'benchmark_fingerprint', '') or '')
    try:
        current_fingerprint = app.model_fingerprint(model)
    except Exception:
        current_fingerprint = ''
    if status in ('pending', 'failed', 'aborted', 'running'):
        return True, status or 'pending'
    if saved_fingerprint and current_fingerprint and saved_fingerprint != current_fingerprint:
        return True, 'stale fingerprint'
    if not get_measured_profile(model, 'auto'):
        return True, 'missing measured auto profile'
    return True, 'missing fresh benchmark'


def _machine_headroom_score(profile: Dict[str, object]) -> float:
    ram = int(profile.get('ram_available', 0) or 0)
    vram = int(profile.get('gpu_memory_free', 0) or 0)
    score = min(1.0, (ram / 1024**3) / 8.0) if ram else 0.35
    if vram:
        score = max(score, min(1.0, (vram / 1024**3) / 2.0))
    pressure = float(profile.get('process_pressure_score', 0.0) or 0.0)
    if pressure:
        score *= max(0.45, 1.0 - pressure * 0.35)
    return max(0.0, min(1.0, score))


def _machine_stability_score(profile: Dict[str, object]) -> float:
    score = 1.0
    if int(profile.get('retry_attempt', 1) or 1) > 1:
        score -= 0.15
    detail = compact_message(str(profile.get('detail', '') or '')).lower()
    if detail not in ('', '1 samples', '2 samples', '3 samples'):
        score -= 0.05
    return max(0.0, min(1.0, score))


def machine_benchmark_rows(app: AppConfig, models: Optional[List[ModelConfig]] = None) -> List[Dict[str, object]]:
    source_models = list(models if models is not None else getattr(app, 'models', []) or [])
    rows: List[Dict[str, object]] = []
    for model in source_models:
        if not benchmark_profile_is_fresh(app, model):
            continue
        auto = get_measured_profile(model, 'auto')
        fast = get_measured_profile(model, 'fast_chat')
        long = get_measured_profile(model, 'long_context')
        opencode = get_measured_profile(model, 'opencode_ready')
        if not auto:
            continue
        opencode_floor = observed_opencode_context_floor(model)
        auto_ctx_slot = int(auto.get('ctx_per_slot', auto.get('ctx', 0)) or 0)
        row = {
            'model_id': model.id,
            'name': getattr(model, 'name', '') or model.id,
            'runtime': getattr(model, 'runtime', 'llama.cpp'),
            'architecture': architecture_label(model),
            'architecture_type': getattr(model, 'architecture_type', 'unknown') or 'unknown',
            'quant': str(getattr(model, 'path', '') or ''),
            'auto_tokens_per_sec': float(auto.get('tokens_per_sec', 0.0) or 0.0),
            'auto_ctx_per_slot': auto_ctx_slot,
            'auto_parallel': int(auto.get('parallel', 1) or 1),
            'fast_tokens_per_sec': float(fast.get('tokens_per_sec', auto.get('tokens_per_sec', 0.0)) or 0.0),
            'long_ctx_per_slot': int(long.get('ctx_per_slot', auto_ctx_slot) or 0),
            'opencode_ctx_per_slot': int(opencode.get('ctx_per_slot', auto_ctx_slot) or 0),
            'opencode_floor': int(opencode_floor or 0),
            'opencode_meets_floor': bool(not opencode_floor or int(opencode.get('ctx_per_slot', auto_ctx_slot) or 0) >= opencode_floor),
            'ram_available': int(auto.get('ram_available', 0) or 0),
            'gpu_memory_free': int(auto.get('gpu_memory_free', 0) or 0),
            'process_pressure_level': auto.get('process_pressure_level', ''),
            'process_pressure_score': float(auto.get('process_pressure_score', 0.0) or 0.0),
            'process_pressure_detail': str(auto.get('process_pressure_detail', '') or ''),
            'stability_score': _machine_stability_score(auto),
            'headroom_score': _machine_headroom_score(auto),
            'benchmarked_at': str(auto.get('benchmarked_at') or getattr(model, 'default_benchmark_at', '') or ''),
            'selection_reason': str(auto.get('selection_reason', '') or 'fresh measured auto profile'),
        }
        rows.append(row)

    max_tps = max([float(row.get('auto_tokens_per_sec', 0.0) or 0.0) for row in rows] or [0.0]) or 1.0
    max_ctx = max([int(row.get('auto_ctx_per_slot', 0) or 0) for row in rows] or [0]) or 1
    for row in rows:
        tps_norm = float(row.get('auto_tokens_per_sec', 0.0) or 0.0) / max_tps
        ctx_norm = int(row.get('auto_ctx_per_slot', 0) or 0) / max_ctx
        headroom = float(row.get('headroom_score', 0.0) or 0.0)
        stability = float(row.get('stability_score', 0.0) or 0.0)
        score = (
            0.50 * tps_norm
            + 0.30 * ctx_norm
            + 0.12 * headroom
            + 0.08 * stability
        )
        row['machine_score'] = round(score * 100.0, 2)
        row['machine_reason'] = (
            f'auto {float(row.get("auto_tokens_per_sec", 0.0) or 0.0):.2f} tok/s '
            f'({int(round(tps_norm * 100))}% of fastest), '
            f'ctx/slot {int(row.get("auto_ctx_per_slot", 0) or 0)} '
            f'({int(round(ctx_norm * 100))}% of highest), '
            f'headroom {int(round(headroom * 100))}%, '
            f'stability {int(round(stability * 100))}%, '
            f'pressure {row.get("process_pressure_level") or "unknown"}'
        )
    return sorted(rows, key=lambda row: (-float(row.get('machine_score', 0.0) or 0.0), str(row.get('model_id', ''))))


def _machine_winner(rows: List[Dict[str, object]], key: str) -> Dict[str, object]:
    if not rows:
        return {}
    if key == 'fastest_chat':
        winner = max(rows, key=lambda row: (float(row.get('fast_tokens_per_sec', 0.0) or 0.0), float(row.get('machine_score', 0.0) or 0.0)))
        return {
            'label': 'Fastest Chat',
            'model_id': winner.get('model_id', ''),
            'metric': f'{float(winner.get("fast_tokens_per_sec", 0.0) or 0.0):.2f} tok/s',
            'reason': 'highest measured Fast Chat throughput',
            'row': dict(winner),
        }
    if key == 'longest_context':
        winner = max(rows, key=lambda row: (int(row.get('long_ctx_per_slot', 0) or 0), float(row.get('auto_tokens_per_sec', 0.0) or 0.0)))
        return {
            'label': 'Longest Context',
            'model_id': winner.get('model_id', ''),
            'metric': f'{int(winner.get("long_ctx_per_slot", 0) or 0)} ctx/slot',
            'reason': 'largest measured Long Context ctx/slot',
            'row': dict(winner),
        }
    if key == 'opencode_ready':
        meets_floor = [row for row in rows if bool(row.get('opencode_meets_floor'))]
        pool = meets_floor or rows
        winner = max(pool, key=lambda row: (int(row.get('opencode_ctx_per_slot', 0) or 0), float(row.get('auto_tokens_per_sec', 0.0) or 0.0)))
        floor = int(winner.get('opencode_floor', 0) or 0)
        if meets_floor and floor:
            reason = f'meets observed OpenCode floor {floor}'
        elif meets_floor:
            reason = 'best measured OpenCode-ready profile; no floor observed'
        else:
            reason = f'fallback: no model met observed OpenCode floor {floor}'
        return {
            'label': 'OpenCode-ready',
            'model_id': winner.get('model_id', ''),
            'metric': f'{int(winner.get("opencode_ctx_per_slot", 0) or 0)} ctx/slot',
            'reason': reason,
            'row': dict(winner),
        }
    winner = max(rows, key=lambda row: (float(row.get('machine_score', 0.0) or 0.0), float(row.get('auto_tokens_per_sec', 0.0) or 0.0)))
    return {
        'label': 'Machine Pick',
        'model_id': winner.get('model_id', ''),
        'metric': f'{float(winner.get("machine_score", 0.0) or 0.0):.2f}',
        'reason': str(winner.get('machine_reason', '') or 'weighted machine score'),
        'row': dict(winner),
    }


def machine_best_summary(app: AppConfig, models: Optional[List[ModelConfig]] = None) -> Dict[str, object]:
    rows = machine_benchmark_rows(app, models=models)
    categories = {
        'fastest_chat': _machine_winner(rows, 'fastest_chat'),
        'longest_context': _machine_winner(rows, 'longest_context'),
        'opencode_ready': _machine_winner(rows, 'opencode_ready'),
        'machine_pick': _machine_winner(rows, 'machine_pick'),
    }
    return {
        'rows': rows,
        'categories': categories,
        'machine_pick': categories.get('machine_pick') or {},
        'benchmarked_count': len(rows),
    }


def record_matches_profile(record: Dict[str, object], profile: Dict[str, object]) -> bool:
    if not record or not profile:
        return False
    record_ctx = int(record.get('ctx', 0) or 0)
    profile_ctx = int(profile.get('ctx', 0) or 0)
    record_parallel = int(record.get('parallel', 1) or 1)
    profile_parallel = int(profile.get('parallel', 1) or 1)
    if record_ctx != profile_ctx or record_parallel != profile_parallel:
        return False
    profile_variant = str(profile.get('variant', '') or '')
    if profile_variant and str(record.get('variant', '') or 'default') != profile_variant:
        return False
    record_tps = float(record.get('tokens_per_sec', 0.0) or 0.0)
    profile_tps = float(profile.get('tokens_per_sec', 0.0) or 0.0)
    return abs(record_tps - profile_tps) < 0.05 or profile_tps <= 0


def add_spectrum_label(record: Dict[str, object], label: str):
    current = str(record.get('spectrum_label', '') or '').strip()
    labels = [item.strip() for item in current.split(',') if item.strip()]
    display = SPECTRUM_LABELS.get(label, label)
    if display not in labels:
        labels.append(display)
    record['spectrum_label'] = ', '.join(labels)


def annotate_spectrum_records(
    records: List[Dict[str, object]],
    winners: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    for record in records:
        if record.get('status') not in ('ok', 'probe ok'):
            add_spectrum_label(record, 'failed')
        if record.get('break_point'):
            add_spectrum_label(record, 'break_point')
    successful = [
        record for record in records
        if record.get('status') == 'ok' and str(record.get('measurement_type', 'full') or 'full') == 'full'
    ]
    if not successful:
        return records
    possible = min(
        successful,
        key=lambda item: (
            int(item.get('ctx_per_slot', 0) or 0),
            -float(item.get('tokens_per_sec', 0.0) or 0.0),
        ),
    )
    add_spectrum_label(possible, 'possible')
    winner_labels = {
        'fast_chat': 'fastest',
        'auto': 'ideal',
        'long_context': 'longest',
        'opencode_ready': 'opencode',
    }
    for key, label in winner_labels.items():
        profile = winners.get(key) or {}
        if str(profile.get('status', 'ok') or 'ok') != 'ok':
            continue
        for record in successful:
            if record_matches_profile(record, profile):
                add_spectrum_label(record, 'winner')
                add_spectrum_label(record, label)
                break
    max_tps = max(float(item.get('tokens_per_sec', 0.0) or 0.0) for item in successful) or 1.0
    max_ctx = max(int(item.get('ctx_per_slot', 0) or 0) for item in successful) or 1

    def runner_pool(key: str) -> List[Dict[str, object]]:
        if key == 'fast_chat':
            return sorted(
                [item for item in successful if item.get('objective') == 'fast_chat'],
                key=lambda item: (float(item.get('tokens_per_sec', 0.0) or 0.0), int(item.get('ctx_per_slot', 0) or 0)),
                reverse=True,
            )
        if key == 'long_context':
            return sorted(
                [item for item in successful if item.get('objective') == 'long_context'],
                key=lambda item: (int(item.get('ctx_per_slot', 0) or 0), float(item.get('tokens_per_sec', 0.0) or 0.0)),
                reverse=True,
            )
        if key == 'opencode_ready':
            return sorted(
                [item for item in successful if item.get('objective') == 'opencode_ready'],
                key=lambda item: (int(item.get('ctx_per_slot', 0) or 0), float(item.get('tokens_per_sec', 0.0) or 0.0)),
                reverse=True,
            )
        return sorted(
            successful,
            key=lambda item: (
                0.55 * (float(item.get('tokens_per_sec', 0.0) or 0.0) / max_tps)
                + 0.35 * (int(item.get('ctx_per_slot', 0) or 0) / max_ctx)
            ),
            reverse=True,
        )

    for key in ('fast_chat', 'long_context', 'opencode_ready', 'auto'):
        profile = winners.get(key) or {}
        runner_candidates = [item for item in runner_pool(key) if not record_matches_profile(item, profile)]
        if runner_candidates:
            add_spectrum_label(runner_candidates[0], 'runner_up')
    return records


def opencode_profile_status_text(winners: Dict[str, Dict[str, object]]) -> str:
    profile = winners.get('opencode_ready') or {}
    if not profile:
        return 'opencode not ready'
    ctx = int(profile.get('ctx_per_slot', profile.get('ctx', 0)) or 0)
    if str(profile.get('status', 'ok') or 'ok') == 'ok':
        return f'opencode ctx/slot={ctx}'
    floor = int(profile.get('context_required', profile.get('opencode_floor', 0)) or 0)
    if floor:
        return f'opencode not ready (best ctx/slot={ctx}, floor={floor})'
    return f'opencode not ready (best ctx/slot={ctx})'


def skipped_adaptive_profile(key: str, reason: str, status: str = 'skipped_budget') -> Dict[str, object]:
    return {
        'status': status,
        'kind': 'adaptive_profile',
        'profile_key': key,
        'objective': key,
        'tokens_per_sec': 0.0,
        'seconds': 0.0,
        'ctx': 0,
        'ctx_per_slot': 0,
        'parallel': 1,
        'selection_score': 0.0,
        'selection_reason': reason,
        'reason': reason,
    }


def fill_missing_adaptive_profiles(
    winners: Dict[str, Dict[str, object]],
    reason: str,
    status: str = 'skipped_budget',
) -> Dict[str, Dict[str, object]]:
    updated = dict(winners or {})
    for key in ADAPTIVE_PROFILE_KEYS:
        if key not in updated:
            updated[key] = skipped_adaptive_profile(key, reason, status=status)
    return updated


def adaptive_profile_status_text(winners: Dict[str, Dict[str, object]], key: str) -> str:
    profile = winners.get(key) or {}
    status = str(profile.get('status', '') or '')
    if status == 'ok':
        if key == 'fast_chat':
            return f'fast={float(profile.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s'
        if key == 'long_context':
            return f'long ctx/slot={int(profile.get("ctx_per_slot", profile.get("ctx", 0)) or 0)}'
        if key == 'auto':
            return f'auto ctx={int(profile.get("ctx", 0) or 0)} parallel={int(profile.get("parallel", 1) or 1)}'
    if not profile:
        return f'{key}=missing'
    return f'{key}={status or "not_ready"}'


def adaptive_profiles_complete(winners: Dict[str, Dict[str, object]]) -> bool:
    return all(str((winners.get(key) or {}).get('status', '') or '') == 'ok' for key in ('fast_chat', 'long_context', 'auto'))


def benchmark_run_summary(winners: Dict[str, Dict[str, object]], records: Optional[List[Dict[str, object]]] = None) -> str:
    low_speed_count = sum(1 for row in list(records or []) if low_speed_guardrail_reason(row))
    rejected_partial_count = sum(
        1 for row in list(records or [])
        if str(row.get('rejection_reason', '') or row.get('selection_rejection_reason', '') or '') == 'rejected_slow_partial_offload'
    )
    if not winners:
        if low_speed_count:
            return f'no winners, {low_speed_count} TQ3 low-speed profile(s) held back'
        if rejected_partial_count:
            return f'no winners, {rejected_partial_count} slow partial-offload profile(s) rejected'
        return 'no winners'
    parts = []
    fast = winners.get('fast_chat') or {}
    long = winners.get('long_context') or {}
    auto = winners.get('auto') or {}
    if fast:
        parts.append(f'fast={float(fast.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s')
    if long:
        parts.append(f'long={int(long.get("ctx_per_slot", 0) or 0)} ctx/slot')
    if auto:
        parts.append(f'auto={int(auto.get("ctx", 0) or 0)} ctx')
    moe = winners.get('moe_placement') or {}
    if moe:
        candidate = str(moe.get('measured_candidate_name', '') or moe.get('runtime_profile', '') or 'moe')
        parts.append(f'moe={candidate} {float(moe.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s')
    max_safe = winners.get('max_context_safe') or {}
    max_experimental = winners.get('max_context_experimental') or {}
    if max_safe or max_experimental:
        safe_ctx = int(max_safe.get('ctx_per_slot', max_safe.get('ctx', 0)) or 0)
        exp_ctx = int(max_experimental.get('ctx_per_slot', max_experimental.get('ctx', 0)) or 0)
        safe_status = str(max_safe.get('status', '') or '')
        if safe_ctx and safe_status == 'ok':
            parts.append(f'maxctx safe={safe_ctx}')
        if exp_ctx:
            parts.append(f'maxctx experimental={exp_ctx}')
    if winners.get('opencode_ready'):
        parts.append(opencode_profile_status_text(winners))
    if any(
        str(row.get('engine', '') or '') == 'buun' and bool(row.get('runtime_fit', False))
        for row in list(winners.values()) + list(records or [])
    ):
        parts.append('buun fit profile')
    failed_count = sum(
        1 for row in list(records or [])
        if str(row.get('status', '') or '').lower() not in ('ok', 'probe ok', 'skipped')
    )
    if failed_count and any(str(row.get('status', '') or '').lower() == 'ok' for row in list(winners.values())):
        parts.append(f'{failed_count} candidate failure(s), winners saved')
    if low_speed_count:
        parts.append(f'{low_speed_count} TQ3 low-speed profile(s) held back')
    if rejected_partial_count:
        parts.append(f'{rejected_partial_count} slow partial-offload profile(s) rejected')
    return ', '.join(parts) if parts else 'no winners'


def upsert_benchmark_run(model: ModelConfig, run: Dict[str, object], limit: int = BENCHMARK_HISTORY_LIMIT):
    run_id = str(run.get('id', '') or '')
    existing = list(getattr(model, 'benchmark_runs', []) or [])
    filtered = [item for item in existing if str(item.get('id', '') or '') != run_id]
    filtered.insert(0, dict(run))
    model.benchmark_runs = filtered[: max(1, int(limit or BENCHMARK_HISTORY_LIMIT))]


def close_stale_running_benchmark_runs(
    model: ModelConfig,
    ended_at: str,
    reason: str = 'aborted_stale_previous_run',
) -> bool:
    changed = False
    runs: List[Dict[str, object]] = []
    for item in list(getattr(model, 'benchmark_runs', []) or []):
        run = dict(item)
        if str(run.get('status', '') or '').strip().lower() == 'running':
            run['status'] = 'aborted'
            run['ended_at'] = ended_at
            run['stale_reason'] = reason
            summary = str(run.get('summary', '') or '').strip()
            run['summary'] = f'{summary}; {reason}' if summary else reason
            changed = True
        runs.append(run)
    if runs:
        model.benchmark_runs = runs[:BENCHMARK_HISTORY_LIMIT]
    if str(getattr(model, 'default_benchmark_status', '') or '').strip().lower() == 'running':
        model.default_benchmark_status = 'aborted'
        model.default_benchmark_at = ended_at
        changed = True
    return changed


def persist_running_benchmark_progress(
    app,
    model: ModelConfig,
    run_id: str,
    kind: str,
    records: List[Dict[str, object]],
    started_at: str,
    hardware: str = '',
):
    running_model = ModelConfig(**asdict(model))
    close_stale_running_benchmark_runs(running_model, started_at)
    running_model.last_benchmark_results = [dict(row) for row in records]
    running_model.default_benchmark_status = 'running'
    running_model.default_benchmark_at = started_at
    run = build_benchmark_run(run_id, kind, 'running', records, {}, started_at, hardware=hardware)
    upsert_benchmark_run(running_model, run)
    app.add_or_update(running_model)


def build_benchmark_run(
    run_id: str,
    kind: str,
    status: str,
    records: List[Dict[str, object]],
    winners: Dict[str, Dict[str, object]],
    started_at: str,
    ended_at: str = '',
    hardware: str = '',
) -> Dict[str, object]:
    successful = [row for row in records if row.get('status') == 'ok']
    failed = [row for row in records if row.get('status') not in ('ok', 'probe ok', 'skipped')]
    elapsed = 0.0
    for row in records:
        elapsed += float(row.get('seconds', 0.0) or 0.0)
    run = {
        'id': run_id,
        'kind': kind,
        'status': status,
        'started_at': started_at,
        'ended_at': ended_at,
        'elapsed_seconds': round(elapsed, 2),
        'records': [dict(row) for row in records],
        'winners': {key: dict(value) for key, value in winners.items()},
        'summary': benchmark_run_summary(winners, records),
        'successful': len(successful),
        'failed': len(failed),
        'hardware': hardware,
    }
    run['benchmark_profiles'] = sorted({
        str(row.get('benchmark_profile', '') or '')
        for row in records
        if str(row.get('benchmark_profile', '') or '')
    })
    run['benchmark_purposes'] = sorted({
        str(row.get('benchmark_purpose', '') or '')
        for row in records
        if str(row.get('benchmark_purpose', '') or '')
    })
    run['engines'] = sorted({
        str(row.get('engine', '') or '')
        for row in records
        if str(row.get('engine', '') or '')
    })
    run['measurement_outputs'] = sorted({
        int(row.get('measurement_output', 0) or 0)
        for row in records
        if int(row.get('measurement_output', 0) or 0) > 0
    })
    run.update(current_process_pressure_payload())
    return run


def adaptive_record_from_candidate(
    candidate: ModelConfig,
    objective: str,
    status: str,
    tokens_per_sec: float = 0.0,
    seconds: float = 0.0,
    detail: str = '',
    ram_available: int = 0,
    gpu_memory_free: int = 0,
    startup_seconds: float = 0.0,
    ready_seconds: float = 0.0,
    warmup_seconds: float = 0.0,
    prompt_tokens: int = 0,
    generated_tokens: int = 0,
    process_snapshots: Optional[Dict[str, Dict[str, object]]] = None,
    engine: str = '',
    server_bin: str = '',
    runtime_profile: str = '',
    benchmark_profile: str = '',
    benchmark_purpose: str = '',
    kv_preset: str = '',
    flash_attn_mode: str = '',
    flash_attn: str = '',
    kv_family: str = 'default',
    kv_quality_tier: str = '',
    kv_compression_tier: str = '',
    kv_score_penalty: float = 0.0,
    benchmark_depth: str = '',
    runtime_fit: bool = False,
    fit: bool = False,
    fit_context: int = 0,
    fit_target: str = '',
    runtime_no_warmup: bool = False,
    gpu_layers_mode: str = '',
    batch_size: int = 0,
    ubatch_size: int = 0,
    ctk: str = '',
    ctv: str = '',
    detected_head_dim: int = 0,
    model_quant: str = '',
    model_family: str = '',
    tq3_status: str = '',
    tq3_weight_format: str = '',
    binary_path: str = '',
    help_supported_cache_types: Optional[List[str]] = None,
    runtime_log_path: str = '',
    failure_excerpt: str = '',
    fit_discovery_phase: str = '',
    viable_ngl: int = 0,
    viable_ngl_source: str = '',
    fit_selected_ngl: int = 0,
    fit_selected_ngl_source: str = '',
    fit_log_excerpt: str = '',
    placement_strategy: str = '',
    cpu_moe: bool = False,
    n_cpu_moe: int = 0,
    tensor_overrides: Optional[List[str]] = None,
    moe_placement_mode: str = '',
    fit_assisted_moe_placement: bool = False,
    startup_result: str = '',
    failure_category: str = '',
    failure_reason: str = '',
    suggested_fix: str = '',
    command: str = '',
    prompt_tokens_per_sec: float = 0.0,
    peak_vram_used: int = 0,
    gpu_memory_total: int = 0,
    ctx: Optional[int] = None,
    output: Optional[int] = None,
    measurement_output: int = 0,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    min_p: Optional[float] = None,
    seed: Optional[int] = None,
    samplers: str = '',
    kv_key: str = '',
    kv_value: str = '',
    no_context_shift: bool = False,
    cache_prompt: Optional[bool] = None,
    cache_reuse: int = 0,
    reasoning: str = '',
    reasoning_budget: Optional[int] = None,
    reasoning_format: str = '',
    reasoning_mode: str = '',
    preserve_thinking: bool = False,
    preserve_thinking_source: str = '',
    chat_template_kwargs: Optional[Dict[str, object]] = None,
    unsupported_launch_flags: Optional[List[str]] = None,
    effective_server_args: Optional[List[str]] = None,
    effective_server_command: str = '',
    mtp_enabled: bool = False,
    mtp_draft_n_max: int = 0,
    spec_type: str = '',
) -> Dict[str, object]:
    if not startup_result:
        startup_result = 'READY' if status == 'ok' else 'FAILED' if status in ('start failed', 'not ready') else ''
    tensor_override_values = (
        [str(item).strip() for item in tensor_overrides if str(item).strip()]
        if isinstance(tensor_overrides, list)
        else [str(item).strip() for item in list(getattr(candidate, 'tensor_overrides', []) or []) if str(item).strip()]
    )
    placement_strategy = str(placement_strategy or getattr(candidate, 'moe_placement_strategy', '') or '').strip()
    cpu_moe = bool(cpu_moe or getattr(candidate, 'cpu_moe', False))
    n_cpu_moe = max(0, int(n_cpu_moe or getattr(candidate, 'n_cpu_moe', 0) or 0))
    if not placement_strategy:
        if cpu_moe:
            placement_strategy = 'cpu_moe_all'
        elif n_cpu_moe > 0:
            placement_strategy = f'n_cpu_moe_{n_cpu_moe}'
        elif tensor_override_values:
            placement_strategy = 'tensor_override'
    record = {
        'objective': objective,
        'preset': objective,
        'tier': 'measured',
        'status': status,
        'tokens_per_sec': round(float(tokens_per_sec), 2),
        'decode_tokens_per_sec': round(float(tokens_per_sec), 2),
        'generation_tokens_per_sec': round(float(tokens_per_sec), 2),
        'prompt_tokens_per_sec': round(float(prompt_tokens_per_sec), 2),
        'total_tokens_per_sec': round(float(tokens_per_sec), 2),
        'seconds': round(float(seconds), 2),
        'startup_seconds': round(float(startup_seconds), 2),
        'ready_seconds': round(float(ready_seconds), 2),
        'warmup_seconds': round(float(warmup_seconds), 2),
        'prompt_tokens': int(prompt_tokens or 0),
        'generated_tokens': int(generated_tokens or 0),
        'ctx': int(ctx if ctx is not None else getattr(candidate, 'ctx', 0) or 0),
        'ctx_per_slot': ctx_per_slot(candidate),
        'parallel': int(getattr(candidate, 'parallel', 0) or 0),
        'threads': int(getattr(candidate, 'threads', 0) or 0),
        'ngl': int(getattr(candidate, 'ngl', 0) or 0),
        'output': int(output if output is not None else getattr(candidate, 'output', 0) or 0),
        'measurement_output': int(measurement_output or 0),
        'cache_ram': int(getattr(candidate, 'cache_ram', 0) or 0),
        'temp': float(temp if temp is not None else getattr(candidate, 'temp', 0.7) or 0.7),
        'top_p': top_p,
        'top_k': top_k,
        'repeat_penalty': repeat_penalty,
        'presence_penalty': presence_penalty,
        'min_p': min_p,
        'seed': seed,
        'samplers': samplers,
        'memory_reserve_percent': int(getattr(candidate, 'memory_reserve_percent', 0) or 0),
        'extra_args': list(getattr(candidate, 'extra_args', []) or []),
        'ram_available': int(ram_available or 0),
        'gpu_memory_free': int(gpu_memory_free or 0),
        'gpu_memory_total': int(gpu_memory_total or 0),
        'peak_vram_used': int(peak_vram_used or 0),
        'engine': engine or getattr(candidate, 'runtime', 'llama.cpp'),
        'server_bin': server_bin,
        'runtime_profile': runtime_profile,
        'benchmark_profile': benchmark_profile,
        'benchmark_purpose': benchmark_purpose,
        'kv_preset': kv_preset,
        'flash_attn_mode': flash_attn_mode,
        'flash_attn': flash_attn or flash_attn_mode,
        'kv_family': kv_family,
        'kv_quality_tier': kv_quality_tier,
        'kv_compression_tier': kv_compression_tier,
        'kv_score_penalty': round(float(kv_score_penalty or 0.0), 4),
        'benchmark_depth': benchmark_depth,
        'runtime_fit': bool(runtime_fit),
        'fit': bool(fit or runtime_fit),
        'fit_context': int(fit_context or 0),
        'fit_target': fit_target,
        'runtime_no_warmup': bool(runtime_no_warmup),
        'batch_size': int(batch_size or 0),
        'ubatch_size': int(ubatch_size or 0),
        'ctk': ctk,
        'ctv': ctv,
        'kv_key': kv_key or ctk,
        'kv_value': kv_value or ctv,
        'detected_head_dim': int(detected_head_dim or 0),
        'model_quant': model_quant,
        'model_family': model_family,
        'tq3_status': tq3_status,
        'tq3_weight_format': tq3_weight_format,
        'binary_path': binary_path,
        'help_supported_cache_types': list(help_supported_cache_types or []),
        'runtime_log_path': runtime_log_path,
        'failure_excerpt': failure_excerpt,
        'fit_discovery_phase': fit_discovery_phase,
        'viable_ngl': int(viable_ngl or 0),
        'viable_ngl_source': viable_ngl_source,
        'fit_selected_ngl': int(fit_selected_ngl or 0),
        'fit_selected_ngl_source': fit_selected_ngl_source,
        'fit_log_excerpt': fit_log_excerpt,
        'gpu_layers_mode': gpu_layers_mode,
        'placement_strategy': placement_strategy,
        'cpu_moe': bool(cpu_moe),
        'n_cpu_moe': int(n_cpu_moe or 0),
        'tensor_overrides': tensor_override_values,
        'moe_placement_mode': moe_placement_mode,
        'fit_assisted_moe_placement': bool(fit_assisted_moe_placement),
        'mtp_enabled': bool(mtp_enabled or getattr(candidate, 'mtp_enabled', False)),
        'mtp_draft_n_max': clamp_mtp_draft(mtp_draft_n_max or getattr(candidate, 'mtp_draft_n_max', 3), default=3),
        'spec_type': spec_type or ('mtp' if bool(mtp_enabled or getattr(candidate, 'mtp_enabled', False)) else ''),
        'startup_result': startup_result,
        'failure_category': failure_category,
        'failure_reason': concise_failure(failure_reason, limit=500),
        'suggested_fix': suggested_fix,
        'command': command,
        'no_context_shift': bool(no_context_shift),
        'cache_prompt': cache_prompt,
        'cache_reuse': int(cache_reuse or 0),
        'reasoning': reasoning,
        'reasoning_budget': reasoning_budget,
        'reasoning_format': reasoning_format,
        'reasoning_mode': reasoning_mode or (
            str(reasoning or '')
            + (f'/{reasoning_format}' if reasoning_format else '')
            + (f' budget={reasoning_budget}' if reasoning_budget is not None else '')
        ).strip(),
        'preserve_thinking': bool(preserve_thinking),
        'preserve_thinking_source': preserve_thinking_source,
        'chat_template_kwargs': dict(chat_template_kwargs or {}),
        'unsupported_launch_flags': list(unsupported_launch_flags or []),
        'effective_server_args': [str(item) for item in list(effective_server_args or [])],
        'effective_server_command': effective_server_command or command,
        'detail': concise_failure(detail, limit=500),
        'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
    }
    record.update(architecture_payload(candidate))
    if process_snapshots:
        record['process_snapshots'] = {key: dict(value) for key, value in process_snapshots.items()}
        if process_snapshots.get('after_generation'):
            record.update(process_snapshots['after_generation'])
        elif process_snapshots.get('after_ready'):
            record.update(process_snapshots['after_ready'])
        elif process_snapshots.get('before'):
            record.update(process_snapshots['before'])
    else:
        record.update(current_process_pressure_payload())
    return record


def runtime_record_context(
    app: AppConfig,
    candidate: ModelConfig,
    runtime_profile: Optional[RuntimeProfile] = None,
    benchmark_profile: Optional[BenchmarkLaunchProfile] = None,
    command: str = '',
) -> Dict[str, object]:
    try:
        profile = runtime_profile or app.runtime_profile_from_model(
            candidate,
            int(getattr(candidate, 'ctx', 0) or 0),
            int(getattr(candidate, 'parallel', 1) or 1),
            int(getattr(candidate, 'ngl', 0) or 0),
        )
    except Exception:
        profile = runtime_profile
    effective_args = benchmark_effective_server_args(app, candidate, profile, benchmark_profile)
    if not command:
        command = shlex.join(effective_args) if effective_args else benchmark_command_preview(app, candidate, profile, benchmark_profile)
    engine = ''
    server_bin = ''
    if hasattr(app, 'active_engine_key_for_model'):
        try:
            engine = app.active_engine_key_for_model(candidate)
        except Exception:
            engine = ''
    if not engine and profile is not None:
        engine = profile.engine_id
    if hasattr(app, 'runtime_server_command'):
        try:
            server_bin = app.runtime_server_command(str(getattr(candidate, 'runtime', 'llama.cpp') or 'llama.cpp'))
        except Exception:
            server_bin = ''
    runtime_log_path = ''
    if hasattr(app, 'logfile'):
        try:
            runtime_log_path = str(app.logfile(candidate.id))
        except Exception:
            runtime_log_path = ''
    kv_preset = getattr(profile, 'kv_preset', '') if profile is not None else ''
    ctk, ctv = kv_modes_from_preset(kv_preset)
    turbo_profile = turbo_kv_profile_for_preset(kv_preset)
    kv_family = getattr(profile, 'kv_family', '') if profile is not None else ''
    if not kv_family:
        if turbo_profile is not None and any(mode.startswith('turbo') for mode in (ctk, ctv)):
            kv_family = 'turbo'
        elif kv_preset and kv_preset != 'default':
            kv_family = 'cache'
        else:
            kv_family = 'default'
    supported_cache_types: List[str] = []
    unsupported_launch_flags: List[str] = []
    if hasattr(app, 'engine_capabilities'):
        try:
            capabilities = app.engine_capabilities()
            supported_cache_types = [str(item) for item in list(getattr(capabilities, 'supported_kv_modes', ()) or ())]
            if benchmark_profile is not None and str(getattr(candidate, 'runtime', 'llama.cpp') or 'llama.cpp') != 'vllm':
                _args, unsupported_launch_flags = benchmark_profile_server_args(benchmark_profile, capabilities)
        except Exception:
            supported_cache_types = []
    context = {
        'engine': engine or getattr(candidate, 'runtime', 'llama.cpp'),
        'server_bin': server_bin,
        'binary_path': server_bin,
        'runtime_log_path': runtime_log_path,
        'runtime_profile': getattr(profile, 'name', '') if profile is not None else '',
        'kv_preset': kv_preset,
        'ctk': ctk,
        'ctv': ctv,
        'detected_head_dim': turboquant_head_dim(candidate),
        'model_quant': extract_quant(candidate),
        'model_family': getattr(candidate, 'model_family', '') or getattr(candidate, 'architecture', '') or '',
        'tq3_status': getattr(candidate, 'tq3_status', 'unknown'),
        'tq3_weight_format': getattr(candidate, 'tq3_weight_format', ''),
        'help_supported_cache_types': supported_cache_types,
        'flash_attn_mode': getattr(profile, 'flash_attn', '') if profile is not None else '',
        'kv_family': kv_family,
        'kv_quality_tier': (
            getattr(profile, 'kv_quality_tier', '') if profile is not None and getattr(profile, 'kv_quality_tier', '') else
            turbo_profile.quality_tier if turbo_profile is not None else ''
        ),
        'kv_compression_tier': (
            getattr(profile, 'kv_compression_tier', '') if profile is not None and getattr(profile, 'kv_compression_tier', '') else
            turbo_profile.compression_tier if turbo_profile is not None else ''
        ),
        'kv_score_penalty': (
            getattr(profile, 'kv_score_penalty', 0.0) if profile is not None and getattr(profile, 'kv_score_penalty', 0.0) else
            turbo_profile.score_penalty if turbo_profile is not None else 0.0
        ),
        'benchmark_depth': getattr(profile, 'benchmark_depth', '') if profile is not None else '',
        'runtime_fit': bool(getattr(profile, 'fit', False)) if profile is not None else False,
        'fit_context': int(getattr(profile, 'fit_context', 0) or 0) if profile is not None else 0,
        'runtime_no_warmup': bool(getattr(profile, 'no_warmup', False)) if profile is not None else False,
        'batch_size': int(getattr(profile, 'batch_size', 0) or 0) if profile is not None else 0,
        'ubatch_size': int(getattr(profile, 'ubatch_size', 0) or 0) if profile is not None else 0,
        'fit_discovery_phase': getattr(profile, 'fit_discovery_phase', '') if profile is not None else '',
        'viable_ngl': int(getattr(profile, 'viable_ngl', 0) or 0) if profile is not None else 0,
        'viable_ngl_source': getattr(profile, 'viable_ngl_source', '') if profile is not None else '',
        'fit_selected_ngl': int(getattr(profile, 'fit_selected_ngl', 0) or 0) if profile is not None else 0,
        'fit_selected_ngl_source': getattr(profile, 'fit_selected_ngl_source', '') if profile is not None else '',
        'fit_log_excerpt': getattr(profile, 'fit_log_excerpt', '') if profile is not None else '',
        'placement_strategy': getattr(profile, 'placement_strategy', '') if profile is not None else '',
        'cpu_moe': bool(getattr(profile, 'cpu_moe', False)) if profile is not None else False,
        'n_cpu_moe': int(getattr(profile, 'n_cpu_moe', 0) or 0) if profile is not None else 0,
        'tensor_overrides': list(getattr(profile, 'tensor_overrides', ()) or ()) if profile is not None else [],
        'moe_placement_mode': runtime_profile_moe_placement_mode(profile),
        'fit_assisted_moe_placement': runtime_profile_moe_placement_mode(profile) == 'fit_assisted',
        'mtp_enabled': bool(getattr(profile, 'mtp_enabled', False)) if profile is not None else bool(getattr(candidate, 'mtp_enabled', False)),
        'mtp_draft_n_max': (
            clamp_mtp_draft(getattr(profile, 'mtp_draft_n_max', 0), default=3)
            if profile is not None and int(getattr(profile, 'mtp_draft_n_max', 0) or 0) > 0
            else clamp_mtp_draft(getattr(candidate, 'mtp_draft_n_max', 3), default=3)
        ),
        'spec_type': 'mtp' if (profile is not None and bool(getattr(profile, 'mtp_enabled', False))) or bool(getattr(candidate, 'mtp_enabled', False)) else '',
        'gpu_layers_mode': (
            'fit' if profile is not None and getattr(profile, 'gpu_layers', None) is None and getattr(profile, 'fit', False)
            else 'fixed' if profile is not None and getattr(profile, 'gpu_layers', None) is not None
            else ''
        ),
        'command': command,
        'effective_server_args': effective_args,
        'effective_server_command': command,
    }
    if benchmark_profile is not None:
        context.update(benchmark_launch_metadata(benchmark_profile, unsupported_launch_flags))
    return context


def apply_failure_context(record: Dict[str, object], detail: str, default_category: str = 'SERVER_TIMEOUT') -> Dict[str, object]:
    classification = classify_benchmark_failure(detail, default_category=default_category)
    record.update(classification)
    record['startup_result'] = 'FAILED'
    if not record.get('detail'):
        record['detail'] = concise_failure(detail, limit=500)
    return record


def benchmark_adaptive_candidate(
    app: AppConfig,
    candidate: ModelConfig,
    objective: str,
    progress: Optional[Callable[[str], None]],
    cancel_token: Optional[CancelToken],
    runtime_profile: Optional[RuntimeProfile] = None,
    benchmark_profile: Optional[BenchmarkLaunchProfile] = None,
    benchmark_purpose: str = 'serve_default',
    benchmark_depth: str = 'full',
    deadline: Optional[BenchmarkDeadline] = None,
) -> Tuple[Dict[str, object], Optional[Dict[str, object]]]:
    check_cancelled(cancel_token)
    process_snapshots: Dict[str, Dict[str, object]] = {'before': current_process_pressure_payload()}
    before_hw = app.hardware_profile(refresh=True)
    try:
        detected_capabilities = app.engine_capabilities()
    except Exception:
        detected_capabilities = None
    launch_profile = benchmark_profile or build_benchmark_launch_profile(
        candidate,
        runtime_profile,
        detected_capabilities,
        purpose=benchmark_purpose,
        depth=benchmark_depth,
    )
    runtime_context = runtime_record_context(app, candidate, runtime_profile, benchmark_profile=launch_profile)
    if deadline is not None and deadline.expired():
        return budget_exhausted_record(
            candidate,
            objective,
            'benchmark budget expired before candidate launch',
            runtime_context,
        ), None
    estimated_safe_ctx = candidate_safe_context_estimate(candidate, before_hw)
    observed_floor = observed_opencode_context_floor(candidate)
    guardrail_state = MemoryGuardrailState()
    admission = memory_guardrail_admission(
        before_hw,
        candidate,
        estimated_safe_ctx,
        pressure_payload=process_snapshots.get('before'),
        observed_floor=observed_floor,
        state=guardrail_state,
    )
    if admission.should_skip:
        if progress:
            progress(
                f'adaptive {objective} skipped by memory guardrail: '
                f'ctx={candidate.ctx} safe_ctx={estimated_safe_ctx} {admission.reason}'
            )
        return memory_guardrail_skip_record(candidate, objective, admission, runtime_context, process_snapshots), None
    start_at = time.monotonic()
    if runtime_profile is not None:
        ok, msg = app.start(candidate, runtime_profile=runtime_profile, benchmark_profile=launch_profile)
    else:
        ok, msg = app.start(candidate, benchmark_profile=launch_profile)
    startup_seconds = time.monotonic() - start_at
    if not ok:
        process_snapshots['after_start'] = current_process_pressure_payload()
        record = adaptive_record_from_candidate(
            candidate,
            objective,
            'start failed',
            detail=msg,
            startup_seconds=startup_seconds,
            process_snapshots=process_snapshots,
            **runtime_context,
        )
        apply_failure_context(record, msg, default_category='SERVER_TIMEOUT')
        enrich_fit_discovery_metadata(record, app, candidate, runtime_profile, success=False)
        return record, None
    watchdog_stop = None
    watchdog_thread = None
    watchdog_stop, watchdog_thread = start_memory_guardrail_watchdog(
        lambda: _guardrail_profile(app),
        lambda: app.stop(candidate, managed_only=True),
        guardrail_state,
        candidate_ctx=int(getattr(candidate, 'ctx', 0) or 0),
        safe_ctx=estimated_safe_ctx,
        observed_floor=observed_floor,
        required_for_floor=_candidate_required_for_opencode_floor(candidate, observed_floor),
        pressure_score=_pressure_score_from_payload(process_snapshots.get('before')),
        phase='runtime',
    )
    try:
        ready_start = time.monotonic()
        ready_timeout = deadline.cap_timeout(BENCHMARK_READY_TIMEOUT) if deadline is not None else BENCHMARK_READY_TIMEOUT
        if ready_timeout <= 0.0:
            return budget_exhausted_record(
                candidate,
                objective,
                'benchmark budget expired before server readiness wait',
                runtime_context,
            ), None
        ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=ready_timeout, cancel_token=cancel_token)
        ready_seconds = time.monotonic() - ready_start
        process_snapshots['after_ready'] = current_process_pressure_payload()
        if not ready_ok:
            if guardrail_state.stop_decision is not None:
                ready_msg = guardrail_state.stop_decision.reason
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'memory guardrail stopped' if guardrail_state.stop_decision is not None else 'not ready',
                detail=ready_msg,
                startup_seconds=startup_seconds,
                ready_seconds=ready_seconds,
                process_snapshots=process_snapshots,
                **runtime_context,
            )
            if guardrail_state.stop_decision is not None:
                apply_memory_guardrail_record(record, guardrail_state.stop_decision, guardrail_state)
            else:
                apply_failure_context(record, ready_msg, default_category='SERVER_TIMEOUT')
            enrich_fit_discovery_metadata(record, app, candidate, runtime_profile, success=False)
            return record, None
        if guardrail_state.stop_decision is not None:
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'memory guardrail stopped',
                detail=guardrail_state.stop_decision.reason,
                startup_seconds=startup_seconds,
                ready_seconds=ready_seconds,
                process_snapshots=process_snapshots,
                **runtime_context,
            )
            apply_memory_guardrail_record(record, guardrail_state.stop_decision, guardrail_state)
            enrich_fit_discovery_metadata(record, app, candidate, runtime_profile, success=False)
            return record, None
        if int(getattr(candidate, 'last_good_ctx', 0) or 0) > 0:
            candidate.ctx = int(candidate.last_good_ctx)
        if int(getattr(candidate, 'last_good_parallel', 0) or 0) > 0:
            candidate.parallel = int(candidate.last_good_parallel)
        runtime_context['startup_result'] = 'READY'
        if progress:
            progress(
                f'adaptive {objective} ready: ctx={candidate.ctx} slot={ctx_per_slot(candidate)} '
                f'parallel={candidate.parallel}; measuring...'
            )
        if deadline is not None and deadline.expired():
            return budget_exhausted_record(
                candidate,
                objective,
                'benchmark budget expired before warmup request',
                runtime_context,
            ), None
        warmup_start = time.monotonic()
        benchmark_completion(
            candidate,
            max_tokens=min(BENCHMARK_WARMUP_TOKENS, max(1, launch_profile.measurement_output)),
            timeout=BENCHMARK_WARMUP_TIMEOUT,
            cancel_token=cancel_token,
            launch_profile=launch_profile,
            deadline=deadline,
        )
        warmup_seconds = time.monotonic() - warmup_start
        if guardrail_state.stop_decision is not None:
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'memory guardrail stopped',
                detail=guardrail_state.stop_decision.reason,
                startup_seconds=startup_seconds,
                ready_seconds=ready_seconds,
                warmup_seconds=warmup_seconds,
                process_snapshots=process_snapshots,
                **runtime_context,
            )
            return apply_memory_guardrail_record(record, guardrail_state.stop_decision, guardrail_state), None
        if deadline is not None and deadline.expired():
            return budget_exhausted_record(
                candidate,
                objective,
                'benchmark budget expired before sample request',
                runtime_context,
            ), None
        bench_timeout = deadline.cap_timeout(BENCHMARK_SAMPLE_TIMEOUT) if deadline is not None else BENCHMARK_SAMPLE_TIMEOUT
        bench_ok, bench = benchmark_completion_suite(
            candidate,
            max_tokens=max(1, launch_profile.measurement_output),
            timeout=bench_timeout,
            cancel_token=cancel_token,
            launch_profile=launch_profile,
            deadline=deadline,
        )
        if not bench_ok:
            process_snapshots['after_generation'] = current_process_pressure_payload()
            detail = guardrail_state.stop_decision.reason if guardrail_state.stop_decision is not None else str(bench.get('error', 'unknown error'))
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'memory guardrail stopped' if guardrail_state.stop_decision is not None else 'benchmark failed',
                detail=detail,
                startup_seconds=startup_seconds,
                ready_seconds=ready_seconds,
                warmup_seconds=warmup_seconds,
                process_snapshots=process_snapshots,
                **runtime_context,
            )
            if guardrail_state.stop_decision is not None:
                apply_memory_guardrail_record(record, guardrail_state.stop_decision, guardrail_state)
            else:
                apply_failure_context(record, str(bench.get('error', 'unknown error')), default_category='API_TIMEOUT')
            enrich_fit_discovery_metadata(record, app, candidate, runtime_profile, success=False)
            return record, None
        snap = app.hardware_profile(refresh=True)
        process_snapshots['after_generation'] = current_process_pressure_payload()
        score = float(bench.get('tokens_per_sec', 0.0) or 0.0)
        elapsed = float(bench.get('elapsed', 0.0) or 0.0)
        prompt_tps = (int(bench.get('prompt_tokens', 0) or 0) / elapsed) if elapsed > 0 else 0.0
        min_free = min(
            value for value in (
                int(getattr(before_hw, 'gpu_memory_free', 0) or 0),
                int(getattr(snap, 'gpu_memory_free', 0) or 0),
            )
            if value >= 0
        )
        total_vram = int(getattr(snap, 'gpu_memory_total', 0) or getattr(before_hw, 'gpu_memory_total', 0) or 0)
        peak_vram = max(0, total_vram - min_free) if total_vram else 0
        record = adaptive_record_from_candidate(
            candidate,
            objective,
            'ok',
            tokens_per_sec=score,
            seconds=elapsed,
            detail=f'{int(bench.get("sample_count", 1) or 1)} samples',
            ram_available=int(getattr(snap, 'memory_available', 0) or 0),
            gpu_memory_free=int(getattr(snap, 'gpu_memory_free', 0) or 0),
            gpu_memory_total=total_vram,
            peak_vram_used=peak_vram,
            startup_seconds=startup_seconds,
            ready_seconds=ready_seconds,
            warmup_seconds=warmup_seconds,
            prompt_tokens=int(bench.get('prompt_tokens', 0) or 0),
            prompt_tokens_per_sec=prompt_tps,
            generated_tokens=int(bench.get('completion_tokens', 0) or 0),
            process_snapshots=process_snapshots,
            **runtime_context,
        )
        apply_memory_guardrail_record(record, state=guardrail_state)
        enrich_fit_discovery_metadata(record, app, candidate, runtime_profile, success=True)
        measured = dict(record)
        measured['model'] = ModelConfig(**asdict(candidate))
        return record, measured
    finally:
        if watchdog_stop is not None:
            watchdog_stop.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=1.0)
        app.stop(candidate, managed_only=True)
        sleep_with_cancel(0.5, cancel_token)


def select_adaptive_candidate_mix(
    candidates: List[Tuple[str, ModelConfig, str]],
    limit: int = ADAPTIVE_MAX_MEASUREMENTS,
) -> List[Tuple[str, ModelConfig, str]]:
    limit = max(1, int(limit or 1))
    selected: List[Tuple[str, ModelConfig, str]] = []
    seen = set()

    def key(item: Tuple[str, ModelConfig, str]):
        objective, candidate, _label = item
        return (
            objective,
            int(getattr(candidate, 'ctx', 0) or 0),
            int(getattr(candidate, 'parallel', 1) or 1),
            tuple(getattr(candidate, 'extra_args', []) or []),
        )

    def add(item: Tuple[str, ModelConfig, str]):
        if len(selected) >= limit:
            return
        item_key = key(item)
        if item_key in seen:
            return
        seen.add(item_key)
        selected.append(item)

    buckets: Dict[str, List[Tuple[str, ModelConfig, str]]] = {}
    for item in candidates:
        buckets.setdefault(item[0], []).append(item)

    for objective in ('long_context', 'fast_chat', 'opencode_ready'):
        bucket = buckets.get(objective, [])
        if not bucket:
            continue
        ordered = sorted(
            bucket,
            key=lambda item: (
                ctx_per_slot(item[1]),
                int(getattr(item[1], 'parallel', 1) or 1),
            ),
        )
        add(ordered[0])
        add(ordered[-1])
        if objective == 'fast_chat':
            add(max(ordered, key=lambda item: int(getattr(item[1], 'parallel', 1) or 1)))

    remaining = sorted(
        candidates,
        key=lambda item: (
            {'fast_chat': 0, 'long_context': 1, 'opencode_ready': 2}.get(item[0], 9),
            -ctx_per_slot(item[1]),
            -int(getattr(item[1], 'parallel', 1) or 1),
        ),
    )
    for item in remaining:
        add(item)
        if len(selected) >= limit:
            break
    return selected


def adaptive_benchmark_candidates(
    app: AppConfig,
    model: ModelConfig,
    profile: HardwareProfile,
    progress: Optional[Callable[[str], None]],
    cancel_token: Optional[CancelToken],
    deadline: float,
) -> List[Tuple[str, ModelConfig, str]]:
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    variants = ['default']
    if getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp' and profile.has_usable_gpu():
        variants.append('q8_kv')
    contexts_by_variant: Dict[str, List[int]] = {}
    probe_completed = 0
    probe_total = max(1, len(variants) * ADAPTIVE_MAX_CONTEXT_PROBES)
    try:
        capabilities = app.engine_capabilities()
    except Exception:
        capabilities = None
    for variant in variants:
        if time.monotonic() >= deadline:
            break
        upper = adaptive_context_upper_bound(model, profile, 'long_context', parallel=1, variant=variant)
        if progress:
            progress(f'adaptive context search {variant}: estimated upper ctx={upper}')
        emit_benchmark_event(
            progress,
            'benchmark_phase',
            model,
            'server',
            message=f'context search {variant}: estimated upper ctx={upper}',
            phase='context search',
            completed=probe_completed,
            total=probe_total,
            candidate=f'{variant} ctx<= {upper}',
        )

        def probe(value: int, variant=variant) -> bool:
            nonlocal probe_completed
            if time.monotonic() >= deadline:
                return False
            candidate = configure_adaptive_candidate(model, profile, 'long_context', value, 1, variant)
            launch_profile = build_benchmark_launch_profile(
                candidate,
                None,
                capabilities,
                purpose='serve_default',
                depth='fast',
            )
            runtime_context = runtime_record_context(app, candidate, benchmark_profile=launch_profile)
            ok, msg = app.start(candidate, benchmark_profile=launch_profile)
            if not ok:
                if progress:
                    progress(f'context probe {variant} ctx={value} start failed: {concise_failure(msg)}')
                probe_completed += 1
                emit_benchmark_event(
                    progress,
                    'benchmark_probe',
                    model,
                    'server',
                    message=f'context probe {variant} ctx={value}: start failed',
                    phase='context search',
                    completed=probe_completed,
                    total=probe_total,
                    candidate=f'{variant} ctx={value}',
                    record=adaptive_record_from_candidate(
                        candidate,
                        'long_context',
                        'start failed',
                        detail=msg,
                        **runtime_context,
                    ),
                )
                return False
            try:
                ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=BENCHMARK_READY_TIMEOUT, cancel_token=cancel_token)
                if progress:
                    state = 'ready' if ready_ok else 'not ready'
                    progress(f'context probe {variant} ctx={value}: {state} {concise_failure(ready_msg)}')
                probe_completed += 1
                emit_benchmark_event(
                    progress,
                    'benchmark_probe',
                    model,
                    'server',
                    message=f'context probe {variant} ctx={value}: {"ready" if ready_ok else "not ready"}',
                    phase='context search',
                    completed=probe_completed,
                    total=probe_total,
                    candidate=f'{variant} ctx={value}',
                    record=adaptive_record_from_candidate(
                        candidate,
                        'long_context',
                        'probe ok' if ready_ok else 'probe failed',
                        detail=ready_msg,
                        startup_result='READY' if ready_ok else 'FAILED',
                        **runtime_context,
                    ),
                )
                return ready_ok
            finally:
                app.stop(candidate, managed_only=True)
                sleep_with_cancel(0.25, cancel_token)

        successes, _failures = adaptive_context_search(ctx_min, upper, probe, max_probes=ADAPTIVE_MAX_CONTEXT_PROBES)
        contexts_by_variant[variant] = successes or [ctx_min]

    candidates: List[Tuple[str, ModelConfig, str]] = []
    seen = set()

    def add(objective: str, ctx: int, parallel: int, variant: str):
        key = (objective, round_context(ctx), parallel, variant)
        if key in seen:
            return
        seen.add(key)
        candidate = configure_adaptive_candidate(model, profile, objective, ctx, parallel, variant)
        label = f'{objective}/{variant}'
        candidates.append((objective, candidate, label))

    for variant, contexts in contexts_by_variant.items():
        ordered = sorted(set(contexts))
        spectrum_contexts = sorted(set(ordered[:1] + ordered[-4:]))
        for ctx in spectrum_contexts:
            add('long_context', ctx, 1, variant)
            add('opencode_ready', ctx, 1, variant)
        for ctx in ordered:
            for parallel in adaptive_parallel_values(model, profile, 'fast_chat', ctx, variant):
                add('fast_chat', ctx, parallel, variant)

    candidates.sort(
        key=lambda item: (
            {'fast_chat': 0, 'long_context': 1, 'opencode_ready': 2}.get(item[0], 9),
            -ctx_per_slot(item[1]),
            int(getattr(item[1], 'parallel', 1) or 1),
        )
    )
    return select_adaptive_candidate_mix(candidates, ADAPTIVE_MAX_MEASUREMENTS)


def exhaustive_variants(model: ModelConfig, profile: HardwareProfile) -> List[str]:
    variants = ['default']
    if getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp' and profile.has_usable_gpu():
        variants.append('q8_kv')
    return variants


def exhaustive_parallel_values(profile: HardwareProfile) -> List[int]:
    max_parallel = max(1, min(16, int(getattr(profile, 'cpu_logical', 0) or 1)))
    values = []
    parallel = 1
    while parallel <= max_parallel:
        values.append(parallel)
        parallel *= 2
    return values or [1]


def parallel_refinement_values(profile: HardwareProfile, best_parallel: int, tested: set) -> List[int]:
    max_parallel = max(1, min(16, int(getattr(profile, 'cpu_logical', 0) or 1)))
    best_parallel = max(1, int(best_parallel or 1))
    values = []
    for parallel in (best_parallel - 1, best_parallel + 1):
        if 1 <= parallel <= max_parallel and parallel not in tested:
            values.append(parallel)
    return sorted(values)


def fast_benchmark_parallel_values(profile: HardwareProfile, model: Optional[ModelConfig] = None) -> List[int]:
    max_parallel = max(1, min(4 if model is not None and model_is_moe(model) else 16, int(getattr(profile, 'cpu_logical', 0) or 1)))
    return [value for value in FAST_BENCHMARK_PARALLEL_TARGETS if value <= max_parallel] or [1]


def fast_benchmark_contexts(model: ModelConfig, profile: HardwareProfile) -> List[int]:
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072) or 131072))
    seed = configure_adaptive_candidate(model, profile, 'long_context', ctx_min, 1, 'default')
    estimate = candidate_safe_context_estimate(seed, profile)
    points = [ctx_min, *FAST_BENCHMARK_CONTEXT_TARGETS]
    if estimate > 0:
        points.append(round_context_down(estimate, CONTEXT_REFINE_STEP))
    clamped = []
    for point in points:
        value = max(ctx_min, min(ctx_max, int(point or ctx_min)))
        clamped.append(value)
    return sorted(set(clamped))


def candidate_safe_context_estimate(candidate: ModelConfig, profile: HardwareProfile) -> int:
    ctx_min = max(256, int(getattr(candidate, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(candidate, 'ctx_max', 131072) or 131072))
    return estimate_safe_context_for_profile(
        candidate,
        profile,
        int(getattr(candidate, 'memory_reserve_percent', 30) or 30),
        max(1, int(getattr(candidate, 'parallel', 1) or 1)),
        ctx_min,
        ctx_max,
    )


def context_pressure_score(pressure_payload: Optional[Dict[str, object]] = None) -> float:
    payload = pressure_payload if pressure_payload is not None else current_process_pressure_payload()
    try:
        return max(0.0, min(1.0, float(payload.get('process_pressure_score', 0.0) or 0.0)))
    except Exception:
        return 0.0


def health_context_ceiling(
    model: ModelConfig,
    profile: HardwareProfile,
    ctx_min: int,
    absolute_ctx_max: int,
    pressure_payload: Optional[Dict[str, object]] = None,
    objective: str = 'long_context',
    variant: str = 'default',
) -> int:
    ctx_min = max(256, int(ctx_min or 2048))
    absolute_ctx_max = max(ctx_min, int(absolute_ctx_max or ctx_min))
    pressure = context_pressure_score(pressure_payload)
    if pressure < CONTEXT_HEALTH_PRESSURE_CAP:
        return absolute_ctx_max
    seed = configure_adaptive_candidate(model, profile, objective, ctx_min, 1, variant)
    seed.ctx_max = absolute_ctx_max
    safe_ctx = candidate_safe_context_estimate(seed, profile)
    if safe_ctx <= 0:
        return ctx_min
    return max(ctx_min, min(absolute_ctx_max, round_context_down(safe_ctx, CONTEXT_REFINE_STEP)))


def _context_milestones_to(ceiling: int) -> List[int]:
    ceiling = max(1, int(ceiling or 1))
    values = [ctx for ctx in RUNTIME_CONTEXT_MILESTONES if ctx <= ceiling]
    if ceiling not in values:
        values.append(ceiling)
    current = RUNTIME_CONTEXT_MILESTONES[-1]
    while current < ceiling:
        current = min(ceiling, current * 2)
        if current not in values:
            values.append(current)
        if current >= ceiling:
            break
    return sorted(set(values))


def dynamic_context_growth_targets(
    model: ModelConfig,
    profile: HardwareProfile,
    ctx_min: int,
    absolute_ctx_max: int,
    depth: str = 'full',
    pressure_payload: Optional[Dict[str, object]] = None,
    observed_floor: int = 0,
    objective: str = 'long_context',
    variant: str = 'default',
) -> List[int]:
    benchmark_depth = 'fast' if str(depth or '').strip().lower() == 'fast' else 'full'
    ctx_min = max(256, int(ctx_min or 2048))
    absolute_ctx_max = max(ctx_min, int(absolute_ctx_max or ctx_min))
    health_ceiling = health_context_ceiling(
        model,
        profile,
        ctx_min,
        absolute_ctx_max,
        pressure_payload=pressure_payload,
        objective=objective,
        variant=variant,
    )
    health_ceiling = max(ctx_min, min(absolute_ctx_max, health_ceiling))
    floor_target = round_context_up(observed_floor, CONTEXT_REFINE_STEP) if int(observed_floor or 0) > 0 else 0
    milestones = _context_milestones_to(health_ceiling)
    values = {ctx_min}

    if benchmark_depth == 'fast':
        values.update(ctx for ctx in (8_192, 16_384) if ctx_min <= ctx <= health_ceiling)
        if floor_target > 32_768 and floor_target <= health_ceiling:
            values.add(floor_target)
        elif health_ceiling > 32_768:
            above = [ctx for ctx in milestones if ctx > 32_768]
            if above:
                values.add(min(above))
        ordered = sorted(ctx for ctx in values if ctx_min <= ctx <= health_ceiling)
        if len(ordered) <= FAST_RUNTIME_CONTEXT_TARGET_LIMIT:
            return ordered
        required = {ctx_min}
        if floor_target > 32_768 and floor_target <= health_ceiling:
            required.add(floor_target)
        required.add(max(ordered))
        selected = sorted(required)
        for ctx in ordered:
            if len(selected) >= FAST_RUNTIME_CONTEXT_TARGET_LIMIT:
                break
            if ctx not in selected:
                selected.append(ctx)
        return sorted(set(selected))

    values.update(ctx for ctx in milestones if ctx_min <= ctx <= health_ceiling)
    if floor_target > 0 and floor_target <= health_ceiling:
        values.add(floor_target)
    values.add(health_ceiling)
    return sorted(ctx for ctx in values if ctx_min <= ctx <= health_ceiling)


def tq3_model_card_q4_tq3_kv_enabled(model: ModelConfig) -> bool:
    text = ' '.join(
        str(getattr(model, key, '') or '')
        for key in ('id', 'name', 'path', 'alias', 'model_family', 'architecture')
    ).lower()
    return (
        'ytan2000' in text
        and 'qwen3.6' in text
        and '35b' in text
        and 'tq3' in text
    )


def tq3_kv_profiles_for_model(
    model: ModelConfig,
    capabilities,
    depth: str,
) -> List[TurboKvProfile]:
    profiles = supported_turbo_kv_profiles(capabilities, depth, engine_id='tq3')
    allowed = {str(item).strip().lower() for item in tuple(getattr(capabilities, 'supported_kv_modes', ()) or ())}
    if (
        tq3_model_card_q4_tq3_kv_enabled(model)
        and 'q4_0' in allowed
        and 'tq3_0' in allowed
        and (depth or '').strip().lower() != 'fast'
        and not any(item.kv_preset == 'q4_0/tq3_0' for item in profiles)
    ):
        profiles.append(TurboKvProfile(
            'q4_0/tq3_0',
            'Qwen3.6 q4/TQ3',
            'model_specific',
            'qwen3.6-tq3',
            'full',
            0.12,
            False,
            'model-card',
        ))
    return profiles


def sibling_llama_bench_for_server(server_bin: str) -> str:
    parts = shlex.split(server_bin or '')
    if not parts:
        return 'llama-bench'
    first = os.path.expanduser(parts[0])
    if '/' in first or first.startswith('.'):
        return str(Path(first).expanduser().with_name('llama-bench'))
    return 'llama-bench'


def _command_exists_for_app(app, command: str) -> bool:
    if hasattr(app, 'command_exists'):
        try:
            return bool(app.command_exists(command))
        except Exception:
            pass
    parts = shlex.split(command or '')
    if not parts:
        return False
    first = os.path.expanduser(parts[0])
    if '/' in first or first.startswith('.'):
        return Path(first).expanduser().exists()
    return subprocess.run(['sh', '-c', f'command -v {shlex.quote(first)} >/dev/null 2>&1']).returncode == 0


def _tq3_raw_profile_key(runtime_profile: RuntimeProfile) -> Tuple[object, ...]:
    return (
        runtime_profile.gpu_layers,
        runtime_profile.kv_preset,
        bool(runtime_profile.cpu_moe),
        int(runtime_profile.n_cpu_moe or 0),
        tuple(runtime_profile.tensor_overrides or ()),
    )


TQ3_RAW_BENCH_CASES: Tuple[Tuple[str, int, int], ...] = (
    ('raw_pp', 1024, 0),
    ('raw_tg', 0, 64),
    ('raw_combined', 1024, 64),
)


def _tq3_raw_profile_rank(runtime_profile: RuntimeProfile, original_index: int) -> Tuple[int, int, int]:
    n_cpu_moe = int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0)
    if n_cpu_moe == 32:
        priority = 0
    elif n_cpu_moe == 30:
        priority = 1
    elif n_cpu_moe > 0:
        priority = 20 + abs(n_cpu_moe - 32)
    elif bool(getattr(runtime_profile, 'cpu_moe', False)):
        priority = 100
    else:
        priority = 999
    return priority, original_index, n_cpu_moe


def _tq3_raw_runtime_profiles(runtime_profiles: List[RuntimeProfile], depth: str) -> List[RuntimeProfile]:
    candidates: List[Tuple[Tuple[int, int, int], RuntimeProfile]] = []
    seen = set()
    fast = (depth or '').strip().lower() == 'fast'
    limit = 1 if fast else 2
    for index, profile in enumerate(runtime_profiles):
        if profile.engine_id != 'tq3' or profile.fit:
            continue
        if not (profile.cpu_moe or int(profile.n_cpu_moe or 0) > 0):
            continue
        if str(profile.kv_preset or '') != 'q8_0/q8_0':
            continue
        if tuple(profile.tensor_overrides or ()):
            continue
        if (
            str(getattr(profile, 'reasoning', '') or '').strip()
            or int(getattr(profile, 'reasoning_budget', -1) or -1) >= 0
            or str(getattr(profile, 'reasoning_format', '') or '').strip()
        ):
            continue
        key = _tq3_raw_profile_key(profile)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((_tq3_raw_profile_rank(profile, index), profile))
    candidates.sort(key=lambda item: item[0])
    return [profile for _rank, profile in candidates[:limit]]


def tq3_raw_presearch_case_total(runtime_profiles: List[RuntimeProfile], depth: str) -> int:
    return len(_tq3_raw_runtime_profiles(runtime_profiles, depth)) * len(TQ3_RAW_BENCH_CASES)


def tq3_llama_bench_command(
    llama_bench_bin: str,
    model: ModelConfig,
    runtime_profile: RuntimeProfile,
    prompt_tokens: int,
    generated_tokens: int,
    threads: int,
) -> List[str]:
    cmd = [
        llama_bench_bin,
        '-m', str(getattr(model, 'path', '') or ''),
        '-p', str(int(prompt_tokens)),
        '-n', str(int(generated_tokens)),
    ]
    if runtime_profile.gpu_layers is not None:
        cmd += ['-ngl', str(int(runtime_profile.gpu_layers))]
    if runtime_profile.cpu_moe:
        cmd.append('-cmoe')
    elif int(runtime_profile.n_cpu_moe or 0) > 0:
        cmd += ['-ncmoe', str(int(runtime_profile.n_cpu_moe or 0))]
    key_mode, value_mode = kv_modes_from_preset(runtime_profile.kv_preset)
    if key_mode and value_mode:
        cmd += ['-ctk', key_mode, '-ctv', value_mode]
    if int(runtime_profile.batch_size or 0) > 0:
        cmd += ['-b', str(int(runtime_profile.batch_size))]
    if int(runtime_profile.ubatch_size or 0) > 0:
        cmd += ['-ub', str(int(runtime_profile.ubatch_size))]
    if str(runtime_profile.flash_attn or '').strip().lower() in ('on', 'auto', ''):
        cmd += ['-fa', '1']
    if int(threads or 0) > 0:
        cmd += ['-t', str(int(threads))]
    return cmd


def parse_llama_bench_tokens_per_sec(output: str) -> float:
    values = []
    for match in re.finditer(r'([0-9]+(?:\.[0-9]+)?)\s*(?:±[^|\n]+)?\s*tok/s', output or '', re.IGNORECASE):
        try:
            values.append(float(match.group(1)))
        except Exception:
            pass
    return values[-1] if values else 0.0


def _raw_bench_candidate_model(model: ModelConfig, runtime_profile: RuntimeProfile) -> ModelConfig:
    candidate = ModelConfig(**asdict(model))
    candidate.ctx = max(1, int(runtime_profile.ctx_size or candidate.ctx or 1))
    candidate.parallel = max(1, int(runtime_profile.parallel or 1))
    if runtime_profile.gpu_layers is not None:
        candidate.ngl = int(runtime_profile.gpu_layers)
    candidate.moe_placement_strategy = str(runtime_profile.placement_strategy or '')
    candidate.cpu_moe = bool(runtime_profile.cpu_moe)
    candidate.n_cpu_moe = int(runtime_profile.n_cpu_moe or 0)
    candidate.tensor_overrides = [str(item) for item in tuple(runtime_profile.tensor_overrides or ())]
    return candidate


def tq3_moe_cpu_placement_threads(
    model: ModelConfig,
    runtime_profile: Optional[RuntimeProfile] = None,
    hardware: Optional[HardwareProfile] = None,
) -> int:
    current = max(1, int(getattr(model, 'threads', 0) or 1))
    if runtime_profile is not None and str(getattr(runtime_profile, 'engine_id', '') or '') != 'tq3':
        return current
    if not model_is_moe(model):
        return current
    cpu_placement = bool(getattr(model, 'cpu_moe', False)) or int(getattr(model, 'n_cpu_moe', 0) or 0) > 0
    if runtime_profile is not None:
        cpu_placement = (
            cpu_placement
            or bool(getattr(runtime_profile, 'cpu_moe', False))
            or int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0) > 0
        )
    if not cpu_placement:
        return current
    logical = int(getattr(hardware, 'cpu_logical', 0) or 0) if hardware is not None else 0
    logical = logical or (os.cpu_count() or current)
    return max(current, min(max(1, logical), 12))


def _terminate_process_group(process: subprocess.Popen):
    if process.poll() is not None:
        return
    try:
        if hasattr(os, 'killpg'):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.terminate()
        except Exception:
            return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, 'killpg'):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _run_tq3_raw_process(
    cmd: List[str],
    timeout_seconds: float,
    cancel_token: Optional[CancelToken] = None,
) -> Dict[str, object]:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
        )
    except FileNotFoundError as exc:
        return {
            'returncode': 127,
            'stdout': str(exc),
            'seconds': time.monotonic() - started,
            'timed_out': False,
            'cancelled': False,
        }
    except Exception as exc:
        return {
            'returncode': -1,
            'stdout': str(exc),
            'seconds': time.monotonic() - started,
            'timed_out': False,
            'cancelled': False,
        }

    timed_out = False
    cancelled = False
    timeout_seconds = max(1.0, float(timeout_seconds or 1.0))
    while process.poll() is None:
        if cancel_token is not None and cancel_token.is_cancelled():
            cancelled = True
            _terminate_process_group(process)
            break
        if time.monotonic() - started >= timeout_seconds:
            timed_out = True
            _terminate_process_group(process)
            break
        if cancel_token is not None and cancel_token.wait(0.5):
            continue
        time.sleep(0.5)

    try:
        stdout, _stderr = process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        try:
            stdout, _stderr = process.communicate(timeout=1)
        except Exception:
            stdout = ''
    return {
        'returncode': int(process.returncode if process.returncode is not None else -1),
        'stdout': stdout or '',
        'seconds': time.monotonic() - started,
        'timed_out': timed_out,
        'cancelled': cancelled,
    }


def run_tq3_raw_llama_bench_presearch(
    app,
    model: ModelConfig,
    hardware: HardwareProfile,
    runtime_profiles: List[RuntimeProfile],
    depth: str,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
    on_record: Optional[Callable[[Dict[str, object]], None]] = None,
    completed_offset: int = 0,
    total: Optional[int] = None,
    run_kind: str = 'server',
    deadline: Optional[BenchmarkDeadline] = None,
) -> Tuple[List[Dict[str, object]], List[RuntimeProfile]]:
    try:
        engine = app.active_engine_key_for_model(model)
    except Exception:
        engine = ''
    if engine != ENGINE_TQ3 or not model_is_moe(model):
        return [], []

    server_install = resolve_engine_install(app, ENGINE_TQ3)
    server_bin = str(server_install.resolved_command or '')
    llama_bench_bin = sibling_llama_bench_for_server(server_bin)
    raw_profiles = _tq3_raw_runtime_profiles(runtime_profiles, depth)
    if not raw_profiles:
        return [], []
    if not _command_exists_for_app(app, llama_bench_bin):
        candidate = _raw_bench_candidate_model(model, raw_profiles[0])
        detail = (
            f'ENGINE_BINARY_MISSING: llama.cpp-tq3 llama-bench not found: {llama_bench_bin}. '
            f'Build llama-bench next to {server_bin or "llama-server"}.'
        )
        record = adaptive_record_from_candidate(
            candidate,
            'raw_engine_search',
            'start failed',
            detail=detail,
            engine=ENGINE_TQ3,
            server_bin=server_bin,
            binary_path=llama_bench_bin,
            runtime_profile=raw_profiles[0].name,
            kv_preset=raw_profiles[0].kv_preset,
            benchmark_depth=depth,
            placement_strategy=raw_profiles[0].placement_strategy,
            cpu_moe=raw_profiles[0].cpu_moe,
            n_cpu_moe=raw_profiles[0].n_cpu_moe,
            tensor_overrides=list(raw_profiles[0].tensor_overrides or ()),
            failure_category='ENGINE_BINARY_MISSING',
            failure_reason=detail,
            suggested_fix='Build llama-bench or set TQ3_LLAMA_SERVER_BIN to the matching llama-server.',
        )
        record['benchmark_kind'] = 'raw_engine_search'
        record['measurement_type'] = 'raw_engine_search'
        record['llama_bench_bin'] = llama_bench_bin
        if on_record:
            on_record(dict(record))
        return [record], []

    records: List[Dict[str, object]] = []
    scores: Dict[Tuple[object, ...], float] = {}
    profiles_by_key = {_tq3_raw_profile_key(item): item for item in raw_profiles}
    fast = (depth or '').strip().lower() == 'fast'
    phase_budget = TQ3_RAW_PRESEARCH_FAST_BUDGET_SECONDS if fast else TQ3_RAW_PRESEARCH_FULL_BUDGET_SECONDS
    case_timeout = TQ3_RAW_PRESEARCH_FAST_CASE_TIMEOUT if fast else TQ3_RAW_PRESEARCH_FULL_CASE_TIMEOUT
    raw_total = len(raw_profiles) * len(TQ3_RAW_BENCH_CASES)
    phase_deadline = min(
        time.monotonic() + phase_budget,
        deadline.end if deadline is not None else time.monotonic() + phase_budget,
    )
    raw_index = 0
    cancelled = False
    failed_cases = 0
    if progress:
        progress(
            f'TQ3 raw llama-bench pre-search: {len(raw_profiles)} profile(s), '
            f'{raw_total} case(s), timeout<={int(case_timeout)}s, budget={int(phase_budget)}s, binary={llama_bench_bin}'
        )
    for profile_index, runtime_profile in enumerate(raw_profiles, start=1):
        check_cancelled(cancel_token)
        candidate = _raw_bench_candidate_model(model, runtime_profile)
        threads = tq3_moe_cpu_placement_threads(candidate, runtime_profile, hardware)
        key = _tq3_raw_profile_key(runtime_profile)
        for measurement_type, prompt_tokens, generated_tokens in TQ3_RAW_BENCH_CASES:
            check_cancelled(cancel_token)
            if failed_cases >= TQ3_RAW_PRESEARCH_FAILURE_LIMIT:
                if progress:
                    progress(f'TQ3 raw llama-bench skipped remaining cases after {failed_cases} failed case(s)')
                break
            raw_index += 1
            remaining = phase_deadline - time.monotonic()
            if remaining <= 0:
                if progress:
                    progress(f'TQ3 raw llama-bench phase budget exhausted after {len(records)} case(s)')
                break
            cmd = tq3_llama_bench_command(
                llama_bench_bin,
                model,
                runtime_profile,
                prompt_tokens,
                generated_tokens,
                threads,
            )
            timeout = min(case_timeout, max(1.0, remaining))
            command_preview = shlex.join(str(item) for item in cmd)
            candidate_label = f'{runtime_profile.name}/{measurement_type}'
            if progress:
                progress(f'TQ3 raw llama-bench command: {command_preview}')
            emit_benchmark_event(
                progress,
                'benchmark_candidate',
                model,
                run_kind,
                message=(
                    f'TQ3 raw llama-bench {measurement_type} '
                    f'{raw_index}/{raw_total}: {runtime_profile.name}'
                ),
                phase='raw_engine_search',
                completed=int(completed_offset + raw_index - 1),
                total=int(total if total is not None else raw_total),
                candidate=candidate_label,
                command=command_preview,
            )
            output = ''
            status = 'benchmark failed'
            tps = 0.0
            exit_code = -1
            result = _run_tq3_raw_process(cmd, timeout, cancel_token)
            output = str(result.get('stdout', '') or '')
            raw_returncode = result.get('returncode', -1)
            exit_code = int(raw_returncode if raw_returncode is not None else -1)
            seconds = float(result.get('seconds', 0.0) or 0.0)
            timed_out = bool(result.get('timed_out', False))
            cancelled_case = bool(result.get('cancelled', False))
            if timed_out:
                output = output or f'raw llama-bench timed out after {timeout:.1f}s'
                status = 'benchmark failed'
            elif cancelled_case:
                output = output or 'raw llama-bench cancelled by user'
                status = 'aborted'
                cancelled = True
            else:
                tps = parse_llama_bench_tokens_per_sec(output)
                status = 'ok' if exit_code == 0 and tps > 0 else 'benchmark failed'
            record = adaptive_record_from_candidate(
                candidate,
                'raw_engine_search',
                status,
                tokens_per_sec=tps,
                prompt_tokens_per_sec=tps if generated_tokens == 0 else 0.0,
                seconds=seconds,
                detail=concise_failure(output, limit=500),
                engine=ENGINE_TQ3,
                server_bin=server_bin,
                binary_path=llama_bench_bin,
                runtime_profile=runtime_profile.name,
                kv_preset=runtime_profile.kv_preset,
                benchmark_depth=depth,
                placement_strategy=runtime_profile.placement_strategy,
                cpu_moe=runtime_profile.cpu_moe,
                n_cpu_moe=runtime_profile.n_cpu_moe,
                tensor_overrides=list(runtime_profile.tensor_overrides or ()),
                batch_size=runtime_profile.batch_size,
                ubatch_size=runtime_profile.ubatch_size,
                ctk=kv_modes_from_preset(runtime_profile.kv_preset)[0],
                ctv=kv_modes_from_preset(runtime_profile.kv_preset)[1],
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                command=command_preview,
            )
            record['benchmark_kind'] = 'raw_engine_search'
            record['measurement_type'] = measurement_type
            record['llama_bench_bin'] = llama_bench_bin
            record['exit_code'] = exit_code
            record['raw_command'] = command_preview
            record['stdout_excerpt'] = benchmark_failure_excerpt(output, limit=500)
            record['stderr_excerpt'] = ''
            record['raw_phase'] = 'raw_engine_search'
            record['raw_profile_index'] = profile_index
            record['raw_profile_total'] = len(raw_profiles)
            record['case_index'] = raw_index
            record['case_total'] = raw_total
            record['timeout_seconds'] = round(float(timeout), 2)
            record['phase_budget_seconds'] = int(phase_budget)
            record['timed_out'] = timed_out
            record['cancelled'] = cancelled_case
            if status != 'ok':
                failed_cases += 1
                if timed_out:
                    apply_failure_context(record, output, default_category='RAW_ENGINE_TIMEOUT')
                elif cancelled_case:
                    record['failure_category'] = ''
                    record['failure_reason'] = 'cancelled'
                    record['suggested_fix'] = ''
                else:
                    apply_failure_context(record, output, default_category='API_TIMEOUT')
            else:
                weight = 1.0 if measurement_type in ('raw_tg', 'raw_combined') else 0.25
                scores[key] = max(scores.get(key, 0.0), float(tps) * weight)
            records.append(record)
            if on_record:
                on_record(dict(record))
            emit_benchmark_event(
                progress,
                'benchmark_result',
                model,
                run_kind,
                message=(
                    f'TQ3 raw llama-bench {measurement_type} {status}: '
                    f'{float(tps or 0.0):.2f} tok/s {runtime_profile.name}'
                ),
                phase='raw_engine_search',
                completed=int(completed_offset + raw_index),
                total=int(total if total is not None else raw_total),
                candidate=candidate_label,
                record=record,
            )
            if cancelled:
                break
        if cancelled or failed_cases >= TQ3_RAW_PRESEARCH_FAILURE_LIMIT or phase_deadline <= time.monotonic():
            break

    promoted_keys = [
        key for key, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if _score > 0
    ][:2]
    promoted = [profiles_by_key[key] for key in promoted_keys if key in profiles_by_key]
    if cancelled:
        raise CancelledError(cancel_token.reason if cancel_token is not None else 'cancelled')
    return records, promoted


def dedupe_runtime_profiles(runtime_profiles: List[RuntimeProfile]) -> List[RuntimeProfile]:
    selected: List[RuntimeProfile] = []
    seen = set()
    for profile in runtime_profiles:
        key = (
            profile.engine_id,
            profile.name,
            profile.ctx_size,
            profile.gpu_layers,
            profile.kv_preset,
            profile.placement_strategy,
            profile.cpu_moe,
            profile.n_cpu_moe,
            tuple(profile.tensor_overrides or ()),
            profile.reasoning,
            profile.reasoning_budget,
            profile.reasoning_format,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(profile)
    return selected


def current_model_moe_placement(model: ModelConfig) -> Optional[MoePlacementCandidate]:
    tensor_overrides = tuple(
        str(item).strip()
        for item in list(getattr(model, 'tensor_overrides', []) or [])
        if str(item).strip()
    )
    cpu_moe = bool(getattr(model, 'cpu_moe', False))
    n_cpu_moe = max(0, int(getattr(model, 'n_cpu_moe', 0) or 0))
    placement = str(getattr(model, 'moe_placement_strategy', '') or '').strip()
    if not (placement or cpu_moe or n_cpu_moe > 0 or tensor_overrides):
        return None
    if placement.startswith('measured:moe_tuning:'):
        placement = placement.rsplit(':', 1)[-1]
    if not placement:
        if cpu_moe:
            placement = 'cpu_moe_all'
        elif n_cpu_moe > 0:
            placement = f'n_cpu_moe_{n_cpu_moe}'
        else:
            placement = 'tensor_override'
    return MoePlacementCandidate(
        name=placement,
        gpu_layers=None,
        cpu_moe=cpu_moe,
        n_cpu_moe=n_cpu_moe,
        tensor_overrides=tensor_overrides,
        expected_vram_saving_hint='current saved MoE placement',
        risk='baseline',
    )


def moe_placement_candidate_from_payload(payload: Dict[str, object]) -> Optional[MoePlacementCandidate]:
    normalized = moe_placement_payload(payload)
    if not (
        normalized.get('strategy')
        or normalized.get('cpu_moe')
        or int(normalized.get('n_cpu_moe', 0) or 0) > 0
        or normalized.get('tensor_overrides')
    ):
        return None
    return MoePlacementCandidate(
        name=str(normalized.get('strategy') or infer_moe_placement_strategy(
            bool(normalized.get('cpu_moe', False)),
            int(normalized.get('n_cpu_moe', 0) or 0),
            normalized.get('tensor_overrides', []),
        ) or 'profile_moe_placement'),
        gpu_layers=None,
        cpu_moe=bool(normalized.get('cpu_moe', False)),
        n_cpu_moe=max(0, int(normalized.get('n_cpu_moe', 0) or 0)),
        tensor_overrides=tuple(_normalized_tensor_overrides(normalized.get('tensor_overrides', []))),
        expected_vram_saving_hint='profile-level measured MoE placement',
        risk='baseline',
    )


def _moe_placement_signature(placement: MoePlacementCandidate) -> Tuple[bool, int, Tuple[str, ...]]:
    return (
        bool(getattr(placement, 'cpu_moe', False)),
        max(0, int(getattr(placement, 'n_cpu_moe', 0) or 0)),
        tuple(getattr(placement, 'tensor_overrides', ()) or ()),
    )


def _runtime_moe_label_suffix(placement: Optional[MoePlacementCandidate]) -> str:
    if placement is None:
        return ''
    if bool(getattr(placement, 'cpu_moe', False)):
        return '+moe_cpu'
    n_cpu_moe = max(0, int(getattr(placement, 'n_cpu_moe', 0) or 0))
    if n_cpu_moe > 0:
        return f'+moe_ncpu{n_cpu_moe}'
    if tuple(getattr(placement, 'tensor_overrides', ()) or ()):
        return '+moe_ot'
    return ''


def runtime_profile_moe_placement_mode(runtime_profile: Optional[RuntimeProfile]) -> str:
    if runtime_profile is None:
        return ''
    placement = str(getattr(runtime_profile, 'placement_strategy', '') or '').strip()
    has_placement = bool(
        bool(getattr(runtime_profile, 'cpu_moe', False))
        or max(0, int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0)) > 0
        or tuple(getattr(runtime_profile, 'tensor_overrides', ()) or ())
        or placement == 'cpu_moe_all'
        or placement.startswith('n_cpu_moe_')
        or placement in ('tensor_override', 'experts_cpu_override')
    )
    if not has_placement:
        return 'fit_only' if bool(getattr(runtime_profile, 'fit', False)) else ''
    return 'fit_assisted' if bool(getattr(runtime_profile, 'fit', False)) else 'locked'


def active_engine_runtime_profiles(
    app: AppConfig,
    model: ModelConfig,
    profile: HardwareProfile,
    depth: str = 'full',
) -> List[RuntimeProfile]:
    benchmark_depth = 'fast' if str(depth or '').strip().lower() == 'fast' else 'full'
    try:
        engine = app.active_engine_key_for_model(model)
    except Exception:
        engine = getattr(model, 'runtime', 'llama.cpp')
    if getattr(model, 'runtime', 'llama.cpp') != 'llama.cpp':
        return []
    if not str(getattr(model, 'path', '') or '').lower().endswith('.gguf'):
        return []
    if engine == 'tq3' and (getattr(model, 'tq3_status', '') or 'unknown').strip().lower() != 'native':
        return []
    try:
        capabilities = app.engine_capabilities()
    except Exception:
        capabilities = default_engine_capabilities(engine)

    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072) or 131072))
    arch = architecture_payload(model)
    native_ctx = int(arch.get('native_context_length', 0) or 0)
    if native_ctx > 0:
        ctx_max = max(ctx_min, min(ctx_max, native_ctx))
    size = int(arch.get('model_file_size', 0) or model_file_size(model) or 0)
    has_gpu = bool(profile and profile.has_usable_gpu())
    vram_free = int(getattr(profile, 'gpu_memory_free', 0) or 0)
    fits_gpu = bool(has_gpu and size > 0 and size <= int(vram_free * 0.82))
    heavy_for_gpu = bool(has_gpu and size > 0 and size > int(vram_free * 0.82))
    moe = model_is_moe(model)
    if engine == 'turboquant':
        turbo_profiles = turboquant_auto_profiles(model, capabilities, benchmark_depth)
    elif engine == 'tq3':
        turbo_profiles = tq3_kv_profiles_for_model(model, capabilities, benchmark_depth)
    else:
        turbo_profiles = supported_turbo_kv_profiles(capabilities, benchmark_depth, engine_id=engine)
    turboquant_status = (getattr(model, 'turboquant_status', '') or 'unknown').strip().lower()
    turboquant_compatible = turboquant_status in ('native', 'padded')
    supports_turbo = bool(
        engine == 'buun'
        and getattr(getattr(app, 'runtime_profile', None), 'supports_turbo_kv', False)
        and capabilities.supports_ctk_ctv
        and has_gpu
        and turbo_profiles
        and turboquant_compatible
    )
    supports_buun_fit = bool(engine == 'buun' and has_gpu and capabilities.supports_fit)
    fit_only_buun = bool(supports_buun_fit and (heavy_for_gpu or (moe and not fits_gpu)))
    supports_cache_kv = bool(capabilities.supports_cache_type_kv and engine != 'buun')
    base_ctx = max(ctx_min, min(ctx_max, 8192 if (moe or heavy_for_gpu or supports_turbo) else 4096))
    if engine == 'tq3' and moe:
        base_ctx = max(ctx_min, min(ctx_max, 4096))
    pressure_payload = current_process_pressure_payload()
    opencode_floor = observed_opencode_context_floor(model)
    context_points = dynamic_context_growth_targets(
        model,
        profile,
        ctx_min,
        ctx_max,
        depth=benchmark_depth,
        pressure_payload=pressure_payload,
        observed_floor=opencode_floor,
    )
    profiles: List[RuntimeProfile] = []
    seen = set()
    moe_placements = generate_moe_placement_candidates(model, profile, capabilities, engine, benchmark_depth)
    context_profile_placements = profile_moe_placements_from_model(model)
    saved_placement = current_model_moe_placement(model)
    if saved_placement is not None:
        saved_signature = _moe_placement_signature(saved_placement)
        moe_placements = [saved_placement] + [
            item for item in moe_placements
            if _moe_placement_signature(item) != saved_signature
        ]
    baseline_placement = moe_placements[0] if moe_placements else None

    def placement_key(placement: Optional[MoePlacementCandidate]) -> Tuple[str, bool, int, Tuple[str, ...]]:
        if placement is None:
            return '', False, 0, ()
        return (
            placement.name,
            bool(placement.cpu_moe),
            int(placement.n_cpu_moe or 0),
            tuple(placement.tensor_overrides or ()),
        )

    def apply_placement(
        runtime_profile: RuntimeProfile,
        placement: Optional[MoePlacementCandidate],
        name: Optional[str] = None,
    ) -> RuntimeProfile:
        if placement is None:
            return runtime_profile
        gpu_layers = runtime_profile.gpu_layers
        if placement.gpu_layers is not None:
            gpu_layers = int(placement.gpu_layers)
        profile_name = name or runtime_profile.name
        if name is None:
            suffix = _runtime_moe_label_suffix(placement)
            if suffix and suffix not in profile_name:
                profile_name = f'{profile_name}{suffix}'
            if suffix and runtime_profile.fit and '+fit_assisted' not in profile_name:
                profile_name = f'{profile_name}+fit_assisted'
        return replace(
            runtime_profile,
            name=profile_name,
            gpu_layers=gpu_layers,
            placement_strategy=placement.name,
            cpu_moe=bool(placement.cpu_moe),
            n_cpu_moe=max(0, int(placement.n_cpu_moe or 0)),
            tensor_overrides=tuple(placement.tensor_overrides or ()),
        )

    def with_tq3_reasoning_off_candidates(items: List[RuntimeProfile]) -> List[RuntimeProfile]:
        if not (
            engine == 'tq3'
            and moe
            and capabilities.supports_reasoning
            and capabilities.supports_reasoning_budget
            and capabilities.supports_reasoning_format
        ):
            return items
        selected: List[RuntimeProfile] = []
        seen_reasoning_shapes = set()

        def add_reasoning_seed(item: RuntimeProfile):
            key = (
                int(item.ctx_size or 0),
                item.gpu_layers,
                item.kv_preset,
                item.placement_strategy,
                item.cpu_moe,
                item.n_cpu_moe,
            )
            if key in seen_reasoning_shapes:
                return
            seen_reasoning_shapes.add(key)
            selected.append(item)

        placement_items = [
            item for item in items
            if not item.fit and (bool(item.cpu_moe) or int(item.n_cpu_moe or 0) > 0)
        ]
        for item in placement_items:
            if int(item.ctx_size or 0) == base_ctx:
                add_reasoning_seed(item)
                if len(selected) >= 2:
                    break
        target_contexts = [max(base_ctx, 16_384)]
        tool_ctx = int(opencode_floor or 0) or 32_768
        if benchmark_depth != 'fast':
            target_contexts.append(max(base_ctx, tool_ctx))
        for target in target_contexts:
            match = next(
                (
                    item for item in placement_items
                    if int(item.ctx_size or 0) >= target
                    and str(item.kv_preset or '') == 'q8_0/q8_0'
                ),
                None,
            )
            if match is not None:
                add_reasoning_seed(match)
        limit = 2 if benchmark_depth == 'fast' else 4
        selected = selected[:limit]
        if not selected:
            return items
        result: List[RuntimeProfile] = []
        emitted = set()
        selected_ids = {id(item) for item in selected}
        for item in items:
            result.append(item)
            if id(item) not in selected_ids:
                continue
            key = (item.name, item.ctx_size, item.kv_preset, item.placement_strategy, item.n_cpu_moe, item.cpu_moe)
            if key in emitted:
                continue
            emitted.add(key)
            result.append(replace(
                item,
                name=f'{item.name}_reasoning_off',
                reasoning='off',
                reasoning_budget=0,
                reasoning_format='deepseek',
            ))
        return result

    def finalized_profiles() -> List[RuntimeProfile]:
        if not moe_placements:
            return with_tq3_reasoning_off_candidates(profiles)
        updated: List[RuntimeProfile] = [
            item if (
                bool(getattr(item, 'cpu_moe', False))
                or int(getattr(item, 'n_cpu_moe', 0) or 0) > 0
                or tuple(getattr(item, 'tensor_overrides', ()) or ())
                or str(getattr(item, 'placement_strategy', '') or '').strip()
            ) else apply_placement(item, baseline_placement)
            for item in profiles
        ]
        profile_keys = {
            (
                item.name,
                item.ctx_size,
                item.gpu_layers,
                item.kv_preset,
                item.placement_strategy,
                item.cpu_moe,
                item.n_cpu_moe,
                item.tensor_overrides,
            )
            for item in updated
        }
        preferred_seed_names = (
            'partial_gpu_probe',
            'kv_compression_probe',
            'gpu_layer_sweep_full',
        )
        seeds = [
            item for item in updated
            if not item.fit
            and item.gpu_layers is not None
            and item.ctx_size == base_ctx
            and (
                item.name in preferred_seed_names
                or item.name.startswith('kv_compression_probe_')
                or item.name.startswith('gpu_layer_sweep_')
            )
        ]
        if not seeds:
            seeds = [
                item for item in updated
                if not item.fit and item.gpu_layers is not None and item.ctx_size == base_ctx
            ]
        if not seeds and has_gpu and updated:
            source = updated[0]
            seeds = [replace(
                source,
                name='moe_placement_probe',
                ctx_size=base_ctx,
                gpu_layers=partial_ngl,
                fit=False,
                fit_context=0,
                no_warmup=bool(capabilities.supports_no_warmup),
            )]
        seed_limit = 1 if benchmark_depth == 'fast' else 2
        selected_seeds: List[RuntimeProfile] = []
        for seed in seeds:
            if len(selected_seeds) >= seed_limit:
                break
            if seed.kv_preset not in {item.kv_preset for item in selected_seeds}:
                selected_seeds.append(seed)
        placement_updates: List[RuntimeProfile] = []
        for placement in moe_placements[1:]:
            for seed in selected_seeds:
                if placement.name == 'baseline_ngl':
                    candidate = replace(
                        seed,
                        name=f'{seed.name}_{placement.name}',
                        gpu_layers=partial_ngl,
                        placement_strategy='baseline_ngl',
                        cpu_moe=False,
                        n_cpu_moe=0,
                        tensor_overrides=(),
                    )
                else:
                    candidate = apply_placement(seed, placement, name=f'{seed.name}_{placement.name}')
                key = (
                    candidate.name,
                    candidate.ctx_size,
                    candidate.gpu_layers,
                    candidate.kv_preset,
                    candidate.placement_strategy,
                    candidate.cpu_moe,
                    candidate.n_cpu_moe,
                    candidate.tensor_overrides,
                )
                if key in profile_keys:
                    continue
                profile_keys.add(key)
                placement_updates.append(candidate)
        if placement_updates and engine == 'tq3' and moe:
            insert_at = next((idx for idx, item in enumerate(updated) if int(item.ctx_size or 0) > base_ctx), len(updated))
            return with_tq3_reasoning_off_candidates(updated[:insert_at] + placement_updates + updated[insert_at:])
        updated.extend(placement_updates)
        return with_tq3_reasoning_off_candidates(updated)

    def kv_for_strategy(strategy: str) -> str:
        if supports_cache_kv and strategy in ('kv_compression_probe', 'context_growth_sweep'):
            return 'q8_0/q8_0'
        return 'default'

    def estimate_partial_ngl() -> int:
        if not has_gpu:
            return 0
        if fits_gpu:
            return 999
        if size <= 0 or vram_free <= 0:
            return 16 if moe else 8
        layer_hint = int(arch.get('layer_count', 0) or 0)
        if layer_hint <= 0:
            layer_hint = 32 if moe else 24
        ratio = max(0.10, min(0.95, vram_free / max(1, size)))
        return max(1, min(layer_hint, int(round(layer_hint * ratio))))

    partial_ngl = estimate_partial_ngl()
    if heavy_for_gpu and partial_ngl > 0:
        partial_ngl = min(partial_ngl, 64)

    def has_locked_moe_placement() -> bool:
        return bool(baseline_placement is not None and _runtime_moe_label_suffix(baseline_placement))

    def locked_moe_ngl() -> int:
        try:
            current_ngl = int(getattr(model, 'ngl', 0) or 0)
        except Exception:
            current_ngl = 0
        return current_ngl if current_ngl > 0 else int(partial_ngl or 0)

    def add(
        name: str,
        ctx: int,
        ngl: Optional[int],
        kv_preset: str,
        batch: int = 256,
        ubatch: int = 128,
        kv_profile=None,
        fit: bool = False,
        fit_context: int = 0,
        no_warmup: bool = False,
        fit_discovery_phase: str = '',
        viable_ngl: int = 0,
        viable_ngl_source: str = '',
        placement: Optional[MoePlacementCandidate] = None,
        mtp_enabled: bool = False,
        mtp_draft_n_max: int = 0,
    ):
        ctx = max(ctx_min, min(ctx_max, int(ctx or base_ctx)))
        ngl_key = 'fit' if ngl is None else int(ngl)
        effective_placement = placement
        if effective_placement is None and saved_placement is not None and context_profile_placements:
            profile_key = moe_profile_key_for_context(ctx)
            payload = context_profile_placements.get(profile_key)
            if payload is None and profile_key == 'hermes_ready':
                payload = context_profile_placements.get('long_context')
            if payload is None:
                payload = context_profile_placements.get('auto')
            effective_placement = moe_placement_candidate_from_payload(payload or {})
        if effective_placement is None:
            effective_placement = baseline_placement
        key = (
            name,
            ctx,
            ngl_key,
            kv_preset,
            bool(fit),
            int(fit_context or 0),
            bool(no_warmup),
            placement_key(effective_placement),
            bool(mtp_enabled),
            int(mtp_draft_n_max or 0),
        )
        if key in seen:
            return
        seen.add(key)
        if kv_profile is not None:
            family = 'turbo' if any(mode.startswith('turbo') for mode in kv_modes_from_preset(kv_preset)) else 'cache'
        else:
            family = 'cache' if kv_preset and kv_preset != 'default' else 'default'
        profile_item = RuntimeProfile(
            engine_id=engine,
            name=name,
            ctx_size=ctx,
            gpu_layers=None if ngl is None else int(ngl),
            parallel=1,
            kv_preset=kv_preset,
            flash_attn='on',
            batch_size=batch,
            ubatch_size=ubatch,
            fit=bool(fit),
            fit_context=max(ctx_min, min(ctx, int(fit_context or ctx_min))) if fit else 0,
            no_warmup=bool(no_warmup),
            kv_family=family,
            kv_quality_tier=getattr(kv_profile, 'quality_tier', '') if kv_profile is not None else '',
            kv_compression_tier=getattr(kv_profile, 'compression_tier', '') if kv_profile is not None else '',
            kv_score_penalty=float(getattr(kv_profile, 'score_penalty', 0.0) or 0.0) if kv_profile is not None else 0.0,
            benchmark_depth=benchmark_depth,
            fit_discovery_phase=fit_discovery_phase,
            viable_ngl=max(0, int(viable_ngl or 0)),
            viable_ngl_source=viable_ngl_source,
            mtp_enabled=bool(mtp_enabled),
            mtp_draft_n_max=clamp_mtp_draft(mtp_draft_n_max, default=3) if mtp_enabled else 0,
        )
        profiles.append(apply_placement(profile_item, effective_placement))

    def fit_context_for(ctx: int) -> int:
        return min(int(ctx or base_ctx), max(ctx_min, 4096))

    def fit_growth_contexts() -> Tuple[int, ...]:
        available = [ctx for ctx in context_points if ctx > base_ctx and ctx <= ctx_max]
        floor = int(observed_opencode_context_floor(model) or 0)
        if floor > base_ctx:
            eligible = [ctx for ctx in context_points if ctx <= ctx_max]
            floor_ctx = _nearest_context_at_or_above(eligible, floor)
            if floor_ctx > base_ctx:
                available.append(floor_ctx)
        if benchmark_depth == 'fast':
            selected = []
            next_ctx = next((ctx for ctx in context_points if ctx > base_ctx and ctx <= ctx_max), 0)
            if next_ctx:
                selected.append(next_ctx)
            high_ctx = next((ctx for ctx in context_points if ctx > 32_768 and ctx <= ctx_max), 0)
            if high_ctx:
                selected.append(high_ctx)
            if floor > base_ctx:
                eligible = [ctx for ctx in context_points if ctx <= ctx_max]
                floor_ctx = _nearest_context_at_or_above(eligible, floor)
                if floor_ctx > base_ctx:
                    selected.append(floor_ctx)
            available = selected
        return tuple(sorted(set(ctx for ctx in available if ctx > base_ctx and ctx <= ctx_max)))

    def add_buun_fit_growth_profiles(include_turbo_ladder: bool):
        growth_contexts = fit_growth_contexts()
        if not growth_contexts:
            return
        if supports_turbo:
            if include_turbo_ladder:
                ladder_profiles = [
                    item for item in turbo_profiles
                    if not item.scalar and (
                        benchmark_depth == 'full'
                        or item.kv_preset in ('turbo4/turbo4', 'turbo3_tcq/turbo3_tcq', 'turbo3_tcq/turbo2_tcq')
                    )
                ]
                preferred = next((item for item in ladder_profiles if item.kv_preset == 'turbo4/turbo4'), None)
                preferred_profiles = [preferred or ladder_profiles[0]] if ladder_profiles else []
            else:
                preferred = next((item for item in turbo_profiles if item.kv_preset == 'turbo4/turbo4'), None)
                preferred_profiles = [preferred or turbo_profiles[0]] if turbo_profiles else []
                ladder_profiles = preferred_profiles
            for ctx in growth_contexts:
                ctx_profiles = ladder_profiles if ctx <= 32_768 else preferred_profiles
                for kv_profile in ctx_profiles:
                    add(
                        f'fit_context_growth_sweep_{ctx}_{kv_profile.name_slug}',
                        ctx,
                        None,
                        kv_profile.kv_preset,
                        batch=128,
                        ubatch=64,
                        kv_profile=kv_profile,
                        fit=True,
                        fit_context=fit_context_for(ctx),
                        no_warmup=capabilities.supports_no_warmup,
                    )
        for ctx in growth_contexts:
            add(
                f'fit_context_growth_sweep_{ctx}',
                ctx,
                None,
                'default',
                batch=128,
                ubatch=64,
                fit=True,
                fit_context=fit_context_for(ctx),
                no_warmup=capabilities.supports_no_warmup,
            )

    if engine == ENGINE_LLAMA_CPP_MTP:
        mtp_ngl = 999 if fits_gpu else partial_ngl
        add('mtp_baseline', base_ctx, mtp_ngl, 'default', batch=128, ubatch=64, mtp_enabled=False)
        supports_mtp = bool(
            getattr(capabilities, 'supports_spec_type', False)
            and getattr(capabilities, 'supports_mtp', False)
            and getattr(capabilities, 'supports_spec_draft_n_max', False)
        )
        if supports_mtp and model_mtp_allowed(model) and (
            bool(getattr(model, 'mtp_enabled', False))
            or str(getattr(model, 'supports_mtp', 'auto') or 'auto').strip().lower() == 'yes'
            or mtp_support_auto_hint(model)
        ):
            for draft in MTP_DRAFT_VALUES:
                add(
                    f'mtp_draft_n{draft}',
                    base_ctx,
                    mtp_ngl,
                    'default',
                    batch=128,
                    ubatch=64,
                    mtp_enabled=True,
                    mtp_draft_n_max=draft,
                )
        return finalized_profiles()

    if engine == 'tq3':
        baseline_profile = next((item for item in turbo_profiles if item.kv_preset == 'q8_0/q8_0'), None)
        baseline_kv = baseline_profile.kv_preset if baseline_profile is not None else 'q8_0/q8_0'

        if not has_gpu:
            add('cpu_probe', min(base_ctx, max(ctx_min, 4096)), 0, baseline_kv, batch=128, ubatch=64, kv_profile=baseline_profile)
            return finalized_profiles()

        add('partial_gpu_probe', base_ctx, partial_ngl, baseline_kv, batch=128, ubatch=64, kv_profile=baseline_profile)
        if moe:
            growth_target = next((ctx for ctx in context_points if ctx > base_ctx and ctx <= ctx_max), 0)
            if growth_target:
                suffix = baseline_profile.name_slug if baseline_profile is not None else 'q8_0_q8_0'
                add(
                    f'context_growth_sweep_{growth_target}_{suffix}',
                    growth_target,
                    partial_ngl,
                    baseline_kv,
                    batch=128,
                    ubatch=64,
                    kv_profile=baseline_profile,
                )
            return finalized_profiles()

        for kv_profile in turbo_profiles:
            if kv_profile.kv_preset == baseline_kv:
                continue
            add(
                f'kv_compression_probe_{kv_profile.name_slug}',
                base_ctx,
                partial_ngl,
                kv_profile.kv_preset,
                kv_profile=kv_profile,
            )
        if benchmark_depth == 'full':
            sweep_kv = baseline_kv
            if fits_gpu:
                add('gpu_layer_sweep_full', base_ctx, 999, sweep_kv, kv_profile=baseline_profile)
            else:
                sweep_center = max(4, partial_ngl)
                sweep_values = [max(1, sweep_center - 4), sweep_center, sweep_center + 4, sweep_center + 8, sweep_center + 12]
                for ngl in sorted(set(value for value in sweep_values if value > 0)):
                    add(f'gpu_layer_sweep_ngl{ngl}', base_ctx, ngl, sweep_kv, kv_profile=baseline_profile)
        growth_contexts = tuple(ctx for ctx in context_points if ctx > base_ctx and ctx <= ctx_max)
        growth_kv = baseline_kv
        for ctx in growth_contexts:
            suffix = baseline_profile.name_slug if baseline_profile is not None else 'q8_0_q8_0'
            add(f'context_growth_sweep_{ctx}_{suffix}', ctx, partial_ngl, growth_kv, kv_profile=baseline_profile)
        return finalized_profiles()

    if engine == 'turboquant':
        baseline_profile = next((item for item in turbo_profiles if item.kv_preset == 'q8_0/q8_0'), None)
        safe_profile = next((item for item in turbo_profiles if item.kv_preset == 'q8_0/turbo4'), None)
        balanced_profile = next((item for item in turbo_profiles if item.kv_preset == 'q8_0/turbo3'), None)
        preferred_profile = safe_profile or baseline_profile or (turbo_profiles[0] if turbo_profiles else None)
        baseline_kv = baseline_profile.kv_preset if baseline_profile is not None else 'q8_0/q8_0'

        if not has_gpu:
            add('cpu_probe', min(base_ctx, max(ctx_min, 4096)), 0, baseline_kv, batch=128, ubatch=64, kv_profile=baseline_profile)
            return finalized_profiles()

        if capabilities.supports_fit:
            discovery_ctx = max(ctx_min, min(ctx_max, max(8192, chat_min_ctx_per_slot(model))))
            discovery_profile = preferred_profile or baseline_profile
            discovery_kv = discovery_profile.kv_preset if discovery_profile is not None else baseline_kv
            if has_locked_moe_placement():
                add(
                    'moe_locked_probe',
                    discovery_ctx,
                    locked_moe_ngl(),
                    discovery_kv,
                    batch=128,
                    ubatch=64,
                    kv_profile=discovery_profile,
                )
            add(
                f'fit_weight_discovery_{(discovery_profile.name_slug if discovery_profile is not None else "q8_0_q8_0")}',
                discovery_ctx,
                None,
                discovery_kv,
                batch=128,
                ubatch=64,
                kv_profile=discovery_profile,
                fit=True,
                fit_context=fit_context_for(discovery_ctx),
                no_warmup=capabilities.supports_no_warmup,
                fit_discovery_phase='weight_fit',
            )

            gpu_total = int(getattr(profile, 'gpu_memory_total', 0) or 0)
            small_gpu = bool(0 < gpu_total <= 9 * 1024**3)
            growth_contexts = tuple(
                ctx for ctx in context_points
                if ctx > discovery_ctx and ctx <= ctx_max and not (small_gpu and ctx >= 262144)
            )
            growth_profiles = []
            for item in (safe_profile, balanced_profile):
                if item is not None and item.kv_preset not in {profile.kv_preset for profile in growth_profiles}:
                    growth_profiles.append(item)
            if not growth_profiles and discovery_profile is not None:
                growth_profiles = [discovery_profile]
            for kv_profile in growth_profiles:
                for ctx in growth_contexts:
                    add(
                        f'fit_context_growth_sweep_{ctx}_{kv_profile.name_slug}',
                        ctx,
                        None,
                        kv_profile.kv_preset,
                        batch=128,
                        ubatch=64,
                        kv_profile=kv_profile,
                        fit=True,
                        fit_context=fit_context_for(ctx),
                        no_warmup=capabilities.supports_no_warmup,
                        fit_discovery_phase='context_growth',
                    )
            return finalized_profiles()

        add('partial_gpu_probe', base_ctx, partial_ngl, baseline_kv, kv_profile=baseline_profile)
        for kv_profile in turbo_profiles:
            if kv_profile.kv_preset == baseline_kv:
                continue
            name = f'kv_compression_probe_{kv_profile.name_slug}'
            add(
                name,
                base_ctx,
                partial_ngl,
                kv_profile.kv_preset,
                kv_profile=kv_profile,
            )
        if benchmark_depth == 'full':
            sweep_kv = preferred_profile.kv_preset if preferred_profile is not None else baseline_kv
            if fits_gpu:
                add('gpu_layer_sweep_full', base_ctx, 999, sweep_kv, kv_profile=preferred_profile)
            else:
                sweep_center = max(4, partial_ngl)
                sweep_values = [max(1, sweep_center - 4), sweep_center, sweep_center + 4, sweep_center + 8, sweep_center + 12]
                for ngl in sorted(set(value for value in sweep_values if value > 0)):
                    add(f'gpu_layer_sweep_ngl{ngl}', base_ctx, ngl, sweep_kv, kv_profile=preferred_profile)

        growth_contexts = tuple(ctx for ctx in context_points if ctx > base_ctx and ctx <= ctx_max)
        growth_kv = preferred_profile.kv_preset if preferred_profile is not None else baseline_kv
        for ctx in growth_contexts:
            suffix = preferred_profile.name_slug if preferred_profile is not None else 'q8_0_q8_0'
            add(f'context_growth_sweep_{ctx}_{suffix}', ctx, partial_ngl, growth_kv, kv_profile=preferred_profile)
        return finalized_profiles()

    if not (engine == 'buun' and has_gpu):
        add('cpu_probe', min(base_ctx, max(ctx_min, 4096)), 0, 'default', batch=128, ubatch=64)
    if has_gpu:
        initial_fit_kv = ''
        if supports_buun_fit:
            if supports_turbo:
                turbo4_profile = next((item for item in turbo_profiles if item.kv_preset == 'turbo4/turbo4'), turbo_profiles[0])
                initial_fit_kv = turbo4_profile.kv_preset
                if has_locked_moe_placement():
                    add(
                        'moe_locked_probe',
                        base_ctx,
                        locked_moe_ngl(),
                        turbo4_profile.kv_preset,
                        batch=128,
                        ubatch=64,
                        kv_profile=turbo4_profile,
                        no_warmup=capabilities.supports_no_warmup,
                    )
                add(
                    'fit_turbokv_probe',
                    base_ctx,
                    None,
                    turbo4_profile.kv_preset,
                    batch=128,
                    ubatch=64,
                    kv_profile=turbo4_profile,
                    fit=True,
                    fit_context=fit_context_for(base_ctx),
                    no_warmup=capabilities.supports_no_warmup,
                )
                add(
                    'fit_default_probe',
                    base_ctx,
                    None,
                    'default',
                    batch=128,
                    ubatch=64,
                    fit=True,
                    fit_context=fit_context_for(base_ctx),
                    no_warmup=capabilities.supports_no_warmup,
                )
            else:
                initial_fit_kv = 'default'
                if has_locked_moe_placement():
                    add(
                        'moe_locked_probe',
                        base_ctx,
                        locked_moe_ngl(),
                        'default',
                        batch=128,
                        ubatch=64,
                        no_warmup=capabilities.supports_no_warmup,
                    )
                add(
                    'fit_default_probe',
                    base_ctx,
                    None,
                    'default',
                    batch=128,
                    ubatch=64,
                    fit=True,
                    fit_context=fit_context_for(base_ctx),
                    no_warmup=capabilities.supports_no_warmup,
                )
            if not fit_only_buun:
                add_buun_fit_growth_profiles(include_turbo_ladder=False)
        if fit_only_buun:
            if supports_turbo:
                for kv_profile in turbo_profiles:
                    if kv_profile.kv_preset == initial_fit_kv:
                        continue
                    add(
                        f'fit_kv_compression_probe_{kv_profile.name_slug}',
                        base_ctx,
                        None,
                        kv_profile.kv_preset,
                        batch=128,
                        ubatch=64,
                        kv_profile=kv_profile,
                        fit=True,
                        fit_context=fit_context_for(base_ctx),
                        no_warmup=capabilities.supports_no_warmup,
                    )
                add_buun_fit_growth_profiles(include_turbo_ladder=True)
            else:
                add_buun_fit_growth_profiles(include_turbo_ladder=False)
            return finalized_profiles()
        add('partial_gpu_probe', base_ctx, partial_ngl, kv_for_strategy('partial_gpu_probe'))
        if supports_turbo:
            for kv_profile in turbo_profiles:
                name = 'kv_compression_probe' if kv_profile.kv_preset == 'turbo4/turbo4' else f'kv_compression_probe_{kv_profile.name_slug}'
                add(
                    name,
                    base_ctx,
                    partial_ngl,
                    kv_profile.kv_preset,
                    kv_profile=kv_profile,
                    no_warmup=capabilities.supports_no_warmup,
                )
        elif supports_cache_kv:
            add('kv_compression_probe', base_ctx, partial_ngl, kv_for_strategy('kv_compression_probe'))
        if benchmark_depth == 'full':
            sweep_kv_profile = next((item for item in turbo_profiles if item.kv_preset == 'turbo4/turbo4'), None) if supports_turbo else None
            sweep_kv = sweep_kv_profile.kv_preset if sweep_kv_profile is not None else kv_for_strategy('gpu_layer_sweep')
            if fits_gpu:
                add('gpu_layer_sweep_full', base_ctx, 999, sweep_kv, kv_profile=sweep_kv_profile)
            else:
                sweep_center = max(4, partial_ngl)
                sweep_values = [max(1, sweep_center - 4), sweep_center, sweep_center + 4, sweep_center + 8, sweep_center + 12]
                for ngl in sorted(set(value for value in sweep_values if value > 0)):
                    add(f'gpu_layer_sweep_ngl{ngl}', base_ctx, ngl, sweep_kv, kv_profile=sweep_kv_profile)
    else:
        if supports_cache_kv:
            add('kv_compression_probe', base_ctx, 0, kv_for_strategy('kv_compression_probe'))

    context_seed_ngl = partial_ngl if has_gpu else 0
    if supports_turbo:
        sweep_kv_profile = next((item for item in turbo_profiles if item.kv_preset == 'turbo4/turbo4'), turbo_profiles[0] if turbo_profiles else None)
        growth_contexts = tuple(ctx for ctx in context_points if ctx > base_ctx and ctx <= ctx_max)
        if sweep_kv_profile is not None:
            for ctx in growth_contexts:
                add(f'context_growth_sweep_{ctx}_{sweep_kv_profile.name_slug}', ctx, context_seed_ngl, sweep_kv_profile.kv_preset, kv_profile=sweep_kv_profile)
        for ctx in growth_contexts:
            add(f'context_growth_sweep_{ctx}', ctx, context_seed_ngl, 'default')
    else:
        context_kv = kv_for_strategy('context_growth_sweep')
        for ctx in context_points:
            if ctx > base_ctx and ctx <= ctx_max:
                add(f'context_growth_sweep_{ctx}', ctx, context_seed_ngl, context_kv)
    return finalized_profiles()


def model_for_runtime_profile(model: ModelConfig, runtime_profile: RuntimeProfile) -> ModelConfig:
    candidate = ModelConfig(**asdict(model))
    candidate.ctx = max(1, int(runtime_profile.ctx_size or candidate.ctx))
    candidate.ngl = int(runtime_profile.gpu_layers if runtime_profile.gpu_layers is not None else candidate.ngl)
    candidate.parallel = max(1, int(runtime_profile.parallel or 1))
    candidate.flash_attn = str(runtime_profile.flash_attn or 'on').strip().lower() != 'off'
    candidate.moe_placement_strategy = str(getattr(runtime_profile, 'placement_strategy', '') or '')
    candidate.cpu_moe = bool(getattr(runtime_profile, 'cpu_moe', False))
    candidate.n_cpu_moe = max(0, int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0))
    candidate.tensor_overrides = [str(item) for item in tuple(getattr(runtime_profile, 'tensor_overrides', ()) or ())]
    candidate.mtp_enabled = bool(getattr(runtime_profile, 'mtp_enabled', False))
    candidate.mtp_draft_n_max = model_mtp_draft_n_max(candidate, runtime_profile)
    candidate.threads = tq3_moe_cpu_placement_threads(candidate, runtime_profile)
    candidate.optimize_mode = 'manual'
    candidate.optimize_tier = 'measured'
    return candidate


def benchmark_config_fingerprint(candidate: ModelConfig) -> str:
    payload = {
        'runtime': getattr(candidate, 'runtime', 'llama.cpp'),
        'path': getattr(candidate, 'path', ''),
        'ctx': int(getattr(candidate, 'ctx', 0) or 0),
        'parallel': int(getattr(candidate, 'parallel', 1) or 1),
        'threads': int(getattr(candidate, 'threads', 0) or 0),
        'ngl': int(getattr(candidate, 'ngl', 0) or 0),
        'output': int(getattr(candidate, 'output', 0) or 0),
        'cache_ram': int(getattr(candidate, 'cache_ram', 0) or 0),
        'mtp_enabled': bool(getattr(candidate, 'mtp_enabled', False)),
        'mtp_draft_n_max': clamp_mtp_draft(getattr(candidate, 'mtp_draft_n_max', 3), default=3),
        'temp': float(getattr(candidate, 'temp', 0.7) or 0.7),
        'top_p': float(getattr(candidate, 'top_p', 0.95) or 0.95),
        'top_k': int(getattr(candidate, 'top_k', 40) or 0),
        'repeat_penalty': float(getattr(candidate, 'repeat_penalty', 1.0) or 1.0),
        'presence_penalty': float(getattr(candidate, 'presence_penalty', 0.0) or 0.0),
        'no_context_shift': bool(getattr(candidate, 'no_context_shift', False)),
        'preserve_thinking': getattr(candidate, 'preserve_thinking', 'auto') or 'auto',
        'flash_attn': bool(getattr(candidate, 'flash_attn', True)),
        'jinja': bool(getattr(candidate, 'jinja', True)),
        'memory_reserve_percent': int(getattr(candidate, 'memory_reserve_percent', 0) or 0),
        'extra_args': list(getattr(candidate, 'extra_args', []) or []),
        'launch_overrides': dict(getattr(candidate, 'launch_overrides', {}) or {}),
        'architecture_type': getattr(candidate, 'architecture_type', 'unknown') or 'unknown',
        'architecture': getattr(candidate, 'architecture', '') or '',
        'expert_count': int(getattr(candidate, 'expert_count', 0) or 0),
        'expert_used_count': int(getattr(candidate, 'expert_used_count', 0) or 0),
        'active_expert_ratio': float(getattr(candidate, 'active_expert_ratio', 0.0) or 0.0),
        'moe_placement_strategy': getattr(candidate, 'moe_placement_strategy', '') or '',
        'cpu_moe': bool(getattr(candidate, 'cpu_moe', False)),
        'n_cpu_moe': int(getattr(candidate, 'n_cpu_moe', 0) or 0),
        'tensor_overrides': list(getattr(candidate, 'tensor_overrides', []) or []),
    }
    arch_record = architecture_payload(candidate)
    for key in (
        'model_file_size',
        'native_context_length',
        'attention_key_length',
        'attention_value_length',
        'attention_head_count',
        'attention_head_count_kv',
    ):
        payload[key] = arch_record.get(key, 0)
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def runtime_profile_config_fingerprint(candidate: ModelConfig, runtime_profile: RuntimeProfile) -> str:
    payload = {
        'candidate': benchmark_config_fingerprint(candidate),
        'engine_id': runtime_profile.engine_id,
        'runtime_profile': runtime_profile.name,
        'gpu_layers': runtime_profile.gpu_layers,
        'kv_preset': runtime_profile.kv_preset,
        'flash_attn': runtime_profile.flash_attn,
        'batch_size': int(runtime_profile.batch_size or 0),
        'ubatch_size': int(runtime_profile.ubatch_size or 0),
        'fit': bool(runtime_profile.fit),
        'fit_context': int(runtime_profile.fit_context or 0),
        'no_warmup': bool(runtime_profile.no_warmup),
        'fit_discovery_phase': str(getattr(runtime_profile, 'fit_discovery_phase', '') or ''),
        'viable_ngl': int(getattr(runtime_profile, 'viable_ngl', 0) or 0),
        'placement_strategy': str(getattr(runtime_profile, 'placement_strategy', '') or ''),
        'cpu_moe': bool(getattr(runtime_profile, 'cpu_moe', False)),
        'n_cpu_moe': int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0),
        'tensor_overrides': list(getattr(runtime_profile, 'tensor_overrides', ()) or ()),
        'reasoning': str(getattr(runtime_profile, 'reasoning', '') or ''),
        'reasoning_budget': int(getattr(runtime_profile, 'reasoning_budget', -1) or -1),
        'reasoning_format': str(getattr(runtime_profile, 'reasoning_format', '') or ''),
        'mtp_enabled': bool(getattr(runtime_profile, 'mtp_enabled', False)),
        'mtp_draft_n_max': int(getattr(runtime_profile, 'mtp_draft_n_max', 0) or 0),
        'extra_args': list(runtime_profile.extra_args or ()),
    }
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def runtime_profile_is_buun_fit(runtime_profile: RuntimeProfile) -> bool:
    return (
        str(getattr(runtime_profile, 'engine_id', '') or '') == 'buun'
        and runtime_profile_uses_fit(runtime_profile)
    )


def runtime_profile_is_fixed_buun(runtime_profile: RuntimeProfile) -> bool:
    return (
        str(getattr(runtime_profile, 'engine_id', '') or '') == 'buun'
        and runtime_profile_is_fixed_gpu_layers(runtime_profile)
    )


def runtime_profile_uses_fit(runtime_profile: RuntimeProfile) -> bool:
    return bool(getattr(runtime_profile, 'fit', False)) and getattr(runtime_profile, 'gpu_layers', None) is None


def runtime_profile_is_fixed_gpu_layers(runtime_profile: RuntimeProfile) -> bool:
    return getattr(runtime_profile, 'gpu_layers', None) is not None


def runtime_profile_fit_phase(runtime_profile: RuntimeProfile) -> str:
    return str(getattr(runtime_profile, 'fit_discovery_phase', '') or '')


def runtime_profile_with_fit_ceiling(
    runtime_profile: RuntimeProfile,
    viable_ngl: int,
    source: str,
    excerpt: str = '',
) -> RuntimeProfile:
    return replace(
        runtime_profile,
        viable_ngl=max(0, int(viable_ngl or 0)),
        viable_ngl_source=source or 'unknown',
        fit_selected_ngl=max(0, int(viable_ngl or 0)),
        fit_selected_ngl_source=source or 'unknown',
        fit_log_excerpt=excerpt,
    )


def fit_ceiling_from_records(records: List[Dict[str, object]]) -> Tuple[int, str, str]:
    for record in reversed(records or []):
        try:
            viable_ngl = int(record.get('viable_ngl', 0) or 0)
        except Exception:
            viable_ngl = 0
        if viable_ngl <= 0:
            try:
                viable_ngl = int(record.get('fit_selected_ngl', 0) or 0)
            except Exception:
                viable_ngl = 0
        source = str(record.get('viable_ngl_source', '') or record.get('fit_selected_ngl_source', '') or '')
        excerpt = str(record.get('fit_log_excerpt', '') or record.get('failure_excerpt', '') or '')
        if viable_ngl > 0 or source:
            return viable_ngl, source or 'unknown', excerpt
    return 0, 'unknown', ''


def smart_should_continue_optional(
    started_monotonic: float,
    measured: List[Dict[str, object]],
    model: ModelConfig,
    profile: HardwareProfile,
    now: Optional[float] = None,
) -> bool:
    now = time.monotonic() if now is None else float(now)
    started = float(started_monotonic if started_monotonic is not None else now)
    if now - started < SMART_BENCHMARK_SOFT_BUDGET_SECONDS:
        return True
    successful = [
        item for item in measured
        if item.get('status') == 'ok' and str(item.get('measurement_type', 'full') or 'full') == 'full'
    ]
    if not successful:
        return True
    has_chat = any(int(item.get('ctx_per_slot', 0) or 0) >= chat_min_ctx_per_slot(model) for item in successful)
    has_single_slot = any(int(item.get('parallel', 1) or 1) == 1 for item in successful)
    return not (has_chat and has_single_slot)


def smart_should_try_q8(
    model: ModelConfig,
    profile: HardwareProfile,
    default_best_ctx: int,
    default_break_ctx: int = 0,
) -> bool:
    if getattr(model, 'runtime', 'llama.cpp') != 'llama.cpp' or not profile.has_usable_gpu():
        return False
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072) or 131072))
    if default_break_ctx and int(default_best_ctx or 0) < ctx_max:
        return True
    default_seed = configure_adaptive_candidate(model, profile, 'long_context', ctx_min, 1, 'default')
    q8_seed = configure_adaptive_candidate(model, profile, 'long_context', ctx_min, 1, 'q8_kv')
    default_safe = candidate_safe_context_estimate(default_seed, profile)
    q8_safe = candidate_safe_context_estimate(q8_seed, profile)
    if default_safe <= 0:
        return q8_safe >= ctx_min
    return q8_safe >= int(default_safe * (1.0 + SMART_Q8_CONTEXT_GAIN_THRESHOLD))


def enrich_exhaustive_record(
    record: Dict[str, object],
    candidate: ModelConfig,
    variant: str,
    retry_attempt: int,
    estimated_safe_ctx: int,
    scan_level: str = 'broad',
    break_point: bool = False,
    measurement_type: str = 'full',
    planner_reason: str = '',
    reused_from: str = '',
) -> Dict[str, object]:
    record['variant'] = variant
    record['retry_attempt'] = retry_attempt
    record['scan_level'] = scan_level
    record['break_point'] = bool(break_point)
    record['estimated_safe_ctx'] = int(estimated_safe_ctx or 0)
    record['measurement_type'] = measurement_type or 'full'
    record['planner_reason'] = planner_reason or scan_level
    record['config_fingerprint'] = benchmark_config_fingerprint(candidate)
    if reused_from:
        record['reused_from'] = reused_from
    if estimated_safe_ctx and estimated_safe_ctx < int(getattr(candidate, 'ctx', 0) or 0):
        detail = str(record.get('detail', '') or '')
        suffix = f'estimate warned safe_ctx={estimated_safe_ctx}'
        record['detail'] = concise_failure(f'{detail}; {suffix}' if detail else suffix, limit=500)
    return record


def emit_exhaustive_result(
    progress: Optional[Callable[[object], None]],
    model: ModelConfig,
    record: Dict[str, object],
    completed: int,
    total: int,
    candidate_label: str,
    run_kind: str = 'server',
):
    benchmark_label = 'fast benchmark' if run_kind == 'server_fast' else 'smart bounded'
    status = str(record.get('status', '') or '')
    if status in ('ok', 'probe ok'):
        message = (
            f'{benchmark_label} {record.get("objective")} {status}: '
            f'{float(record.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s '
            f'ctx={record.get("ctx")} slot={record.get("ctx_per_slot")} '
            f'par={record.get("parallel")} variant={record.get("variant")}'
        )
    else:
        category = str(record.get('failure_category', '') or status)
        message = (
            f'{benchmark_label} {record.get("objective")} {status}: {category} '
            f'ctx={record.get("ctx")} slot={record.get("ctx_per_slot")} '
            f'par={record.get("parallel")} variant={record.get("variant")}'
        )
    emit_benchmark_event(
        progress,
        'benchmark_result',
        model,
        run_kind,
        message=message,
        phase=f'measuring {benchmark_label} candidates',
        completed=completed,
        total=total,
        candidate=candidate_label,
        record=record,
    )


def benchmark_exhaustive_candidate_with_retry(
    app: AppConfig,
    base_model: ModelConfig,
    profile: HardwareProfile,
    objective: str,
    ctx: int,
    parallel: int,
    variant: str,
    progress: Optional[Callable[[object], None]],
    cancel_token: Optional[CancelToken],
    completed: int,
    total: int,
    scan_level: str = 'broad',
    run_kind: str = 'server',
    planner_reason: str = '',
    measurement_type: str = 'full',
    deadline: Optional[BenchmarkDeadline] = None,
) -> Tuple[bool, bool, List[Dict[str, object]], List[Dict[str, object]], int]:
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    candidate_label = f'{objective}/{variant}/{scan_level} ctx={ctx} par={parallel}'
    benchmark_label = 'fast benchmark' if run_kind == 'server_fast' else 'smart bounded'
    for attempt in (1, 2):
        check_cancelled(cancel_token)
        try:
            engine = app.active_engine_key_for_model(base_model)
        except Exception:
            engine = str(getattr(base_model, 'runtime', '') or '')
        if not deadline_has_candidate_budget(deadline, engine=engine):
            if progress:
                progress(f'{benchmark_label} candidate {candidate_label} skipped: benchmark budget exhausted')
            return False, True, records, measured, completed
        candidate = configure_adaptive_candidate(base_model, profile, objective, ctx, parallel, variant)
        estimated_safe_ctx = candidate_safe_context_estimate(candidate, profile)
        try:
            capabilities = app.engine_capabilities()
        except Exception:
            capabilities = None
        launch_depth = 'fast' if run_kind == 'server_fast' or scan_level == 'fast' else 'full'
        launch_profile = build_benchmark_launch_profile(
            candidate,
            None,
            capabilities,
            purpose='serve_default',
            depth=launch_depth,
        )
        command_preview = benchmark_command_preview(app, candidate, benchmark_profile=launch_profile)
        if progress:
            progress(
                f'{benchmark_label} candidate {candidate_label} attempt={attempt} '
                f'estimated_safe_ctx={estimated_safe_ctx}'
            )
        emit_benchmark_event(
            progress,
            'benchmark_candidate',
            base_model,
            run_kind,
            message=f'{benchmark_label} candidate {candidate_label} attempt={attempt}',
            phase=f'measuring {benchmark_label} candidates',
            completed=completed,
            total=total,
            candidate=candidate_label,
            command=command_preview,
        )
        record, measured_item = benchmark_adaptive_candidate(
            app,
            candidate,
            objective,
            progress,
            cancel_token,
            benchmark_profile=launch_profile,
            benchmark_purpose='serve_default',
            benchmark_depth=launch_depth,
            deadline=deadline,
        )
        completed += 1
        ok = record.get('status') == 'ok'
        break_point = not ok and attempt == 2
        enrich_exhaustive_record(
            record,
            candidate,
            variant,
            attempt,
            estimated_safe_ctx,
            scan_level=scan_level,
            break_point=break_point,
            measurement_type=measurement_type,
            planner_reason=planner_reason or scan_level,
        )
        records.append(record)
        if measured_item:
            measured_item['variant'] = variant
            measured_item['retry_attempt'] = attempt
            measured_item['scan_level'] = scan_level
            measured_item['measurement_type'] = measurement_type
            measured_item['planner_reason'] = planner_reason or scan_level
            measured_item['config_fingerprint'] = benchmark_config_fingerprint(candidate)
            measured.append(measured_item)
        emit_exhaustive_result(progress, base_model, record, completed, total, candidate_label, run_kind=run_kind)
        if ok:
            return True, False, records, measured, completed
        deterministic_failures = {
            'BUDGET_EXHAUSTED',
            'CLI_INVALID',
            'MEMORY_GUARDRAIL',
            'MEMORY_FIT_FAILED',
            'FIXED_GPU_LAYERS_FIT_FAILED',
            'CUDA_OOM_WEIGHTS',
            'CUDA_OOM_KV',
            'KV_MODE_INCOMPATIBLE',
            'BUUN_FIT_FAILED',
            'BUUN_CPU_WARMUP_ABORT',
        }
        if str(record.get('failure_category', '') or '') in deterministic_failures:
            return False, True, records, measured, completed
        if tq3_terminal_candidate_failure(record):
            if progress:
                progress(
                    f'{benchmark_label} candidate {candidate_label} failed with '
                    f'{record.get("failure_category") or record.get("status")}; not retrying TQ3 timeout/low-speed profile.'
                )
            return False, True, records, measured, completed
        if attempt == 1 and progress:
            progress(f'{benchmark_label} candidate {candidate_label} failed once; retrying to confirm break...')
    return False, True, records, measured, completed


def benchmark_runtime_profile_with_retry(
    app: AppConfig,
    base_model: ModelConfig,
    runtime_profile: RuntimeProfile,
    objective: str,
    progress: Optional[Callable[[object], None]],
    cancel_token: Optional[CancelToken],
    completed: int,
    total: int,
    run_kind: str = 'server',
    max_attempts: int = 2,
    benchmark_depth: str = 'full',
    benchmark_purpose: str = 'serve_default',
    deadline: Optional[BenchmarkDeadline] = None,
) -> Tuple[bool, bool, List[Dict[str, object]], List[Dict[str, object]], int]:
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    ngl_label = 'fit' if runtime_profile.gpu_layers is None else str(runtime_profile.gpu_layers)
    candidate_label = (
        f'{runtime_profile.name} ctx={runtime_profile.ctx_size} '
        f'ngl={ngl_label} par={runtime_profile.parallel} kv={runtime_profile.kv_preset}'
    )
    attempts = max(1, int(max_attempts or 1))
    for attempt in range(1, attempts + 1):
        check_cancelled(cancel_token)
        if not deadline_has_candidate_budget(deadline, runtime_profile=runtime_profile):
            if progress:
                progress(f'runtime profile candidate {candidate_label} skipped: benchmark budget exhausted')
            break
        candidate = model_for_runtime_profile(base_model, runtime_profile)
        profile_fingerprint = runtime_profile_config_fingerprint(candidate, runtime_profile)
        try:
            capabilities = app.engine_capabilities()
        except Exception:
            capabilities = None
        launch_profile = build_benchmark_launch_profile(
            candidate,
            runtime_profile,
            capabilities,
            purpose=benchmark_purpose,
            depth=benchmark_depth,
        )
        command_preview = benchmark_command_preview(app, candidate, runtime_profile, launch_profile)
        if progress:
            progress(f'runtime profile candidate {candidate_label} attempt={attempt}')
        emit_benchmark_event(
            progress,
            'benchmark_candidate',
            base_model,
            run_kind,
            message=f'runtime profile candidate {candidate_label} attempt={attempt}',
            phase='measuring runtime profiles',
            completed=completed,
            total=total,
            candidate=candidate_label,
            command=command_preview,
        )
        record, measured_item = benchmark_adaptive_candidate(
            app,
            candidate,
            objective,
            progress,
            cancel_token,
            runtime_profile=runtime_profile,
            benchmark_profile=launch_profile,
            benchmark_purpose=benchmark_purpose,
            benchmark_depth=benchmark_depth,
            deadline=deadline,
        )
        completed += 1
        ok = record.get('status') == 'ok'
        enrich_exhaustive_record(
            record,
            candidate,
            runtime_profile.name or 'runtime_profile',
            attempt,
            0,
            scan_level=f'runtime_profile_{benchmark_depth}',
            break_point=not ok and attempt == attempts,
            measurement_type='full',
            planner_reason=runtime_profile.name or 'runtime_profile',
        )
        record['config_fingerprint'] = profile_fingerprint
        record['benchmark_depth'] = runtime_profile.benchmark_depth or benchmark_depth
        records.append(record)
        if measured_item:
            measured_item['variant'] = runtime_profile.name or 'runtime_profile'
            measured_item['retry_attempt'] = attempt
            measured_item['scan_level'] = f'runtime_profile_{benchmark_depth}'
            measured_item['measurement_type'] = 'full'
            measured_item['planner_reason'] = runtime_profile.name or 'runtime_profile'
            measured_item['config_fingerprint'] = profile_fingerprint
            measured_item['benchmark_depth'] = runtime_profile.benchmark_depth or benchmark_depth
            measured.append(measured_item)
        emit_exhaustive_result(progress, base_model, record, completed, total, candidate_label, run_kind=run_kind)
        if ok:
            return True, False, records, measured, completed
        deterministic_failures = {
            'BUDGET_EXHAUSTED',
            'CLI_INVALID',
            'MEMORY_GUARDRAIL',
            'MEMORY_FIT_FAILED',
            'FIXED_GPU_LAYERS_FIT_FAILED',
            'CUDA_OOM_WEIGHTS',
            'CUDA_OOM_KV',
            'KV_MODE_INCOMPATIBLE',
            'BUUN_FIT_FAILED',
            'BUUN_CPU_WARMUP_ABORT',
        }
        if str(record.get('failure_category', '') or '') in deterministic_failures:
            if progress:
                progress(f'runtime profile candidate {candidate_label} failed with {record.get("failure_category")}; moving to a different profile.')
            break
        if tq3_terminal_candidate_failure(record, runtime_profile):
            if progress:
                progress(
                    f'runtime profile candidate {candidate_label} failed with '
                    f'{record.get("failure_category") or record.get("status")}; not retrying TQ3 timeout/low-speed profile.'
                )
            break
        if attempt < attempts and progress:
            progress(f'runtime profile candidate {candidate_label} failed once; retrying to confirm break...')
    return False, True, records, measured, completed


def benchmark_frontier_probe_candidate(
    app: AppConfig,
    candidate: ModelConfig,
    objective: str,
    progress: Optional[Callable[[str], None]],
    cancel_token: Optional[CancelToken],
    deadline: Optional[BenchmarkDeadline] = None,
) -> Tuple[Dict[str, object], bool]:
    check_cancelled(cancel_token)
    try:
        capabilities = app.engine_capabilities()
    except Exception:
        capabilities = None
    launch_profile = build_benchmark_launch_profile(
        candidate,
        None,
        capabilities,
        purpose='serve_default',
        depth='fast',
    )
    runtime_context = runtime_record_context(app, candidate, benchmark_profile=launch_profile)
    if deadline is not None and deadline.expired():
        return budget_exhausted_record(candidate, objective, 'benchmark budget expired before frontier probe', runtime_context), False
    ok, msg = app.start(candidate, benchmark_profile=launch_profile)
    if not ok:
        record = adaptive_record_from_candidate(candidate, objective, 'start failed', detail=msg, **runtime_context)
        apply_failure_context(record, msg, default_category='SERVER_TIMEOUT')
        return record, False
    try:
        ready_timeout = deadline.cap_timeout(BENCHMARK_READY_TIMEOUT) if deadline is not None else BENCHMARK_READY_TIMEOUT
        if ready_timeout <= 0.0:
            return budget_exhausted_record(candidate, objective, 'benchmark budget expired before frontier readiness wait', runtime_context), False
        ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=ready_timeout, cancel_token=cancel_token)
        if not ready_ok:
            record = adaptive_record_from_candidate(candidate, objective, 'not ready', detail=ready_msg, **runtime_context)
            apply_failure_context(record, ready_msg, default_category='SERVER_TIMEOUT')
            return record, False
        if int(getattr(candidate, 'last_good_ctx', 0) or 0) > 0:
            candidate.ctx = int(candidate.last_good_ctx)
        if int(getattr(candidate, 'last_good_parallel', 0) or 0) > 0:
            candidate.parallel = int(candidate.last_good_parallel)
        if progress:
            progress(f'frontier probe ready: ctx={candidate.ctx} slot={ctx_per_slot(candidate)} variant={getattr(candidate, "variant", "default")}')
        warm_ok, warm = benchmark_completion(
            candidate,
            max_tokens=min(BENCHMARK_WARMUP_TOKENS, max(1, launch_profile.measurement_output)),
            timeout=BENCHMARK_WARMUP_TIMEOUT,
            cancel_token=cancel_token,
            launch_profile=launch_profile,
            deadline=deadline,
        )
        if not warm_ok:
            detail = str(warm.get('error', 'warmup failed'))
            record = adaptive_record_from_candidate(candidate, objective, 'benchmark failed', detail=detail, startup_result='READY', **runtime_context)
            apply_failure_context(record, detail, default_category='API_TIMEOUT')
            return record, False
        snap = app.hardware_profile(refresh=True)
        elapsed = float(warm.get('elapsed', 0.0) or 0.0)
        prompt_tps = (int(warm.get('prompt_tokens', 0) or 0) / elapsed) if elapsed > 0 else 0.0
        record = adaptive_record_from_candidate(
            candidate,
            objective,
            'probe ok',
            tokens_per_sec=float(warm.get('tokens_per_sec', 0.0) or 0.0),
            seconds=elapsed,
            detail='warmup probe',
            ram_available=int(getattr(snap, 'memory_available', 0) or 0),
            gpu_memory_free=int(getattr(snap, 'gpu_memory_free', 0) or 0),
            gpu_memory_total=int(getattr(snap, 'gpu_memory_total', 0) or 0),
            startup_result='READY',
            prompt_tokens=int(warm.get('prompt_tokens', 0) or 0),
            prompt_tokens_per_sec=prompt_tps,
            generated_tokens=int(warm.get('completion_tokens', 0) or 0),
            **runtime_context,
        )
        return record, True
    finally:
        app.stop(candidate, managed_only=True)
        sleep_with_cancel(0.25, cancel_token)


def benchmark_smart_probe_with_retry(
    app: AppConfig,
    base_model: ModelConfig,
    profile: HardwareProfile,
    ctx: int,
    variant: str,
    progress: Optional[Callable[[object], None]],
    cancel_token: Optional[CancelToken],
    completed: int,
    total: int,
    planner_reason: str = 'frontier',
    deadline: Optional[BenchmarkDeadline] = None,
) -> Tuple[bool, bool, List[Dict[str, object]], int]:
    records: List[Dict[str, object]] = []
    candidate_label = f'long_context/{variant}/{planner_reason} ctx={ctx} par=1'
    for attempt in (1, 2):
        check_cancelled(cancel_token)
        try:
            engine = app.active_engine_key_for_model(base_model)
        except Exception:
            engine = str(getattr(base_model, 'runtime', '') or '')
        if not deadline_has_candidate_budget(deadline, engine=engine):
            if progress:
                progress(f'smart frontier probe {candidate_label} skipped: benchmark budget exhausted')
            return False, True, records, completed
        candidate = configure_adaptive_candidate(base_model, profile, 'long_context', ctx, 1, variant)
        estimated_safe_ctx = candidate_safe_context_estimate(candidate, profile)
        command_preview = benchmark_command_preview(app, candidate)
        if progress:
            progress(
                f'smart frontier probe {candidate_label} attempt={attempt} '
                f'estimated_safe_ctx={estimated_safe_ctx}'
            )
        emit_benchmark_event(
            progress,
            'benchmark_probe',
            base_model,
            'server',
            message=f'smart frontier probe {candidate_label} attempt={attempt}',
            phase='smart frontier probes',
            completed=completed,
            total=total,
            candidate=candidate_label,
            command=command_preview,
        )
        record, ok = benchmark_frontier_probe_candidate(app, candidate, 'long_context', progress, cancel_token, deadline=deadline)
        completed += 1
        break_point = not ok and attempt == 2
        enrich_exhaustive_record(
            record,
            candidate,
            variant,
            attempt,
            estimated_safe_ctx,
            scan_level=planner_reason,
            break_point=break_point,
            measurement_type='probe',
            planner_reason=planner_reason,
        )
        records.append(record)
        emit_exhaustive_result(progress, base_model, record, completed, total, candidate_label)
        if ok:
            return True, False, records, completed
        if tq3_terminal_candidate_failure(record):
            if progress:
                progress(
                    f'smart frontier probe {candidate_label} failed with '
                    f'{record.get("failure_category") or record.get("status")}; not retrying TQ3 timeout/low-speed profile.'
                )
            return False, True, records, completed
        if attempt == 1 and progress:
            progress(f'smart frontier probe {candidate_label} failed once; retrying to confirm break...')
    return False, True, records, completed


def benchmark_exhaustive_profiles(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'server', progress, cancel_token)
    if not preflight_ok:
        return False, preflight_msg

    profile = app.hardware_profile(refresh=True)
    started_at = datetime.now().isoformat(timespec='seconds')
    run_id = f'server-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072) or 131072))
    chat_floor = chat_min_ctx_per_slot(model)
    opencode_floor = observed_opencode_context_floor(model)
    runtime_profiles = active_engine_runtime_profiles(app, model, profile, depth='full')
    total = max(
        len(runtime_profiles) * 2 if runtime_profiles else 0,
        24,
        SMART_FRONTIER_MAX_PROBES * 2 + 16,
    )
    raw_total = tq3_raw_presearch_case_total(runtime_profiles, 'full')
    if raw_total:
        total += raw_total
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    current: Optional[ModelConfig] = None
    completed = 0
    started_monotonic = time.monotonic()
    deadline = BenchmarkDeadline.from_end(started_monotonic + SMART_BENCHMARK_SOFT_BUDGET_SECONDS)
    budget_exhausted = False
    full_by_fingerprint: Dict[str, Dict[str, object]] = {}
    disabled_runtime_kv: set[Tuple[str, str]] = set()
    disabled_runtime_memory: set[Tuple[str, ...]] = set()

    running_model = ModelConfig(**asdict(model))
    if close_stale_running_benchmark_runs(running_model, started_at):
        model.benchmark_runs = list(running_model.benchmark_runs)
        model.default_benchmark_status = running_model.default_benchmark_status
        model.default_benchmark_at = running_model.default_benchmark_at
    running_model.default_benchmark_status = 'running'
    running_run = build_benchmark_run(run_id, 'server', 'running', [], {}, started_at, hardware=profile.short_summary())
    upsert_benchmark_run(running_model, running_run)
    app.add_or_update(running_model)

    pressure_payload = current_process_pressure_payload()
    moe_note = ''
    if model_is_moe(model):
        moe_note = (
            f'MoE detected from {getattr(model, "classification_source", "unknown") or "unknown"}: '
            f'{int(getattr(model, "expert_count", 0) or 0)} experts, '
            f'{int(getattr(model, "expert_used_count", 0) or 0)} active. '
            'Using conservative memory/offload search. '
        )
    start_msg = (
        f'smart bounded benchmark started: ctx={ctx_min}..{ctx_max}, '
        f'chat_floor={chat_floor}, opencode_floor={opencode_floor or "none"}, '
        f'soft_budget={SMART_BENCHMARK_SOFT_BUDGET_SECONDS // 60}m hard_deadline=on, '
        f'arch={architecture_label(model)}, runtime_profiles={len(runtime_profiles)}, '
        f'{moe_note}{profile.short_summary()} '
        f'{pressure_payload.get("process_pressure_detail", "")}'
    )
    if progress:
        progress(start_msg)
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'server',
        message=start_msg,
        phase='starting',
        completed=0,
        total=total,
    )

    def persist_raw_record(record: Dict[str, object]):
        nonlocal completed
        records.append(dict(record))
        completed += 1
        persist_running_benchmark_progress(
            app,
            model,
            run_id,
            'server',
            records,
            started_at,
            profile.short_summary(),
        )

    _raw_records, raw_promotions = run_tq3_raw_llama_bench_presearch(
        app,
        model,
        profile,
        runtime_profiles,
        'full',
        progress,
        cancel_token,
        on_record=persist_raw_record,
        completed_offset=completed,
        total=total,
        run_kind='server',
        deadline=deadline,
    )
    if raw_promotions:
        runtime_profiles = dedupe_runtime_profiles(raw_promotions + runtime_profiles)
        if progress:
            progress(f'TQ3 raw llama-bench promoted {len(raw_promotions)} candidate(s) into server validation')

    def optional_refinement_allowed() -> bool:
        return smart_should_continue_optional(started_monotonic, measured, model, profile)

    def run_full_measurement(
        objective: str,
        ctx: int,
        parallel: int,
        variant: str,
        planner_reason: str,
        optional: bool = False,
    ) -> Tuple[bool, bool]:
        nonlocal completed, current, total
        nonlocal budget_exhausted
        check_cancelled(cancel_token)
        if not deadline_has_candidate_budget(deadline, engine=app.active_engine_key_for_model(model)):
            budget_exhausted = True
            if progress:
                progress('smart bounded global budget exhausted; selecting from measured candidates')
            return False, True
        if optional and not optional_refinement_allowed():
            if progress:
                progress(f'smart bounded skipped optional {planner_reason}: soft budget reached and winners exist')
            return False, False
        current = configure_adaptive_candidate(model, profile, objective, ctx, parallel, variant)
        fingerprint = benchmark_config_fingerprint(current)
        if fingerprint in full_by_fingerprint:
            source = full_by_fingerprint[fingerprint]
            if progress:
                progress(
                    f'smart bounded reused {objective}/{variant} ctx={ctx} par={parallel} '
                    f'from {source.get("objective", "measured")}/{source.get("planner_reason", "full")}'
                )
            return True, False
        ok, broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
            app,
            model,
            profile,
            objective,
            ctx,
            parallel,
            variant,
            progress,
            cancel_token,
            completed,
            total,
            scan_level=planner_reason,
            planner_reason=planner_reason,
            measurement_type='full',
            deadline=deadline,
        )
        records.extend(new_records)
        measured.extend(new_measured)
        for item in new_measured:
            full_by_fingerprint[str(item.get('config_fingerprint', '') or fingerprint)] = item
        return ok, broke

    def run_frontier(variant: str, planner_reason: str = 'frontier') -> Tuple[List[int], List[int], int]:
        nonlocal completed, current, total
        successes: List[int] = []
        failures: List[int] = []
        tested = set()

        def probe(value: int) -> bool:
            nonlocal completed, current
            nonlocal budget_exhausted
            if not deadline_has_candidate_budget(deadline, engine=app.active_engine_key_for_model(model)):
                budget_exhausted = True
                return False
            value = max(ctx_min, min(ctx_max, round_context(value)))
            if value in tested:
                return value in successes
            tested.add(value)
            current = configure_adaptive_candidate(model, profile, 'long_context', value, 1, variant)
            ok, broke, new_records, completed = benchmark_smart_probe_with_retry(
                app,
                model,
                profile,
                value,
                variant,
                progress,
                cancel_token,
                completed,
                total,
                planner_reason=planner_reason,
                deadline=deadline,
            )
            records.extend(new_records)
            (successes if ok else failures).append(value)
            return ok

        emit_benchmark_event(
            progress,
            'benchmark_phase',
            model,
            'server',
            message=f'smart frontier search {variant}',
            phase=f'smart frontier search {variant}',
            completed=completed,
            total=total,
            candidate=variant,
        )
        probe_successes, probe_failures = adaptive_context_search(
            ctx_min,
            ctx_max,
            probe,
            max_probes=SMART_FRONTIER_MAX_PROBES,
        )
        successes = sorted(set(successes + probe_successes))
        failures = sorted(set(failures + probe_failures))
        if successes and failures and optional_refinement_allowed():
            first_break = min(ctx for ctx in failures if ctx > max(successes)) if any(ctx > max(successes) for ctx in failures) else min(failures)
            for ctx in smart_break_refinement_contexts(max(successes), first_break, tested):
                probe(ctx)
            successes = sorted(set(successes))
            failures = sorted(set(failures))
        break_ctx = min(failures) if failures else 0
        if progress:
            progress(
                f'smart frontier {variant}: {len(successes)} viable probe(s), '
                f'break={break_ctx or "none"}'
            )
        return successes, failures, break_ctx

    def run_fast_chat_race(ctx: int, variant: str):
        nonlocal completed, current
        nonlocal budget_exhausted
        max_parallel = max(1, min(4 if model_is_moe(model) else 16, int(getattr(profile, 'cpu_logical', 0) or 1)))
        parallel_values = [value for value in (1, 2, 4, 8, 16) if value <= max_parallel]
        best_parallel = 0
        best_tps = 0.0
        non_improving = 0
        tested_parallel = set()
        for parallel in parallel_values:
            check_cancelled(cancel_token)
            if not deadline_has_candidate_budget(deadline, engine=app.active_engine_key_for_model(model)):
                budget_exhausted = True
                break
            if ctx // max(1, parallel) < chat_floor:
                break
            ok, broke = run_full_measurement('fast_chat', ctx, parallel, variant, 'chat_parallel')
            tested_parallel.add(parallel)
            latest = [item for item in measured if item.get('status') == 'ok' and item.get('objective') == 'fast_chat' and int(item.get('ctx', 0) or 0) == ctx and int(item.get('parallel', 1) or 1) == parallel and str(item.get('variant', '') or 'default') == variant]
            tps = max([float(item.get('tokens_per_sec', 0.0) or 0.0) for item in latest] or [0.0])
            if ok and tps > best_tps * (1.0 + SMART_PARALLEL_IMPROVEMENT_THRESHOLD):
                best_tps = tps
                best_parallel = parallel
                non_improving = 0
            elif ok:
                non_improving += 1
            if broke or non_improving >= SMART_PARALLEL_NON_IMPROVING_LIMIT:
                break
        if best_parallel:
            for parallel in parallel_refinement_values(profile, best_parallel, tested_parallel):
                check_cancelled(cancel_token)
                if ctx // max(1, parallel) >= chat_floor:
                    run_full_measurement('fast_chat', ctx, parallel, variant, 'parallel_refine', optional=True)

    try:
        if runtime_profiles:
            if progress:
                progress(f'smart bounded using {len(runtime_profiles)} active-engine runtime profile(s)')
            emit_benchmark_event(
                progress,
                'benchmark_phase',
                model,
                'server',
                message=f'active-engine runtime profile search: {len(runtime_profiles)} candidate(s)',
                phase='runtime profile search',
                completed=completed,
                total=total,
            )
            fit_succeeded_engines: set[str] = set()
            fit_ceiling_by_engine: Dict[str, Tuple[int, str, str]] = {}
            fixed_fit_engine_skipped = 0
            for runtime_profile in runtime_profiles:
                skip_reason = runtime_profile_skip_reason(runtime_profile, disabled_runtime_kv)
                if not skip_reason:
                    skip_reason = runtime_profile_memory_skip_reason(runtime_profile, disabled_runtime_memory)
                if skip_reason:
                    if progress:
                        progress(f'smart bounded skipped {runtime_profile.name}: {skip_reason}')
                    continue
                runtime_engine = str(getattr(runtime_profile, 'engine_id', '') or '')
                fit_phase = runtime_profile_fit_phase(runtime_profile)
                if fit_phase == 'context_growth' and runtime_engine not in fit_succeeded_engines:
                    if progress:
                        progress(f'smart bounded skipped {runtime_profile.name}: weight-fit discovery has no viable ceiling yet')
                    continue
                if fit_phase == 'context_growth' and runtime_engine in fit_ceiling_by_engine:
                    runtime_profile = runtime_profile_with_fit_ceiling(runtime_profile, *fit_ceiling_by_engine[runtime_engine])
                if runtime_engine in fit_succeeded_engines and runtime_profile_is_fixed_gpu_layers(runtime_profile):
                    fixed_fit_engine_skipped += 1
                    continue
                if not deadline_has_candidate_budget(deadline, runtime_profile=runtime_profile):
                    budget_exhausted = True
                    if progress:
                        progress('smart bounded runtime profile budget reached; selecting from measured candidates')
                    break
                objective = 'opencode_ready' if runtime_profile.ctx_size >= 16384 else 'long_context'
                ok, _broke, new_records, new_measured, completed = benchmark_runtime_profile_with_retry(
                    app,
                    model,
                    runtime_profile,
                    objective,
                    progress,
                    cancel_token,
                    completed,
                    total,
                    max_attempts=2,
                    benchmark_depth='full',
                    deadline=deadline,
                )
                records.extend(new_records)
                measured.extend(new_measured)
                for item in new_measured:
                    full_by_fingerprint[str(item.get('config_fingerprint', '') or '')] = item
                if ok and runtime_profile_uses_fit(runtime_profile):
                    fit_succeeded_engines.add(runtime_engine)
                    if fit_phase == 'weight_fit' or runtime_engine not in fit_ceiling_by_engine:
                        fit_ceiling_by_engine[runtime_engine] = fit_ceiling_from_records(new_records)
                    if progress:
                        progress(f'smart bounded {runtime_engine} fit profile succeeded; skipping fixed-NGL fallback probes')
                elif not ok and new_records:
                    memory_key = runtime_profile_memory_disable_key(new_records[-1], runtime_profile)
                    if memory_key and memory_key not in disabled_runtime_memory:
                        disabled_runtime_memory.add(memory_key)
                        if progress:
                            progress(f'smart bounded pruned memory-risky runtime shape after {new_records[-1].get("failure_category")}')
                    disable_key = runtime_profile_kv_disable_key(new_records[-1], runtime_profile)
                    if disable_key[0] and disable_key not in disabled_runtime_kv:
                        disabled_runtime_kv.add(disable_key)
                        if progress:
                            progress(f'smart bounded disabled {disable_key[0]} {disable_key[1]} after incompatible KV startup')
            if fixed_fit_engine_skipped and progress:
                progress(f'smart bounded skipped {fixed_fit_engine_skipped} fixed-NGL profile(s) after fit success')
        else:
            default_successes, default_failures, default_break = run_frontier('default')
            if not default_successes:
                default_successes = [ctx_min]
            default_contexts = smart_measurement_contexts(
                default_successes,
                default_failures,
                ctx_min,
                ctx_max,
                chat_floor,
                opencode_floor,
            )
            for ctx in default_contexts:
                run_full_measurement('long_context', ctx, 1, 'default', 'frontier')

            if opencode_floor:
                opencode_ctx = _nearest_context_at_or_above(default_successes, opencode_floor)
                if opencode_ctx and opencode_ctx not in default_contexts:
                    run_full_measurement('long_context', opencode_ctx, 1, 'default', 'opencode_floor', optional=True)

            long_records = [
                row for row in records
                if row.get('status') == 'ok'
                and row.get('objective') == 'long_context'
                and row.get('variant') == 'default'
                and row.get('measurement_type') == 'full'
            ]
            tested_full_contexts = {int(row.get('ctx', 0) or 0) for row in long_records}
            for ctx in context_knee_refinement_contexts(long_records, tested_full_contexts, ctx_max)[:2]:
                run_full_measurement('long_context', ctx, 1, 'default', 'speed_knee', optional=True)

            for ctx in smart_fast_contexts(default_successes, chat_floor):
                run_fast_chat_race(ctx, 'default')

            default_best_ctx = max(default_successes or [0])
            if smart_should_try_q8(model, profile, default_best_ctx, default_break) and optional_refinement_allowed():
                q8_successes, q8_failures, _q8_break = run_frontier('q8_kv', planner_reason='q8_probe')
                q8_contexts = smart_measurement_contexts(
                    q8_successes,
                    q8_failures,
                    ctx_min,
                    ctx_max,
                    chat_floor,
                    opencode_floor,
                )[:3]
                for ctx in q8_contexts:
                    run_full_measurement('long_context', ctx, 1, 'q8_kv', 'q8_probe', optional=True)
                for ctx in smart_fast_contexts(q8_successes, chat_floor)[:1]:
                    run_fast_chat_race(ctx, 'q8_kv')
    except CancelledError:
        if current is not None:
            app.stop(current, managed_only=True)
            records.append(enrich_exhaustive_record(
                adaptive_record_from_candidate(current, 'smart_bounded', 'aborted', detail='user requested abort'),
                current,
                str(getattr(current, 'variant', 'default') or 'default'),
                1,
                0,
                measurement_type='full',
                planner_reason='abort',
            ))
        ended_at = datetime.now().isoformat(timespec='seconds')
        aborted_model = ModelConfig(**asdict(model))
        aborted_model.last_benchmark_results = records
        aborted_model.default_benchmark_status = 'aborted'
        aborted_model.default_benchmark_at = ended_at
        run = build_benchmark_run(run_id, 'server', 'aborted', records, {}, started_at, ended_at, profile.short_summary())
        upsert_benchmark_run(aborted_model, run)
        app.add_or_update(aborted_model)
        msg = '⚠ aborted; managed processes stopped'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_aborted',
            model,
            'server',
            message=msg,
            phase='aborted',
            completed=completed,
            total=completed,
            records=records,
        )
        return False, msg

    winners = select_measured_profiles(model, measured, profile)
    propagate_rejection_reasons(records, measured)
    ended_at = datetime.now().isoformat(timespec='seconds')
    if not winners:
        saved, preserved = failed_benchmark_model_state(app, model, records, ended_at)
        saved.benchmark_fingerprint = app.model_fingerprint(saved)
        run = build_benchmark_run(run_id, 'server', 'failed', records, {}, started_at, ended_at, profile.short_summary())
        upsert_benchmark_run(saved, run)
        app.add_or_update(saved)
        if preserved:
            msg = preserved_profiles_message('smart bounded benchmark found no better working candidate', records)
        else:
            summary = benchmark_failure_summary(records, 'no measured candidates completed')
            msg = f'❌ smart bounded benchmark failed: {summary}'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_error',
            model,
            'server',
            message=msg,
            phase='failed',
            completed=completed,
            total=completed,
            records=records,
        )
        return False, msg

    missing_reason = (
        'No valid candidate measured before benchmark budget expired'
        if budget_exhausted or deadline.expired()
        else 'No valid candidate measured for this profile in the bounded benchmark run'
    )
    winners = fill_missing_adaptive_profiles(winners, missing_reason, status='skipped_budget' if budget_exhausted or deadline.expired() else 'skipped')
    run_status = 'done' if adaptive_profiles_complete(winners) else ('skipped_budget' if not any(str((winners.get(key) or {}).get('status', '')) == 'ok' for key in ADAPTIVE_PROFILE_KEYS) else 'partial')
    winners = attach_profile_frontier(model, measured, winners, profile, generated_at=ended_at)
    annotate_spectrum_records(records, winners)
    saved = ModelConfig(**asdict(model))
    saved.last_benchmark_results = records
    saved.measured_profiles = winners
    saved.benchmark_fingerprint = app.model_fingerprint(saved)
    saved.default_benchmark_at = ended_at
    run = build_benchmark_run(run_id, 'server', run_status, records, winners, started_at, ended_at, profile.short_summary())
    upsert_benchmark_run(saved, run)

    auto_profile = winners['auto']
    apply_measured_profile(saved, 'auto')
    saved.measured_profiles = winners
    saved.last_benchmark_tokens_per_sec = round(float(auto_profile.get('tokens_per_sec', 0.0) or 0.0), 2)
    saved.last_benchmark_seconds = round(float(auto_profile.get('seconds', 0.0) or 0.0), 2)
    saved.last_benchmark_profile = (
        f'auto/smart-bounded {saved.last_benchmark_tokens_per_sec:.2f} tok/s '
        f'ctx={auto_profile.get("ctx")} slot={auto_profile.get("ctx_per_slot")} {profile.short_summary()}'
    )
    saved.default_benchmark_status = run_status
    app.add_or_update(saved)
    sync_msg = sync_opencode_after_tuning(app)
    prefix = '✅ smart bounded profiles saved' if run_status == 'done' else '⚠ smart bounded partial profiles saved'
    msg = (
        f'{prefix}: {adaptive_profile_status_text(winners, "fast_chat")}, '
        f'{adaptive_profile_status_text(winners, "long_context")}, '
        f'{opencode_profile_status_text(winners)}, '
        f'{adaptive_profile_status_text(winners, "auto")} | status={run_status} | {sync_msg}'
    )
    if progress:
        progress(msg)
    emit_benchmark_event(
        progress,
        'benchmark_done',
        saved,
        'server',
        message=msg,
        phase='complete',
        completed=completed,
        total=completed,
        records=records,
    )
    return True, msg


def _safe_tuning_path_part(value: object) -> str:
    text = str(value or '').strip()
    clean = re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('._')
    return clean or 'unknown'


def _moe_tuning_layer_count(model: ModelConfig) -> int:
    try:
        return max(0, int(gguf_layer_count(model) or 0))
    except Exception:
        return 0


def _moe_tuning_capabilities(app, runtime_profile: RuntimeProfile):
    try:
        return app.engine_capabilities()
    except Exception:
        return default_engine_capabilities(str(getattr(runtime_profile, 'engine_id', '') or 'llama.cpp'))


def _moe_tuning_baseline_profile(app, model: ModelConfig, depth: str) -> RuntimeProfile:
    profile = app.runtime_profile_from_model(
        model,
        int(getattr(model, 'ctx', 0) or 0),
        int(getattr(model, 'parallel', 1) or 1),
        int(getattr(model, 'ngl', 0) or 0),
    )
    return replace(profile, name='baseline_current', benchmark_depth=depth)


def _moe_tuning_effective_args(
    app,
    model: ModelConfig,
    runtime_profile: RuntimeProfile,
    depth: str,
) -> List[str]:
    candidate_model = model_for_runtime_profile(model, runtime_profile)
    capabilities = _moe_tuning_capabilities(app, runtime_profile)
    launch_profile = build_benchmark_launch_profile(
        candidate_model,
        runtime_profile,
        capabilities,
        purpose='moe_tuning',
        depth=depth,
    )
    return benchmark_effective_server_args(app, candidate_model, runtime_profile, launch_profile)


def moe_context_bucket_specs(model: ModelConfig, depth: str = 'full') -> List[Dict[str, object]]:
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', getattr(model, 'ctx', 32768)) or 32768))
    raw = [
        ('fast_chat', '8k', 8192),
        ('auto', '32k', 32768),
        ('hermes_ready', '64k', 65536),
        ('long_context', '131k', 131072),
    ]
    if str(depth or '').strip().lower() != 'full':
        raw = raw[:2]
    specs: List[Dict[str, object]] = []
    seen_ctx = set()
    for profile_key, label, target in raw:
        ctx = max(ctx_min, min(ctx_max, int(target)))
        if ctx in seen_ctx:
            continue
        seen_ctx.add(ctx)
        specs.append({'profile_key': profile_key, 'label': label, 'ctx': ctx})
    return specs


def _context_validation_n_cpu_moe_values(winner_n_cpu_moe: int, layer_count: int, depth: str) -> List[int]:
    center = max(0, int(winner_n_cpu_moe or 0))
    layers = max(0, int(layer_count or 0))
    if center <= 0 or layers <= 0:
        return []
    deltas = (-4, -2, 0, 2, 4) if str(depth or '').strip().lower() == 'full' else (0, 2, 4)
    values = {
        max(1, min(layers, center + delta))
        for delta in deltas
    }
    return sorted(values, key=lambda value: (abs(value - center), value))


def generate_context_moe_validation_candidates(
    baseline: RuntimeProfile,
    winner: Dict[str, object],
    capabilities,
    layer_count: int,
    bucket_ctx: int,
    bucket_key: str,
    depth: str,
) -> List[TuningCandidate]:
    baseline = replace(baseline, ctx_size=max(1, int(bucket_ctx or baseline.ctx_size or 1)))
    candidates: List[TuningCandidate] = []
    winner_payload = profile_moe_placement(winner)
    winner_candidate = moe_placement_candidate_from_payload(winner_payload)
    if winner_candidate is not None:
        candidates.append(TuningCandidate(
            name=f'{bucket_key}_{winner_candidate.name}',
            runtime_profile=replace(
                baseline,
                name=f'{bucket_key}_{winner_candidate.name}',
                gpu_layers=baseline.gpu_layers,
                placement_strategy=winner_candidate.name,
                cpu_moe=winner_candidate.cpu_moe,
                n_cpu_moe=winner_candidate.n_cpu_moe,
                tensor_overrides=winner_candidate.tensor_overrides,
            ),
            source='context_validation',
            risk='baseline',
            expected_effect=f'validate current MoE placement at {bucket_ctx} ctx',
            phase='context_validation',
        ))
    if bool(getattr(capabilities, 'supports_n_cpu_moe', False)):
        for value in _context_validation_n_cpu_moe_values(
            int(winner_payload.get('n_cpu_moe', 0) or 0),
            layer_count,
            depth,
        ):
            name = f'{bucket_key}_n_cpu_moe_{value}'
            candidates.append(TuningCandidate(
                name=name,
                runtime_profile=replace(
                    baseline,
                    name=name,
                    gpu_layers=999,
                    placement_strategy=f'n_cpu_moe_{value}',
                    n_cpu_moe=value,
                    cpu_moe=False,
                    tensor_overrides=(),
                ),
                source='context_validation',
                risk='normal',
                expected_effect=f'validate n_cpu_moe={value} at {bucket_ctx} ctx',
                phase='context_validation',
            ))
    if bool(getattr(capabilities, 'supports_cpu_moe', False)):
        name = f'{bucket_key}_cpu_moe_all'
        candidates.append(TuningCandidate(
            name=name,
            runtime_profile=replace(
                baseline,
                name=name,
                gpu_layers=999,
                placement_strategy='cpu_moe_all',
                cpu_moe=True,
                n_cpu_moe=0,
                tensor_overrides=(),
            ),
            source='context_validation',
            risk='safe',
            expected_effect=f'validate all experts on CPU at {bucket_ctx} ctx',
            phase='context_validation',
        ))

    deduped: List[TuningCandidate] = []
    seen = set()
    limit = 6 if str(depth or '').strip().lower() == 'full' else 4
    for candidate in candidates:
        profile = candidate.runtime_profile
        key = (profile.cpu_moe, profile.n_cpu_moe, tuple(profile.tensor_overrides or ()), profile.ctx_size)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return deduped


def _record_vram_headroom(record: Dict[str, object]) -> int:
    values: List[int] = []
    for key in ('memory_guardrail_min_gpu_memory_free', 'gpu_memory_free', 'vram_headroom_bytes'):
        try:
            value = int(record.get(key, 0) or 0)
        except Exception:
            value = 0
        if value > 0:
            values.append(value)
    return min(values) if values else 0


def _record_peak_ram(record: Dict[str, object]) -> int:
    for key in ('process_rss', 'memory_used', 'ram_used', 'peak_ram_bytes'):
        try:
            value = int(record.get(key, 0) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _enrich_moe_tuning_records(
    records: Sequence[Dict[str, object]],
    candidate: TuningCandidate,
    baseline_profile: RuntimeProfile,
    run_id: str,
    objective,
    effective_args: Sequence[str],
    early_stop_text: str = '',
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    profile = candidate.runtime_profile
    args = [str(item) for item in list(effective_args or [])]
    args_preview = shlex.join(args) if args else ''
    for record in records:
        row = dict(record)
        row['benchmark_kind'] = 'moe_tuning'
        row['benchmark_purpose'] = 'moe_tuning'
        row['objective'] = 'moe_placement'
        row['measured_profile_key'] = 'moe_placement'
        row['tuning_run_id'] = run_id
        row['measured_candidate_name'] = candidate.name
        row['tuning_phase'] = candidate.phase
        row['tuning_source'] = candidate.source
        row['tuning_risk'] = candidate.risk
        row['tuning_expected_effect'] = candidate.expected_effect
        row['target_vram_headroom_bytes'] = int(objective.target_vram_headroom_bytes)
        row['vram_headroom_bytes'] = _record_vram_headroom(row)
        row['peak_ram_bytes'] = _record_peak_ram(row)
        row['effective_server_args'] = args
        row['effective_server_args_preview'] = args_preview
        row['ngl_required_for_moe_tuning'] = bool(
            profile.gpu_layers is not None
            and profile.gpu_layers != baseline_profile.gpu_layers
        )
        if early_stop_text:
            row['early_stop_reason'] = early_stop_text
        row['tuning_score'] = score_tuning_record(row, objective, candidate.risk)
        if str(row.get('status', '') or '').lower() == 'ok':
            row['selection_reason'] = 'measured successful MoE placement candidate'
        else:
            row['selection_reason'] = str(row.get('failure_category', '') or row.get('status', '') or 'failed')
        enriched.append(row)
    return enriched


def _run_context_aware_moe_validation(
    app,
    model: ModelConfig,
    baseline_profile: RuntimeProfile,
    capabilities,
    layer_count: int,
    depth_key: str,
    primary_winner: Dict[str, object],
    base_objective,
    run_id: str,
    progress: Optional[Callable[[object], None]],
    cancel_token: Optional[CancelToken],
    completed: int,
    total: int,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], int]:
    if not primary_winner or str(depth_key or '').strip().lower() != 'full':
        return [], {}, completed
    records: List[Dict[str, object]] = []
    placements: Dict[str, Dict[str, object]] = {}
    for bucket in moe_context_bucket_specs(model, depth_key):
        check_cancelled(cancel_token)
        profile_key = str(bucket.get('profile_key') or '')
        label = str(bucket.get('label') or profile_key)
        ctx = max(1, int(bucket.get('ctx', 0) or 0))
        bucket_objective = TuningObjective(
            purpose='moe_placement_context',
            target_context=ctx,
            min_context=ctx,
            target_vram_headroom_bytes=int(base_objective.target_vram_headroom_bytes),
            max_trials=6,
            depth=depth_key,
        )
        candidates = generate_context_moe_validation_candidates(
            baseline_profile,
            primary_winner,
            capabilities,
            layer_count,
            ctx,
            profile_key,
            depth_key,
        )
        if not candidates:
            continue
        if progress:
            progress(f'MoE context validation {profile_key} ctx={ctx}: {len(candidates)} candidate(s)')
        bucket_records: List[Dict[str, object]] = []
        for candidate in candidates:
            check_cancelled(cancel_token)
            runtime_profile = candidate.runtime_profile
            effective_args = _moe_tuning_effective_args(app, model, runtime_profile, depth_key)
            ok, _broke, new_records, _new_measured, completed = benchmark_runtime_profile_with_retry(
                app,
                model,
                runtime_profile,
                'moe_placement_context',
                progress,
                cancel_token,
                completed,
                total,
                run_kind='moe_tuning',
                max_attempts=1,
                benchmark_depth=depth_key,
                benchmark_purpose='moe_context_validation',
            )
            enriched = _enrich_moe_tuning_records(
                new_records,
                candidate,
                baseline_profile,
                run_id,
                bucket_objective,
                effective_args,
            )
            for row in enriched:
                row['objective'] = 'moe_placement_context'
                row['benchmark_purpose'] = 'moe_context_validation'
                row['measured_profile_key'] = f'moe_placement:{profile_key}'
                row['profile_key'] = profile_key
                row['context_bucket'] = label
                row['target_context'] = ctx
                row['context_validation'] = True
                row['context_validation_winner'] = False
                row['tuning_score'] = score_tuning_record(row, bucket_objective, candidate.risk)
            records.extend(enriched)
            bucket_records.extend(enriched)
            if progress:
                suffix = 'ok' if ok else (enriched[-1].get('failure_category') if enriched else 'failed')
                progress(f'MoE context validation {profile_key}/{candidate.name}: {suffix}')
        bucket_winner = select_measured_tuning_winner(bucket_records, bucket_objective)
        if bucket_winner:
            for row in records:
                if (
                    row.get('profile_key') == profile_key
                    and row.get('measured_candidate_name') == bucket_winner.get('measured_candidate_name')
                ):
                    row['context_validation_winner'] = True
            payload = moe_placement_payload(bucket_winner, source_name='context_validation')
            payload.update({
                'profile_key': profile_key,
                'context_bucket': label,
                'target_context': ctx,
                'ctx': int(bucket_winner.get('ctx', ctx) or ctx),
                'tokens_per_sec': float(bucket_winner.get('tokens_per_sec', 0.0) or 0.0),
                'measured_candidate_name': str(bucket_winner.get('measured_candidate_name', '') or ''),
                'tuning_run_id': run_id,
            })
            placements[profile_key] = payload
    return records, placements, completed


def _tuning_candidate_payload(candidate: TuningCandidate) -> Dict[str, object]:
    profile = candidate.runtime_profile
    return {
        'name': candidate.name,
        'source': candidate.source,
        'risk': candidate.risk,
        'phase': candidate.phase,
        'expected_effect': candidate.expected_effect,
        'runtime_profile': profile.name,
        'ctx': profile.ctx_size,
        'ngl': profile.gpu_layers,
        'parallel': profile.parallel,
        'kv_preset': profile.kv_preset,
        'cpu_moe': profile.cpu_moe,
        'n_cpu_moe': profile.n_cpu_moe,
        'tensor_overrides': list(profile.tensor_overrides),
    }


def _write_moe_tuning_logs(
    app,
    model: ModelConfig,
    run_id: str,
    engine: str,
    capabilities,
    objective,
    candidates: Sequence[TuningCandidate],
    records: Sequence[Dict[str, object]],
    winner: Dict[str, object],
    warnings: Sequence[str],
    early_stop_text: str,
    layer_count: int,
    profile: HardwareProfile,
) -> Tuple[str, str]:
    root = (
        CACHE_DIR
        / 'tuning'
        / _safe_tuning_path_part(getattr(model, 'id', 'model'))
        / _safe_tuning_path_part(engine or 'engine')
    )
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    json_path = root / f'{stamp}-{_safe_tuning_path_part(run_id)}.json'
    log_path = root / f'{stamp}-{_safe_tuning_path_part(run_id)}.log'
    try:
        fingerprint = app.model_fingerprint(model)
    except Exception:
        fingerprint = benchmark_config_fingerprint(model)
    payload = {
        'run_id': run_id,
        'model_id': getattr(model, 'id', ''),
        'model_path': getattr(model, 'path', ''),
        'model_fingerprint': fingerprint,
        'engine_id': engine,
        'runtime_capabilities': asdict(capabilities) if hasattr(capabilities, '__dataclass_fields__') else {},
        'runtime_binary': (
            app.runtime_server_command(getattr(model, 'runtime', 'llama.cpp'))
            if hasattr(app, 'runtime_server_command')
            else getattr(model, 'runtime', 'llama.cpp')
        ),
        'runtime_version': app.runtime_binary_version(model) if hasattr(app, 'runtime_binary_version') else '',
        'hardware_profile': asdict(profile) if hasattr(profile, '__dataclass_fields__') else {},
        'objective': asdict(objective),
        'layer_count': int(layer_count or 0),
        'candidates': [_tuning_candidate_payload(candidate) for candidate in candidates],
        'results': [dict(row) for row in records],
        'winner': dict(winner),
        'warnings': list(warnings or []),
        'early_stop_reason': early_stop_text,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')
    lines = [
        f'MoE tuning run {run_id}',
        f'model={getattr(model, "id", "")} engine={engine} layer_count={layer_count or "unknown"}',
        f'winner={winner.get("measured_candidate_name", "-") if winner else "-"}',
    ]
    if early_stop_text:
        lines.append(f'early_stop={early_stop_text}')
    for warning in warnings:
        lines.append(f'warning={warning}')
    for row in records:
        lines.append(
            'candidate={candidate} status={status} tps={tps:.2f} headroom={headroom} failure={failure}'.format(
                candidate=row.get('measured_candidate_name', row.get('runtime_profile', '-')),
                status=row.get('status', '-'),
                tps=float(row.get('tokens_per_sec', 0.0) or 0.0),
                headroom=int(row.get('vram_headroom_bytes', 0) or 0),
                failure=row.get('failure_category', '') or row.get('detail', ''),
            )
        )
    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(json_path), str(log_path)


def _moe_tuning_warnings(records: Sequence[Dict[str, object]], layer_count: int, early_stop_text: str) -> List[str]:
    warnings: List[str] = []
    if layer_count <= 0:
        warnings.append('layer count unavailable; n_cpu_moe ladder was skipped')
    failed = [
        row for row in records
        if str(row.get('status', '') or '').lower() not in ('ok', 'probe ok', 'skipped')
    ]
    oom_failures = [
        row for row in failed
        if str(row.get('failure_category', '') or '') in ('CUDA_OOM_WEIGHTS', 'CUDA_OOM_KV', 'MEMORY_GUARDRAIL')
    ]
    if oom_failures:
        names = ', '.join(str(row.get('measured_candidate_name', '') or row.get('runtime_profile', 'candidate')) for row in oom_failures[:3])
        warnings.append(f'unsafe/OOM candidate(s): {names}')
    experimental = [
        row for row in records
        if str(row.get('tuning_risk', '') or '').lower() == 'experimental'
    ]
    if experimental:
        warnings.append('tensor override mode is experimental')
    if early_stop_text:
        warnings.append(early_stop_text)
    deduped: List[str] = []
    for item in warnings:
        clean = compact_message(item)
        if clean and clean not in deduped:
            deduped.append(clean)
    return deduped


def benchmark_moe_placement_tuning(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
    depth: str = 'fast',
) -> Tuple[bool, str]:
    depth_key = 'full' if str(depth or '').strip().lower() == 'full' else 'fast'
    profile = app.hardware_profile(refresh=True)
    baseline_profile = _moe_tuning_baseline_profile(app, model, depth_key)
    capabilities = _moe_tuning_capabilities(app, baseline_profile)
    try:
        engine = app.active_engine_key_for_model(model)
    except Exception:
        engine = baseline_profile.engine_id
    eligibility = moe_tuning_eligibility_reason(model, profile, capabilities, engine)
    if eligibility:
        msg = f'MoE tuning skipped: {eligibility}'
        if progress:
            progress(msg)
        return False, msg

    preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'moe_tuning', progress, cancel_token)
    if not preflight_ok:
        return False, preflight_msg

    profile = app.hardware_profile(refresh=True)
    started_at = datetime.now().isoformat(timespec='seconds')
    run_id = f'moe-tuning-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    layer_count = _moe_tuning_layer_count(model)
    objective = tuning_objective_for_model(model, profile, depth_key)
    coarse_candidates = generate_moe_tuning_candidates(
        baseline_profile,
        capabilities,
        profile,
        layer_count,
        depth=depth_key,
    )
    total = max(1, objective.max_trials)
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    completed = 0
    early_stop_text = ''
    extra_probe_used = False
    probe_after_unsafe_pending = False
    all_candidates: List[TuningCandidate] = list(coarse_candidates)
    current_candidate_model: Optional[ModelConfig] = None
    primary_winner: Dict[str, object] = {}
    profile_moe_placements: Dict[str, Dict[str, object]] = {}

    running_model = ModelConfig(**asdict(model))
    if close_stale_running_benchmark_runs(running_model, started_at):
        model.benchmark_runs = list(running_model.benchmark_runs)
    running_run = build_benchmark_run(run_id, 'moe_tuning', 'running', [], {}, started_at, hardware=profile.short_summary())
    running_run['depth'] = depth_key
    upsert_benchmark_run(running_model, running_run)
    app.add_or_update(running_model)

    start_msg = (
        f'MoE placement tuning started: depth={depth_key}, candidates={len(coarse_candidates)}, '
        f'layer_count={layer_count or "unknown"}, engine={engine}, {profile.short_summary()}'
    )
    if progress:
        progress(start_msg)
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'moe_tuning',
        message=start_msg,
        phase='moe tuning',
        completed=0,
        total=total,
    )

    def measured_candidate_names() -> set[str]:
        return {
            str(row.get('measured_candidate_name', '') or '')
            for row in records
            if str(row.get('measured_candidate_name', '') or '')
        }

    def persist_progress():
        running = ModelConfig(**asdict(model))
        running_run_local = build_benchmark_run(
            run_id,
            'moe_tuning',
            'running',
            records,
            {},
            started_at,
            hardware=profile.short_summary(),
        )
        running_run_local['depth'] = depth_key
        if early_stop_text:
            running_run_local['early_stop_reason'] = early_stop_text
        upsert_benchmark_run(running, running_run_local)
        app.add_or_update(running)

    def run_candidate(candidate: TuningCandidate) -> bool:
        nonlocal completed, early_stop_text, extra_probe_used, probe_after_unsafe_pending, current_candidate_model
        check_cancelled(cancel_token)
        if len(measured_candidate_names()) >= objective.max_trials:
            return False
        was_probe_after_unsafe = probe_after_unsafe_pending
        probe_after_unsafe_pending = False
        runtime_profile = candidate.runtime_profile
        current_candidate_model = model_for_runtime_profile(model, runtime_profile)
        effective_args = _moe_tuning_effective_args(app, model, runtime_profile, depth_key)
        if progress:
            progress(f'MoE tuning candidate {candidate.name} ({candidate.phase})')
        ok, _broke, new_records, new_measured, completed = benchmark_runtime_profile_with_retry(
            app,
            model,
            runtime_profile,
            'moe_placement',
            progress,
            cancel_token,
            completed,
            total,
            run_kind='moe_tuning',
            max_attempts=1,
            benchmark_depth=depth_key,
            benchmark_purpose='moe_tuning',
        )
        enriched = _enrich_moe_tuning_records(
            new_records,
            candidate,
            baseline_profile,
            run_id,
            objective,
            effective_args,
            early_stop_text,
        )
        records.extend(enriched)
        measured.extend(new_measured)
        persist_progress()
        if was_probe_after_unsafe:
            if not early_stop_text:
                early_stop_text = 'full stopped after one nearby probe following unsafe n_cpu_moe value'
            if records:
                records[-1]['early_stop_reason'] = early_stop_text
            return False
        if not ok and enriched:
            stop_now, next_extra_probe_used, reason = early_stop_reason(enriched[-1], depth_key, extra_probe_used)
            if reason:
                early_stop_text = reason
                records[-1]['early_stop_reason'] = early_stop_text
            extra_probe_used = next_extra_probe_used
            if stop_now:
                return False
            if next_extra_probe_used:
                probe_after_unsafe_pending = True
        return True

    try:
        for candidate in coarse_candidates:
            if not run_candidate(candidate):
                break

        coarse_winner = select_measured_tuning_winner(records, objective)
        refinements: List[TuningCandidate] = []
        if coarse_winner and not early_stop_text and layer_count > 0 and int(coarse_winner.get('n_cpu_moe', 0) or 0) > 0:
            refinements = [
                candidate for candidate in generate_refinement_candidates(baseline_profile, coarse_winner, layer_count)
                if candidate.name not in measured_candidate_names()
            ]
            remaining = max(0, objective.max_trials - len(measured_candidate_names()))
            refinements = refinements[:remaining]
        if refinements and progress:
            progress(f'MoE tuning refinement around {coarse_winner.get("measured_candidate_name")}: {len(refinements)} candidate(s)')
        all_candidates.extend(refinements)
        for candidate in refinements:
            if not run_candidate(candidate):
                break

        primary_winner = select_measured_tuning_winner(records, objective)
        if primary_winner:
            context_records, profile_moe_placements, completed = _run_context_aware_moe_validation(
                app,
                model,
                baseline_profile,
                capabilities,
                layer_count,
                depth_key,
                primary_winner,
                objective,
                run_id,
                progress,
                cancel_token,
                completed,
                total,
            )
            records.extend(context_records)
    except CancelledError:
        if current_candidate_model is not None:
            app.stop(current_candidate_model, managed_only=True)
            records.append(adaptive_record_from_candidate(
                current_candidate_model,
                'moe_placement',
                'aborted',
                detail='user requested abort',
            ))
        ended_at = datetime.now().isoformat(timespec='seconds')
        aborted_model = ModelConfig(**asdict(model))
        run = build_benchmark_run(run_id, 'moe_tuning', 'aborted', records, {}, started_at, ended_at, profile.short_summary())
        if early_stop_text:
            run['early_stop_reason'] = early_stop_text
        upsert_benchmark_run(aborted_model, run)
        app.add_or_update(aborted_model)
        msg = 'MoE tuning aborted; managed processes stopped'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_aborted',
            model,
            'moe_tuning',
            message=msg,
            phase='aborted',
            completed=completed,
            total=completed,
            records=records,
        )
        return False, msg

    winner = primary_winner or select_measured_tuning_winner(records, objective)
    ended_at = datetime.now().isoformat(timespec='seconds')
    warnings = _moe_tuning_warnings(records, layer_count, early_stop_text)
    status = 'done' if winner else 'failed'
    log_json = ''
    log_text = ''
    try:
        log_json, log_text = _write_moe_tuning_logs(
            app,
            model,
            run_id,
            engine,
            capabilities,
            objective,
            all_candidates,
            records,
            winner,
            warnings,
            early_stop_text,
            layer_count,
            profile,
        )
    except Exception as exc:
        warnings.append(f'tuning log write failed: {compact_message(str(exc))}')

    saved = ModelConfig(**asdict(model))
    measured_profiles = dict(getattr(saved, 'measured_profiles', {}) or {})
    winners: Dict[str, Dict[str, object]] = {}
    if winner:
        winner_profile = dict(winner)
        winner_profile['status'] = 'ok'
        winner_profile['tuning_run_id'] = run_id
        winner_profile['benchmark_depth'] = depth_key
        winner_profile['measured_profile_key'] = 'moe_placement'
        winner_profile['tuning_summary'] = 'fastest measured MoE placement candidate above safety scoring gates'
        if profile_moe_placements:
            winner_profile['profile_moe_placements'] = {
                key: dict(value)
                for key, value in profile_moe_placements.items()
            }
            winner_profile['context_validation_summary'] = (
                f'{len(profile_moe_placements)} context bucket placement(s) validated'
            )
        if log_json:
            winner_profile['tuning_log_json'] = log_json
        if log_text:
            winner_profile['tuning_log_path'] = log_text
        if early_stop_text:
            winner_profile['early_stop_reason'] = early_stop_text
        if warnings:
            winner_profile['warnings'] = list(warnings)
        measured_profiles['moe_placement'] = winner_profile
        winners['moe_placement'] = winner_profile
    saved.measured_profiles = measured_profiles
    run = build_benchmark_run(run_id, 'moe_tuning', status, records, winners, started_at, ended_at, profile.short_summary())
    run['depth'] = depth_key
    run['warnings'] = list(warnings)
    if profile_moe_placements:
        run['profile_moe_placements'] = {
            key: dict(value)
            for key, value in profile_moe_placements.items()
        }
    if early_stop_text:
        run['early_stop_reason'] = early_stop_text
    if log_json:
        run['tuning_log_json'] = log_json
    if log_text:
        run['tuning_log_path'] = log_text
    upsert_benchmark_run(saved, run)
    app.add_or_update(saved)

    if not winner:
        summary = benchmark_failure_summary(records, 'no measured MoE placement candidates completed')
        msg = f'MoE placement tuning failed: {summary}'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_error',
            model,
            'moe_tuning',
            message=msg,
            phase='failed',
            completed=completed,
            total=completed,
            records=records,
        )
        return False, msg

    candidate_name = str(winner.get('measured_candidate_name', '') or winner.get('runtime_profile', '') or 'candidate')
    msg = (
        f'MoE placement recommendation saved: {candidate_name} '
        f'{float(winner.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s '
        f'headroom={int(winner.get("vram_headroom_bytes", 0) or 0) // (1024 ** 2)} MiB'
    )
    if warnings:
        msg = f'{msg}; warnings={len(warnings)}'
    if progress:
        progress(msg)
    emit_benchmark_event(
        progress,
        'benchmark_done',
        saved,
        'moe_tuning',
        message=msg,
        phase='complete',
        completed=completed,
        total=completed,
        records=records,
    )
    return True, msg


def benchmark_max_context_probe(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'max_context_probe', progress, cancel_token)
    if not preflight_ok:
        return False, preflight_msg

    profile = app.hardware_profile(refresh=True)
    started_at = datetime.now().isoformat(timespec='seconds')
    run_id = f'max-context-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    runtime_profiles = max_context_probe_runtime_profiles(app, model, profile)
    if not runtime_profiles:
        msg = 'max context probe skipped: no supported context/KV candidates'
        if progress:
            progress(msg)
        return False, msg

    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    completed = 0
    total = max(1, len(runtime_profiles))
    current_profile: Optional[RuntimeProfile] = None

    running_model = ModelConfig(**asdict(model))
    if close_stale_running_benchmark_runs(running_model, started_at):
        model.benchmark_runs = list(running_model.benchmark_runs)
    running_run = build_benchmark_run(run_id, 'max_context_probe', 'running', [], {}, started_at, hardware=profile.short_summary())
    upsert_benchmark_run(running_model, running_run)
    app.add_or_update(running_model)

    start_msg = (
        f'max context probe started: {len(runtime_profiles)} candidate(s), '
        f'targets={",".join(str(item) for item in max_context_probe_targets(model))}, '
        f'target_headroom={target_vram_headroom_bytes(profile) // (1024 ** 2)} MiB, '
        f'{profile.short_summary()}'
    )
    if progress:
        progress(start_msg)
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'max_context_probe',
        message=start_msg,
        phase='max context probe',
        completed=0,
        total=total,
    )

    try:
        for runtime_profile in runtime_profiles:
            check_cancelled(cancel_token)
            current_profile = runtime_profile
            ok, _broke, new_records, new_measured, completed = benchmark_runtime_profile_with_retry(
                app,
                model,
                runtime_profile,
                'max_context_probe',
                progress,
                cancel_token,
                completed,
                total,
                run_kind='max_context_probe',
                max_attempts=1,
                benchmark_depth='full',
                benchmark_purpose='max_context_probe',
            )
            records.extend(new_records)
            measured.extend(new_measured)
            running = ModelConfig(**asdict(model))
            run = build_benchmark_run(run_id, 'max_context_probe', 'running', records, {}, started_at, hardware=profile.short_summary())
            upsert_benchmark_run(running, run)
            app.add_or_update(running)
            if progress:
                status = 'ok' if ok else (new_records[-1].get('failure_category') if new_records else 'failed')
                progress(f'max context probe {runtime_profile.name}: {status}')
    except CancelledError:
        if current_profile is not None:
            app.stop(model_for_runtime_profile(model, current_profile), managed_only=True)
        ended_at = datetime.now().isoformat(timespec='seconds')
        aborted_model = ModelConfig(**asdict(model))
        aborted_model.last_benchmark_results = records
        run = build_benchmark_run(run_id, 'max_context_probe', 'aborted', records, {}, started_at, ended_at, profile.short_summary())
        upsert_benchmark_run(aborted_model, run)
        app.add_or_update(aborted_model)
        msg = 'max context probe aborted; managed processes stopped'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_aborted',
            model,
            'max_context_probe',
            message=msg,
            phase='aborted',
            completed=completed,
            total=completed,
            records=records,
        )
        return False, msg

    ended_at = datetime.now().isoformat(timespec='seconds')
    winners = select_max_context_probe_profiles(model, measured, profile)
    if not winners:
        saved = ModelConfig(**asdict(model))
        saved.last_benchmark_results = records
        run = build_benchmark_run(run_id, 'max_context_probe', 'failed', records, {}, started_at, ended_at, profile.short_summary())
        upsert_benchmark_run(saved, run)
        app.add_or_update(saved)
        msg = f'❌ max context probe failed: {benchmark_failure_summary(records, "no measured max-context candidates completed")}'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_error',
            model,
            'max_context_probe',
            message=msg,
            phase='failed',
            completed=completed,
            total=completed,
            records=records,
        )
        return False, msg

    saved = ModelConfig(**asdict(model))
    saved.last_benchmark_results = records
    measured_profiles = dict(getattr(saved, 'measured_profiles', {}) or {})
    measured_profiles.update(winners)
    saved.measured_profiles = measured_profiles
    run = build_benchmark_run(run_id, 'max_context_probe', 'done', records, winners, started_at, ended_at, profile.short_summary())
    run['target_vram_headroom_bytes'] = target_vram_headroom_bytes(profile)
    upsert_benchmark_run(saved, run)
    app.add_or_update(saved)

    safe = winners.get('max_context_safe') or {}
    experimental = winners.get('max_context_experimental') or {}
    safe_ctx = int(safe.get('ctx_per_slot', safe.get('ctx', 0)) or 0)
    experimental_ctx = int(experimental.get('ctx_per_slot', experimental.get('ctx', 0)) or 0)
    safe_status = str(safe.get('status', '') or 'unknown')
    msg = (
        f'✅ max context probe saved: safe={safe_ctx if safe_status == "ok" else "not_ready"} '
        f'experimental={experimental_ctx} ctx/slot'
    )
    if progress:
        progress(msg)
    emit_benchmark_event(
        progress,
        'benchmark_done',
        saved,
        'max_context_probe',
        message=msg,
        phase='complete',
        completed=completed,
        total=completed,
        records=records,
    )
    return True, msg


def benchmark_fast_profiles(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'server_fast', progress, cancel_token)
    if not preflight_ok:
        return False, preflight_msg

    profile = app.hardware_profile(refresh=True)
    started_at = datetime.now().isoformat(timespec='seconds')
    run_id = f'server-fast-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    runtime_profiles = active_engine_runtime_profiles(app, model, profile, depth='fast')
    contexts = fast_benchmark_contexts(model, profile)
    parallel_values = fast_benchmark_parallel_values(profile, model)
    total = max(1, len(runtime_profiles) if runtime_profiles else len(contexts) * (2 + len(parallel_values)) * 2)
    raw_total = tq3_raw_presearch_case_total(runtime_profiles, 'fast')
    if raw_total:
        total += raw_total
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    current: Optional[ModelConfig] = None
    completed = 0
    started_monotonic = time.monotonic()
    deadline = BenchmarkDeadline.from_end(started_monotonic + FAST_RUNTIME_PROFILE_BUDGET_SECONDS)
    budget_exhausted = False
    disabled_runtime_kv: set[Tuple[str, str]] = set()
    disabled_runtime_memory: set[Tuple[str, ...]] = set()

    running_model = ModelConfig(**asdict(model))
    if close_stale_running_benchmark_runs(running_model, started_at):
        model.benchmark_runs = list(running_model.benchmark_runs)
        model.default_benchmark_status = running_model.default_benchmark_status
        model.default_benchmark_at = running_model.default_benchmark_at
    running_model.default_benchmark_status = 'running'
    running_run = build_benchmark_run(run_id, 'server_fast', 'running', [], {}, started_at, hardware=profile.short_summary())
    upsert_benchmark_run(running_model, running_run)
    app.add_or_update(running_model)

    pressure_payload = current_process_pressure_payload()
    if runtime_profiles:
        start_msg = (
            f'fast benchmark started: runtime_profiles={len(runtime_profiles)}, '
            f'budget={FAST_RUNTIME_PROFILE_BUDGET_SECONDS // 60}m, '
            f'hard_deadline=on, '
            f'arch={architecture_label(model)}, {profile.short_summary()} '
            f'{pressure_payload.get("process_pressure_detail", "")}'
        )
    else:
        start_msg = (
            f'fast benchmark started: contexts={",".join(str(ctx) for ctx in contexts)}, '
            f'parallel={",".join(str(value) for value in parallel_values)}, default variant, '
            f'budget={FAST_RUNTIME_PROFILE_BUDGET_SECONDS // 60}m, hard_deadline=on, '
            f'arch={architecture_label(model)}, {profile.short_summary()} '
            f'{pressure_payload.get("process_pressure_detail", "")}'
        )
    if progress:
        progress(start_msg)
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'server_fast',
        message=start_msg,
        phase='fast benchmark',
        completed=0,
        total=total,
    )

    def persist_raw_record(record: Dict[str, object]):
        nonlocal completed
        records.append(dict(record))
        completed += 1
        persist_running_benchmark_progress(
            app,
            model,
            run_id,
            'server_fast',
            records,
            started_at,
            profile.short_summary(),
        )

    _raw_records, raw_promotions = run_tq3_raw_llama_bench_presearch(
        app,
        model,
        profile,
        runtime_profiles,
        'fast',
        progress,
        cancel_token,
        on_record=persist_raw_record,
        completed_offset=completed,
        total=total,
        run_kind='server_fast',
        deadline=deadline,
    )
    if raw_promotions:
        runtime_profiles = dedupe_runtime_profiles(raw_promotions + runtime_profiles)
        if progress:
            progress(f'TQ3 raw llama-bench promoted {len(raw_promotions)} candidate(s) into server validation')

    try:
        if runtime_profiles:
            fit_succeeded_engines: set[str] = set()
            fit_ceiling_by_engine: Dict[str, Tuple[int, str, str]] = {}
            fixed_fit_engine_skipped = 0
            for runtime_profile in runtime_profiles:
                check_cancelled(cancel_token)
                skip_reason = runtime_profile_skip_reason(runtime_profile, disabled_runtime_kv)
                if not skip_reason:
                    skip_reason = runtime_profile_memory_skip_reason(runtime_profile, disabled_runtime_memory)
                if skip_reason:
                    if progress:
                        progress(f'fast skipped {runtime_profile.name}: {skip_reason}')
                    continue
                runtime_engine = str(getattr(runtime_profile, 'engine_id', '') or '')
                fit_phase = runtime_profile_fit_phase(runtime_profile)
                if fit_phase == 'context_growth' and runtime_engine not in fit_succeeded_engines:
                    if progress:
                        progress(f'fast skipped {runtime_profile.name}: weight-fit discovery has no viable ceiling yet')
                    continue
                if fit_phase == 'context_growth' and runtime_engine in fit_ceiling_by_engine:
                    runtime_profile = runtime_profile_with_fit_ceiling(runtime_profile, *fit_ceiling_by_engine[runtime_engine])
                if runtime_engine in fit_succeeded_engines and runtime_profile_is_fixed_gpu_layers(runtime_profile):
                    fixed_fit_engine_skipped += 1
                    continue
                if not deadline_has_candidate_budget(deadline, runtime_profile=runtime_profile):
                    budget_exhausted = True
                    if progress:
                        progress('fast runtime profile budget reached; selecting from measured candidates')
                    break
                objective = 'opencode_ready' if runtime_profile.ctx_size >= 16384 else 'long_context'
                ok, _broke, new_records, new_measured, completed = benchmark_runtime_profile_with_retry(
                    app,
                    model,
                    runtime_profile,
                    objective,
                    progress,
                    cancel_token,
                    completed,
                    total,
                    run_kind='server_fast',
                    max_attempts=1,
                    benchmark_depth='fast',
                    deadline=deadline,
                )
                records.extend(new_records)
                measured.extend(new_measured)
                if ok and runtime_profile_uses_fit(runtime_profile):
                    fit_succeeded_engines.add(runtime_engine)
                    if fit_phase == 'weight_fit' or runtime_engine not in fit_ceiling_by_engine:
                        fit_ceiling_by_engine[runtime_engine] = fit_ceiling_from_records(new_records)
                    if progress:
                        progress(f'fast {runtime_engine} fit profile succeeded; skipping fixed-NGL fallback probes')
                elif not ok and new_records:
                    memory_key = runtime_profile_memory_disable_key(new_records[-1], runtime_profile)
                    if memory_key and memory_key not in disabled_runtime_memory:
                        disabled_runtime_memory.add(memory_key)
                        if progress:
                            progress(f'fast pruned memory-risky runtime shape after {new_records[-1].get("failure_category")}')
                    disable_key = runtime_profile_kv_disable_key(new_records[-1], runtime_profile)
                    if disable_key[0] and disable_key not in disabled_runtime_kv:
                        disabled_runtime_kv.add(disable_key)
                        if progress:
                            progress(f'fast disabled {disable_key[0]} {disable_key[1]} after incompatible KV startup')
            if fixed_fit_engine_skipped and progress:
                progress(f'fast skipped {fixed_fit_engine_skipped} fixed-NGL profile(s) after fit success')
        else:
            chat_floor = chat_min_ctx_per_slot(model)
            for ctx in contexts:
                check_cancelled(cancel_token)
                if not deadline_has_candidate_budget(deadline, engine=app.active_engine_key_for_model(model)):
                    budget_exhausted = True
                    break
                current = configure_adaptive_candidate(model, profile, 'long_context', ctx, 1, 'default')
                ok, broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                    app,
                    model,
                    profile,
                    'long_context',
                    ctx,
                    1,
                    'default',
                    progress,
                    cancel_token,
                    completed,
                    total,
                    scan_level='fast',
                    run_kind='server_fast',
                    deadline=deadline,
                )
                records.extend(new_records)
                measured.extend(new_measured)
                if not ok and broke:
                    if progress:
                        progress(f'fast benchmark stopped at confirmed context break ctx={ctx}')
                    break

                current = configure_adaptive_candidate(model, profile, 'opencode_ready', ctx, 1, 'default')
                ok, _broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                    app,
                    model,
                    profile,
                    'opencode_ready',
                    ctx,
                    1,
                    'default',
                    progress,
                    cancel_token,
                    completed,
                    total,
                    scan_level='fast',
                    run_kind='server_fast',
                    deadline=deadline,
                )
                records.extend(new_records)
                measured.extend(new_measured)

                for parallel in parallel_values:
                    check_cancelled(cancel_token)
                    if not deadline_has_candidate_budget(deadline, engine=app.active_engine_key_for_model(model)):
                        budget_exhausted = True
                        break
                    if ctx // max(1, parallel) < chat_floor:
                        continue
                    current = configure_adaptive_candidate(model, profile, 'fast_chat', ctx, parallel, 'default')
                    ok, broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                        app,
                        model,
                        profile,
                        'fast_chat',
                        ctx,
                        parallel,
                        'default',
                        progress,
                        cancel_token,
                        completed,
                        total,
                        scan_level='fast',
                        run_kind='server_fast',
                        deadline=deadline,
                    )
                    records.extend(new_records)
                    measured.extend(new_measured)
                    if not ok and broke:
                        if progress:
                            progress(f'fast benchmark stopped parallel sweep ctx={ctx} parallel={parallel}')
                        break
    except CancelledError:
        if current is not None:
            app.stop(current, managed_only=True)
            records.append(enrich_exhaustive_record(
                adaptive_record_from_candidate(current, 'fast', 'aborted', detail='user requested abort'),
                current,
                'default',
                1,
                0,
                scan_level='fast',
            ))
        ended_at = datetime.now().isoformat(timespec='seconds')
        aborted_model = ModelConfig(**asdict(model))
        aborted_model.last_benchmark_results = records
        aborted_model.default_benchmark_status = 'aborted'
        aborted_model.default_benchmark_at = ended_at
        run = build_benchmark_run(run_id, 'server_fast', 'aborted', records, {}, started_at, ended_at, profile.short_summary())
        upsert_benchmark_run(aborted_model, run)
        app.add_or_update(aborted_model)
        msg = '⚠ aborted; managed processes stopped'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_aborted',
            model,
            'server_fast',
            message=msg,
            phase='aborted',
            completed=completed,
            total=completed,
            records=records,
        )
        return False, msg

    winners = select_measured_profiles(model, measured, profile)
    propagate_rejection_reasons(records, measured)
    ended_at = datetime.now().isoformat(timespec='seconds')
    if not winners:
        saved, preserved = failed_benchmark_model_state(app, model, records, ended_at)
        saved.benchmark_fingerprint = app.model_fingerprint(saved)
        run = build_benchmark_run(run_id, 'server_fast', 'failed', records, {}, started_at, ended_at, profile.short_summary())
        upsert_benchmark_run(saved, run)
        app.add_or_update(saved)
        msg = (
            preserved_profiles_message('fast benchmark found no better working candidate', records)
            if preserved
            else f'❌ fast benchmark failed: {benchmark_failure_summary(records, "no measured candidates completed")}'
        )
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_error',
            model,
            'server_fast',
            message=msg,
            phase='failed',
            completed=completed,
            total=completed,
            records=records,
        )
        return False, msg

    winners = attach_profile_frontier(model, measured, winners, profile, generated_at=ended_at)
    annotate_spectrum_records(records, winners)
    saved = ModelConfig(**asdict(model))
    saved.last_benchmark_results = records
    saved.measured_profiles = winners
    saved.benchmark_fingerprint = app.model_fingerprint(saved)
    saved.default_benchmark_at = ended_at
    run = build_benchmark_run(run_id, 'server_fast', run_status, records, winners, started_at, ended_at, profile.short_summary())
    upsert_benchmark_run(saved, run)

    auto_profile = winners['auto']
    apply_measured_profile(saved, 'auto')
    saved.measured_profiles = winners
    saved.last_benchmark_tokens_per_sec = round(float(auto_profile.get('tokens_per_sec', 0.0) or 0.0), 2)
    saved.last_benchmark_seconds = round(float(auto_profile.get('seconds', 0.0) or 0.0), 2)
    saved.last_benchmark_profile = (
        f'auto/fast {saved.last_benchmark_tokens_per_sec:.2f} tok/s '
        f'ctx={auto_profile.get("ctx")} slot={auto_profile.get("ctx_per_slot")} {profile.short_summary()}'
    )
    saved.default_benchmark_status = run_status
    app.add_or_update(saved)
    sync_msg = sync_opencode_after_tuning(app)
    prefix = '✅ fast profiles saved' if run_status == 'done' else '⚠ fast partial profiles saved'
    msg = (
        f'{prefix}: {adaptive_profile_status_text(winners, "fast_chat")}, '
        f'{adaptive_profile_status_text(winners, "long_context")}, '
        f'{opencode_profile_status_text(winners)}, '
        f'{adaptive_profile_status_text(winners, "auto")} | status={run_status} | {sync_msg}'
    )
    if progress:
        progress(msg)
    emit_benchmark_event(
        progress,
        'benchmark_done',
        saved,
        'server_fast',
        message=msg,
        phase='complete',
        completed=completed,
        total=completed,
        records=records,
    )
    return True, msg


def benchmark_adaptive_profiles(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
    time_budget_seconds: int = ADAPTIVE_BENCHMARK_TIME_BUDGET_SECONDS,
) -> Tuple[bool, str]:
    preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'server', progress, cancel_token)
    if not preflight_ok:
        return False, preflight_msg
    profile = app.hardware_profile(refresh=True)
    deadline = time.monotonic() + max(60, int(time_budget_seconds or ADAPTIVE_BENCHMARK_TIME_BUDGET_SECONDS))
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    running_model = ModelConfig(**asdict(model))
    running_model.default_benchmark_status = 'running'
    app.add_or_update(running_model)
    pressure_payload = current_process_pressure_payload()
    if progress:
        progress(
            f'adaptive benchmark started: budget≈{int(time_budget_seconds // 60)}m, '
            f'arch={architecture_label(model)}, {profile.short_summary()} '
            f'{pressure_payload.get("process_pressure_detail", "")}'
        )
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'server',
        message=(
            f'adaptive benchmark started: budget≈{int(time_budget_seconds // 60)}m, '
            f'arch={architecture_label(model)}, {profile.short_summary()} '
            f'{pressure_payload.get("process_pressure_detail", "")}'
        ),
        phase='starting',
        completed=0,
        total=0,
    )

    current: Optional[ModelConfig] = None
    try:
        candidates = adaptive_benchmark_candidates(app, model, profile, progress, cancel_token, deadline)
        if progress:
            progress(f'adaptive benchmark measuring {len(candidates)} profile candidate(s)')
        emit_benchmark_event(
            progress,
            'benchmark_phase',
            model,
            'server',
            message=f'measuring {len(candidates)} adaptive profile candidate(s)',
            phase='measuring candidates',
            completed=0,
            total=len(candidates),
        )
        for idx, (objective, candidate, label) in enumerate(candidates, start=1):
            check_cancelled(cancel_token)
            if time.monotonic() >= deadline:
                record = adaptive_record_from_candidate(candidate, objective, 'time budget exhausted', detail='adaptive benchmark budget reached')
                records.append(record)
                emit_benchmark_event(
                    progress,
                    'benchmark_result',
                    model,
                    'server',
                    message='adaptive benchmark budget reached',
                    phase='measuring candidates',
                    completed=idx,
                    total=len(candidates),
                    candidate=label,
                    record=record,
                )
                break
            current = candidate
            if progress:
                progress(
                    f'adaptive candidate {idx}/{len(candidates)} {label}: '
                    f'ctx={candidate.ctx} slot={ctx_per_slot(candidate)} parallel={candidate.parallel}'
                )
            emit_benchmark_event(
                progress,
                'benchmark_candidate',
                model,
                'server',
                message=(
                    f'adaptive candidate {idx}/{len(candidates)} {label}: '
                    f'ctx={candidate.ctx} slot={ctx_per_slot(candidate)} parallel={candidate.parallel}'
                ),
                phase='measuring candidates',
                completed=idx - 1,
                total=len(candidates),
                candidate=label,
            )
            record, measured_item = benchmark_adaptive_candidate(app, candidate, objective, progress, cancel_token)
            records.append(record)
            if measured_item:
                measured.append(measured_item)
                if progress:
                    progress(
                        f'adaptive {objective} scored {float(record.get("tokens_per_sec", 0.0)):.2f} tok/s '
                        f'ctx={record.get("ctx")} slot={record.get("ctx_per_slot")}'
                    )
            elif progress:
                progress(f'adaptive {objective} failed: {record.get("status")} {record.get("detail")}')
            emit_benchmark_event(
                progress,
                'benchmark_result',
                model,
                'server',
                message=(
                    f'adaptive {objective} {record.get("status")}: '
                    f'{float(record.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s '
                    f'ctx={record.get("ctx")} slot={record.get("ctx_per_slot")}'
                ),
                phase='measuring candidates',
                completed=idx,
                total=len(candidates),
                candidate=label,
                record=record,
            )
    except CancelledError:
        if current is not None:
            app.stop(current, managed_only=True)
            records.append(adaptive_record_from_candidate(current, 'adaptive', 'aborted', detail='user requested abort'))
        aborted_model = ModelConfig(**asdict(model))
        aborted_model.last_benchmark_results = records
        aborted_model.default_benchmark_status = 'aborted'
        aborted_model.default_benchmark_at = datetime.now().isoformat(timespec='seconds')
        app.add_or_update(aborted_model)
        msg = '⚠ aborted; managed processes stopped'
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_aborted',
            model,
            'server',
            message=msg,
            phase='aborted',
            records=records,
        )
        return False, msg

    winners = select_measured_profiles(model, measured, profile)
    propagate_rejection_reasons(records, measured)
    if not winners:
        ended_at = datetime.now().isoformat(timespec='seconds')
        saved, preserved = failed_benchmark_model_state(app, model, records, ended_at)
        saved.benchmark_fingerprint = app.model_fingerprint(saved)
        app.add_or_update(saved)
        msg = (
            preserved_profiles_message('adaptive benchmark found no better working candidate', records)
            if preserved
            else f'❌ adaptive benchmark failed: {benchmark_failure_summary(records, "no measured candidates completed")}'
        )
        if progress:
            progress(msg)
        emit_benchmark_event(
            progress,
            'benchmark_error',
            model,
            'server',
            message=msg,
            phase='failed',
            records=records,
        )
        return False, msg

    ended_at = datetime.now().isoformat(timespec='seconds')
    budget_expired = time.monotonic() >= deadline
    winners = fill_missing_adaptive_profiles(
        winners,
        'No valid candidate measured before benchmark budget expired' if budget_expired else 'No valid candidate measured for this profile',
        status='skipped_budget' if budget_expired else 'skipped',
    )
    run_status = 'done' if adaptive_profiles_complete(winners) else 'partial'
    winners = attach_profile_frontier(model, measured, winners, profile, generated_at=ended_at)
    annotate_spectrum_records(records, winners)
    saved = ModelConfig(**asdict(model))
    saved.last_benchmark_results = records
    saved.measured_profiles = winners
    saved.benchmark_fingerprint = app.model_fingerprint(saved)
    saved.default_benchmark_at = ended_at

    auto_profile = winners['auto']
    apply_measured_profile(saved, 'auto')
    saved.measured_profiles = winners
    saved.last_benchmark_tokens_per_sec = round(float(auto_profile.get('tokens_per_sec', 0.0) or 0.0), 2)
    saved.last_benchmark_seconds = round(float(auto_profile.get('seconds', 0.0) or 0.0), 2)
    saved.last_benchmark_profile = (
        f'auto/measured {saved.last_benchmark_tokens_per_sec:.2f} tok/s '
        f'ctx={auto_profile.get("ctx")} slot={auto_profile.get("ctx_per_slot")} {profile.short_summary()}'
    )
    saved.default_benchmark_status = run_status
    app.add_or_update(saved)
    sync_msg = sync_opencode_after_tuning(app)
    prefix = '✅ adaptive profiles saved' if run_status == 'done' else '⚠ adaptive partial profiles saved'
    msg = (
        f'{prefix}: {adaptive_profile_status_text(winners, "fast_chat")}, '
        f'{adaptive_profile_status_text(winners, "long_context")}, '
        f'opencode ctx/slot={winners["opencode_ready"]["ctx_per_slot"]}, '
        f'{adaptive_profile_status_text(winners, "auto")} | status={run_status} | {sync_msg}'
    )
    if progress:
        progress(msg)
    emit_benchmark_event(
        progress,
        'benchmark_done',
        saved,
        'server',
        message=msg,
        phase='complete',
        completed=len(records),
        total=len(records),
        records=records,
    )
    return True, msg


def benchmark_raw_speed_profile(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'raw_speed', progress, cancel_token)
    if not preflight_ok:
        return False, preflight_msg

    run_id = f'raw-speed-{int(time.time())}'
    started_at = datetime.now().isoformat(timespec='seconds')
    profile = app.hardware_profile(refresh=True)
    try:
        runtime_profile = app.runtime_profile_from_model(
            model,
            int(getattr(model, 'ctx', 0) or 0),
            int(getattr(model, 'parallel', 1) or 1),
            int(getattr(model, 'ngl', 0) or 0),
        )
    except Exception:
        runtime_profile = None
    total = 1
    start_msg = f'raw speed benchmark started: deterministic request, {profile.short_summary()}'
    if progress:
        progress(start_msg)
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'raw_speed',
        message=start_msg,
        phase='raw speed',
        completed=0,
        total=total,
    )
    records: List[Dict[str, object]] = []
    completed = 0
    try:
        check_cancelled(cancel_token)
        candidate = model_for_runtime_profile(model, runtime_profile) if runtime_profile is not None else ModelConfig(**asdict(model))
        try:
            capabilities = app.engine_capabilities()
        except Exception:
            capabilities = None
        launch_profile = build_benchmark_launch_profile(
            candidate,
            runtime_profile,
            capabilities,
            purpose='raw_speed',
            depth='fast',
        )
        command_preview = benchmark_command_preview(app, candidate, runtime_profile, launch_profile)
        emit_benchmark_event(
            progress,
            'benchmark_candidate',
            model,
            'raw_speed',
            message='raw speed candidate',
            phase='raw speed',
            completed=0,
            total=total,
            candidate='raw_speed',
            command=command_preview,
        )
        record, _measured_item = benchmark_adaptive_candidate(
            app,
            candidate,
            'raw_speed',
            progress,
            cancel_token,
            runtime_profile=runtime_profile,
            benchmark_profile=launch_profile,
            benchmark_purpose='raw_speed',
            benchmark_depth='fast',
        )
        completed = 1
        record['config_fingerprint'] = (
            runtime_profile_config_fingerprint(candidate, runtime_profile)
            if runtime_profile is not None else benchmark_config_fingerprint(candidate)
        )
        records.append(record)
        emit_exhaustive_result(progress, model, record, completed, total, 'raw_speed', run_kind='raw_speed')
    except CancelledError:
        ended_at = datetime.now().isoformat(timespec='seconds')
        run = build_benchmark_run(run_id, 'raw_speed', 'aborted', records, {}, started_at, ended_at, profile.short_summary())
        saved = ModelConfig(**asdict(model))
        upsert_benchmark_run(saved, run)
        app.add_or_update(saved)
        msg = '⚠ raw speed benchmark aborted; managed processes stopped'
        if progress:
            progress(msg)
        emit_benchmark_event(progress, 'benchmark_aborted', model, 'raw_speed', message=msg, phase='aborted', completed=completed, total=total, records=records)
        return False, msg

    ended_at = datetime.now().isoformat(timespec='seconds')
    status = 'done' if records and records[-1].get('status') == 'ok' else 'failed'
    run = build_benchmark_run(run_id, 'raw_speed', status, records, {}, started_at, ended_at, profile.short_summary())
    saved = ModelConfig(**asdict(model))
    upsert_benchmark_run(saved, run)
    app.add_or_update(saved)
    if status == 'done':
        score = float(records[-1].get('tokens_per_sec', 0.0) or 0.0)
        msg = f'✅ raw speed benchmark saved: {score:.2f} tok/s'
        event = 'benchmark_done'
        ok = True
    else:
        msg = f'❌ raw speed benchmark failed: {benchmark_failure_summary(records, "no raw speed measurement completed")}'
        event = 'benchmark_error'
        ok = False
    if progress:
        progress(msg)
    emit_benchmark_event(progress, event, model, 'raw_speed', message=msg, phase='complete', completed=completed, total=total, records=records)
    return ok, msg


def benchmark_best_optimization(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    return benchmark_exhaustive_profiles(app, model, progress=progress, cancel_token=cancel_token)


def benchmark_all_models_runner(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    return benchmark_adaptive_profiles(
        app,
        model,
        progress=progress,
        cancel_token=cancel_token,
        time_budget_seconds=ALL_MODELS_ADAPTIVE_TIME_BUDGET_SECONDS,
    )


def safe_bootstrap_benchmark(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    preflight_ok, preflight_msg = benchmark_preflight_cleanup(app, model, 'safe_bootstrap', progress, cancel_token)
    if not preflight_ok:
        return False, preflight_msg
    profile = app.hardware_profile(refresh=True)
    return _run_server_benchmark_candidates(
        app,
        model,
        safe_bootstrap_candidate_models(model, profile),
        profile,
        'safe bootstrap benchmark',
        progress=progress,
        cancel_token=cancel_token,
        update_default_status=True,
    )


def _get_model_pid(app: AppConfig, model: ModelConfig, discover: bool = True, managed_only: bool = False) -> Optional[int]:
    try:
        return app.get_pid(model, discover=discover, managed_only=managed_only)
    except TypeError:
        return app.get_pid(model)


def _batch_summary_record(
    model: ModelConfig,
    status: str,
    detail: str,
    reason: str = '',
) -> Dict[str, object]:
    profile = get_measured_profile(model, 'auto')
    record = {
        'objective': 'deep_all',
        'model_id': model.id,
        'status': status,
        'tokens_per_sec': round(float(profile.get('tokens_per_sec', 0.0) or 0.0), 2),
        'seconds': round(float(profile.get('seconds', 0.0) or 0.0), 2),
        'ctx': int(profile.get('ctx', getattr(model, 'ctx', 0)) or 0),
        'ctx_per_slot': int(profile.get('ctx_per_slot', ctx_per_slot(model)) or 0),
        'parallel': int(profile.get('parallel', getattr(model, 'parallel', 1)) or 1),
        'variant': str(profile.get('variant', '') or 'auto'),
        'measurement_type': 'batch',
        'planner_reason': reason,
        'detail': concise_failure(detail, limit=500),
        'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
    }
    record.update(architecture_payload(model))
    record.update(current_process_pressure_payload())
    return record


def benchmark_all_models_deep(
    app: AppConfig,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
    force: bool = False,
    benchmark_runner: Optional[Callable[..., Tuple[bool, str]]] = None,
    start_runner: Optional[Callable[..., Tuple[bool, str]]] = None,
) -> Tuple[bool, str]:
    runner = benchmark_runner or benchmark_all_models_runner
    restarter = start_runner or start_model_with_progress
    models = list(getattr(app, 'models', []) or [])
    total = len(models)
    completed = 0
    skipped = 0
    failed = 0
    restored = 0
    benchmarked = 0
    summary_records: List[Dict[str, object]] = []

    def emit_batch(
        event: str,
        message: str,
        model: Optional[ModelConfig] = None,
        phase: str = '',
        record: Optional[Dict[str, object]] = None,
        records: Optional[List[Dict[str, object]]] = None,
    ):
        if not progress:
            return
        payload: Dict[str, object] = {
            'event': event,
            'run_kind': 'server_all',
            'model_id': model.id if model else '',
            'message': compact_message(message),
            'phase': phase or compact_message(message),
            'completed': completed,
            'total': total,
            'candidate': model.id if model else '',
            'batch_completed': completed,
            'batch_total': total,
            'batch_skipped': skipped,
            'batch_failed': failed,
            'batch_restored': restored,
        }
        if record is not None:
            payload['record'] = dict(record)
        if records is not None:
            payload['records'] = [dict(item) for item in records]
        progress(payload)

    def forward_inner(model: ModelConfig, index: int, payload: object):
        if not progress:
            return
        prefix = f'[{index}/{total}] {model.id}: '
        if isinstance(payload, dict):
            forwarded = dict(payload)
            event = str(forwarded.get('event', '') or '')
            if event in ('benchmark_started', 'benchmark_done', 'benchmark_error', 'benchmark_aborted'):
                forwarded['event'] = 'benchmark_phase'
            forwarded['run_kind'] = 'server_all'
            forwarded['model_id'] = model.id
            forwarded['completed'] = completed
            forwarded['total'] = total
            forwarded['batch_completed'] = completed
            forwarded['batch_total'] = total
            forwarded['batch_skipped'] = skipped
            forwarded['batch_failed'] = failed
            forwarded['batch_restored'] = restored
            message = compact_message(str(forwarded.get('message') or forwarded.get('phase') or event or 'benchmark update'))
            forwarded['message'] = prefix + message if message else prefix.rstrip()
            forwarded['phase'] = str(forwarded.get('phase') or 'benchmarking')
            candidate = compact_message(str(forwarded.get('candidate', '') or ''))
            forwarded['candidate'] = f'{model.id} {candidate}'.strip()
            if isinstance(forwarded.get('record'), dict):
                record = dict(forwarded.get('record') or {})
                record.setdefault('model_id', model.id)
                forwarded['record'] = record
            if isinstance(forwarded.get('records'), list):
                records = []
                for item in forwarded.get('records') or []:
                    if isinstance(item, dict):
                        record = dict(item)
                        record.setdefault('model_id', model.id)
                        records.append(record)
                forwarded['records'] = records
            progress(forwarded)
            return
        emit_batch('benchmark_phase', prefix + compact_message(str(payload)), model, phase='benchmarking')

    if total <= 0:
        msg = 'No models configured for deep benchmark all.'
        emit_batch('benchmark_done', msg, phase='complete')
        return True, msg

    label = 'force refresh' if force else 'missing/stale/failed'
    pressure_payload = current_process_pressure_payload()
    emit_batch(
        'benchmark_started',
        f'deep benchmark all started: {total} model(s), mode={label}, {pressure_payload.get("process_pressure_detail", "")}',
        phase='starting',
    )

    try:
        for index, original_model in enumerate(models, start=1):
            check_cancelled(cancel_token)
            model = app.get_model(original_model.id) or original_model
            include, reason = deep_benchmark_model_decision(app, model, force=force)
            if not include:
                skipped += 1
                completed += 1
                record = _batch_summary_record(model, 'skipped', reason, reason=reason)
                summary_records.append(record)
                emit_batch('benchmark_result', f'{model.id} skipped: {reason}', model, phase='skipped', record=record)
                continue

            restore_after = False
            status, detail = app.health(model)
            managed_pid = _get_model_pid(app, model, discover=False, managed_only=True)
            any_pid = managed_pid or _get_model_pid(app, model)
            running = status in ('READY', 'LOADING', 'STARTING') or bool(any_pid)
            if running and not managed_pid:
                skipped += 1
                completed += 1
                detail_text = f'unmanaged server is running ({detail})'
                record = _batch_summary_record(model, 'skipped', detail_text, reason='unmanaged running')
                summary_records.append(record)
                emit_batch('benchmark_result', f'{model.id} skipped: {detail_text}', model, phase='skipped', record=record)
                continue
            if running and managed_pid:
                emit_batch('benchmark_phase', f'{model.id}: stopping managed server before benchmark', model, phase='stopping')
                stop_ok, stop_msg = app.stop(model, managed_only=True)
                if not stop_ok:
                    failed += 1
                    completed += 1
                    record = _batch_summary_record(model, 'stop failed', stop_msg, reason='managed stop failed')
                    summary_records.append(record)
                    emit_batch('benchmark_result', f'{model.id} stop failed: {stop_msg}', model, phase='failed', record=record)
                    continue
                restore_after = True
                sleep_with_cancel(0.3, cancel_token)

            emit_batch('benchmark_phase', f'{model.id}: deep benchmark started ({reason})', model, phase='benchmarking')
            ok = False
            result = ''
            try:
                ok, result = runner(
                    app,
                    model,
                    progress=lambda payload, model=model, index=index: forward_inner(model, index, payload),
                    cancel_token=cancel_token,
                )
                if cancel_token is not None and cancel_token.is_cancelled():
                    raise CancelledError(cancel_token.reason)
            except CancelledError:
                raise
            except Exception as exc:
                ok = False
                result = f'deep benchmark failed: {exc}'

            saved_model = app.get_model(model.id) or model
            if ok:
                benchmarked += 1
                record = _batch_summary_record(saved_model, 'ok', result, reason=reason)
            else:
                failed += 1
                record = _batch_summary_record(saved_model, 'failed', result, reason=reason)
            summary_records.append(record)
            emit_batch(
                'benchmark_result',
                f'{model.id}: {"done" if ok else "failed"} - {concise_failure(result)}',
                saved_model,
                phase='benchmark complete',
                record=record,
            )

            if restore_after:
                check_cancelled(cancel_token)
                emit_batch('benchmark_phase', f'{model.id}: restoring managed server', saved_model, phase='restoring')
                restore_ok, restore_msg = restarter(
                    app,
                    saved_model,
                    progress=lambda text, model=saved_model: emit_batch(
                        'benchmark_phase',
                        f'{model.id}: restore: {compact_message(str(text))}',
                        model,
                        phase='restoring',
                    ),
                    cancel_token=cancel_token,
                )
                if restore_ok:
                    restored += 1
                    emit_batch('benchmark_phase', f'{model.id}: restored managed server', saved_model, phase='restored')
                else:
                    failed += 1
                    emit_batch('benchmark_phase', f'{model.id}: restore failed: {restore_msg}', saved_model, phase='restore failed')

            completed += 1
            emit_batch('benchmark_phase', f'{model.id}: batch step complete', saved_model, phase='model complete')
    except CancelledError:
        msg = '⚠ aborted; managed processes stopped'
        emit_batch('benchmark_aborted', msg, phase='aborted', records=summary_records)
        return False, msg

    summary = machine_best_summary(app)
    pick = summary.get('machine_pick') if isinstance(summary, dict) else {}
    pick_id = str((pick or {}).get('model_id', '') or '')
    pick_text = f' Machine Pick: {pick_id}.' if pick_id else ''
    prefix = '✅' if failed == 0 else '❌'
    result_word = 'complete' if failed == 0 else 'completed with failures'
    msg = (
        f'{prefix} deep benchmark all {result_word}: {benchmarked} benchmarked, '
        f'{skipped} skipped, {failed} failed, {restored} restored.'
        f'{pick_text}'
    )
    emit_batch('benchmark_done', msg, phase='complete', records=summary_records)
    return failed == 0, msg
