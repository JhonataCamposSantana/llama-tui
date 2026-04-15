import json
import re
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from urllib import request

from .control import CancelToken, CancelledError, check_cancelled, sleep_with_cancel
from .gguf import set_model_extra_arg, strip_extra_args
from .hardware import HardwareProfile
from .models import ModelConfig
from .optimize import (
    apply_hardware_baseline,
    apply_best_optimization,
    apply_optimization_preset,
    estimate_safe_context_for_profile,
    choose_best_preset,
    select_best_tier,
)
from .textutil import compact_message

BENCHMARK_MAX_CANDIDATES = 6
BENCHMARK_WARMUP_TOKENS = 16
BENCHMARK_SAMPLE_TOKENS = 96
BENCHMARK_WARMUP_TIMEOUT = 120
BENCHMARK_SAMPLE_TIMEOUT = 240
BENCHMARK_READY_TIMEOUT = 180
SAFE_BOOTSTRAP_PRESETS = (
    ('max_context', 'safe'),
    ('tokens_per_sec', 'safe'),
)
SAFE_BOOTSTRAP_Q8_TARGET_CTX = 4096
ADAPTIVE_BENCHMARK_TIME_BUDGET_SECONDS = 20 * 60
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


def sync_opencode_after_tuning(app: AppConfig) -> str:
    if not app.opencode.path:
        return 'opencode.path unset; skipped opencode sync'
    ok, msg = app.generate_opencode()
    return msg if ok else f'opencode sync failed: {msg}'
def append_model_log(app: AppConfig, model: ModelConfig, text: str):
    app.append_log(model.id, text)


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
    if record is not None:
        payload['record'] = dict(record)
    if records is not None:
        payload['records'] = [dict(item) for item in records]
    progress(payload)


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
    if isinstance(profile.get('extra_args'), list):
        model.extra_args = [str(item) for item in profile.get('extra_args', [])]
    model.optimize_mode = f'measured_{key}'
    model.optimize_tier = 'measured'
    return True, (
        f'measured {key}: ctx={model.ctx} parallel={model.parallel} '
        f'threads={model.threads} ngl={model.ngl} '
        f'{float(profile.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s'
    )


def model_from_measured_profile(model: ModelConfig, key: str) -> Optional[ModelConfig]:
    candidate = ModelConfig(**asdict(model))
    ok, _msg = apply_measured_profile(candidate, key)
    return candidate if ok else None


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


def adaptive_batch_sizes(ctx: int, parallel: int, objective: str) -> Tuple[int, int]:
    ctx = max(1, int(ctx or 1))
    parallel = max(1, int(parallel or 1))
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
        batch, ubatch = adaptive_batch_sizes(candidate.ctx, candidate.parallel, objective)
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


def adaptive_parallel_values(model: ModelConfig, profile: HardwareProfile, objective: str, ctx: int, variant: str) -> List[int]:
    if objective != 'fast_chat':
        return [1]
    max_cpu = max(1, min(16, int(getattr(profile, 'cpu_logical', 0) or 8)))
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
        ok, measured_msg = apply_measured_profile(model, measured_key)
        if ok:
            if progress:
                progress(f'trying measured {measured_key} profile: {measured_msg}')
            app.add_or_update(model)
            sync_msg = sync_opencode_after_tuning(app)
            ok, msg = app.start(model)
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
            progress(f'{model.id}: {measured_msg}; using estimated launch profile.')
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
def launch_opencode_stack(
    app: AppConfig,
    model: ModelConfig,
    workspace: str,
    include_vscode: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    check_cancelled(cancel_token)
    valid, workspace_path, reason = app.validate_workspace_path(workspace)
    if not valid or workspace_path is None:
        return False, f'❌ {reason}'
    if not getattr(model, 'enabled', True):
        return False, f'❌ {model.id} is disabled; enable it before launching OpenCode.'

    app.opencode.last_workspace_path = str(workspace_path)
    app.save()

    started_for_stack = False
    status, _detail = app.health(model)
    if status == 'READY':
        if progress:
            progress(f'{model.id} already ready; using current server for OpenCode.')
    elif status in ('LOADING', 'STARTING') or app.get_pid(model):
        if progress:
            progress(f'{model.id} is starting; waiting for readiness before OpenCode launch...')
        ready_ok, ready_msg = app.wait_until_ready(model, timeout=180, cancel_token=cancel_token)
        if not ready_ok:
            return False, concise_failure(ready_msg)
    else:
        if progress:
            progress(f'{model.id} is stopped; launching OpenCode-ready profile before OpenCode...')
        ready_ok, ready_msg = launch_with_failsafe(app, model, 'opencode_ready', 'auto', progress=progress, cancel_token=cancel_token)
        if not ready_ok:
            return False, concise_failure(ready_msg)
        started_for_stack = True

    if cancel_token is not None and cancel_token.is_cancelled():
        if started_for_stack:
            app.stop(model, managed_only=True)
        check_cancelled(cancel_token)
    if not (getattr(app.opencode, 'path', '') or '').strip():
        return False, '❌ Set opencode.path first in settings.'
    sync_ok, sync_msg = app.generate_opencode()
    if not sync_ok:
        return False, f'❌ {sync_msg}'
    if progress:
        progress(sync_msg)

    if not app.command_exists('opencode'):
        return False, '❌ opencode command not found in PATH.'
    terminal_ok, _terminal_cmd, terminal_msg = app.build_terminal_command(
        f'OpenCode {model.id}',
        workspace_path,
        app.build_opencode_shell_command(model, workspace_path),
    )
    if not terminal_ok:
        return False, f'❌ {terminal_msg}'

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
    open_ok, open_msg = app.launch_opencode_terminal(model, workspace_path)
    if not open_ok:
        detail = f'❌ {open_msg}'
        if warnings:
            detail += ' | warnings: ' + '; '.join(warnings)
        return False, detail
    if progress:
        progress(open_msg)

    stack_label = 'full-stack' if include_vscode else 'OpenCode'
    detail = f'✅ launched {stack_label} for {model.id} in {workspace_path}'
    if warnings:
        detail += ' | warnings: ' + '; '.join(warnings)
    return True, detail
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
    max_tokens: int = 64,
    timeout: int = 180,
    prompt: Optional[str] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, Dict]:
    check_cancelled(cancel_token)
    prompt = prompt or BENCHMARK_PROMPTS[0]
    payload = {
        'model': model.alias,
        'messages': [
            {'role': 'system', 'content': 'You are a concise local model benchmark assistant.'},
            {'role': 'user', 'content': prompt},
        ],
        'max_tokens': max_tokens,
        'temperature': 0,
        'stream': False,
    }
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
) -> Tuple[bool, Dict]:
    samples = []
    failures = []
    for prompt in BENCHMARK_PROMPTS:
        check_cancelled(cancel_token)
        ok, bench = benchmark_completion(model, max_tokens=max_tokens, timeout=timeout, prompt=prompt, cancel_token=cancel_token)
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
    if progress:
        progress(f'{label} started: {len(candidates)} candidate(s), {profile.short_summary()}')
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
    ):
        benchmark_records.append({
            'preset': preset,
            'tier': tier,
            'status': status,
            'tokens_per_sec': round(float(score), 2),
            'seconds': round(float(elapsed), 2),
            'ctx': int(getattr(candidate, 'ctx', 0) or 0),
            'parallel': int(getattr(candidate, 'parallel', 0) or 0),
            'threads': int(getattr(candidate, 'threads', 0) or 0),
            'ngl': int(getattr(candidate, 'ngl', 0) or 0),
            'detail': concise_failure(detail, limit=500),
            'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
        })

    current: Optional[Tuple[str, str, ModelConfig]] = None
    try:
        for attempt, (preset, tier, candidate, tune_msg) in enumerate(candidates, start=1):
            check_cancelled(cancel_token)
            current = (preset, tier, candidate)
            if progress:
                progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier}: {tune_msg}')
            ok, msg = app.start(candidate)
            if not ok:
                if progress:
                    progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} failed to start: {concise_failure(msg)}')
                add_benchmark_record(preset, tier, candidate, 'start failed', detail=msg)
                failures.append(f'{preset}/{tier}: start failed ({concise_failure(msg)})')
                continue

            try:
                if progress:
                    progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} started; waiting for readiness...')
                ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=BENCHMARK_READY_TIMEOUT, cancel_token=cancel_token)
                if not ready_ok:
                    if progress:
                        progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} not ready: {concise_failure(ready_msg)}')
                    add_benchmark_record(preset, tier, candidate, 'not ready', detail=ready_msg)
                    failures.append(f'{preset}/{tier}: {concise_failure(ready_msg)}')
                    continue

                if progress:
                    progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} ready; warming benchmark prompt...')
                benchmark_completion(
                    candidate,
                    max_tokens=BENCHMARK_WARMUP_TOKENS,
                    timeout=BENCHMARK_WARMUP_TIMEOUT,
                    cancel_token=cancel_token,
                )
                if progress:
                    progress(
                        f'candidate {attempt}/{len(candidates)} {preset}/{tier} '
                        f'measuring {len(BENCHMARK_PROMPTS)}x{BENCHMARK_SAMPLE_TOKENS}-token completion suite...'
                    )
                bench_ok, bench = benchmark_completion_suite(
                    candidate,
                    max_tokens=BENCHMARK_SAMPLE_TOKENS,
                    timeout=BENCHMARK_SAMPLE_TIMEOUT,
                    cancel_token=cancel_token,
                )
                if not bench_ok:
                    if progress:
                        progress(f'candidate {attempt}/{len(candidates)} {preset}/{tier} benchmark failed: {bench.get("error", "unknown error")}')
                    add_benchmark_record(preset, tier, candidate, 'benchmark failed', detail=str(bench.get('error', 'unknown error')))
                    failures.append(f'{preset}/{tier}: benchmark failed ({bench.get("error", "unknown error")})')
                    continue

                score = float(bench['tokens_per_sec'])
                elapsed = float(bench['elapsed'])
                if progress:
                    progress(
                        f'candidate {attempt}/{len(candidates)} {preset}/{tier} '
                        f'scored median {score:.2f} tok/s across {int(bench.get("sample_count", 1))} sample(s)'
                    )
                add_benchmark_record(preset, tier, candidate, 'ok', score=score, elapsed=elapsed)
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
            preset, tier, candidate = current
            add_benchmark_record(preset, tier, candidate, 'aborted', detail='user requested abort')
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
        msg = f'❌ {label} failed: {details}'
        recorded_model = clone_model_config(model)
        recorded_model.last_benchmark_results = benchmark_records
        if update_default_status:
            recorded_model.benchmark_fingerprint = app.model_fingerprint(recorded_model)
            recorded_model.default_benchmark_status = 'failed'
            recorded_model.default_benchmark_at = datetime.now().isoformat(timespec='seconds')
        app.add_or_update(recorded_model)
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
    return {
        'status': 'ok',
        'objective': key,
        'ctx': int(getattr(candidate, 'ctx', 0) or 0),
        'ctx_per_slot': ctx_per_slot(candidate),
        'parallel': int(getattr(candidate, 'parallel', 1) or 1),
        'threads': int(getattr(candidate, 'threads', 0) or 0),
        'ngl': int(getattr(candidate, 'ngl', 0) or 0),
        'output': int(getattr(candidate, 'output', 0) or 0),
        'cache_ram': int(getattr(candidate, 'cache_ram', 0) or 0),
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
        'benchmarked_at': str(record.get('benchmarked_at') or datetime.now().isoformat(timespec='seconds')),
        'hardware': profile.short_summary(),
    }


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
        floor = max(floor, parse_context_requirement(str(row.get('detail', ''))))
        for task in row.get('task_details', []) or []:
            if isinstance(task, dict):
                floor = max(floor, parse_context_requirement(str(task.get('detail', ''))))
                floor = max(floor, parse_context_requirement(' '.join(str(x) for x in task.get('stderr_tail', []) or [])))
                floor = max(floor, parse_context_requirement(' '.join(str(x) for x in task.get('stdout_tail', []) or [])))
    return floor


def select_measured_profiles(
    model: ModelConfig,
    measured: List[Dict[str, object]],
    profile: HardwareProfile,
) -> Dict[str, Dict[str, object]]:
    successful = [item for item in measured if item.get('status') == 'ok']
    if not successful:
        return {}
    max_tps = max(float(item.get('tokens_per_sec', 0.0) or 0.0) for item in successful) or 1.0
    max_ctx = max(int(item.get('ctx_per_slot', 0) or 0) for item in successful) or 1
    fast_floor = chat_min_ctx_per_slot(model)
    opencode_floor = observed_opencode_context_floor(model)

    fast_pool = [item for item in successful if int(item.get('ctx_per_slot', 0) or 0) >= fast_floor] or successful
    long_pool = successful
    opencode_pool = [
        item for item in successful
        if int(item.get('parallel', 1) or 1) == 1 and int(item.get('ctx_per_slot', 0) or 0) >= opencode_floor
    ] or [item for item in successful if int(item.get('parallel', 1) or 1) == 1] or successful

    fast = max(fast_pool, key=lambda item: (float(item.get('tokens_per_sec', 0.0) or 0.0), int(item.get('ctx_per_slot', 0) or 0)))
    long = max(long_pool, key=lambda item: (int(item.get('ctx_per_slot', 0) or 0), float(item.get('tokens_per_sec', 0.0) or 0.0)))
    opencode = max(opencode_pool, key=lambda item: (int(item.get('ctx_per_slot', 0) or 0), float(item.get('tokens_per_sec', 0.0) or 0.0)))

    def auto_score(item: Dict[str, object]) -> float:
        tps_norm = float(item.get('tokens_per_sec', 0.0) or 0.0) / max_tps
        ctx_norm = int(item.get('ctx_per_slot', 0) or 0) / max_ctx
        ram = int(item.get('ram_available', 0) or 0)
        vram = int(item.get('gpu_memory_free', 0) or 0)
        headroom = min(1.0, (ram / 1024**3) / 8.0)
        if vram:
            headroom = max(headroom, min(1.0, (vram / 1024**3) / 2.0))
        return 0.55 * tps_norm + 0.35 * ctx_norm + 0.10 * headroom

    auto = max(successful, key=auto_score)
    winners = {
        'fast_chat': fast,
        'long_context': long,
        'opencode_ready': opencode,
        'auto': auto,
    }
    return {
        key: adaptive_profile_dict(key, item['model'], item, profile)
        for key, item in winners.items()
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
        if record.get('status') != 'ok':
            add_spectrum_label(record, 'failed')
        if record.get('break_point'):
            add_spectrum_label(record, 'break_point')
    successful = [record for record in records if record.get('status') == 'ok']
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


def benchmark_run_summary(winners: Dict[str, Dict[str, object]]) -> str:
    if not winners:
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
    return ', '.join(parts) if parts else 'no winners'


def upsert_benchmark_run(model: ModelConfig, run: Dict[str, object], limit: int = BENCHMARK_HISTORY_LIMIT):
    run_id = str(run.get('id', '') or '')
    existing = list(getattr(model, 'benchmark_runs', []) or [])
    filtered = [item for item in existing if str(item.get('id', '') or '') != run_id]
    filtered.insert(0, dict(run))
    model.benchmark_runs = filtered[: max(1, int(limit or BENCHMARK_HISTORY_LIMIT))]


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
    failed = [row for row in records if row.get('status') != 'ok']
    elapsed = 0.0
    for row in records:
        elapsed += float(row.get('seconds', 0.0) or 0.0)
    return {
        'id': run_id,
        'kind': kind,
        'status': status,
        'started_at': started_at,
        'ended_at': ended_at,
        'elapsed_seconds': round(elapsed, 2),
        'records': [dict(row) for row in records],
        'winners': {key: dict(value) for key, value in winners.items()},
        'summary': benchmark_run_summary(winners),
        'successful': len(successful),
        'failed': len(failed),
        'hardware': hardware,
    }


def adaptive_record_from_candidate(
    candidate: ModelConfig,
    objective: str,
    status: str,
    tokens_per_sec: float = 0.0,
    seconds: float = 0.0,
    detail: str = '',
    ram_available: int = 0,
    gpu_memory_free: int = 0,
) -> Dict[str, object]:
    return {
        'objective': objective,
        'preset': objective,
        'tier': 'measured',
        'status': status,
        'tokens_per_sec': round(float(tokens_per_sec), 2),
        'seconds': round(float(seconds), 2),
        'ctx': int(getattr(candidate, 'ctx', 0) or 0),
        'ctx_per_slot': ctx_per_slot(candidate),
        'parallel': int(getattr(candidate, 'parallel', 0) or 0),
        'threads': int(getattr(candidate, 'threads', 0) or 0),
        'ngl': int(getattr(candidate, 'ngl', 0) or 0),
        'output': int(getattr(candidate, 'output', 0) or 0),
        'cache_ram': int(getattr(candidate, 'cache_ram', 0) or 0),
        'temp': float(getattr(candidate, 'temp', 0.7) or 0.7),
        'memory_reserve_percent': int(getattr(candidate, 'memory_reserve_percent', 0) or 0),
        'extra_args': list(getattr(candidate, 'extra_args', []) or []),
        'ram_available': int(ram_available or 0),
        'gpu_memory_free': int(gpu_memory_free or 0),
        'detail': concise_failure(detail, limit=500),
        'benchmarked_at': datetime.now().isoformat(timespec='seconds'),
    }


def benchmark_adaptive_candidate(
    app: AppConfig,
    candidate: ModelConfig,
    objective: str,
    progress: Optional[Callable[[str], None]],
    cancel_token: Optional[CancelToken],
) -> Tuple[Dict[str, object], Optional[Dict[str, object]]]:
    check_cancelled(cancel_token)
    ok, msg = app.start(candidate)
    if not ok:
        record = adaptive_record_from_candidate(candidate, objective, 'start failed', detail=msg)
        return record, None
    try:
        ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=BENCHMARK_READY_TIMEOUT, cancel_token=cancel_token)
        if not ready_ok:
            return adaptive_record_from_candidate(candidate, objective, 'not ready', detail=ready_msg), None
        if int(getattr(candidate, 'last_good_ctx', 0) or 0) > 0:
            candidate.ctx = int(candidate.last_good_ctx)
        if int(getattr(candidate, 'last_good_parallel', 0) or 0) > 0:
            candidate.parallel = int(candidate.last_good_parallel)
        if progress:
            progress(
                f'adaptive {objective} ready: ctx={candidate.ctx} slot={ctx_per_slot(candidate)} '
                f'parallel={candidate.parallel}; measuring...'
            )
        benchmark_completion(
            candidate,
            max_tokens=BENCHMARK_WARMUP_TOKENS,
            timeout=BENCHMARK_WARMUP_TIMEOUT,
            cancel_token=cancel_token,
        )
        bench_ok, bench = benchmark_completion_suite(
            candidate,
            max_tokens=BENCHMARK_SAMPLE_TOKENS,
            timeout=BENCHMARK_SAMPLE_TIMEOUT,
            cancel_token=cancel_token,
        )
        if not bench_ok:
            return adaptive_record_from_candidate(candidate, objective, 'benchmark failed', detail=str(bench.get('error', 'unknown error'))), None
        snap = app.hardware_profile(refresh=True)
        score = float(bench.get('tokens_per_sec', 0.0) or 0.0)
        elapsed = float(bench.get('elapsed', 0.0) or 0.0)
        record = adaptive_record_from_candidate(
            candidate,
            objective,
            'ok',
            tokens_per_sec=score,
            seconds=elapsed,
            detail=f'{int(bench.get("sample_count", 1) or 1)} samples',
            ram_available=int(getattr(snap, 'memory_available', 0) or 0),
            gpu_memory_free=int(getattr(snap, 'gpu_memory_free', 0) or 0),
        )
        measured = dict(record)
        measured['model'] = ModelConfig(**asdict(candidate))
        return record, measured
    finally:
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
            ok, msg = app.start(candidate)
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
                    record=adaptive_record_from_candidate(candidate, 'long_context', 'start failed', detail=msg),
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


def enrich_exhaustive_record(
    record: Dict[str, object],
    candidate: ModelConfig,
    variant: str,
    retry_attempt: int,
    estimated_safe_ctx: int,
    scan_level: str = 'broad',
    break_point: bool = False,
) -> Dict[str, object]:
    record['variant'] = variant
    record['retry_attempt'] = retry_attempt
    record['scan_level'] = scan_level
    record['break_point'] = bool(break_point)
    record['estimated_safe_ctx'] = int(estimated_safe_ctx or 0)
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
):
    emit_benchmark_event(
        progress,
        'benchmark_result',
        model,
        'server',
        message=(
            f'coarse-to-fine {record.get("objective")} {record.get("status")}: '
            f'{float(record.get("tokens_per_sec", 0.0) or 0.0):.2f} tok/s '
            f'ctx={record.get("ctx")} slot={record.get("ctx_per_slot")} '
            f'par={record.get("parallel")} variant={record.get("variant")}'
        ),
        phase='measuring coarse-to-fine candidates',
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
) -> Tuple[bool, bool, List[Dict[str, object]], List[Dict[str, object]], int]:
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    candidate_label = f'{objective}/{variant}/{scan_level} ctx={ctx} par={parallel}'
    for attempt in (1, 2):
        check_cancelled(cancel_token)
        candidate = configure_adaptive_candidate(base_model, profile, objective, ctx, parallel, variant)
        estimated_safe_ctx = candidate_safe_context_estimate(candidate, profile)
        if progress:
            progress(
                f'exhaustive candidate {candidate_label} attempt={attempt} '
                f'estimated_safe_ctx={estimated_safe_ctx}'
            )
        emit_benchmark_event(
            progress,
            'benchmark_candidate',
            base_model,
            'server',
            message=f'coarse-to-fine candidate {candidate_label} attempt={attempt}',
            phase='measuring coarse-to-fine candidates',
            completed=completed,
            total=total,
            candidate=candidate_label,
        )
        record, measured_item = benchmark_adaptive_candidate(app, candidate, objective, progress, cancel_token)
        completed += 1
        ok = record.get('status') == 'ok'
        break_point = not ok and attempt == 2
        enrich_exhaustive_record(record, candidate, variant, attempt, estimated_safe_ctx, scan_level=scan_level, break_point=break_point)
        records.append(record)
        if measured_item:
            measured_item['variant'] = variant
            measured_item['retry_attempt'] = attempt
            measured_item['scan_level'] = scan_level
            measured.append(measured_item)
        emit_exhaustive_result(progress, base_model, record, completed, total, candidate_label)
        if ok:
            return True, False, records, measured, completed
        if attempt == 1 and progress:
            progress(f'coarse-to-fine candidate {candidate_label} failed once; retrying to confirm break...')
    return False, True, records, measured, completed


def benchmark_exhaustive_profiles(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[object], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    status, _detail = app.health(model)
    if status in ('READY', 'LOADING', 'STARTING') or app.get_pid(model):
        return False, '❌ Stop the model before running coarse-to-fine benchmark profiles.'

    profile = app.hardware_profile(refresh=True)
    started_at = datetime.now().isoformat(timespec='seconds')
    run_id = f'server-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048) or 2048))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072) or 131072))
    contexts = exhaustive_context_ladder(ctx_min, ctx_max)
    variants = exhaustive_variants(model, profile)
    broad_parallel_count = len(exhaustive_parallel_values(profile))
    total = max(1, len(variants) * len(contexts) * (2 + broad_parallel_count) * 2)
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    current: Optional[ModelConfig] = None
    completed = 0

    running_model = ModelConfig(**asdict(model))
    running_model.default_benchmark_status = 'running'
    running_run = build_benchmark_run(run_id, 'server', 'running', [], {}, started_at, hardware=profile.short_summary())
    upsert_benchmark_run(running_model, running_run)
    app.add_or_update(running_model)

    start_msg = (
        f'coarse-to-fine benchmark started: ctx={ctx_min}..{ctx_max}, '
        f'steps={COARSE_CONTEXT_LOW_STEP}/{COARSE_CONTEXT_MID_STEP}/{COARSE_CONTEXT_HIGH_STEP}, '
        f'variants={",".join(variants)}, {profile.short_summary()}'
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

    try:
        for variant in variants:
            check_cancelled(cancel_token)
            variant_contexts: List[int] = []
            context_levels: Dict[int, str] = {}
            long_records: List[Dict[str, object]] = []
            tested_contexts = set()
            confirmed_break_ctx = 0
            emit_benchmark_event(
                progress,
                'benchmark_phase',
                model,
                'server',
                message=f'coarse long-context sweep {variant}',
                phase=f'coarse long-context sweep {variant}',
                completed=completed,
                total=total,
                candidate=variant,
            )
            for ctx in contexts:
                check_cancelled(cancel_token)
                current = configure_adaptive_candidate(model, profile, 'long_context', ctx, 1, variant)
                tested_contexts.add(ctx)
                ok, broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                    app,
                    model,
                    profile,
                    'long_context',
                    ctx,
                    1,
                    variant,
                    progress,
                    cancel_token,
                    completed,
                    total,
                    scan_level='broad',
                )
                records.extend(new_records)
                long_records.extend(new_records)
                measured.extend(new_measured)
                if not ok and broke:
                    confirmed_break_ctx = ctx
                    if progress:
                        progress(f'coarse long-context sweep {variant} stopped at break ctx={ctx}')
                    break
                variant_contexts.append(ctx)
                context_levels[ctx] = 'broad'

            if progress:
                progress(f'coarse-to-fine variant {variant} found {len(variant_contexts)} launchable context step(s)')
            if not variant_contexts:
                continue

            if confirmed_break_ctx:
                last_success_ctx = max(variant_contexts)
                refine_contexts = break_refinement_contexts(last_success_ctx, confirmed_break_ctx, tested_contexts)
                if refine_contexts:
                    emit_benchmark_event(
                        progress,
                        'benchmark_phase',
                        model,
                        'server',
                        message=f'break refinement {variant}: {last_success_ctx}..{confirmed_break_ctx}',
                        phase=f'break refinement {variant}',
                        completed=completed,
                        total=total,
                        candidate=variant,
                    )
                for ctx in refine_contexts:
                    check_cancelled(cancel_token)
                    current = configure_adaptive_candidate(model, profile, 'long_context', ctx, 1, variant)
                    tested_contexts.add(ctx)
                    ok, broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                        app,
                        model,
                        profile,
                        'long_context',
                        ctx,
                        1,
                        variant,
                        progress,
                        cancel_token,
                        completed,
                        total,
                        scan_level='break_refine',
                    )
                    records.extend(new_records)
                    long_records.extend(new_records)
                    measured.extend(new_measured)
                    if not ok and broke:
                        confirmed_break_ctx = ctx
                        if progress:
                            progress(f'break refinement {variant} confirmed closer break ctx={ctx}')
                        break
                    variant_contexts.append(ctx)
                    context_levels[ctx] = 'break_refine'

            knee_contexts = context_knee_refinement_contexts(long_records, tested_contexts, ctx_max)
            if knee_contexts:
                emit_benchmark_event(
                    progress,
                    'benchmark_phase',
                    model,
                    'server',
                    message=f'knee refinement {variant}: {len(knee_contexts)} context(s)',
                    phase=f'knee refinement {variant}',
                    completed=completed,
                    total=total,
                    candidate=variant,
                )
            for ctx in knee_contexts:
                check_cancelled(cancel_token)
                current = configure_adaptive_candidate(model, profile, 'long_context', ctx, 1, variant)
                tested_contexts.add(ctx)
                ok, _broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                    app,
                    model,
                    profile,
                    'long_context',
                    ctx,
                    1,
                    variant,
                    progress,
                    cancel_token,
                    completed,
                    total,
                    scan_level='knee_refine',
                )
                records.extend(new_records)
                long_records.extend(new_records)
                measured.extend(new_measured)
                if ok:
                    variant_contexts.append(ctx)
                    context_levels[ctx] = 'knee_refine'

            variant_contexts = sorted(set(variant_contexts))

            emit_benchmark_event(
                progress,
                'benchmark_phase',
                model,
                'server',
                message=f'OpenCode-ready sweep {variant}',
                phase=f'opencode-ready sweep {variant}',
                completed=completed,
                total=total,
                candidate=variant,
            )
            for ctx in variant_contexts:
                check_cancelled(cancel_token)
                current = configure_adaptive_candidate(model, profile, 'opencode_ready', ctx, 1, variant)
                ok, broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                    app,
                    model,
                    profile,
                    'opencode_ready',
                    ctx,
                    1,
                    variant,
                    progress,
                    cancel_token,
                    completed,
                    total,
                    scan_level=context_levels.get(ctx, 'broad'),
                )
                records.extend(new_records)
                measured.extend(new_measured)
                if not ok and broke:
                    if progress:
                        progress(f'exhaustive opencode-ready sweep {variant} stopped at break ctx={ctx}')
                    break

            emit_benchmark_event(
                progress,
                'benchmark_phase',
                model,
                'server',
                message=f'coarse fast-chat sweep {variant}',
                phase=f'coarse fast-chat sweep {variant}',
                completed=completed,
                total=total,
                candidate=variant,
            )
            chat_floor = chat_min_ctx_per_slot(model)
            for ctx in variant_contexts:
                fast_records_for_context: List[Dict[str, object]] = []
                tested_parallel = set()
                for parallel in exhaustive_parallel_values(profile):
                    check_cancelled(cancel_token)
                    if ctx // max(1, parallel) < chat_floor:
                        break
                    current = configure_adaptive_candidate(model, profile, 'fast_chat', ctx, parallel, variant)
                    tested_parallel.add(parallel)
                    ok, broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                        app,
                        model,
                        profile,
                        'fast_chat',
                        ctx,
                        parallel,
                        variant,
                        progress,
                        cancel_token,
                        completed,
                        total,
                        scan_level=context_levels.get(ctx, 'broad'),
                    )
                    records.extend(new_records)
                    fast_records_for_context.extend(new_records)
                    measured.extend(new_measured)
                    if not ok and broke:
                        if progress:
                            progress(
                                f'coarse fast-chat parallel sweep {variant} ctx={ctx} '
                                f'stopped at parallel={parallel}'
                            )
                        break
                successful_fast = [record for record in fast_records_for_context if record.get('status') == 'ok']
                if successful_fast:
                    best_parallel = int(max(
                        successful_fast,
                        key=lambda record: float(record.get('tokens_per_sec', 0.0) or 0.0),
                    ).get('parallel', 1) or 1)
                    for parallel in parallel_refinement_values(profile, best_parallel, tested_parallel):
                        check_cancelled(cancel_token)
                        if ctx // max(1, parallel) < chat_floor:
                            continue
                        current = configure_adaptive_candidate(model, profile, 'fast_chat', ctx, parallel, variant)
                        ok, _broke, new_records, new_measured, completed = benchmark_exhaustive_candidate_with_retry(
                            app,
                            model,
                            profile,
                            'fast_chat',
                            ctx,
                            parallel,
                            variant,
                            progress,
                            cancel_token,
                            completed,
                            total,
                            scan_level='parallel_refine',
                        )
                        records.extend(new_records)
                        measured.extend(new_measured)
            if progress:
                progress(f'coarse-to-fine variant {variant} finished {len(variant_contexts)} context step(s)')
    except CancelledError:
        if current is not None:
            app.stop(current, managed_only=True)
            records.append(enrich_exhaustive_record(
                adaptive_record_from_candidate(current, 'coarse-to-fine', 'aborted', detail='user requested abort'),
                current,
                str(getattr(current, 'variant', 'default') or 'default'),
                1,
                0,
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
    annotate_spectrum_records(records, winners)
    ended_at = datetime.now().isoformat(timespec='seconds')
    saved = ModelConfig(**asdict(model))
    saved.last_benchmark_results = records
    saved.measured_profiles = winners
    saved.benchmark_fingerprint = app.model_fingerprint(saved)
    saved.default_benchmark_at = ended_at
    status_text = 'done' if winners else 'failed'
    run = build_benchmark_run(run_id, 'server', status_text, records, winners, started_at, ended_at, profile.short_summary())
    upsert_benchmark_run(saved, run)

    if not winners:
        saved.default_benchmark_status = 'failed'
        app.add_or_update(saved)
        msg = '❌ exhaustive benchmark failed: no measured candidates completed'
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

    auto_profile = winners['auto']
    apply_measured_profile(saved, 'auto')
    saved.measured_profiles = winners
    saved.last_benchmark_tokens_per_sec = round(float(auto_profile.get('tokens_per_sec', 0.0) or 0.0), 2)
    saved.last_benchmark_seconds = round(float(auto_profile.get('seconds', 0.0) or 0.0), 2)
    saved.last_benchmark_profile = (
        f'auto/coarse-to-fine {saved.last_benchmark_tokens_per_sec:.2f} tok/s '
        f'ctx={auto_profile.get("ctx")} slot={auto_profile.get("ctx_per_slot")} {profile.short_summary()}'
    )
    saved.default_benchmark_status = 'done'
    app.add_or_update(saved)
    sync_msg = sync_opencode_after_tuning(app)
    msg = (
        f'✅ coarse-to-fine profiles saved: fast={winners["fast_chat"]["tokens_per_sec"]:.2f} tok/s, '
        f'long ctx/slot={winners["long_context"]["ctx_per_slot"]}, '
        f'opencode ctx/slot={winners["opencode_ready"]["ctx_per_slot"]}, '
        f'auto ctx={saved.ctx} parallel={saved.parallel} | {sync_msg}'
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


def benchmark_adaptive_profiles(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
    time_budget_seconds: int = ADAPTIVE_BENCHMARK_TIME_BUDGET_SECONDS,
) -> Tuple[bool, str]:
    status, _detail = app.health(model)
    if status in ('READY', 'LOADING', 'STARTING') or app.get_pid(model):
        return False, '❌ Stop the model before running adaptive benchmark profiles.'
    profile = app.hardware_profile(refresh=True)
    deadline = time.monotonic() + max(60, int(time_budget_seconds or ADAPTIVE_BENCHMARK_TIME_BUDGET_SECONDS))
    records: List[Dict[str, object]] = []
    measured: List[Dict[str, object]] = []
    running_model = ModelConfig(**asdict(model))
    running_model.default_benchmark_status = 'running'
    app.add_or_update(running_model)
    if progress:
        progress(f'adaptive benchmark started: budget≈{int(time_budget_seconds // 60)}m, {profile.short_summary()}')
    emit_benchmark_event(
        progress,
        'benchmark_started',
        model,
        'server',
        message=f'adaptive benchmark started: budget≈{int(time_budget_seconds // 60)}m, {profile.short_summary()}',
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
    annotate_spectrum_records(records, winners)
    saved = ModelConfig(**asdict(model))
    saved.last_benchmark_results = records
    saved.measured_profiles = winners
    saved.benchmark_fingerprint = app.model_fingerprint(saved)
    saved.default_benchmark_at = datetime.now().isoformat(timespec='seconds')
    if not winners:
        saved.default_benchmark_status = 'failed'
        app.add_or_update(saved)
        msg = '❌ adaptive benchmark failed: no measured candidates completed'
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

    auto_profile = winners['auto']
    apply_measured_profile(saved, 'auto')
    saved.measured_profiles = winners
    saved.last_benchmark_tokens_per_sec = round(float(auto_profile.get('tokens_per_sec', 0.0) or 0.0), 2)
    saved.last_benchmark_seconds = round(float(auto_profile.get('seconds', 0.0) or 0.0), 2)
    saved.last_benchmark_profile = (
        f'auto/measured {saved.last_benchmark_tokens_per_sec:.2f} tok/s '
        f'ctx={auto_profile.get("ctx")} slot={auto_profile.get("ctx_per_slot")} {profile.short_summary()}'
    )
    saved.default_benchmark_status = 'done'
    app.add_or_update(saved)
    sync_msg = sync_opencode_after_tuning(app)
    msg = (
        f'✅ adaptive profiles saved: fast={winners["fast_chat"]["tokens_per_sec"]:.2f} tok/s, '
        f'long ctx/slot={winners["long_context"]["ctx_per_slot"]}, '
        f'opencode ctx/slot={winners["opencode_ready"]["ctx_per_slot"]}, '
        f'auto ctx={saved.ctx} parallel={saved.parallel} | {sync_msg}'
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


def benchmark_best_optimization(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    return benchmark_exhaustive_profiles(app, model, progress=progress, cancel_token=cancel_token)


def safe_bootstrap_benchmark(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
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
