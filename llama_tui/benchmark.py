import json
import re
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from urllib import request

from .control import CancelToken, CancelledError, check_cancelled, sleep_with_cancel
from .gguf import set_model_extra_arg
from .hardware import HardwareProfile
from .models import ModelConfig
from .optimize import (
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


def sync_opencode_after_tuning(app: AppConfig) -> str:
    if not app.opencode.path:
        return 'opencode.path unset; skipped opencode sync'
    ok, msg = app.generate_opencode()
    return msg if ok else f'opencode sync failed: {msg}'
def append_model_log(app: AppConfig, model: ModelConfig, text: str):
    app.append_log(model.id, text)
def concise_failure(text: str, limit: int = 320) -> str:
    message = compact_message(text)
    if len(message) <= limit:
        return message
    return message[: max(0, limit - 3)] + '...'
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
            progress(f'{model.id} is stopped; launching best profile before OpenCode...')
        ready_ok, ready_msg = launch_with_failsafe(app, model, 'best', 'auto', progress=progress, cancel_token=cancel_token)
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


def benchmark_best_optimization(
    app: AppConfig,
    model: ModelConfig,
    progress: Optional[Callable[[str], None]] = None,
    cancel_token: Optional[CancelToken] = None,
) -> Tuple[bool, str]:
    profile = app.hardware_profile(refresh=True)
    return _run_server_benchmark_candidates(
        app,
        model,
        benchmark_candidate_models(model, profile),
        profile,
        'benchmark optimization',
        progress=progress,
        cancel_token=cancel_token,
        update_default_status=True,
    )


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
