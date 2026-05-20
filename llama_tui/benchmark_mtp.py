import os
import re
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .engines import ENGINE_LLAMA_CPP_MTP
from .launch_profiles import BenchmarkLaunchProfile
from .models import ModelConfig


def _mtp_env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


MTP_DECODE_HEAVY_PROMPTS = (
    (
        'Write a concise technical checklist for keeping a local language model '
        'server fast and stable. Use short bullet points.'
    ),
    (
        'Explain how to diagnose a CUDA out-of-memory error in a local inference '
        'server. Include practical steps and keep the answer compact.'
    ),
)
MTP_PROMPT_HEAVY_CONTEXT = (
    'You are reviewing a local LLM control-plane benchmark design. The system manages '
    'llama.cpp-family servers, launch profiles, context limits, GPU offload, KV cache '
    'settings, model provenance, OpenCode and Hermes readiness, and measured profile '
    'selection. The benchmark must compare a no-speculative baseline against MTP draft '
    'settings without confusing startup success with throughput. MTP can improve decode '
    'speed when draft acceptance is high, but it can regress prompt processing and add '
    'memory pressure. The optimizer should reject failed launches, empty API responses, '
    'very low acceptance, and profiles that are slower for prompt-heavy workflows. It '
    'should keep separate recommendations for fast chat, safe launch, long context, and '
    'OpenCode readiness. It must not enable MTP for vision/mmproj models, and it should '
    'explain blocked binary capability states in plain language. '
) * 6
MTP_PROMPT_HEAVY_PROMPTS = (
    MTP_PROMPT_HEAVY_CONTEXT
    + 'Summarize the benchmark policy as a decision memo with risks, acceptance criteria, '
    + 'and the recommended next action.',
)
MTP_WORKLOAD_OUTPUT_CAPS = {
    'decode_heavy': {'fast': 128, 'full': 160},
    'prompt_heavy': {'fast': 96, 'full': 112},
}


def mtp_optimizer_workload_specs(depth: str = 'fast') -> Tuple[Dict[str, object], ...]:
    depth_key = 'fast' if str(depth or '').strip().lower() == 'fast' else 'full'
    return (
        {
            'name': 'decode_heavy',
            'prompts': MTP_DECODE_HEAVY_PROMPTS,
            'max_tokens': MTP_WORKLOAD_OUTPUT_CAPS['decode_heavy'][depth_key],
        },
        {
            'name': 'prompt_heavy',
            'prompts': MTP_PROMPT_HEAVY_PROMPTS,
            'max_tokens': MTP_WORKLOAD_OUTPUT_CAPS['prompt_heavy'][depth_key],
        },
    )


def _mtp_workload_profile(
    launch_profile: BenchmarkLaunchProfile,
    workload_name: str,
    max_tokens: int,
) -> BenchmarkLaunchProfile:
    output = max(int(getattr(launch_profile, 'output', 0) or 0), int(max_tokens or 1))
    return replace(
        launch_profile,
        name=f'{launch_profile.name}_{workload_name}',
        output=output,
        measurement_output=max(1, int(max_tokens or 1)),
    )


def _mtp_workload_result(name: str, bench: Dict[str, object]) -> Dict[str, object]:
    elapsed = float(bench.get('elapsed', 0.0) or 0.0)
    prompt_tokens = int(bench.get('prompt_tokens', 0) or 0)
    completion_tokens = int(bench.get('completion_tokens', 0) or 0)
    prompt_workload_tps = round((prompt_tokens / elapsed) if elapsed > 0 else 0.0, 4)
    sample_scores = [float(item) for item in list(bench.get('sample_tokens_per_sec', []) or []) if float(item or 0.0) > 0.0]
    if len(sample_scores) == 2:
        steady_tokens_per_sec = sample_scores[-1]
    elif sample_scores:
        steady_tokens_per_sec = statistics.median(sample_scores)
    else:
        steady_tokens_per_sec = float(bench.get('tokens_per_sec', 0.0) or 0.0)
    return {
        'name': name,
        'elapsed': round(elapsed, 4),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'tokens_per_sec': round(float(steady_tokens_per_sec), 4),
        'first_sample_tokens_per_sec': round(float(sample_scores[0]), 4) if sample_scores else 0.0,
        'steady_sample_tokens_per_sec': round(float(steady_tokens_per_sec), 4),
        'prompt_workload_tokens_per_sec': prompt_workload_tps,
        'sample_count': int(bench.get('sample_count', 0) or 0),
        'sample_tokens_per_sec': list(bench.get('sample_tokens_per_sec', []) or []),
        'error': str(bench.get('error', '') or ''),
    }


CompletionSuite = Callable[..., Tuple[bool, Dict]]


def benchmark_mtp_optimizer_workloads(
    model: ModelConfig,
    launch_profile: BenchmarkLaunchProfile,
    depth: str = 'fast',
    timeout: int = 240,
    cancel_token=None,
    deadline=None,
    completion_suite: Optional[CompletionSuite] = None,
) -> Tuple[bool, Dict[str, object]]:
    if completion_suite is None:
        raise ValueError('completion_suite is required')
    workloads: Dict[str, Dict[str, object]] = {}
    total_elapsed = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_samples = 0
    errors: List[str] = []
    for spec in mtp_optimizer_workload_specs(depth):
        name = str(spec['name'])
        max_tokens = int(spec['max_tokens'])
        workload_profile = _mtp_workload_profile(launch_profile, name, max_tokens)
        ok, bench = completion_suite(
            model,
            max_tokens=max_tokens,
            timeout=timeout,
            cancel_token=cancel_token,
            launch_profile=workload_profile,
            deadline=deadline,
            prompts=tuple(spec['prompts']),
        )
        if not ok:
            errors.append(f'{name}: {bench.get("error", "unknown error")}')
            workloads[name] = {
                'name': name,
                'elapsed': 0.0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'tokens_per_sec': 0.0,
                'prompt_workload_tokens_per_sec': 0.0,
                'sample_count': 0,
                'sample_tokens_per_sec': [],
                'error': str(bench.get('error', 'unknown error')),
            }
            continue
        texts = [str(item or '').strip() for item in list(bench.get('texts', []) or [])]
        if int(bench.get('completion_tokens', 0) or 0) <= 0 or not any(texts):
            errors.append(f'{name}: empty benchmark output')
            bench['error'] = 'empty benchmark output'
        result = _mtp_workload_result(name, bench)
        workloads[name] = result
        total_elapsed += float(result['elapsed'])
        total_prompt_tokens += int(result['prompt_tokens'])
        total_completion_tokens += int(result['completion_tokens'])
        total_samples += int(result['sample_count'])

    if errors:
        return False, {
            'error': '; '.join(errors),
            'mtp_workloads': workloads,
            'elapsed': total_elapsed,
            'completion_tokens': total_completion_tokens,
            'prompt_tokens': total_prompt_tokens,
            'tokens_per_sec': 0.0,
            'prompt_workload_tokens_per_sec': 0.0,
            'sample_count': total_samples,
        }

    decode = workloads.get('decode_heavy', {})
    prompt_heavy = workloads.get('prompt_heavy', {})
    return True, {
        'elapsed': total_elapsed,
        'completion_tokens': total_completion_tokens,
        'prompt_tokens': total_prompt_tokens,
        'tokens_per_sec': float(decode.get('tokens_per_sec', 0.0) or 0.0),
        'prompt_workload_tokens_per_sec': float(prompt_heavy.get('prompt_workload_tokens_per_sec', 0.0) or 0.0),
        'sample_count': total_samples,
        'error': '',
        'mtp_workloads': workloads,
    }


# Minimum draft tokens for an acceptance sample to be treated as reliable.
# A handful of tokens at 100% acceptance is noise, not a result.
MTP_MIN_RELIABLE_DRAFT_TOKENS = _mtp_env_int(
    'LLAMA_TUI_MTP_MIN_DRAFT_TOKENS', 32, minimum=1,
)


def _mtp_record_draft_tokens(item: Dict[str, object]) -> int:
    try:
        return int(item.get('draft_tokens', 0) or 0)
    except (TypeError, ValueError):
        return 0


def best_mtp_acceptance_record(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    # A partial candidate (timed out / guardrail-stopped after producing
    # acceptance metrics) is still a usable winner -- include it, but rank
    # completed `ok` runs and reliable-size samples first so a tiny 100%
    # sample never wins over a real measurement.
    candidates = [
        dict(item)
        for item in records
        if str(item.get('status', '') or '') in ('ok', 'partial')
        and bool(item.get('mtp_enabled'))
        and str(item.get('mtp_risk_level', '') or 'usable') != 'failed'
    ]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda item: (
            1 if str(item.get('status', '') or '') == 'ok' else 0,
            1 if _mtp_record_draft_tokens(item) >= MTP_MIN_RELIABLE_DRAFT_TOKENS else 0,
            float(item.get('accept_rate', 0.0) or 0.0),
            float(item.get('tokens_per_sec', 0.0) or 0.0),
            int(item.get('mtp_draft_n_max', 0) or 0),
        ),
    )


def weighted_mtp_accept_rate(
    records: Sequence[Dict[str, object]],
) -> Tuple[float, int, int, str]:
    """Token-weighted MTP acceptance with a reliability tag.

    Returns ``(accept_rate, total_draft_tokens, sample_count, reliability)``
    where ``reliability`` is one of:

    - ``'reliable'`` — at least one record met ``MTP_MIN_RELIABLE_DRAFT_TOKENS``
      and only those records were included; tiny samples were excluded so they
      could not dominate.
    - ``'sparse'`` — no record met the threshold but at least one tiny MTP
      sample was available. The rate is still computed from the tiny samples
      instead of silently returning zero, so the selector can down-weight the
      result rather than treat it as missing data.
    - ``'none'`` — no MTP-enabled records with positive draft tokens at all.
    """
    reliable_gen = 0
    reliable_acc = 0
    reliable_samples = 0
    sparse_gen = 0
    sparse_acc = 0
    sparse_samples = 0
    for item in records:
        if not bool(item.get('mtp_enabled')):
            continue
        gen = _mtp_record_draft_tokens(item)
        if gen <= 0:
            continue
        try:
            acc = int(item.get('accepted_tokens', 0) or 0)
        except (TypeError, ValueError):
            acc = 0
        if acc <= 0:
            acc = int(round(_record_float(item, 'accept_rate') * gen))
        acc = min(max(acc, 0), gen)
        if gen >= MTP_MIN_RELIABLE_DRAFT_TOKENS:
            reliable_gen += gen
            reliable_acc += acc
            reliable_samples += 1
        else:
            sparse_gen += gen
            sparse_acc += acc
            sparse_samples += 1
    if reliable_gen > 0:
        return round(reliable_acc / reliable_gen, 4), reliable_gen, reliable_samples, 'reliable'
    if sparse_gen > 0:
        return round(sparse_acc / sparse_gen, 4), sparse_gen, sparse_samples, 'sparse'
    return 0.0, 0, 0, 'none'


def _record_float(record: Dict[str, object], key: str) -> float:
    try:
        return float(record.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _record_int(record: Dict[str, object], key: str) -> int:
    try:
        return int(record.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _record_decode_tps(record: Dict[str, object]) -> float:
    """Decode rate for MTP scoring: server-truth /metrics value when available.

    Falls back to the recorded ``decode_tokens_per_sec`` then the client-side
    ``tokens_per_sec``. The client number conflates long-context prefill into
    decode, so it is the least trustworthy for MTP gain comparisons.
    """
    server = _record_float(record, 'server_decode_tokens_per_sec')
    if server > 0:
        return server
    decode = _record_float(record, 'decode_tokens_per_sec')
    if decode > 0:
        return decode
    return _record_float(record, 'tokens_per_sec')


def _mtp_baseline_record(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    for item in records:
        if bool(item.get('mtp_enabled')):
            continue
        if str(item.get('benchmark_phase', '') or '') == 'baseline_no_mtp':
            return dict(item)
    return {}


def _mtp_acceptance_missing(record: Dict[str, object]) -> bool:
    source = str(record.get('mtp_acceptance_source', '') or '').strip().lower()
    return source == 'not_reported' and _record_float(record, 'accept_rate') <= 0.0


def _record_prompt_workload_tps(record: Optional[Dict[str, object]]) -> float:
    if not record:
        return 0.0
    if 'prompt_workload_tokens_per_sec' in record:
        return _record_float(record, 'prompt_workload_tokens_per_sec')
    return _record_float(record, 'prompt_tokens_per_sec')


def _mtp_candidate_risk(record: Dict[str, object], baseline: Dict[str, object]) -> Tuple[str, str]:
    if str(record.get('status', '') or '') != 'ok':
        return 'failed', str(record.get('failure_reason') or record.get('detail') or 'candidate did not complete')
    if not bool(record.get('mtp_enabled')):
        return 'baseline', 'no-MTP baseline'
    if _mtp_acceptance_missing(record):
        return 'risky', 'acceptance rate was not reported by the runtime log'
    accept_rate = _record_float(record, 'accept_rate')
    if accept_rate < 0.60:
        if _record_int(record, 'mtp_draft_n_max') >= 3:
            return 'failed', f'draft_n=3 accept_rate {accept_rate:.0%} is below 60%'
        return 'risky', f'accept_rate {accept_rate:.0%} is below 70%'
    if accept_rate < 0.70:
        return 'risky', f'accept_rate {accept_rate:.0%} is below 70%'
    prefill_cost = _record_float(record, 'prefill_cost_vs_baseline')
    if prefill_cost > 0.50:
        return 'failed', f'prefill cost {prefill_cost:.0%} is above 50%'
    decode_gain = _record_float(record, 'decode_gain_vs_baseline')
    baseline_available = bool(baseline and str(baseline.get('status', '') or '') == 'ok')
    if accept_rate >= 0.80 and baseline_available and decode_gain >= 1.50:
        return 'excellent', f'accept_rate {accept_rate:.0%}, decode gain {decode_gain:.2f}x'
    if accept_rate >= 0.75 and (not baseline_available or decode_gain > 1.0):
        gain_text = f', decode gain {decode_gain:.2f}x' if baseline_available else ''
        return 'good', f'accept_rate {accept_rate:.0%}{gain_text}'
    return 'usable', f'accept_rate {accept_rate:.0%}'


def annotate_mtp_optimizer_records(records: Sequence[Dict[str, object]], spec_type: str = '') -> List[Dict[str, object]]:
    mutable = [dict(item) for item in records]
    baseline = _mtp_baseline_record(mutable)
    baseline_ok = bool(baseline and str(baseline.get('status', '') or '') == 'ok')
    baseline_tps = _record_decode_tps(baseline) if baseline_ok else 0.0
    baseline_prompt_workload_tps = _record_prompt_workload_tps(baseline) if baseline_ok else 0.0
    baseline_seconds = _record_float(baseline, 'seconds') if baseline_ok else 0.0
    baseline_memory = _record_int(baseline, 'peak_vram_used') if baseline_ok else 0
    baseline_id = str(baseline.get('benchmark_phase') or 'baseline_no_mtp') if baseline else ''
    if baseline and str(baseline.get('status', '') or '') == 'skipped_runtime_assert':
        baseline_id = 'baseline_no_mtp:skipped_runtime_assert'

    for record in mutable:
        record['kind'] = 'mtp_optimizer'
        record['mtp_objective'] = 'decode_heavy'
        record['mtp_spec_type'] = str(record.get('spec_type') or spec_type or '')
        record['total_wall_seconds'] = _record_float(record, 'seconds')
        record['loaded_vram_bytes'] = _record_int(record, 'peak_vram_used')
        record['loaded_ram_bytes'] = _record_int(record, 'peak_ram')
        record['final_command'] = str(record.get('command') or record.get('effective_server_command') or '')
        record['baseline_profile_id'] = baseline_id
        if str(record.get('status', '') or '') == 'ok' and bool(record.get('mtp_enabled')) and baseline_ok:
            current_tps = _record_decode_tps(record)
            current_prompt_workload_tps = _record_prompt_workload_tps(record)
            current_seconds = _record_float(record, 'seconds')
            current_memory = _record_int(record, 'peak_vram_used')
            record['decode_gain_vs_baseline'] = round(current_tps / baseline_tps, 4) if baseline_tps > 0 else 0.0
            record['prefill_cost_vs_baseline'] = (
                round(
                    max(
                        0.0,
                        (baseline_prompt_workload_tps - current_prompt_workload_tps) / baseline_prompt_workload_tps,
                    ),
                    4,
                )
                if baseline_prompt_workload_tps > 0 and current_prompt_workload_tps > 0 else 0.0
            )
            record['total_wall_gain_vs_baseline'] = round(baseline_seconds / current_seconds, 4) if baseline_seconds > 0 and current_seconds > 0 else 0.0
            record['memory_delta_vs_baseline'] = int(current_memory - baseline_memory)
        elif str(record.get('benchmark_phase', '') or '') == 'baseline_no_mtp' and str(record.get('status', '') or '') == 'ok':
            record['decode_gain_vs_baseline'] = 1.0
            record['prefill_cost_vs_baseline'] = 0.0
            record['total_wall_gain_vs_baseline'] = 1.0
            record['memory_delta_vs_baseline'] = 0
        else:
            record.setdefault('decode_gain_vs_baseline', 0.0)
            record.setdefault('prefill_cost_vs_baseline', 0.0)
            record.setdefault('total_wall_gain_vs_baseline', 0.0)
            record.setdefault('memory_delta_vs_baseline', 0)
        risk, reason = _mtp_candidate_risk(record, baseline)
        record['mtp_risk_level'] = risk
        record['mtp_recommendation_reason'] = reason
    return mutable


def mtp_optimizer_profile_recommendations(records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    annotated = [dict(item) for item in records]
    recommendations: Dict[str, Dict[str, object]] = {}
    baseline = _mtp_baseline_record(annotated)
    baseline_ok = bool(baseline and str(baseline.get('status', '') or '') == 'ok')
    if baseline:
        recommendations['mtp_baseline_no_spec'] = dict(baseline)
    candidates = [
        dict(item)
        for item in annotated
        if str(item.get('status', '') or '') == 'ok'
        and bool(item.get('mtp_enabled'))
        and str(item.get('mtp_risk_level', '') or '') not in ('', 'failed')
    ]
    if not candidates:
        return recommendations
    risk_rank = {'excellent': 0, 'good': 1, 'usable': 2, 'risky': 3}

    def safer_within_ten_percent(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
        if not items:
            return {}
        fastest = max(items, key=lambda item: (_record_float(item, 'tokens_per_sec'), _record_float(item, 'accept_rate')))
        fastest_tps = _record_float(fastest, 'tokens_per_sec')
        fastest_accept = _record_float(fastest, 'accept_rate')
        safer = [
            item for item in items
            if item is not fastest
            and _record_float(item, 'tokens_per_sec') >= fastest_tps * 0.90
            and _record_float(item, 'accept_rate') >= fastest_accept + 0.05
            and risk_rank.get(str(item.get('mtp_risk_level', '') or ''), 9) < risk_rank.get(str(fastest.get('mtp_risk_level', '') or ''), 9)
        ]
        if safer:
            return max(
                safer,
                key=lambda item: (
                    -risk_rank.get(str(item.get('mtp_risk_level', '') or ''), 9),
                    _record_float(item, 'accept_rate'),
                    _record_float(item, 'tokens_per_sec'),
                ),
            )
        return fastest

    def q8_no_mmap_score(item: Dict[str, object]) -> Tuple[int, int, int]:
        kv = str(item.get('kv_preset', '') or '').strip().lower()
        draft_kv = str(item.get('mtp_draft_kv_preset', '') or '').strip().lower()
        target_q8 = kv == 'q8_0/q8_0' or (
            str(item.get('ctk', '') or '').strip().lower() == 'q8_0'
            and str(item.get('ctv', '') or '').strip().lower() == 'q8_0'
        )
        draft_q8 = draft_kv == 'q8_0/q8_0' or (
            str(item.get('draft_ctk', '') or '').strip().lower() == 'q8_0'
            and str(item.get('draft_ctv', '') or '').strip().lower() == 'q8_0'
        )
        return (
            1 if target_q8 else 0,
            1 if draft_q8 else 0,
            1 if bool(item.get('no_mmap', item.get('runtime_no_mmap', False))) else 0,
        )

    fast_candidates = [
        item for item in candidates
        if not baseline_ok or _record_float(item, 'decode_gain_vs_baseline') > 1.05
    ]
    if fast_candidates:
        recommendations['mtp_fast_chat'] = safer_within_ten_percent(fast_candidates)
    safe_candidates = [
        item for item in candidates
        if str(item.get('mtp_risk_level', '') or '') in ('excellent', 'good', 'usable')
        and (
            not baseline_ok
            or (
                _record_float(item, 'decode_gain_vs_baseline') > 1.0
                and _record_float(item, 'prefill_cost_vs_baseline') <= 0.30
            )
        )
    ]
    if safe_candidates:
        recommendations['mtp_safe'] = max(
            safe_candidates,
            key=lambda item: (
                -risk_rank.get(str(item.get('mtp_risk_level', '') or ''), 9),
                _record_float(item, 'accept_rate'),
                -(_record_int(item, 'mtp_draft_n_max') if _record_int(item, 'mtp_draft_n_max') > 0 else 99),
                _record_float(item, 'tokens_per_sec'),
            ),
        )
        recommendations['mtp_long_context'] = max(
            safe_candidates,
            key=lambda item: (
                _record_int(item, 'ctx') >= 131072,
                _record_int(item, 'ctx'),
                *q8_no_mmap_score(item),
                _record_int(item, 'gpu_memory_free') or _record_int(item, 'final_vram_free_bytes'),
                _record_float(item, 'accept_rate'),
                _record_float(item, 'tokens_per_sec'),
            ),
        )
    opencode_candidates = [
        item for item in safe_candidates
        if baseline_ok
        if _record_float(item, 'total_wall_gain_vs_baseline') > 1.0
        and _record_float(item, 'prefill_cost_vs_baseline') <= 0.30
    ]
    if opencode_candidates:
        recommendations['mtp_opencode_ready'] = max(
            opencode_candidates,
            key=lambda item: (
                _record_float(item, 'total_wall_gain_vs_baseline'),
                _record_float(item, 'tokens_per_sec'),
            ),
        )
    return recommendations


# ============================================================================
# Runtime-log parsing for MTP draft acceptance metrics.
# Extracted from benchmark.py as part of audit finding #6 step 5. The parsers
# below convert llama-server stderr/log text into the structured fields the
# scoring/recommendation code further up the module consumes. Pure functions
# (plus one file read in enrich_mtp_acceptance_metrics, kept tightly scoped).
# ============================================================================


_DRAFT_MTP_STAT_RE = re.compile(
    r'statistics\s+draft-mtp:.*?#gen\s+drafts?\s*=\s*([0-9]+)'
    r'.*?#acc\s+drafts?\s*=\s*([0-9]+)',
    re.IGNORECASE,
)


def _draft_mtp_session_totals(text: str) -> List[Tuple[int, int]]:
    """Return ``[(gen_drafts, acc_drafts), ...]`` -- one entry per server session.

    llama.cpp prints ``statistics draft-mtp: ... #gen drafts = N, #acc drafts = M``
    with counters that are cumulative within a server session and reset to a
    fresh count when a new server starts. Each session therefore contributes its
    final (largest) cumulative line; a drop in ``#gen drafts`` marks a reset.
    """
    sessions: List[Tuple[int, int]] = []
    prev_gen = -1
    last: Optional[Tuple[int, int]] = None
    for match in _DRAFT_MTP_STAT_RE.finditer(text or ''):
        gen = int(match.group(1))
        acc = int(match.group(2))
        if gen < prev_gen and last is not None:
            sessions.append(last)
        last = (gen, acc)
        prev_gen = gen
    if last is not None:
        sessions.append(last)
    return sessions


def parse_mtp_acceptance_metrics(text: str) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    source = str(text or '')
    if not source:
        return payload
    low = source.lower()

    # Preferred path: llama.cpp's own `statistics draft-mtp:` lines. Use the
    # most recent server session (this candidate's launch); the run-summary
    # layer down-weights tiny samples by draft_tokens.
    sessions = _draft_mtp_session_totals(source)
    if sessions:
        gen, acc = sessions[-1]
        if gen > 0:
            payload['draft_tokens'] = gen
            payload['accepted_tokens'] = min(acc, gen)
            payload['accept_rate'] = round(max(0.0, min(1.0, acc / gen)), 4)
            payload['mtp_acceptance_sessions'] = len(sessions)
            return payload

    def first_int(patterns: Sequence[str]) -> int:
        for pattern in patterns:
            match = re.search(pattern, low, flags=re.I)
            if match:
                try:
                    return max(0, int(float(match.group(1))))
                except Exception:
                    continue
        return 0

    draft_tokens = first_int((
        r'\bdraft(?:ed)?[_\s-]*tokens?\s*[:=]\s*([0-9]+)',
        r'\bgenerated[_\s-]*draft[_\s-]*tokens?\s*[:=]\s*([0-9]+)',
        r'\bspec(?:ulative)?[_\s-]*draft(?:ed)?\s*[:=]\s*([0-9]+)',
        r'\bn[_\s-]*draft(?:ed)?\s*[:=]\s*([0-9]+)',
    ))
    accepted_tokens = first_int((
        r'\baccepted[_\s-]*tokens?\s*[:=]\s*([0-9]+)',
        r'\baccepted[_\s-]*draft[_\s-]*tokens?\s*[:=]\s*([0-9]+)',
        r'\bspec(?:ulative)?[_\s-]*accepted\s*[:=]\s*([0-9]+)',
        r'\bn[_\s-]*accepted\s*[:=]\s*([0-9]+)',
    ))
    rate = 0.0
    for pattern in (
        r'\baccept(?:ance)?[_\s-]*rate\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?',
        r'\baccepted\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%',
    ):
        match = re.search(pattern, low, flags=re.I)
        if match:
            try:
                value = float(match.group(1))
                rate = value / 100.0 if value > 1.0 else value
                break
            except Exception:
                continue
    if not rate and draft_tokens > 0 and accepted_tokens > 0:
        rate = min(1.0, accepted_tokens / max(1, draft_tokens))
    if draft_tokens:
        payload['draft_tokens'] = draft_tokens
    if accepted_tokens:
        payload['accepted_tokens'] = accepted_tokens
    if rate:
        payload['accept_rate'] = round(max(0.0, min(1.0, rate)), 4)
    return payload


def parse_mtp_runtime_diagnostics(text: str) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    source = str(text or '')
    if not source:
        return payload
    low = source.lower()
    if 'failed to fit params to free device memory' in low and 'n_gpu_layers already set by user' in low:
        payload['failure_category'] = 'FIXED_GPU_LAYERS_BLOCKED_FIT'
        payload['fit_failed_fixed_gpu_layers'] = True
        payload['suggested_fix'] = 'Use a fit-assisted MTP profile without fixed --n-gpu-layers.'
    if 'tensor overrides to cpu are used with mmap enabled' in low and 'no-mmap' in low:
        payload['warning'] = 'MMAP_CPU_OVERRIDE_SLOWPATH'
        payload['mmap_warning'] = True
        payload['mmap_cpu_override_slowpath'] = True
        payload['recommendation'] = 'Add --no-mmap for MTP profiles with CPU tensor overrides.'
    for line in source.splitlines():
        cpu_mapped = re.search(r'CPU_Mapped model buffer size\s*=\s*([0-9.]+)\s+MiB', line, re.IGNORECASE)
        if cpu_mapped:
            payload['cpu_model_buffer_type'] = 'CPU_Mapped'
            payload['cpu_mapped_model_buffer_mib'] = float(cpu_mapped.group(1))
            continue
        cuda_host = re.search(r'CUDA_Host model buffer size\s*=\s*([0-9.]+)\s+MiB', line, re.IGNORECASE)
        if cuda_host:
            payload['cpu_model_buffer_type'] = 'CUDA_Host'
            payload['cuda_host_model_buffer_mib'] = float(cuda_host.group(1))
            continue
        cuda_model = re.search(r'CUDA\d* model buffer size\s*=\s*([0-9.]+)\s+MiB', line, re.IGNORECASE)
        if cuda_model:
            payload['cuda_model_buffer_mib'] = float(cuda_model.group(1))
            continue
        prompt_eval = re.search(
            r'prompt eval time\s*=.*?/\s*\d+\s+tokens.*?,\s*([0-9.]+)\s+tokens per second',
            line,
            re.IGNORECASE,
        )
        if prompt_eval:
            payload['prompt_eval_tokens_per_sec'] = float(prompt_eval.group(1))
            continue
        eval_time = re.search(
            r'(?<!prompt )eval time\s*=.*?/\s*\d+\s+tokens.*?,\s*([0-9.]+)\s+tokens per second',
            line,
            re.IGNORECASE,
        )
        if eval_time:
            payload['eval_tokens_per_sec'] = float(eval_time.group(1))
            continue
        vram_free = re.search(r'(?:final\s+)?(?:vram|cuda|device).*?free(?: memory)?\s*[:=]\s*([0-9.]+)\s*(GiB|MiB)', line, re.IGNORECASE)
        if vram_free:
            value = float(vram_free.group(1))
            unit = vram_free.group(2).lower()
            payload['final_vram_free_bytes'] = int(value * (1024 ** 3 if unit == 'gib' else 1024 ** 2))
    return payload


def enrich_mtp_acceptance_metrics(record: Dict[str, object]) -> Dict[str, object]:
    is_mtp_record = (
        bool(record.get('mtp_enabled', False))
        or str(record.get('spec_type', '') or '') in ('mtp', 'draft-mtp')
        or str(record.get('engine', '') or '').strip().lower() == ENGINE_LLAMA_CPP_MTP
    )
    if not is_mtp_record:
        return record
    log_path = str(record.get('runtime_log_path', '') or '')
    text = ''
    if log_path:
        try:
            # Read a wide tail so every `statistics draft-mtp:` line for the
            # candidate's server session is captured, not just the last few.
            text = Path(log_path).read_text(encoding='utf-8', errors='replace')[-40000:]
        except Exception:
            text = ''
    metrics = parse_mtp_acceptance_metrics(text)
    diagnostics = parse_mtp_runtime_diagnostics(text)
    if metrics:
        record.update(metrics)
        record['mtp_acceptance_source'] = 'runtime_log'
    else:
        record['mtp_acceptance_source'] = 'not_reported'
    if diagnostics:
        warning = str(diagnostics.pop('warning', '') or '')
        recommendation = str(diagnostics.pop('recommendation', '') or '')
        category = str(diagnostics.get('failure_category', '') or '')
        if category and str(record.get('status', '') or '') != 'ok':
            record['failure_category'] = category
            record['failure_reason'] = record.get('failure_reason') or 'MTP fit was blocked because --n-gpu-layers was fixed.'
            record['suggested_fix'] = record.get('suggested_fix') or diagnostics.get('suggested_fix', '')
        record.update(diagnostics)
        if warning:
            warnings = [str(item) for item in list(record.get('runtime_warnings', []) or [])]
            if warning not in warnings:
                warnings.append(warning)
            record['runtime_warnings'] = warnings
            record['warning'] = warning
        if recommendation:
            record['runtime_recommendation'] = recommendation
    return record
