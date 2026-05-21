"""MoE-tuning pure helpers.

Sixth extraction of audit finding #6 (decompose ``benchmark.py``).
Covers the cohesive pure subset of the MoE expert-placement tuning
slice:

  - layer-count and capability lookups
  - MTP-aware MoE eligibility checks + blocked-reason formatters
  - MTP-aware ``RuntimeProfile`` transform (uses the best saved
    acceptance record to seed ``mtp_draft_n_max``)
  - acceptance-record selection from a model's saved measured profiles
  - NextN/recurrent feature detection
  - context-bucket and validation-ladder specs
  - per-run warning aggregation

The heavy orchestrator ``benchmark_moe_placement_tuning`` stays in
``benchmark.py`` because it depends on ``BenchmarkDeadline``,
``emit_benchmark_event``, ``adaptive_record_from_candidate``,
``build_benchmark_launch_profile``, ``benchmark_effective_server_args``,
``runtime_log_text_for_record``, and many other orchestration helpers.
The orchestrator imports the primitives here and stays a thin
coordinator.
"""

from dataclasses import replace
from typing import Dict, List, Sequence

from .benchmark_mtp import annotate_mtp_optimizer_records
from .gguf import gguf_layer_count
from .model_compat import detect_model_runtime_features
from .models import ModelConfig
from .mtp import MTP_DRAFT_VALUES, clamp_mtp_draft, model_mtp_allowed
from .runtime_profiles import RuntimeProfile, mtp_spec_type_value
from .textutil import compact_message


def _moe_tuning_layer_count(model: ModelConfig) -> int:
    try:
        return max(0, int(gguf_layer_count(model) or 0))
    except Exception:
        return 0


def _model_mtp_acceptance_records(model: ModelConfig) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    measured_profiles = dict(getattr(model, 'measured_profiles', {}) or {})
    mtp_profile = measured_profiles.get('mtp_acceptance')
    if isinstance(mtp_profile, dict):
        records.append(dict(mtp_profile))
    for run in list(getattr(model, 'benchmark_runs', []) or []):
        if not isinstance(run, dict):
            continue
        if str(run.get('benchmark_strategy_id', '') or '') != 'mtp_acceptance_matrix':
            continue
        for row in list(run.get('records', []) or []):
            if isinstance(row, dict):
                records.append(dict(row))
    return records


def _usable_mtp_acceptance_record_for_model(model: ModelConfig) -> Dict[str, object]:
    # Only apply MTP to MoE tuning when we have *throughput-verified* evidence
    # that MTP helps. A 'partial' acceptance record proves the probe accepted
    # drafted tokens (acceptance scraped from server log) but its API request
    # timed out before producing tokens_per_sec, so we have no proof MTP is a
    # net-positive on this hardware/model. Applying MTP based on acceptance
    # alone made baseline tok/s drop ~20% (21.69 -> 18.13) and pushed several
    # n_cpu_moe variants into MEMORY_FIT_FAILED because the MTP draft cache
    # eats VRAM. Require status='ok' so MoE tuning only enables MTP when the
    # probe both ran AND measured a real decode rate. Partial records still
    # surface in best_mtp_acceptance_record() for display.
    records = annotate_mtp_optimizer_records(_model_mtp_acceptance_records(model))
    candidates = [
        dict(item)
        for item in records
        if str(item.get('status', '') or '') == 'ok'
        and bool(item.get('mtp_enabled'))
        and str(item.get('mtp_risk_level', '') or 'usable') != 'failed'
    ]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda item: (
            float(item.get('accept_rate', 0.0) or 0.0),
            float(item.get('tokens_per_sec', 0.0) or 0.0),
            int(item.get('mtp_draft_n_max', 0) or 0),
        ),
    )


def _best_mtp_acceptance_draft_for_model(model: ModelConfig) -> int:
    best = _usable_mtp_acceptance_record_for_model(model)
    if best:
        return clamp_mtp_draft(best.get('mtp_draft_n_max', 0), default=2)
    return 0


def _moe_tuning_mtp_aware(engine: str, model: ModelConfig, capabilities) -> bool:
    # Audit #7: MTP is now a binary capability rather than a dedicated
    # engine. MTP-aware MoE tuning applies to any llama.cpp-family
    # engine when the model is MTP-native AND the binary advertises
    # the speculative MTP flags. vLLM (and any future non-llama.cpp
    # engine) is excluded because the launch path is different.
    if str(engine or '').strip().lower() == 'vllm':
        return False
    features = detect_model_runtime_features(model)
    return bool(
        'mtp_native' in features
        and model_mtp_allowed(model)
        and getattr(capabilities, 'supports_spec_type', False)
        and getattr(capabilities, 'supports_mtp', False)
        and mtp_spec_type_value(capabilities)
        and getattr(capabilities, 'supports_spec_draft_n_max', False)
    )


def _moe_tuning_mtp_required(engine: str, model: ModelConfig) -> bool:
    # Audit #7: an MTP-native model on a llama.cpp-family engine
    # requires MTP launch capability for sensible MoE tuning — no-MTP
    # would be a degraded baseline. vLLM is excluded.
    if str(engine or '').strip().lower() == 'vllm':
        return False
    features = detect_model_runtime_features(model)
    return bool('mtp_native' in features and model_mtp_allowed(model))


def _moe_tuning_mtp_blocked_reason(capabilities) -> str:
    spec_values = tuple(getattr(capabilities, 'spec_type_values', ()) or ()) if capabilities is not None else ()
    supported = ','.join(str(item) for item in spec_values) if spec_values else 'none'
    missing: List[str] = []
    if not getattr(capabilities, 'supports_spec_type', False):
        missing.append('--spec-type')
    if not getattr(capabilities, 'supports_mtp', False) or not mtp_spec_type_value(capabilities):
        missing.append('mtp/draft-mtp spec value')
    if not getattr(capabilities, 'supports_spec_draft_n_max', False):
        missing.append('--spec-draft-n-max')
    missing_text = ', '.join(missing) if missing else 'MTP launch capability'
    return (
        'MTP-aware MoE placement blocked: selected llama.cpp-mtp binary is missing '
        f'{missing_text}; advertised spec types: {supported}'
    )


def _moe_tuning_mtp_acceptance_required_reason(model: ModelConfig) -> str:
    return (
        'MTP-aware MoE placement skipped: no usable MTP acceptance winner is saved. '
        'Run MTP Optimizer first so MoE tuning can reuse a measured draft_n.'
    )


def _mtp_aware_moe_tuning_profile(
    profile: RuntimeProfile,
    model: ModelConfig,
    capabilities,
) -> RuntimeProfile:
    draft_n = _best_mtp_acceptance_draft_for_model(model)
    return replace(
        profile,
        parallel=1,
        mtp_enabled=True,
        mtp_draft_n_max=draft_n if draft_n in MTP_DRAFT_VALUES else 2,
        no_warmup=bool(getattr(profile, 'no_warmup', False) or getattr(capabilities, 'supports_no_warmup', False)),
        benchmark_strategy_id='mtp_aware_moe_placement',
        benchmark_phase='moe_placement_mtp',
        benchmark_metric_group='tg_tps,ttft_ms,tpot_ms,draft_tokens,accepted_tokens,accept_rate,startup_status',
    )


def _model_has_nextn_or_recurrent_features(model: ModelConfig) -> bool:
    features = detect_model_runtime_features(model)
    if 'nextn_native' in features:
        return True
    text = ' '.join(
        str(item or '').lower()
        for item in (
            getattr(model, 'architecture', ''),
            getattr(model, 'model_family', ''),
            getattr(model, 'metadata_summary', ''),
            getattr(model, 'path', ''),
        )
    )
    return any(marker in text for marker in ('nextn', 'next-n', 'next_n', 'recurrent', 'ssm', 'gated_delta_net'))


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
