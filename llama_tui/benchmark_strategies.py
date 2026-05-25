from dataclasses import dataclass
from typing import Optional, Tuple

from .engines import (
    ENGINE_LLAMA_CPP,
    ENGINE_TURBOQUANT,
    normalize_engine_id,
)
from .hardware import HardwareProfile
from .model_compat import detect_model_runtime_features
from .models import ModelConfig
from .runtime_profiles import EngineCapabilities, mtp_spec_type_value


BENCHMARK_OBJECTIVES = (
    'quick_sanity',
    'fast_chat',
    'opencode_agent',
    'long_context',
    'mtp_long_context_probe',
    'max_throughput',
    'memory_fit',
    'quality_safe',
    'experimental_lab',
)


@dataclass(frozen=True)
class BenchmarkPhase:
    id: str
    runner: str
    objective: str
    description: str
    metrics: Tuple[str, ...] = ()
    max_candidates: int = 0
    timeout_seconds: int = 0


@dataclass(frozen=True)
class BenchmarkStrategy:
    id: str
    engine_id: str
    required_features: Tuple[str, ...]
    objectives: Tuple[str, ...]
    phases: Tuple[BenchmarkPhase, ...]
    hard_budget_seconds: int
    max_candidates: int
    retry_policy: str
    reason: str = ''
    metric_groups: Tuple[str, ...] = ()
    blocked_reason: str = ''
    result_statuses: Tuple[str, ...] = (
        'complete',
        'partial',
        'skipped_budget',
        'skipped_incompatible',
        'failed_terminal',
    )


def normalize_benchmark_objective(objective: str) -> str:
    normalized = str(objective or 'quick_sanity').strip().lower()
    return normalized if normalized in BENCHMARK_OBJECTIVES else 'quick_sanity'


def _phase(
    phase_id: str,
    runner: str,
    objective: str,
    description: str,
    metrics: Tuple[str, ...],
    max_candidates: int = 0,
    timeout_seconds: int = 0,
) -> BenchmarkPhase:
    return BenchmarkPhase(
        id=phase_id,
        runner=runner,
        objective=objective,
        description=description,
        metrics=metrics,
        max_candidates=max_candidates,
        timeout_seconds=timeout_seconds,
    )


def _features_for_strategy(
    model: ModelConfig,
    hardware: Optional[HardwareProfile],
    model_size_bytes: int = 0,
) -> Tuple[str, ...]:
    return tuple(sorted(detect_model_runtime_features(model, hardware_profile=hardware, model_size_bytes=model_size_bytes)))


def select_benchmark_strategy(
    engine_id: str,
    model: ModelConfig,
    hardware: Optional[HardwareProfile] = None,
    capabilities: Optional[EngineCapabilities] = None,
    objective: str = 'quick_sanity',
    depth: str = 'fast',
    model_size_bytes: int = 0,
) -> BenchmarkStrategy:
    engine = normalize_engine_id(engine_id)
    requested_objective = normalize_benchmark_objective(objective)
    depth_key = 'fast' if str(depth or '').strip().lower() == 'fast' else 'full'
    features = _features_for_strategy(model, hardware, model_size_bytes=model_size_bytes)
    has_moe = 'moe' in features

    # MTP is a binary capability, not a dedicated engine. Audit #7:
    # ENGINE_LLAMA_CPP_MTP collapsed into ENGINE_LLAMA_CPP, so the gate
    # is now: any llama.cpp-family engine + an MTP-native model OR a
    # MTP-capable binary triggers the acceptance matrix strategy; the
    # strategy reports a blocked_reason when prerequisites are missing.
    mtp_spec_type = mtp_spec_type_value(capabilities)
    spec_values = tuple(getattr(capabilities, 'spec_type_values', ()) or ()) if capabilities is not None else ()
    model_is_mtp = 'mtp_native' in features
    mtp_capable_binary = (
        bool(capabilities is not None)
        and bool(getattr(capabilities, 'supports_spec_type', False))
        and bool(getattr(capabilities, 'supports_mtp', False))
        and bool(mtp_spec_type)
        and bool(getattr(capabilities, 'supports_spec_draft_n_max', False))
    )
    mtp_strategy_applies = (
        engine in (ENGINE_LLAMA_CPP, ENGINE_TURBOQUANT) and model_is_mtp
    )
    if mtp_strategy_applies:
        blocked_reason = ''
        if not model_is_mtp:
            blocked_reason = 'model is not detected as MTP-native'
        elif not mtp_capable_binary:
            supported = ','.join(str(item) for item in spec_values) if spec_values else 'none'
            blocked_reason = (
                'selected binary does not advertise a supported --spec-type '
                f'value (draft-mtp or mtp) and --spec-draft-n-max; advertised spec types: {supported}'
            )
        return BenchmarkStrategy(
            id='mtp_acceptance_matrix',
            engine_id=engine,
            required_features=('mtp_native',),
            objectives=('quick_sanity', 'fast_chat', 'mtp_long_context_probe', 'experimental_lab'),
            phases=() if blocked_reason else (
                _phase('baseline_no_mtp', 'server_openai_api', 'quick_sanity', 'Baseline without MTP speculative decoding', ('tg_tps', 'ttft_ms', 'tpot_ms'), 1, 180),
                _phase('draft_acceptance', 'custom_mtp_probe', 'mtp_long_context_probe', 'Fit-assisted q8/no-mmap draft n=1..3 acceptance matrix', ('tg_tps', 'draft_tokens', 'accepted_tokens', 'accept_rate'), 18 if depth_key == 'fast' else 28, 240),
            ),
            hard_budget_seconds=0 if blocked_reason else (12 * 60 if depth_key == 'fast' else 24 * 60),
            max_candidates=0 if blocked_reason else (20 if depth_key == 'fast' else 32),
            retry_policy='blocked' if blocked_reason else 'terminal_missing_flags_single_start_retry',
            reason=blocked_reason or f'MTP-capable GGUF uses optional no-MTP baseline plus draft acceptance-rate comparison; spec_type={mtp_spec_type}; force np=1 and block vision/mmproj',
            metric_groups=('tg_tps', 'draft_tokens', 'accepted_tokens', 'accept_rate', 'ttft_ms', 'tpot_ms'),
            blocked_reason=blocked_reason,
        )

    if engine == ENGINE_TURBOQUANT:
        return BenchmarkStrategy(
            id='turboquant_kv_sweep',
            engine_id=engine,
            required_features=('local_gguf',),
            objectives=('quick_sanity', 'fast_chat', 'long_context', 'memory_fit', 'quality_safe'),
            phases=(
                _phase('q8_baseline', 'server_openai_api', 'quick_sanity', 'q8_0/q8_0 baseline before compressed KV', ('tg_tps', 'peak_vram', 'quality_risk'), 1, 180),
                _phase('kv_compression', 'server_openai_api', 'memory_fit', 'TurboQuant KV compression ladder, separate from weight-format probing', ('tg_tps', 'context_max_stable', 'peak_vram', 'quality_risk'), 4 if depth_key == 'fast' else 7, 240),
                _phase('context_probe', 'server_openai_api', 'long_context', 'Context growth using measured safe KV profiles', ('context_max_stable', 'peak_vram'), 2 if depth_key == 'fast' else 6, 240),
            ),
            hard_budget_seconds=12 * 60 if depth_key == 'fast' else 45 * 60,
            max_candidates=8 if depth_key == 'fast' else 18,
            retry_policy='oom_terminal_port_race_retry_once',
            reason='TurboQuant should compare KV-cache modes explicitly and score memory savings separately from speed',
            metric_groups=('tg_tps', 'context_max_stable', 'peak_vram', 'quality_risk'),
        )

    if has_moe and engine == ENGINE_LLAMA_CPP:
        return BenchmarkStrategy(
            id='moe_hybrid_placement',
            engine_id=engine,
            required_features=('local_gguf', 'moe'),
            objectives=('quick_sanity', 'fast_chat', 'memory_fit', 'long_context'),
            phases=(
                _phase('moe_placement', 'server_openai_api', 'memory_fit', 'Expert CPU/GPU placement before broader context exploration', ('pp_tps', 'tg_tps', 'peak_vram'), 4 if depth_key == 'fast' else 8, 240),
                _phase('batch_ubatch', 'server_openai_api', 'fast_chat', 'Batch and microbatch probes after placement is known', ('tg_tps', 'tpot_ms', 'peak_ram'), 2 if depth_key == 'fast' else 4, 180),
                _phase('context_probe', 'server_openai_api', 'long_context', 'Context growth only after viable placement', ('context_max_stable', 'peak_vram'), 1 if depth_key == 'fast' else 4, 240),
            ),
            hard_budget_seconds=12 * 60 if depth_key == 'fast' else 45 * 60,
            max_candidates=12 if depth_key == 'fast' else 16,
            retry_policy='oom_terminal_timeout_guarded',
            reason='MoE models need expert placement measured before dense-style batch/context sweeps',
            metric_groups=('pp_tps', 'tg_tps', 'peak_vram', 'peak_ram', 'context_max_stable'),
        )

    return BenchmarkStrategy(
        id='llama_cpp_dense_baseline',
        engine_id=engine or ENGINE_LLAMA_CPP,
        required_features=('local_gguf',),
        objectives=(requested_objective, 'quick_sanity', 'fast_chat', 'long_context') if requested_objective != 'quick_sanity' else ('quick_sanity', 'fast_chat', 'long_context'),
        phases=(
            _phase('pp', 'llama_bench', 'quick_sanity', 'Prompt-processing probe, e.g. -p 512/2048 -n 0', ('pp_tps',), 2, 90),
            _phase('tg', 'llama_bench', 'fast_chat', 'Token-generation probe, e.g. -p 0 -n 64/128', ('tg_tps', 'tpot_ms'), 2, 120),
            _phase('server_sanity', 'server_openai_api', 'quick_sanity', 'Small OpenAI-compatible launch sanity check', ('tg_tps', 'ttft_ms', 'peak_vram'), 3, 180),
            _phase('context_probe', 'server_openai_api', 'long_context', 'Bounded context growth', ('context_max_stable', 'peak_vram'), 1 if depth_key == 'fast' else 5, 240),
        ),
        hard_budget_seconds=10 * 60 if depth_key == 'fast' else 40 * 60,
        max_candidates=6 if depth_key == 'fast' else 14,
        retry_policy='oom_terminal_port_race_retry_once',
        reason='dense llama.cpp-family GGUF can use a standard prompt/generation/server sanity ladder',
        metric_groups=('pp_tps', 'tg_tps', 'ttft_ms', 'tpot_ms', 'peak_vram', 'context_max_stable'),
    )


def strategy_summary(strategy: BenchmarkStrategy) -> str:
    phases = ', '.join(phase.id for phase in strategy.phases)
    return (
        f'{strategy.id}: objectives={",".join(strategy.objectives)} '
        f'phases={phases} budget={strategy.hard_budget_seconds // 60}m '
        f'max_candidates={strategy.max_candidates}'
    )
