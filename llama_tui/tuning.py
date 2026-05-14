from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .hardware import HardwareProfile
from .models import ModelConfig
from .optimize import model_is_moe
from .runtime_profiles import EngineCapabilities, RuntimeProfile


LLAMA_CPP_FAMILY_ENGINES = {'llama.cpp', 'llama.cpp-mtp', 'buun', 'turboquant', 'tq3'}
EXPERTS_CPU_OVERRIDE = '.*ffn_.*_exps.*=CPU'
MIB = 1024 ** 2
GIB = 1024 ** 3


@dataclass(frozen=True)
class TuningObjective:
    purpose: str
    target_context: int
    min_context: int
    target_vram_headroom_bytes: int
    max_trials: int
    depth: str
    prefer_speed: bool = True
    prefer_stability: bool = True


@dataclass(frozen=True)
class TuningCandidate:
    name: str
    runtime_profile: RuntimeProfile
    source: str
    risk: str
    expected_effect: str
    phase: str = 'coarse'


@dataclass(frozen=True)
class TuningResult:
    candidate_name: str
    success: bool
    tokens_per_sec: float
    load_seconds: float
    benchmark_seconds: float
    peak_vram_bytes: int
    peak_ram_bytes: int
    vram_headroom_bytes: int
    ctx_size: int
    error_category: str
    error_excerpt: str
    score: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TuningRecommendation:
    winner: Dict[str, object]
    all_results: List[Dict[str, object]]
    summary: str
    warnings: List[str]


def normalize_tuning_depth(depth: str) -> str:
    return 'full' if str(depth or '').strip().lower() == 'full' else 'fast'


def target_vram_headroom_bytes(profile: Optional[HardwareProfile]) -> int:
    total = int(getattr(profile, 'gpu_memory_total', 0) or 0) if profile is not None else 0
    if total <= 0:
        return 1024 * MIB
    if total <= 12 * GIB:
        return 1024 * MIB
    if total <= 24 * GIB:
        return 1536 * MIB
    return 2048 * MIB


def tuning_trial_budget(depth: str) -> int:
    return 8 if normalize_tuning_depth(depth) == 'fast' else 16


def tuning_objective_for_model(model: ModelConfig, profile: Optional[HardwareProfile], depth: str) -> TuningObjective:
    target = max(1, int(getattr(model, 'ctx', 0) or 1))
    minimum = max(1, int(getattr(model, 'ctx_min', 0) or 1))
    return TuningObjective(
        purpose='moe_placement',
        target_context=target,
        min_context=minimum,
        target_vram_headroom_bytes=target_vram_headroom_bytes(profile),
        max_trials=tuning_trial_budget(depth),
        depth=normalize_tuning_depth(depth),
    )


def moe_tuning_eligibility_reason(
    model: ModelConfig,
    profile: Optional[HardwareProfile],
    capabilities: EngineCapabilities,
    engine_id: str,
) -> str:
    engine = str(engine_id or '').strip().lower()
    if engine not in LLAMA_CPP_FAMILY_ENGINES:
        return f'active engine {engine or "-"} is not eligible for llama.cpp-family MoE tuning'
    if not str(getattr(model, 'path', '') or '').lower().endswith('.gguf'):
        return 'MoE tuning is available only for GGUF models'
    if not model_is_moe(model):
        return 'model is not detected as MoE'
    if not (profile and profile.has_usable_gpu()):
        return 'usable GPU information is required for MoE tuning'
    if not (
        bool(getattr(capabilities, 'supports_cpu_moe', False))
        or bool(getattr(capabilities, 'supports_n_cpu_moe', False))
        or bool(getattr(capabilities, 'supports_override_tensor', False))
    ):
        return 'active engine does not advertise cpu-moe, n-cpu-moe, or override-tensor support'
    return ''


def generate_n_cpu_moe_ladder(layer_count: int, small_gpu: bool = False) -> List[int]:
    layers = max(0, int(layer_count or 0))
    if layers <= 0:
        return []
    ratios = (1.0, 0.85, 0.75, 0.66, 0.50, 0.40, 0.33, 0.25, 0.15, 0.0)
    values = {
        max(0, min(layers, int(round(layers * ratio))))
        for ratio in ratios
    }
    ordered = sorted(values, reverse=True)
    if small_gpu:
        return ordered
    return ordered


def _replace_profile(
    baseline: RuntimeProfile,
    name: str,
    gpu_layers: Optional[int],
    placement_strategy: str,
    cpu_moe: bool = False,
    n_cpu_moe: int = 0,
    tensor_overrides: Sequence[str] = (),
) -> RuntimeProfile:
    return RuntimeProfile(
        engine_id=baseline.engine_id,
        ctx_size=baseline.ctx_size,
        gpu_layers=gpu_layers,
        parallel=baseline.parallel,
        kv_preset=baseline.kv_preset,
        flash_attn=baseline.flash_attn,
        batch_size=baseline.batch_size,
        ubatch_size=baseline.ubatch_size,
        fit=baseline.fit,
        fit_context=baseline.fit_context,
        no_warmup=baseline.no_warmup,
        extra_args=baseline.extra_args,
        name=name,
        kv_family=baseline.kv_family,
        kv_quality_tier=baseline.kv_quality_tier,
        kv_compression_tier=baseline.kv_compression_tier,
        kv_score_penalty=baseline.kv_score_penalty,
        benchmark_depth=baseline.benchmark_depth,
        fit_discovery_phase=baseline.fit_discovery_phase,
        viable_ngl=baseline.viable_ngl,
        viable_ngl_source=baseline.viable_ngl_source,
        fit_selected_ngl=baseline.fit_selected_ngl,
        fit_selected_ngl_source=baseline.fit_selected_ngl_source,
        fit_log_excerpt=baseline.fit_log_excerpt,
        placement_strategy=placement_strategy,
        cpu_moe=bool(cpu_moe),
        n_cpu_moe=max(0, int(n_cpu_moe or 0)),
        tensor_overrides=tuple(str(item) for item in tuple(tensor_overrides or ()) if str(item).strip()),
        reasoning=baseline.reasoning,
        reasoning_budget=baseline.reasoning_budget,
        reasoning_format=baseline.reasoning_format,
        mtp_enabled=baseline.mtp_enabled,
        mtp_draft_n_max=baseline.mtp_draft_n_max,
        benchmark_strategy_id=baseline.benchmark_strategy_id,
        benchmark_objectives=baseline.benchmark_objectives,
        benchmark_phase=baseline.benchmark_phase,
        benchmark_metric_group=baseline.benchmark_metric_group,
    )


def _conservative_moe_gpu_layers(baseline: RuntimeProfile, full_gpu_plausible: bool) -> Optional[int]:
    return 999 if full_gpu_plausible else baseline.gpu_layers


def generate_moe_tuning_candidates(
    baseline: RuntimeProfile,
    capabilities: EngineCapabilities,
    profile: Optional[HardwareProfile],
    layer_count: int,
    depth: str = 'fast',
) -> List[TuningCandidate]:
    normalized_depth = normalize_tuning_depth(depth)
    gpu_total = int(getattr(profile, 'gpu_memory_total', 0) or 0) if profile is not None else 0
    gpu_free = int(getattr(profile, 'gpu_memory_free', 0) or 0) if profile is not None else 0
    small_gpu = bool(0 < gpu_total <= 9 * GIB)
    has_moe_capability = bool(
        getattr(capabilities, 'supports_cpu_moe', False)
        or getattr(capabilities, 'supports_n_cpu_moe', False)
        or getattr(capabilities, 'supports_override_tensor', False)
    )
    full_gpu_plausible = bool(
        has_moe_capability
        and not small_gpu
        and gpu_free > target_vram_headroom_bytes(profile) * 2
    )
    moe_candidate_gpu_layers = _conservative_moe_gpu_layers(baseline, full_gpu_plausible)
    candidates: List[TuningCandidate] = [
        TuningCandidate(
            name='baseline_current',
            runtime_profile=_replace_profile(
                baseline,
                'baseline_current',
                baseline.gpu_layers,
                baseline.placement_strategy,
                baseline.cpu_moe,
                baseline.n_cpu_moe,
                baseline.tensor_overrides,
            ),
            source='baseline',
            risk='baseline',
            expected_effect='current active-engine launch profile',
        )
    ]
    if bool(getattr(capabilities, 'supports_cpu_moe', False)):
        candidates.append(TuningCandidate(
            name='cpu_moe_all',
            runtime_profile=_replace_profile(
                baseline,
                'cpu_moe_all',
                moe_candidate_gpu_layers,
                'cpu_moe_all',
                cpu_moe=True,
            ),
            source='coarse',
            risk='safe',
            expected_effect='all MoE experts on CPU',
        ))
    if bool(getattr(capabilities, 'supports_n_cpu_moe', False)) and layer_count > 0:
        for value in generate_n_cpu_moe_ladder(layer_count, small_gpu=small_gpu):
            if value <= 0:
                continue
            candidates.append(TuningCandidate(
                name=f'n_cpu_moe_{value}',
                runtime_profile=_replace_profile(
                    baseline,
                    f'n_cpu_moe_{value}',
                    moe_candidate_gpu_layers,
                    f'n_cpu_moe_{value}',
                    n_cpu_moe=value,
                ),
                source='coarse',
                risk='safe' if small_gpu and value >= max(1, int(layer_count * 0.5)) else 'normal',
                expected_effect=f'first {value} MoE expert layers on CPU',
            ))
    if full_gpu_plausible:
        candidates.append(TuningCandidate(
            name='full_gpu_no_moe',
            runtime_profile=_replace_profile(
                baseline,
                'full_gpu_no_moe',
                999,
                'full_gpu_no_moe',
            ),
            source='coarse',
            risk='aggressive',
            expected_effect='full GPU layer attempt without new MoE CPU placement',
        ))
    if bool(getattr(capabilities, 'supports_override_tensor', False)):
        candidates.append(TuningCandidate(
            name='experts_cpu_override',
            runtime_profile=_replace_profile(
                baseline,
                'experts_cpu_override',
                moe_candidate_gpu_layers,
                'experts_cpu_override',
                tensor_overrides=(EXPERTS_CPU_OVERRIDE,),
            ),
            source='coarse',
            risk='experimental',
            expected_effect='expert tensors routed to CPU by override pattern',
        ))

    seen = set()
    deduped: List[TuningCandidate] = []
    for candidate in candidates:
        profile_item = candidate.runtime_profile
        key = (
            candidate.name,
            profile_item.gpu_layers,
            profile_item.cpu_moe,
            profile_item.n_cpu_moe,
            profile_item.tensor_overrides,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    limit = 6 if normalized_depth == 'fast' else 10
    return deduped[:limit]


def refinement_n_cpu_moe_values(winner_n_cpu_moe: int, layer_count: int) -> List[int]:
    center = max(0, int(winner_n_cpu_moe or 0))
    layers = max(0, int(layer_count or 0))
    if center <= 0 or layers <= 0:
        return []
    values = {
        max(0, min(layers, center + delta))
        for delta in range(-4, 5)
    }
    values.discard(center)
    return sorted(values, key=lambda value: (abs(value - center), -value))


def generate_refinement_candidates(
    baseline: RuntimeProfile,
    measured_winner: Dict[str, object],
    layer_count: int,
) -> List[TuningCandidate]:
    try:
        center = int(measured_winner.get('n_cpu_moe', 0) or 0)
    except Exception:
        center = 0
    try:
        winner_ngl = int(measured_winner.get('ngl', 0) or 0)
    except Exception:
        winner_ngl = 0
    refinement_gpu_layers = winner_ngl if winner_ngl > 0 else baseline.gpu_layers
    values = refinement_n_cpu_moe_values(center, layer_count)
    candidates: List[TuningCandidate] = []
    for value in values:
        candidates.append(TuningCandidate(
            name=f'n_cpu_moe_{value}',
            runtime_profile=_replace_profile(
                baseline,
                f'n_cpu_moe_{value}',
                refinement_gpu_layers,
                f'n_cpu_moe_{value}',
                n_cpu_moe=value,
            ),
            source='refinement',
            risk='normal',
            expected_effect=f'refine around measured n_cpu_moe={center}',
            phase='refinement',
        ))
    return candidates


def _record_headroom(record: Dict[str, object]) -> int:
    values = [
        int(record.get('vram_headroom_bytes', 0) or 0),
        int(record.get('memory_guardrail_min_gpu_memory_free', 0) or 0),
        int(record.get('gpu_memory_free', 0) or 0),
    ]
    return max(0, min(value for value in values if value > 0)) if any(value > 0 for value in values) else 0


def score_tuning_record(record: Dict[str, object], objective: TuningObjective, risk: str = '') -> float:
    if str(record.get('status', '') or '').lower() != 'ok':
        return 0.0
    score = float(record.get('tokens_per_sec', 0.0) or 0.0)
    if score <= 0:
        return 0.0
    headroom = _record_headroom(record)
    if headroom and headroom < int(objective.target_vram_headroom_bytes):
        score *= 0.25
    if int(record.get('ctx', 0) or 0) < int(objective.target_context or 0):
        score *= 0.50
    if str(risk or record.get('tuning_risk', '') or '').lower() == 'experimental':
        score *= 0.95
    ram_available = int(record.get('ram_available', 0) or record.get('memory_available', 0) or 0)
    if ram_available and ram_available < 2 * GIB:
        score *= 0.75
    return round(score, 4)


def select_measured_tuning_winner(
    records: Sequence[Dict[str, object]],
    objective: TuningObjective,
) -> Dict[str, object]:
    measured = [
        dict(record)
        for record in list(records or [])
        if str(record.get('status', '') or '').lower() == 'ok'
    ]
    if not measured:
        return {}
    for record in measured:
        if 'tuning_score' not in record:
            record['tuning_score'] = score_tuning_record(record, objective, str(record.get('tuning_risk', '') or ''))
    return max(
        measured,
        key=lambda record: (
            float(record.get('tuning_score', 0.0) or 0.0),
            float(record.get('tokens_per_sec', 0.0) or 0.0),
            int(record.get('gpu_memory_free', 0) or 0),
        ),
    )


def unsafe_failure_category(category: str) -> bool:
    return str(category or '') in {
        'CUDA_OOM_WEIGHTS',
        'CUDA_OOM_KV',
        'MEMORY_GUARDRAIL',
        'MEMORY_FIT_FAILED',
        'FIXED_GPU_LAYERS_FIT_FAILED',
    }


def hard_stop_failure_category(category: str) -> bool:
    return str(category or '') in {
        'CUDA_OOM_WEIGHTS',
        'CUDA_OOM_KV',
        'MEMORY_GUARDRAIL',
        'MODEL_LOAD_FAILED',
        'ENGINE_BINARY_MISSING',
    }


def early_stop_reason(
    record: Dict[str, object],
    depth: str,
    extra_probe_used: bool = False,
) -> Tuple[bool, bool, str]:
    category = str(record.get('failure_category', '') or '')
    if not unsafe_failure_category(category):
        return False, extra_probe_used, ''
    candidate = str(record.get('measured_candidate_name', '') or record.get('runtime_profile', '') or 'candidate')
    if normalize_tuning_depth(depth) == 'fast':
        return True, extra_probe_used, f'fast stopped after unsafe {candidate}: {category}'
    if hard_stop_failure_category(category):
        return True, extra_probe_used, f'full stopped after hard unsafe {candidate}: {category}'
    if not extra_probe_used:
        return False, True, f'full allowed one nearby probe after unsafe {candidate}: {category}'
    return True, extra_probe_used, f'full stopped after second unsafe nearby value: {category}'
