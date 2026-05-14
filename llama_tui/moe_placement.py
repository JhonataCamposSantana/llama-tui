from dataclasses import dataclass
from typing import List, Optional, Tuple

from .gguf import gguf_layer_count
from .hardware import HardwareProfile
from .models import ModelConfig
from .optimize import model_is_moe
from .runtime_profiles import EngineCapabilities


LLAMA_CPP_FAMILY_ENGINES = {'llama.cpp', 'llama.cpp-mtp', 'buun', 'turboquant', 'tq3'}
EXPERTS_CPU_OVERRIDE = '.*ffn_.*_exps.*=CPU'


@dataclass(frozen=True)
class MoePlacementCandidate:
    name: str
    gpu_layers: Optional[int]
    cpu_moe: bool = False
    n_cpu_moe: int = 0
    tensor_overrides: Tuple[str, ...] = ()
    expected_vram_saving_hint: str = ''
    risk: str = 'normal'


def _layer_count(model: ModelConfig) -> int:
    try:
        layers = int(gguf_layer_count(model) or 0)
    except Exception:
        layers = 0
    return max(0, layers)


def _n_cpu_moe_ladder(layer_count: int, small_gpu: bool) -> List[int]:
    if layer_count <= 0:
        return []
    raw_values = [
        layer_count,
        int(round(layer_count * 0.75)),
        int(round(layer_count * 0.50)),
        32,
        24,
        16,
        8,
    ]
    values: List[int] = []
    for value in raw_values:
        clamped = max(1, min(layer_count, int(value or 0)))
        if clamped not in values:
            values.append(clamped)
    return sorted(values, reverse=small_gpu)


def _candidate_limit(tier: str, small_gpu: bool = False) -> int:
    normalized = (tier or 'full').strip().lower()
    if small_gpu:
        return 8 if normalized == 'fast' else 12
    return 3 if normalized == 'fast' else 6


def _partial_ngl_ladder(layer_count: int) -> List[int]:
    values: List[int] = []
    ceiling = max(1, int(layer_count or 0)) if layer_count > 0 else 999
    for value in (8, 12, 13, 16, 20):
        clamped = max(1, min(ceiling, int(value)))
        if clamped not in values:
            values.append(clamped)
    return values


def _tq3_small_gpu_candidates(
    capabilities: EngineCapabilities,
    layer_count: int,
) -> List[MoePlacementCandidate]:
    body: List[MoePlacementCandidate] = []
    if capabilities.supports_n_cpu_moe:
        seen_values = set()
        for value in (32, 30, 36, 40):
            if layer_count > 0:
                value = max(1, min(layer_count, int(value)))
            if value in seen_values:
                continue
            seen_values.add(value)
            body.append(MoePlacementCandidate(
                name=f'n_cpu_moe_{value}',
                gpu_layers=999,
                n_cpu_moe=value,
                expected_vram_saving_hint=f'first {value} MoE expert layers on CPU',
                risk='safe',
            ))
    if capabilities.supports_cpu_moe:
        body.append(MoePlacementCandidate(
            name='cpu_moe_all',
            gpu_layers=999,
            cpu_moe=True,
            expected_vram_saving_hint='all MoE experts on CPU',
            risk='safe',
        ))
    body.append(MoePlacementCandidate(
        name='baseline_ngl',
        gpu_layers=None,
        expected_vram_saving_hint='last-resort partial GPU-layer behavior',
        risk='baseline',
    ))
    return body


def generate_moe_placement_candidates(
    model: ModelConfig,
    profile: Optional[HardwareProfile],
    capabilities: EngineCapabilities,
    engine_id: str,
    tier: str,
) -> List[MoePlacementCandidate]:
    engine = (engine_id or '').strip().lower()
    if engine not in LLAMA_CPP_FAMILY_ENGINES:
        return []
    if not model_is_moe(model):
        return []

    has_gpu = bool(profile and profile.has_usable_gpu())
    gpu_total = int(getattr(profile, 'gpu_memory_total', 0) or 0) if profile is not None else 0
    small_gpu = bool(0 < gpu_total <= 9 * 1024**3)
    layer_count = _layer_count(model)
    if engine == 'tq3' and small_gpu:
        return _tq3_small_gpu_candidates(capabilities, layer_count)

    candidates: List[MoePlacementCandidate] = []
    if has_gpu and small_gpu:
        for value in _partial_ngl_ladder(layer_count):
            candidates.append(MoePlacementCandidate(
                name=f'partial_ngl_{value}',
                gpu_layers=value,
                expected_vram_saving_hint=f'conservative {value}-layer GPU offload for small VRAM',
                risk='safe',
            ))
    else:
        candidates.append(MoePlacementCandidate(
            name='baseline_ngl',
            gpu_layers=None,
            expected_vram_saving_hint='current GPU-layer behavior',
            risk='baseline',
        ))
    if not (
        capabilities.supports_cpu_moe
        or capabilities.supports_n_cpu_moe
        or capabilities.supports_override_tensor
    ):
        return candidates[:_candidate_limit(tier, small_gpu)]

    body: List[MoePlacementCandidate] = []

    cpu_all = MoePlacementCandidate(
        name='cpu_moe_all',
        gpu_layers=999,
        cpu_moe=True,
        expected_vram_saving_hint='all MoE experts on CPU',
        risk='safe',
    )
    full_gpu = MoePlacementCandidate(
        name='full_gpu',
        gpu_layers=999,
        expected_vram_saving_hint='no MoE CPU offload',
        risk='aggressive',
    )

    if has_gpu and not small_gpu:
        body.append(full_gpu)
    if capabilities.supports_cpu_moe:
        body.append(cpu_all)
    if capabilities.supports_n_cpu_moe:
        for value in _n_cpu_moe_ladder(layer_count, small_gpu):
            body.append(MoePlacementCandidate(
                name=f'n_cpu_moe_{value}',
                gpu_layers=999,
                n_cpu_moe=value,
                expected_vram_saving_hint=f'first {value} MoE expert layers on CPU',
                risk='safe' if small_gpu and value >= max(1, int(layer_count * 0.5)) else 'normal',
            ))
    if capabilities.supports_override_tensor:
        body.append(MoePlacementCandidate(
            name='experts_cpu_override',
            gpu_layers=999,
            tensor_overrides=(EXPERTS_CPU_OVERRIDE,),
            expected_vram_saving_hint='expert tensors routed to CPU by pattern',
            risk='experimental',
        ))
    if has_gpu and small_gpu:
        body.append(MoePlacementCandidate(
            name='baseline_ngl',
            gpu_layers=None,
            expected_vram_saving_hint='last-resort current GPU-layer behavior',
            risk='baseline',
        ))

    deduped: List[MoePlacementCandidate] = []
    seen = set()
    for candidate in body:
        key = (candidate.name, candidate.gpu_layers, candidate.cpu_moe, candidate.n_cpu_moe, candidate.tensor_overrides)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    limit = _candidate_limit(tier, small_gpu)
    selected = deduped[: max(0, limit - len(candidates))]
    if (tier or '').strip().lower() != 'fast':
        override = next((item for item in deduped if item.name == 'experts_cpu_override'), None)
        if override is not None and override not in selected:
            if len(candidates) + len(selected) >= limit:
                selected = selected[:-1]
            selected.append(override)
    return candidates + selected
