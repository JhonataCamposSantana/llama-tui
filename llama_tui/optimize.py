import os
from typing import Optional

from .discovery import classify_model_type, extract_quant
from .gguf import (
    estimate_kv_bytes_per_token,
    gguf_layer_count,
    has_extra_flag,
    model_file_size,
    selected_cache_type,
)
from .hardware import HardwareProfile
from .models import ModelConfig


def model_likely_fits_gpu(model: ModelConfig, profile: Optional[HardwareProfile], tier: str) -> bool:
    if not profile or not profile.has_usable_gpu():
        return False
    size = model_file_size(model)
    if size <= 0:
        return False
    gpu_weight_budget = {
        'safe': 0.70,
        'moderate': 0.80,
        'extreme': 0.88,
    }.get(tier, 0.80)
    return size <= int(profile.gpu_memory_free * gpu_weight_budget)
def choose_gpu_layers_for_profile(model: ModelConfig, profile: Optional[HardwareProfile], tier: str) -> int:
    if getattr(model, 'runtime', 'llama.cpp') != 'llama.cpp':
        return int(getattr(model, 'ngl', 0) or 0)
    if not profile or not profile.has_usable_gpu():
        return 0
    if model_likely_fits_gpu(model, profile, tier):
        return 999

    layer_count = gguf_layer_count(model)
    size = model_file_size(model)
    if layer_count <= 0 or size <= 0:
        return 0

    reserve_by_tier = {'safe': 35, 'moderate': 25, 'extreme': 15}
    reserve_pct = reserve_by_tier.get(tier, 25)
    usable_gpu = int(profile.gpu_memory_free * ((100 - reserve_pct) / 100.0))
    workspace = estimate_gpu_workspace_bytes(profile)
    min_ctx = max(256, int(getattr(model, 'ctx_min', 2048)))
    kv_floor = estimate_kv_bytes_per_token(model) * min_ctx
    weight_budget = usable_gpu - workspace - kv_floor
    if weight_budget <= 256 * 1024**2:
        return 0

    per_layer = max(1, size / max(1, layer_count))
    layers = int(weight_budget / per_layer)
    return max(0, min(layer_count, layers))
def kv_cache_uses_gpu(model: ModelConfig, profile: Optional[HardwareProfile]) -> bool:
    if not profile or not profile.has_usable_gpu():
        return False
    if getattr(model, 'runtime', 'llama.cpp') != 'llama.cpp':
        return True
    if has_extra_flag(list(getattr(model, 'extra_args', []) or []), '--no-kv-offload', '-nkvo'):
        return False
    try:
        return int(getattr(model, 'ngl', 0) or 0) != 0
    except Exception:
        return True
def estimate_gpu_workspace_bytes(profile: HardwareProfile) -> int:
    total = profile.gpu_memory_total or profile.gpu_memory_free
    if total <= 0:
        return 0
    return min(max(512 * 1024**2, int(total * 0.08)), 2 * 1024**3)
def estimate_gpu_weight_bytes(model: ModelConfig, profile: HardwareProfile, tier: str, usable_gpu: int) -> int:
    size = model_file_size(model)
    if size <= 0:
        return int(usable_gpu * 0.55)
    if getattr(model, 'runtime', 'llama.cpp') == 'vllm':
        return min(int(size * 1.12), int(usable_gpu * 0.90))
    layer_count = gguf_layer_count(model)
    try:
        ngl = int(getattr(model, 'ngl', 0) or 0)
    except Exception:
        ngl = 0
    if layer_count > 0 and 0 < ngl < layer_count:
        return min(int(size * (ngl / layer_count) * 1.15), int(usable_gpu * 0.85))
    if model_likely_fits_gpu(model, profile, tier):
        return min(int(size * 1.08), int(usable_gpu * 0.85))
    offload_fraction = {
        'safe': 0.45,
        'moderate': 0.58,
        'extreme': 0.70,
    }.get(tier, 0.58)
    return min(int(usable_gpu * offload_fraction), int(size * 0.95))
def estimate_gpu_context_for_profile(
    model: ModelConfig,
    profile: Optional[HardwareProfile],
    reserve_pct: int,
    parallel: int,
) -> Optional[int]:
    if not profile or not kv_cache_uses_gpu(model, profile):
        return None
    reserve_pct = max(5, min(80, reserve_pct))
    usable_gpu = int(profile.gpu_memory_free * ((100 - reserve_pct) / 100.0))
    if usable_gpu <= 0:
        return 0
    tier = getattr(model, 'optimize_tier', 'moderate')
    workspace = estimate_gpu_workspace_bytes(profile)
    weights = estimate_gpu_weight_bytes(model, profile, tier, usable_gpu)
    kv_budget = usable_gpu - workspace - weights
    if kv_budget <= 0:
        return 0
    kv_per_token = max(1, estimate_kv_bytes_per_token(model))
    return int(kv_budget // (kv_per_token * max(1, parallel)))
def estimate_ram_model_overhead(model: ModelConfig, profile: HardwareProfile) -> int:
    size = model_file_size(model)
    if size <= 0:
        return 0
    if getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp':
        # GGUF weights are mmap-backed, so MemAvailable should not be charged for
        # the whole file. Keep headroom for page churn, tokenizer data, and CPU layers.
        return min(int(size * 0.20), 2 * 1024**3, int((profile.memory_total or size) * 0.15))
    return min(int(size * 0.50), 4 * 1024**3)
def choose_threads_for_profile(model: ModelConfig, profile: Optional[HardwareProfile], tier: str) -> int:
    logical = (profile.cpu_logical if profile else 0) or (os.cpu_count() or 1)
    physical = (profile.cpu_physical if profile else 0) or max(1, logical // 2)
    tier = tier if tier in ('safe', 'moderate', 'extreme') else 'moderate'
    if getattr(model, 'runtime', 'llama.cpp') == 'vllm':
        return int(getattr(model, 'threads', 1) or 1)

    if model_likely_fits_gpu(model, profile, tier):
        return max(2, min(physical, 8))
    if tier == 'safe':
        return max(2, min(logical, max(2, physical - 2)))
    if tier == 'extreme':
        return max(2, min(logical, max(physical, logical - 1)))
    return max(2, min(logical, physical))
def estimate_safe_context_for_profile(
    model: ModelConfig,
    profile: Optional[HardwareProfile],
    reserve_pct: int,
    parallel: int,
    ctx_min: int,
    ctx_max: int,
) -> int:
    if not profile:
        return ctx_max

    limits = []
    kv_per_token = max(1, estimate_kv_bytes_per_token(model))
    reserve_pct = max(5, min(70, reserve_pct))
    if profile.memory_available > 0:
        usable_ram = int(profile.memory_available * ((100 - reserve_pct) / 100.0))
        if getattr(model, 'runtime', 'llama.cpp') != 'vllm' and not model_likely_fits_gpu(model, profile, getattr(model, 'optimize_tier', 'moderate')):
            usable_ram -= estimate_ram_model_overhead(model, profile)
        if usable_ram > 0:
            limits.append(usable_ram // (kv_per_token * max(1, parallel)))
        else:
            limits.append(0)

    gpu_ctx = estimate_gpu_context_for_profile(model, profile, reserve_pct, parallel)
    if gpu_ctx is not None:
        limits.append(gpu_ctx)

    if not limits:
        return ctx_max
    return max(0, min(ctx_max, int(min(limits))))
def select_best_tier(model: ModelConfig, profile: Optional[HardwareProfile]) -> str:
    if not profile:
        return 'moderate'

    size = model_file_size(model)
    available = profile.memory_available
    total = profile.memory_total
    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048)))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072)))
    if profile.has_usable_gpu():
        safe_ctx = estimate_safe_context_for_profile(model, profile, 35, 1, ctx_min, ctx_max)
        moderate_ctx = estimate_safe_context_for_profile(model, profile, 25, 1, ctx_min, ctx_max)
        extreme_ctx = estimate_safe_context_for_profile(model, profile, 15, 1, ctx_min, ctx_max)
        if safe_ctx < ctx_min:
            return 'safe'
        if model_likely_fits_gpu(model, profile, 'extreme') and extreme_ctx >= min(ctx_max, 32768):
            return 'extreme'
        if model_likely_fits_gpu(model, profile, 'moderate') or moderate_ctx >= min(ctx_max, 8192):
            return 'moderate'
        return 'safe'
    if model_likely_fits_gpu(model, profile, 'extreme'):
        return 'extreme'
    if model_likely_fits_gpu(model, profile, 'moderate'):
        return 'moderate'
    if available and size and size > int(available * 0.65):
        return 'safe'
    if total and total < 18 * 1024**3:
        return 'moderate' if available >= 7 * 1024**3 else 'safe'
    if available >= 20 * 1024**3:
        return 'extreme'
    if available >= 8 * 1024**3:
        return 'moderate'
    return 'safe'
def choose_best_preset(model: ModelConfig, profile: Optional[HardwareProfile]) -> str:
    runtime = getattr(model, 'runtime', 'llama.cpp')
    quant = extract_quant(model).lower()
    model_type = classify_model_type(model)
    size = model_file_size(model)
    available = profile.memory_available if profile else 0
    if runtime == 'vllm':
        return 'tokens_per_sec'
    if profile and profile.has_usable_gpu():
        ctx_min = max(256, int(getattr(model, 'ctx_min', 2048)))
        ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072)))
        tps_ctx = estimate_safe_context_for_profile(model, profile, 30, 4, ctx_min, min(ctx_max, 12288))
        if model_likely_fits_gpu(model, profile, 'moderate') and tps_ctx >= max(ctx_min, 4096):
            return 'tokens_per_sec'
        return 'max_context'
    if size and available and size > int(available * 0.55):
        return 'max_context'
    if model_type == 'CPU' or 'q2' in quant or 'q3' in quant:
        return 'max_context'
    return 'tokens_per_sec'
def apply_hardware_baseline(model: ModelConfig, profile: Optional[HardwareProfile], tier: str):
    if not profile:
        return
    runtime = getattr(model, 'runtime', 'llama.cpp')
    if runtime != 'vllm':
        model.threads = choose_threads_for_profile(model, profile, tier)
        if model_likely_fits_gpu(model, profile, tier):
            model.ngl = 999
            model.flash_attn = True
        elif profile.has_usable_gpu():
            model.ngl = choose_gpu_layers_for_profile(model, profile, tier)
            model.flash_attn = model.ngl > 0
        elif not profile.has_usable_gpu():
            model.ngl = 0
def apply_optimization_preset(
    model: ModelConfig,
    preset: str,
    tier: str = 'moderate',
    profile: Optional[HardwareProfile] = None,
) -> str:
    runtime = getattr(model, 'runtime', 'llama.cpp')
    tier = (tier or getattr(model, 'optimize_tier', 'moderate')).strip().lower()
    if tier not in ('safe', 'moderate', 'extreme'):
        tier = 'moderate'
    model.optimize_tier = tier
    apply_hardware_baseline(model, profile, tier)
    extra_args = list(getattr(model, 'extra_args', []) or [])

    def strip_flags(*flags: str):
        nonlocal extra_args
        cleaned: List[str] = []
        skip_next = False
        flag_set = set(flags)
        for token in extra_args:
            if skip_next:
                skip_next = False
                continue
            if token in flag_set:
                skip_next = True
                continue
            if any(token.startswith(f'{f}=') for f in flag_set):
                continue
            cleaned.append(token)
        extra_args = cleaned

    def set_flag(flag: str, value: str):
        nonlocal extra_args
        strip_flags(flag)
        extra_args += [flag, value]

    ctx_min = max(256, int(getattr(model, 'ctx_min', 2048)))
    ctx_max = max(ctx_min, int(getattr(model, 'ctx_max', 131072)))

    if preset == 'max_context':
        model.optimize_mode = 'max_context_safe'
        model.parallel = 1
        reserve_by_tier = {'safe': 35, 'moderate': 25, 'extreme': 15}
        min_ctx_by_tier = {'safe': 16384, 'moderate': 32768, 'extreme': 65536}
        model.memory_reserve_percent = max(reserve_by_tier[tier], int(getattr(model, 'memory_reserve_percent', 25)))
        model.ctx = max(int(getattr(model, 'ctx', 8192)), min_ctx_by_tier[tier])
        model.ctx = max(ctx_min, min(model.ctx, ctx_max))
        model.output = min(max(model.output, 2048), 4096)
        if runtime == 'llama.cpp':
            batch_by_tier = {'safe': '128', 'moderate': '256', 'extreme': '512'}
            ubatch_by_tier = {'safe': '64', 'moderate': '128', 'extreme': '256'}
            set_flag('--batch-size', batch_by_tier[tier])
            set_flag('--ubatch-size', ubatch_by_tier[tier])
            set_flag('--cache-type-k', 'q8_0')
            set_flag('--cache-type-v', 'q8_0')
        elif runtime == 'vllm':
            util_by_tier = {'safe': '0.75', 'moderate': '0.88', 'extreme': '0.92'}
            seqs_by_tier = {'safe': '1', 'moderate': '2', 'extreme': '3'}
            set_flag('--gpu-memory-utilization', util_by_tier[tier])
            set_flag('--max-num-seqs', seqs_by_tier[tier])
            set_flag('--max-num-batched-tokens', str(max(4096, min(model.ctx, 24576))))
        model.extra_args = extra_args
        safe_ctx = estimate_safe_context_for_profile(model, profile, model.memory_reserve_percent, model.parallel, ctx_min, ctx_max)
        model.ctx = min(model.ctx, safe_ctx) if safe_ctx >= ctx_min else ctx_min
        hw_note = f' | {profile.short_summary()}' if profile else ''
        return f'{model.id}: preset=max_context_safe/{tier} ({runtime}) ctx={model.ctx} parallel={model.parallel} threads={model.threads} ngl={model.ngl}{hw_note}'

    if preset == 'tokens_per_sec':
        model.optimize_mode = 'max_context_safe'
        reserve_by_tier = {'safe': 40, 'moderate': 30, 'extreme': 20}
        target_ctx_by_tier = {'safe': 4096, 'moderate': 8192, 'extreme': 12288}
        par_by_tier = {'safe': 2, 'moderate': 4, 'extreme': 8}
        model.memory_reserve_percent = max(reserve_by_tier[tier], int(getattr(model, 'memory_reserve_percent', 25)))
        model.ctx = max(ctx_min, min(target_ctx_by_tier[tier], ctx_max))
        model.parallel = max(1, min(par_by_tier[tier], int(getattr(model, 'parallel', 1)) + 1))
        model.output = min(model.output, 2048)
        if runtime == 'llama.cpp':
            batch_by_tier = {'safe': '512', 'moderate': '1024', 'extreme': '2048'}
            ubatch_by_tier = {'safe': '256', 'moderate': '512', 'extreme': '1024'}
            set_flag('--batch-size', batch_by_tier[tier])
            set_flag('--ubatch-size', ubatch_by_tier[tier])
            strip_flags('--cache-type-k', '--cache-type-v')
        elif runtime == 'vllm':
            util_by_tier = {'safe': '0.70', 'moderate': '0.80', 'extreme': '0.90'}
            seqs_by_tier = {'safe': '4', 'moderate': '8', 'extreme': '16'}
            btok_by_tier = {'safe': '4096', 'moderate': '8192', 'extreme': '16384'}
            set_flag('--gpu-memory-utilization', util_by_tier[tier])
            set_flag('--max-num-seqs', seqs_by_tier[tier])
            set_flag('--max-num-batched-tokens', btok_by_tier[tier])
        model.extra_args = extra_args
        safe_ctx = estimate_safe_context_for_profile(model, profile, model.memory_reserve_percent, model.parallel, ctx_min, ctx_max)
        model.ctx = min(model.ctx, safe_ctx) if safe_ctx >= ctx_min else ctx_min
        hw_note = f' | {profile.short_summary()}' if profile else ''
        return f'{model.id}: preset=tokens_per_sec/{tier} ({runtime}) ctx={model.ctx} parallel={model.parallel} threads={model.threads} ngl={model.ngl}{hw_note}'

    return f'{model.id}: keeping current settings'
def apply_best_optimization(
    model: ModelConfig,
    tier: str = 'moderate',
    profile: Optional[HardwareProfile] = None,
) -> str:
    if tier == 'auto':
        tier = select_best_tier(model, profile)
    preset = choose_best_preset(model, profile)
    return apply_optimization_preset(model, preset, tier=tier, profile=profile)
