import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .gguf import read_gguf_template_metadata
from .models import ModelConfig
from .runtime_profiles import EngineCapabilities, RuntimeProfile, kv_modes_from_preset, strip_runtime_tuning_args


BENCHMARK_PURPOSES = ('raw_speed', 'serve_default')
SERVE_DEFAULT_FAST_OUTPUT_CAP = 512
SERVE_DEFAULT_FULL_OUTPUT_CAP = 1024
TQ3_MOE_FAST_OUTPUT_CAP = 128
TQ3_MOE_FULL_OUTPUT_CAP = 256
RAW_SPEED_OUTPUT = 512


@dataclass(frozen=True)
class BenchmarkLaunchProfile:
    name: str
    purpose: str
    ctx: int
    output: int
    measurement_output: int
    temp: float
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    min_p: Optional[float] = None
    seed: Optional[int] = None
    samplers: str = ''
    no_context_shift: bool = False
    flash_attn: str = 'auto'
    kv_key_type: str = ''
    kv_value_type: str = ''
    fit: bool = False
    fit_context: int = 0
    fit_target: str = ''
    cache_prompt: Optional[bool] = None
    cache_reuse: int = 0
    reasoning: str = ''
    reasoning_budget: Optional[int] = None
    preserve_thinking: bool = False
    preserve_thinking_source: str = 'default'
    chat_template_kwargs: Dict[str, object] = field(default_factory=dict)
    extra_args: Tuple[str, ...] = field(default_factory=tuple)


def _maybe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ''):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _maybe_int(value: object, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ''):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _maybe_bool(value: object, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or '').strip().lower()
    if text in ('true', '1', 'yes', 'on'):
        return True
    if text in ('false', '0', 'no', 'off'):
        return False
    return default


def _launch_overrides(model: ModelConfig) -> Dict[str, object]:
    raw = getattr(model, 'launch_overrides', {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def preserve_thinking_override_value(model: ModelConfig) -> str:
    override = _launch_overrides(model).get('preserve_thinking', None)
    value = override if override is not None else getattr(model, 'preserve_thinking', 'auto')
    normalized = str(value or 'auto').strip().lower()
    return normalized if normalized in ('auto', 'on', 'off') else 'auto'


def detect_preserve_thinking_default(model: ModelConfig) -> Tuple[bool, str]:
    metadata = read_gguf_template_metadata(getattr(model, 'path', '') or '')
    metadata_text = ' '.join(str(value) for value in metadata.values()).lower()
    model_text = ' '.join(
        str(value or '')
        for value in (
            getattr(model, 'name', ''),
            getattr(model, 'id', ''),
            getattr(model, 'architecture', ''),
            getattr(model, 'model_family', ''),
        )
    ).lower()
    combined = f'{metadata_text} {model_text}'
    if any(marker in metadata_text for marker in ('preserve_thinking', 'enable_thinking', '<think', 'reasoning')):
        return True, 'gguf_template'
    if any(marker in combined for marker in ('qwq', 'deepseek-r1', 'gpt-oss')):
        return True, 'family_hint'
    if 'qwen3' in combined and any(marker in combined for marker in ('think', 'thinking', 'reasoning')):
        return True, 'family_hint'
    return False, 'default'


def resolve_preserve_thinking(model: ModelConfig) -> Tuple[bool, str]:
    override = preserve_thinking_override_value(model)
    if override == 'on':
        return True, 'override'
    if override == 'off':
        return False, 'override'
    return detect_preserve_thinking_default(model)


def _override_sequence(overrides: Dict[str, object], key: str) -> Tuple[str, ...]:
    value = overrides.get(key, ())
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if isinstance(value, str) and value.strip():
        return tuple(value.split())
    return ()


def _runtime_engine_id(runtime_profile: Optional[RuntimeProfile]) -> str:
    return str(getattr(runtime_profile, 'engine_id', '') or '').strip().lower()


def _is_tq3_moe_profile(model: ModelConfig, runtime_profile: Optional[RuntimeProfile]) -> bool:
    if _runtime_engine_id(runtime_profile) != 'tq3':
        return False
    if int(getattr(model, 'expert_count', 0) or 0) > 0:
        return True
    if int(getattr(model, 'expert_used_count', 0) or 0) > 0:
        return True
    text = ' '.join(
        str(getattr(model, name, '') or '')
        for name in ('architecture_type', 'architecture', 'model_family', 'name', 'id')
    ).lower()
    return 'moe' in text


def build_benchmark_launch_profile(
    model: ModelConfig,
    runtime_profile: Optional[RuntimeProfile] = None,
    capabilities: Optional[EngineCapabilities] = None,
    purpose: str = 'serve_default',
    depth: str = 'full',
) -> BenchmarkLaunchProfile:
    purpose_key = str(purpose or 'serve_default').strip().lower()
    if purpose_key not in BENCHMARK_PURPOSES:
        purpose_key = 'serve_default'
    depth_key = 'fast' if str(depth or '').strip().lower() == 'fast' else 'full'
    overrides = _launch_overrides(model)
    ctx = int(getattr(runtime_profile, 'ctx_size', 0) or getattr(model, 'ctx', 0) or 0)
    kv_key, kv_value = kv_modes_from_preset(str(getattr(runtime_profile, 'kv_preset', '') or ''))
    flash = str(getattr(runtime_profile, 'flash_attn', '') or ('on' if getattr(model, 'flash_attn', True) else 'off'))
    if purpose_key == 'raw_speed':
        output = RAW_SPEED_OUTPUT
        measurement_output = RAW_SPEED_OUTPUT
        temp = 0.0
        top_p = None
        top_k = None
        repeat_penalty = 1.0
        presence_penalty = 0.0
        no_context_shift = False
        cache_prompt: Optional[bool] = False
    else:
        output = max(1, int(getattr(model, 'output', 4096) or 4096))
        cap = SERVE_DEFAULT_FAST_OUTPUT_CAP if depth_key == 'fast' else SERVE_DEFAULT_FULL_OUTPUT_CAP
        if _is_tq3_moe_profile(model, runtime_profile):
            cap = TQ3_MOE_FAST_OUTPUT_CAP if depth_key == 'fast' else TQ3_MOE_FULL_OUTPUT_CAP
        measurement_output = max(1, min(output, cap))
        temp = float(getattr(model, 'temp', 0.7) or 0.7)
        top_p = _maybe_float(getattr(model, 'top_p', 0.95), 0.95)
        top_k = _maybe_int(getattr(model, 'top_k', 40), 40)
        repeat_penalty = _maybe_float(getattr(model, 'repeat_penalty', 1.0), 1.0)
        presence_penalty = _maybe_float(getattr(model, 'presence_penalty', 0.0), 0.0)
        no_context_shift = bool(getattr(model, 'no_context_shift', False))
        cache_prompt = None

    output = max(1, _maybe_int(overrides.get('output'), output) or output)
    measurement_output = max(1, min(output, _maybe_int(overrides.get('measurement_output'), measurement_output) or measurement_output))
    temp = _maybe_float(overrides.get('temp'), temp) or 0.0
    top_p = _maybe_float(overrides.get('top_p'), top_p)
    top_k = _maybe_int(overrides.get('top_k'), top_k)
    repeat_penalty = _maybe_float(overrides.get('repeat_penalty'), repeat_penalty)
    presence_penalty = _maybe_float(overrides.get('presence_penalty'), presence_penalty)
    min_p = _maybe_float(overrides.get('min_p'), None)
    seed = _maybe_int(overrides.get('seed'), None)
    samplers = str(overrides.get('samplers', '') or '').strip()
    no_context_shift = bool(_maybe_bool(overrides.get('no_context_shift'), no_context_shift))
    cache_prompt = _maybe_bool(overrides.get('cache_prompt'), cache_prompt)
    cache_reuse = max(0, _maybe_int(overrides.get('cache_reuse'), 0) or 0)
    reasoning = str(overrides.get('reasoning', '') or '').strip().lower()
    if reasoning not in ('', 'on', 'off', 'auto'):
        reasoning = ''
    reasoning_budget = _maybe_int(overrides.get('reasoning_budget'), None)
    fit_target = str(overrides.get('fit_target', '') or '').strip()

    preserve_thinking, preserve_source = resolve_preserve_thinking(model)
    chat_kwargs_raw = overrides.get('chat_template_kwargs', {})
    chat_kwargs = dict(chat_kwargs_raw) if isinstance(chat_kwargs_raw, dict) else {}
    if preserve_thinking:
        chat_kwargs.setdefault('preserve_thinking', True)

    return BenchmarkLaunchProfile(
        name=purpose_key,
        purpose=purpose_key,
        ctx=max(1, ctx),
        output=output,
        measurement_output=measurement_output,
        temp=float(temp),
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        min_p=min_p,
        seed=seed,
        samplers=samplers,
        no_context_shift=no_context_shift,
        flash_attn=flash,
        kv_key_type=kv_key,
        kv_value_type=kv_value,
        fit=bool(getattr(runtime_profile, 'fit', False)) if runtime_profile is not None else False,
        fit_context=int(getattr(runtime_profile, 'fit_context', 0) or 0) if runtime_profile is not None else 0,
        fit_target=fit_target,
        cache_prompt=cache_prompt,
        cache_reuse=cache_reuse,
        reasoning=reasoning,
        reasoning_budget=reasoning_budget,
        preserve_thinking=bool(preserve_thinking),
        preserve_thinking_source=preserve_source,
        chat_template_kwargs=chat_kwargs,
        extra_args=_override_sequence(overrides, 'extra_args'),
    )


def benchmark_profile_request_fields(profile: BenchmarkLaunchProfile, max_tokens: Optional[int] = None) -> Dict[str, object]:
    tokens = max(1, int(max_tokens or profile.measurement_output or 1))
    payload: Dict[str, object] = {
        'max_tokens': tokens,
        'temperature': float(profile.temp),
    }
    if profile.top_p is not None:
        payload['top_p'] = float(profile.top_p)
    if profile.top_k is not None:
        payload['top_k'] = int(profile.top_k)
    if profile.repeat_penalty is not None:
        payload['repeat_penalty'] = float(profile.repeat_penalty)
    if profile.presence_penalty is not None:
        payload['presence_penalty'] = float(profile.presence_penalty)
    if profile.min_p is not None:
        payload['min_p'] = float(profile.min_p)
    if profile.seed is not None:
        payload['seed'] = int(profile.seed)
    return payload


def benchmark_profile_server_args(
    profile: BenchmarkLaunchProfile,
    capabilities: EngineCapabilities,
) -> Tuple[List[str], List[str]]:
    args: List[str] = []
    unsupported: List[str] = []

    def add_supported(flag: str, value: object, supported: bool):
        if value in (None, ''):
            return
        if supported:
            args.extend([flag, str(value)])
        else:
            unsupported.append(flag)

    add_supported('--top-p', profile.top_p, capabilities.supports_top_p)
    add_supported('--top-k', profile.top_k, capabilities.supports_top_k)
    add_supported('--min-p', profile.min_p, capabilities.supports_min_p)
    add_supported('--repeat-penalty', profile.repeat_penalty, capabilities.supports_repeat_penalty)
    add_supported('--presence-penalty', profile.presence_penalty, capabilities.supports_presence_penalty)
    add_supported('--samplers', profile.samplers, capabilities.supports_samplers)
    add_supported('--seed', profile.seed, capabilities.supports_seed)
    if profile.no_context_shift:
        if capabilities.supports_context_shift:
            args.append('--no-context-shift')
        else:
            unsupported.append('--no-context-shift')
    if profile.chat_template_kwargs:
        if capabilities.supports_chat_template_kwargs:
            args.extend(['--chat-template-kwargs', json.dumps(profile.chat_template_kwargs, sort_keys=True)])
        else:
            unsupported.append('--chat-template-kwargs')
    add_supported('--reasoning', profile.reasoning, capabilities.supports_reasoning)
    add_supported('--reasoning-budget', profile.reasoning_budget, capabilities.supports_reasoning_budget)
    if profile.cache_prompt is not None:
        if capabilities.supports_cache_prompt:
            args.append('--cache-prompt' if profile.cache_prompt else '--no-cache-prompt')
        else:
            unsupported.append('--cache-prompt')
    add_supported('--cache-reuse', profile.cache_reuse if profile.cache_reuse > 0 else None, capabilities.supports_cache_reuse)
    add_supported('-fitt', profile.fit_target, capabilities.supports_fit_target)
    args.extend(strip_runtime_tuning_args(profile.extra_args))
    return args, unsupported


def benchmark_launch_metadata(
    profile: BenchmarkLaunchProfile,
    unsupported_flags: Sequence[str] = (),
) -> Dict[str, object]:
    return {
        'benchmark_profile': profile.name,
        'benchmark_purpose': profile.purpose,
        'ctx': int(profile.ctx),
        'output': int(profile.output),
        'measurement_output': int(profile.measurement_output),
        'temp': float(profile.temp),
        'top_p': profile.top_p,
        'top_k': profile.top_k,
        'repeat_penalty': profile.repeat_penalty,
        'presence_penalty': profile.presence_penalty,
        'min_p': profile.min_p,
        'seed': profile.seed,
        'samplers': profile.samplers,
        'kv_key': profile.kv_key_type,
        'kv_value': profile.kv_value_type,
        'flash_attn': profile.flash_attn,
        'fit': bool(profile.fit),
        'fit_context': int(profile.fit_context),
        'fit_target': profile.fit_target,
        'no_context_shift': bool(profile.no_context_shift),
        'cache_prompt': profile.cache_prompt,
        'cache_reuse': int(profile.cache_reuse),
        'reasoning': profile.reasoning,
        'reasoning_budget': profile.reasoning_budget,
        'preserve_thinking': bool(profile.preserve_thinking),
        'preserve_thinking_source': profile.preserve_thinking_source,
        'chat_template_kwargs': dict(profile.chat_template_kwargs),
        'unsupported_launch_flags': list(unsupported_flags or ()),
    }
