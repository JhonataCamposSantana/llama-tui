import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

DEFAULT_BUUN_LLAMA_SERVER = 'buun-llama-server'
BUUN_KV_MODES = ('turbo4', 'turbo3_tcq', 'turbo2_tcq', 'turbo3', 'turbo2')
COMMON_KV_MODES = (
    'f32',
    'f16',
    'bf16',
    'q8_0',
    'q4_0',
    'q4_1',
    'iq4_nl',
    'q5_0',
    'q5_1',
    *BUUN_KV_MODES,
)
FLASH_ATTN_SYNTAXES = ('value', 'flag', 'unsupported')
RUNTIME_TUNING_FLAGS = (
    '--flash-attn',
    '-fa',
    '--cache-type-k',
    '--cache-type-v',
    '--cache-type',
    '-ctk',
    '-ctv',
    '-ct',
    '--batch-size',
    '--ubatch-size',
    '-b',
    '-ub',
)


@dataclass(frozen=True)
class EngineCapabilities:
    flash_attn_syntax: str = 'value'
    flash_attn_flag: str = '--flash-attn'
    supports_ctk_ctv: bool = False
    supports_cache_type_kv: bool = True
    supports_parallel: bool = True
    gpu_layers_flag: str = '--n-gpu-layers'
    supported_kv_modes: Tuple[str, ...] = ()
    help_text: str = ''


@dataclass(frozen=True)
class TurboKvProfile:
    kv_preset: str
    label: str
    quality_tier: str
    compression_tier: str
    benchmark_depth: str
    score_penalty: float = 0.0
    scalar: bool = False

    @property
    def name_slug(self) -> str:
        return self.kv_preset.replace('/', '_').replace('-', '_')


TURBO_KV_PROFILES: Tuple[TurboKvProfile, ...] = (
    TurboKvProfile('turbo4/turbo4', 'turbo4 safe', 'safe', '4.25bpv', 'fast', 0.0),
    TurboKvProfile('turbo3_tcq/turbo3_tcq', 'turbo3 TCQ', 'balanced', '3.25bpv', 'fast', 0.025),
    TurboKvProfile('turbo3_tcq/turbo2_tcq', 'turbo3/turbo2 TCQ', 'aggressive', '2.75bpv', 'fast', 0.06),
    TurboKvProfile('turbo2_tcq/turbo2_tcq', 'turbo2 TCQ', 'max_context', '2.25bpv', 'full', 0.10),
    TurboKvProfile('turbo3/turbo3', 'turbo3 scalar', 'diagnostic', '3.25bpv_scalar', 'full', 0.08, True),
    TurboKvProfile('turbo2/turbo2', 'turbo2 scalar', 'diagnostic', '2.25bpv_scalar', 'full', 0.15, True),
)


def turbo_kv_profile_for_preset(kv_preset: str) -> Optional[TurboKvProfile]:
    normalized = (kv_preset or '').strip().lower()
    for profile in TURBO_KV_PROFILES:
        if profile.kv_preset == normalized:
            return profile
    key_mode, value_mode = kv_modes_from_preset(normalized)
    symmetric = f'{key_mode}/{value_mode}' if key_mode and value_mode else ''
    for profile in TURBO_KV_PROFILES:
        if profile.kv_preset == symmetric:
            return profile
    return None


def supported_turbo_kv_profiles(
    capabilities: EngineCapabilities,
    depth: str = 'full',
) -> List[TurboKvProfile]:
    normalized_depth = (depth or 'full').strip().lower()
    allowed = {mode.strip().lower() for mode in capabilities.supported_kv_modes or BUUN_KV_MODES}
    profiles: List[TurboKvProfile] = []
    for profile in TURBO_KV_PROFILES:
        if normalized_depth == 'fast' and profile.benchmark_depth != 'fast':
            continue
        key_mode, value_mode = kv_modes_from_preset(profile.kv_preset)
        if key_mode in allowed and value_mode in allowed:
            profiles.append(profile)
    return profiles


@dataclass(frozen=True)
class EngineProfile:
    engine_id: str
    label: str
    server_bin: str
    default_args: Tuple[str, ...] = ()
    supported_kv_modes: Tuple[str, ...] = ()
    flash_attn_syntax: str = 'value'
    supports_turbo_kv: bool = False
    experimental: bool = False
    context_override: Optional[int] = None
    kv_mode: str = ''
    kv_key_mode: str = ''
    kv_value_mode: str = ''

    @property
    def engine(self) -> str:
        return self.engine_id

    @property
    def display_name(self) -> str:
        return self.label

    @property
    def server_command(self) -> str:
        return self.server_bin

    @property
    def is_buun(self) -> bool:
        return self.engine_id == 'buun'

    def llama_extra_args(self) -> List[str]:
        if not self.is_buun:
            return []
        key_mode, value_mode = self.buun_kv_pair()
        return ['--flash-attn', 'on', '-ctk', key_mode, '-ctv', value_mode]

    def header_indicator(self) -> str:
        if self.is_buun:
            key_mode, value_mode = self.buun_kv_pair()
            kv = f'key={key_mode} value={value_mode}'
        else:
            kv = (self.kv_mode or '-').strip() or '-'
        ctx = str(self.context_override) if self.context_override is not None else 'model default'
        suffix = ' | Experimental' if self.experimental else ''
        return f'Engine: {self.label} | KV: {kv} | Context: {ctx}{suffix}'

    def buun_kv_pair(self) -> Tuple[str, str]:
        return resolve_buun_kv_modes(self.kv_mode, self.kv_key_mode, self.kv_value_mode)


@dataclass(frozen=True)
class RuntimeProfile:
    engine_id: str
    ctx_size: int
    gpu_layers: int
    parallel: int
    kv_preset: str = 'default'
    flash_attn: str = 'auto'
    batch_size: int = 0
    ubatch_size: int = 0
    extra_args: Tuple[str, ...] = field(default_factory=tuple)
    name: str = ''
    kv_family: str = 'default'
    kv_quality_tier: str = ''
    kv_compression_tier: str = ''
    kv_score_penalty: float = 0.0
    benchmark_depth: str = ''


def resolve_buun_kv_modes(
    kv_mode: str = '',
    kv_key_mode: str = '',
    kv_value_mode: str = '',
) -> Tuple[str, str]:
    base = (kv_mode or 'turbo4').strip() or 'turbo4'
    key_mode = (kv_key_mode or base).strip() or base
    value_mode = (kv_value_mode or base).strip() or base
    return key_mode, value_mode


def make_runtime_profile(
    engine: str,
    default_llama_server: str,
    ctx_override: Optional[int] = None,
    kv_mode: str = '',
    kv_key_mode: str = '',
    kv_value_mode: str = '',
) -> EngineProfile:
    normalized = (engine or 'llama.cpp').strip().lower()
    if normalized == 'buun':
        command = os.environ.get('BUUN_LLAMA_SERVER_BIN') or DEFAULT_BUUN_LLAMA_SERVER
        key_mode, value_mode = resolve_buun_kv_modes(kv_mode, kv_key_mode, kv_value_mode)
        return EngineProfile(
            engine_id='buun',
            label='buun-llama-cpp',
            server_bin=command,
            default_args=(),
            supported_kv_modes=BUUN_KV_MODES,
            flash_attn_syntax='value',
            supports_turbo_kv=True,
            experimental=True,
            context_override=ctx_override,
            kv_mode=(kv_mode or 'turbo4').strip() or 'turbo4',
            kv_key_mode=key_mode,
            kv_value_mode=value_mode,
        )
    return EngineProfile(
        engine_id='llama.cpp',
        label='llama.cpp',
        server_bin=default_llama_server,
        default_args=(),
        supported_kv_modes=('default', 'f16', 'q8_0', 'q4_0'),
        flash_attn_syntax='value',
        supports_turbo_kv=False,
        experimental=False,
        context_override=ctx_override,
        kv_mode=(kv_mode or '').strip(),
        kv_key_mode='',
        kv_value_mode='',
    )


def default_engine_capabilities(engine_id: str = 'llama.cpp') -> EngineCapabilities:
    normalized = (engine_id or 'llama.cpp').strip().lower()
    if normalized == 'buun':
        return EngineCapabilities(
            flash_attn_syntax='value',
            flash_attn_flag='--flash-attn',
            supports_ctk_ctv=True,
            supports_cache_type_kv=False,
            supports_parallel=True,
            gpu_layers_flag='-ngl',
            supported_kv_modes=BUUN_KV_MODES,
        )
    return EngineCapabilities(supported_kv_modes=('f16', 'q8_0', 'q4_0'))


def parse_supported_kv_modes(help_text: str, engine_id: str, defaults: EngineCapabilities) -> Tuple[str, ...]:
    text = (help_text or '').lower()
    found: List[str] = []
    known = set(COMMON_KV_MODES)
    lines = text.splitlines()
    segments: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if 'allowed values:' not in line:
            index += 1
            continue
        _, _, segment = line.partition('allowed values:')
        index += 1
        while index < len(lines):
            continuation = lines[index].strip()
            if not continuation or continuation.startswith('-') or 'allowed values:' in continuation:
                break
            segment = f'{segment} {continuation}'
            index += 1
        segments.append(segment)

    for segment in segments:
        for token in re.split(r'[\s,|]+', segment):
            cleaned = token.strip(' .;:()[]{}')
            if cleaned in known and cleaned not in found:
                found.append(cleaned)
    if found:
        return tuple(found)
    if (engine_id or '').strip().lower() == 'buun' and defaults.supports_ctk_ctv:
        return BUUN_KV_MODES
    return defaults.supported_kv_modes


def parse_engine_capabilities(help_text: str, engine_id: str = 'llama.cpp') -> EngineCapabilities:
    text = help_text or ''
    low = text.lower()
    defaults = default_engine_capabilities(engine_id)
    has_long_flash = '--flash-attn' in low
    has_short_flash = bool(re.search(r'(^|\s)-fa(\s|,|$)', low))
    if not has_long_flash and not has_short_flash:
        flash_syntax = 'unsupported'
        flash_flag = '--flash-attn'
    else:
        flash_flag = '--flash-attn' if has_long_flash else '-fa'
        value_markers = (
            r'flash-attn[^\n]*(on\|off\|auto|auto\|on\|off|on/off/auto|<[^>]*>|=\w+)',
            r'-fa[^\n]*(on\|off\|auto|auto\|on\|off|on/off/auto|<[^>]*>|=\w+)',
        )
        flash_syntax = 'value' if any(re.search(pattern, low) for pattern in value_markers) else defaults.flash_attn_syntax
        if flash_syntax not in FLASH_ATTN_SYNTAXES:
            flash_syntax = 'value'

    supports_ctk_ctv = ('-ctk' in low and '-ctv' in low) or defaults.supports_ctk_ctv
    supports_cache_type_kv = (
        ('--cache-type-k' in low and '--cache-type-v' in low)
        if ('--cache-type-k' in low or '--cache-type-v' in low)
        else defaults.supports_cache_type_kv
    )
    supports_parallel = '--parallel' in low if '--parallel' in low else defaults.supports_parallel
    if '--n-gpu-layers' in low:
        gpu_layers_flag = '--n-gpu-layers'
    elif re.search(r'(^|\s)-ngl(\s|,|$)', low):
        gpu_layers_flag = '-ngl'
    else:
        gpu_layers_flag = defaults.gpu_layers_flag

    supported_kv_modes = parse_supported_kv_modes(text, engine_id, defaults)

    return EngineCapabilities(
        flash_attn_syntax=flash_syntax,
        flash_attn_flag=flash_flag,
        supports_ctk_ctv=supports_ctk_ctv,
        supports_cache_type_kv=supports_cache_type_kv,
        supports_parallel=supports_parallel,
        gpu_layers_flag=gpu_layers_flag,
        supported_kv_modes=supported_kv_modes,
        help_text=text,
    )


@lru_cache(maxsize=32)
def detect_engine_capabilities(server_bin: str, engine_id: str = 'llama.cpp') -> EngineCapabilities:
    parts = shlex.split(server_bin or '')
    if not parts:
        return default_engine_capabilities(engine_id)
    outputs = []
    for help_flag in ('--help', '-h'):
        try:
            result = subprocess.run(
                [*parts, help_flag],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=4,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = f'{result.stdout or ""}\n{result.stderr or ""}'.strip()
        if output:
            outputs.append(output)
        if result.returncode == 0 and output:
            break
    if not outputs:
        return default_engine_capabilities(engine_id)
    return parse_engine_capabilities('\n'.join(outputs), engine_id)


def strip_runtime_tuning_args(args: Sequence[str], *extra_flags: str) -> List[str]:
    flags = set(RUNTIME_TUNING_FLAGS + tuple(extra_flags))
    cleaned: List[str] = []
    tokens = [str(item) for item in list(args or [])]
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in flags:
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith('-'):
                idx += 2
            else:
                idx += 1
            continue
        if any(token.startswith(f'{flag}=') for flag in flags):
            idx += 1
            continue
        cleaned.append(token)
        idx += 1
    return cleaned


def kv_modes_from_preset(kv_preset: str) -> Tuple[str, str]:
    value = (kv_preset or 'default').strip().lower()
    if value in ('', 'default'):
        return '', ''
    if '/' in value:
        left, right = value.split('/', 1)
        return left.strip(), right.strip()
    return value, value


def is_turbo_kv_preset(kv_preset: str) -> bool:
    key_mode, value_mode = kv_modes_from_preset(kv_preset)
    return key_mode.startswith('turbo') or value_mode.startswith('turbo')


def build_flash_attn_args(mode: str, capabilities: EngineCapabilities) -> List[str]:
    normalized = str(mode or 'auto').strip().lower()
    if normalized in ('', 'auto'):
        normalized = 'on'
    if normalized in ('false', '0', 'no', 'off'):
        normalized = 'off'
    if normalized in ('true', '1', 'yes'):
        normalized = 'on'
    if capabilities.flash_attn_syntax == 'unsupported' or normalized == 'off':
        return []
    flag = capabilities.flash_attn_flag or '--flash-attn'
    if capabilities.flash_attn_syntax == 'flag':
        return [flag]
    return [flag, normalized]


def runtime_profile_extra_args(
    engine: EngineProfile,
    runtime_profile: RuntimeProfile,
    capabilities: EngineCapabilities,
    existing_args: Sequence[str] = (),
) -> List[str]:
    args = strip_runtime_tuning_args(existing_args)
    args.extend(build_flash_attn_args(runtime_profile.flash_attn, capabilities))
    kv_key, kv_value = kv_modes_from_preset(runtime_profile.kv_preset)
    if engine.supports_turbo_kv and is_turbo_kv_preset(runtime_profile.kv_preset):
        if capabilities.supports_ctk_ctv:
            args += ['-ctk', kv_key or 'turbo4', '-ctv', kv_value or 'turbo4']
    elif kv_key and kv_value and capabilities.supports_cache_type_kv:
        args += ['--cache-type-k', kv_key, '--cache-type-v', kv_value]
    if runtime_profile.batch_size > 0:
        args += ['--batch-size', str(int(runtime_profile.batch_size))]
    if runtime_profile.ubatch_size > 0:
        args += ['--ubatch-size', str(int(runtime_profile.ubatch_size))]
    args.extend(str(item) for item in runtime_profile.extra_args)
    return args
