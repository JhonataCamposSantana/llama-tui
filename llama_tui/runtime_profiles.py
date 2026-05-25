import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

DEFAULT_TURBOQUANT_LLAMA_SERVER = (
    str(Path.home() / 'llama-cpp-turboquant' / 'build' / 'bin' / 'llama-server')
    if (Path.home() / 'llama-cpp-turboquant' / 'build' / 'bin' / 'llama-server').exists()
    else 'turboquant-llama-server'
)
TURBOQUANT_KV_MODES = ('q8_0', 'turbo4', 'turbo3', 'turbo2')
# Benchmark-validated default value cache for compatible TurboQuant+ models
# (RTX 4060 8GB + i5-13420H): -ctk q8_0 -ctv turbo3 was the measured winner.
# Used only when the user has not pinned a value cache and no benchmark
# winner is persisted; incompatible/small-head models fall back to q8_0.
DEFAULT_TURBOQUANT_VALUE_MODE = 'turbo3'
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
    *TURBOQUANT_KV_MODES,
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
    '--fit',
    '-fit',
    '--fit-ctx',
    '-fitc',
    '--fit-target',
    '-fitt',
    '--cache-ram',
    '--no-mmap',
    '--mlock',
    '--warmup',
    '--no-warmup',
    '--temp',
    '--temperature',
    '--top-p',
    '--top-k',
    '--min-p',
    '--repeat-penalty',
    '--presence-penalty',
    '--samplers',
    '--seed',
    '--context-shift',
    '--no-context-shift',
    '--chat-template-kwargs',
    '--reasoning',
    '-rea',
    '--reasoning-budget',
    '--reasoning-format',
    '--cache-prompt',
    '--no-cache-prompt',
    '--cache-reuse',
    '--cpu-moe',
    '-cmoe',
    '--n-cpu-moe',
    '-ncmoe',
    '--override-tensor',
    '--override-tensors',
    '-ot',
    '--spec-type',
    '--spec-draft-n-max',
    '--spec-draft-type-k',
    '--spec-draft-type-v',
    '--cache-type-k-draft',
    '--cache-type-v-draft',
)


@dataclass(frozen=True)
class CommandStatus:
    command: str
    exists: bool
    executable: bool
    resolved_path: str = ''
    path_like: bool = False


@dataclass(frozen=True)
class MtpBinaryResolution:
    command: str
    source: str
    exists: bool
    executable: bool
    resolved_path: str = ''
    checked_paths: Tuple[str, ...] = ()


def command_file_status(command: str) -> CommandStatus:
    parts = shlex.split(command or '')
    if not parts:
        return CommandStatus('', False, False)
    first = os.path.expanduser(parts[0])
    path_like = '/' in first or first.startswith('.') or first.startswith('~')
    if path_like:
        path = Path(first).expanduser()
        exists = path.exists()
        executable = bool(exists and os.access(path, os.X_OK))
        return CommandStatus(command, exists, executable, str(path), True)
    resolved = shutil.which(first)
    return CommandStatus(command, resolved is not None, resolved is not None, str(resolved or ''), False)


MTP_ENV_BINARY_NAMES = ('llama-server', 'server', 'llama-server-mtp')
MTP_ENV_RELATIVE_CANDIDATES = (
    Path('llama-server'),
    Path('build') / 'bin' / 'llama-server',
    Path('build-mtp') / 'bin' / 'llama-server',
    Path('build-mtp') / 'bin' / 'server',
    Path('build') / 'bin' / 'server',
)


def llama_cpp_mtp_default_candidates(home: Optional[Path] = None) -> Tuple[str, ...]:
    root = Path.home() if home is None else Path(home).expanduser()
    return (
        str(root / 'src' / 'llama.cpp-mtp' / 'build-mtp' / 'bin' / 'llama-server'),
        str(root / 'src' / 'llama.cpp-mtp' / 'build' / 'bin' / 'llama-server'),
        str(root / 'llama.cpp-mtp' / 'build-mtp' / 'bin' / 'llama-server'),
        str(root / 'llama.cpp-mtp' / 'build' / 'bin' / 'llama-server'),
        str(root / 'src' / 'llama.cpp' / 'build-mtp' / 'bin' / 'llama-server'),
        str(root / '.local' / 'bin' / 'llama-server-mtp'),
    )


DEFAULT_LLAMA_CPP_MTP_SERVER = llama_cpp_mtp_default_candidates()[0]


def _select_mtp_candidate(candidates: Sequence[str], source: str) -> MtpBinaryResolution:
    checked = tuple(str(item) for item in candidates if str(item or '').strip())
    if not checked:
        return MtpBinaryResolution(DEFAULT_LLAMA_CPP_MTP_SERVER, source, False, False, checked_paths=())
    statuses = [command_file_status(item) for item in checked]
    chosen = next((item for item in statuses if item.executable), None)
    if chosen is None:
        chosen = next((item for item in statuses if item.exists), None)
    if chosen is None:
        chosen = statuses[0]
    return MtpBinaryResolution(
        command=chosen.command,
        source=source,
        exists=chosen.exists,
        executable=chosen.executable,
        resolved_path=chosen.resolved_path,
        checked_paths=checked,
    )


def llama_cpp_mtp_candidates_from_env(value: str) -> Tuple[str, ...]:
    text = str(value or '').strip()
    if not text:
        return ()
    path = Path(text).expanduser()
    if '/' not in text and not text.startswith('.') and not text.startswith('~'):
        return (text,)
    if path.is_file() or path.name in MTP_ENV_BINARY_NAMES:
        return (str(path),)
    return tuple(str(path / relative) for relative in MTP_ENV_RELATIVE_CANDIDATES)


def resolve_llama_cpp_mtp_binary(env_value: Optional[str] = None) -> MtpBinaryResolution:
    value = os.environ.get('LLAMA_CPP_MTP_PATH') if env_value is None else env_value
    if str(value or '').strip():
        return _select_mtp_candidate(
            llama_cpp_mtp_candidates_from_env(str(value or '')),
            'env:LLAMA_CPP_MTP_PATH',
        )
    default_resolution = _select_mtp_candidate(llama_cpp_mtp_default_candidates(), 'default')
    if default_resolution.executable:
        return default_resolution
    path_resolution = _select_mtp_candidate(('llama-server-mtp',), 'PATH')
    if path_resolution.executable:
        return path_resolution
    return default_resolution


@dataclass(frozen=True)
class EngineCapabilities:
    flash_attn_syntax: str = 'value'
    flash_attn_flag: str = '--flash-attn'
    supports_ctk_ctv: bool = False
    supports_cache_type_kv: bool = True
    supports_parallel: bool = True
    supports_fit: bool = False
    supports_fit_ctx: bool = False
    supports_fit_target: bool = False
    supports_cache_ram: bool = True
    supports_no_mmap: bool = False
    supports_mlock: bool = False
    supports_no_warmup: bool = False
    supports_context_shift: bool = False
    supports_chat_template_kwargs: bool = False
    supports_reasoning: bool = False
    supports_reasoning_budget: bool = False
    supports_reasoning_format: bool = False
    supports_cache_prompt: bool = False
    supports_cache_reuse: bool = False
    supports_top_p: bool = False
    supports_top_k: bool = False
    supports_min_p: bool = False
    supports_repeat_penalty: bool = False
    supports_presence_penalty: bool = False
    supports_samplers: bool = False
    supports_seed: bool = False
    supports_cpu_moe: bool = False
    supports_n_cpu_moe: bool = False
    supports_override_tensor: bool = False
    supports_spec_type: bool = False
    supports_mtp: bool = False
    spec_type_values: Tuple[str, ...] = ()
    mtp_spec_type: str = ''
    mtp_spec_type_value: str = ''
    supports_spec_draft_n_max: bool = False
    supports_spec_draft_type_kv: bool = False
    cpu_moe_flag: str = '-cmoe'
    n_cpu_moe_flag: str = '-ncmoe'
    override_tensor_flag: str = '-ot'
    spec_type_flag: str = '--spec-type'
    spec_draft_n_max_flag: str = '--spec-draft-n-max'
    spec_draft_type_k_flag: str = '--spec-draft-type-k'
    spec_draft_type_v_flag: str = '--spec-draft-type-v'
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
    profile_class: str = ''

    @property
    def name_slug(self) -> str:
        return self.kv_preset.replace('/', '_').replace('-', '_')


TURBO_KV_PROFILES: Tuple[TurboKvProfile, ...] = (
    TurboKvProfile('turbo4/turbo4', 'turbo4 safe', 'safe', '4.25bpv', 'fast', 0.0, False, 'safe'),
    TurboKvProfile('turbo3_tcq/turbo3_tcq', 'turbo3 TCQ', 'balanced', '3.25bpv', 'fast', 0.025, False, 'balanced'),
    TurboKvProfile('turbo3_tcq/turbo2_tcq', 'turbo3/turbo2 TCQ', 'aggressive', '2.75bpv', 'fast', 0.06, False, 'aggressive'),
    TurboKvProfile('turbo2_tcq/turbo2_tcq', 'turbo2 TCQ', 'max_context', '2.25bpv', 'full', 0.10, False, 'extreme'),
    TurboKvProfile('turbo3/turbo3', 'turbo3 scalar', 'diagnostic', '3.25bpv_scalar', 'full', 0.08, True, 'diagnostic'),
    TurboKvProfile('turbo2/turbo2', 'turbo2 scalar', 'diagnostic', '2.25bpv_scalar', 'full', 0.15, True, 'diagnostic'),
)

TURBOQUANT_KV_PROFILES: Tuple[TurboKvProfile, ...] = (
    TurboKvProfile('q8_0/q8_0', 'q8 baseline', 'baseline', '8.5bpv', 'fast', 0.0, False, 'baseline'),
    TurboKvProfile('q8_0/turbo4', 'safe V turbo4', 'safe', '6.375bpv', 'fast', 0.0, False, 'safe'),
    TurboKvProfile('q8_0/turbo3', 'balanced V turbo3', 'balanced', '5.8125bpv', 'fast', 0.025, False, 'balanced'),
    TurboKvProfile('q8_0/turbo2', 'extreme V turbo2', 'extreme_v', '5.3125bpv', 'full', 0.08, False, 'extreme V'),
    TurboKvProfile('turbo4/turbo4', 'symmetric turbo4', 'symmetric', '4.25bpv', 'full', 0.06, False, 'symmetric'),
    TurboKvProfile('turbo3/turbo3', 'symmetric turbo3', 'symmetric', '3.125bpv', 'full', 0.10, False, 'symmetric'),
    TurboKvProfile('turbo2/turbo2', 'symmetric turbo2', 'symmetric', '2.125bpv', 'full', 0.18, False, 'symmetric'),
)

def turbo_kv_profile_for_preset(kv_preset: str) -> Optional[TurboKvProfile]:
    normalized = (kv_preset or '').strip().lower()
    for profile_set in (TURBO_KV_PROFILES, TURBOQUANT_KV_PROFILES):
        for profile in profile_set:
            if profile.kv_preset == normalized:
                return profile
    key_mode, value_mode = kv_modes_from_preset(normalized)
    symmetric = f'{key_mode}/{value_mode}' if key_mode and value_mode else ''
    for profile_set in (TURBO_KV_PROFILES, TURBOQUANT_KV_PROFILES):
        for profile in profile_set:
            if profile.kv_preset == symmetric:
                return profile
    return None


def supported_turbo_kv_profiles(
    capabilities: EngineCapabilities,
    depth: str = 'full',
    engine_id: str = 'turboquant',
) -> List[TurboKvProfile]:
    normalized_depth = (depth or 'full').strip().lower()
    defaults = TURBOQUANT_KV_MODES
    profile_set = TURBOQUANT_KV_PROFILES
    allowed = {mode.strip().lower() for mode in capabilities.supported_kv_modes or defaults}
    profiles: List[TurboKvProfile] = []
    for profile in profile_set:
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
    kv_value_explicit: bool = False

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
    def is_turboquant(self) -> bool:
        return self.engine_id == 'turboquant'

    @property
    def is_llama_cpp_mtp(self) -> bool:
        return self.engine_id == 'llama.cpp-mtp'

    def header_indicator(self) -> str:
        if self.is_turboquant:
            key_mode, value_mode = self.engine_kv_pair()
            kv = f'key={key_mode} value={value_mode}'
        elif self.is_llama_cpp_mtp:
            kv = 'default | MTP Experimental'
        else:
            kv = (self.kv_mode or '-').strip() or '-'
        ctx = str(self.context_override) if self.context_override is not None else 'model default'
        suffix = ' | Experimental' if self.experimental else ''
        return f'Engine: {self.label} | KV: {kv} | Context: {ctx}{suffix}'

    def turboquant_kv_pair(self) -> Tuple[str, str]:
        return resolve_turboquant_kv_modes(self.kv_mode, self.kv_key_mode, self.kv_value_mode)

    def engine_kv_pair(self) -> Tuple[str, str]:
        return self.turboquant_kv_pair()


@dataclass(frozen=True)
class RuntimeProfile:
    engine_id: str
    ctx_size: int
    gpu_layers: Optional[int]
    parallel: int
    kv_preset: str = 'default'
    flash_attn: str = 'auto'
    batch_size: int = 0
    ubatch_size: int = 0
    fit: bool = False
    fit_context: int = 0
    fit_target: int = 0
    cache_ram: Optional[int] = None
    no_mmap: bool = False
    no_warmup: bool = False
    extra_args: Tuple[str, ...] = field(default_factory=tuple)
    name: str = ''
    kv_family: str = 'default'
    kv_quality_tier: str = ''
    kv_compression_tier: str = ''
    kv_score_penalty: float = 0.0
    benchmark_depth: str = ''
    fit_discovery_phase: str = ''
    viable_ngl: int = 0
    viable_ngl_source: str = ''
    fit_selected_ngl: int = 0
    fit_selected_ngl_source: str = ''
    fit_log_excerpt: str = ''
    placement_strategy: str = ''
    cpu_moe: bool = False
    n_cpu_moe: int = 0
    tensor_overrides: Tuple[str, ...] = field(default_factory=tuple)
    threads: int = 0
    reasoning: str = ''
    reasoning_budget: int = -1
    reasoning_format: str = ''
    mtp_enabled: bool = False
    mtp_draft_n_max: int = 0
    mtp_draft_kv_preset: str = ''
    mtp_spec_type: str = ''
    benchmark_strategy_id: str = ''
    benchmark_objectives: Tuple[str, ...] = field(default_factory=tuple)
    benchmark_phase: str = ''
    benchmark_metric_group: str = ''


@dataclass(frozen=True)
class MtpArgDiagnostics:
    enabled: bool = False
    selected_spec_type: str = ''
    draft_n_max: int = 0
    added_flags: Tuple[str, ...] = ()
    skipped_flags: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    blocked_reason: str = ''


def llama_cpp_mtp_server_from_env() -> str:
    return resolve_llama_cpp_mtp_binary().command


def resolve_turboquant_kv_modes(
    kv_mode: str = '',
    kv_key_mode: str = '',
    kv_value_mode: str = '',
) -> Tuple[str, str]:
    base = (kv_mode or 'q8_0').strip() or 'q8_0'
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
    kv_value_explicit: Optional[bool] = None,
) -> EngineProfile:
    normalized = (engine or 'llama.cpp').strip().lower()
    if normalized == 'turboquant':
        command = os.environ.get('TURBOQUANT_LLAMA_SERVER_BIN') or DEFAULT_TURBOQUANT_LLAMA_SERVER
        key_mode, value_mode = resolve_turboquant_kv_modes(kv_mode, kv_key_mode, kv_value_mode)
        # Distinguish "user pinned a value cache" from "value defaulted to q8_0".
        # --kv (shorthand) and --kv-value both pin the value; --kv-key alone does not.
        explicit = (
            bool((str(kv_value_mode or '') + str(kv_mode or '')).strip())
            if kv_value_explicit is None
            else bool(kv_value_explicit)
        )
        return EngineProfile(
            engine_id='turboquant',
            label='TurboQuant+',
            server_bin=command,
            default_args=(),
            supported_kv_modes=TURBOQUANT_KV_MODES,
            flash_attn_syntax='value',
            supports_turbo_kv=True,
            experimental=True,
            context_override=ctx_override,
            kv_mode=(kv_mode or 'q8_0').strip() or 'q8_0',
            kv_key_mode=key_mode,
            kv_value_mode=value_mode,
            kv_value_explicit=explicit,
        )
    if normalized in ('llama.cpp-mtp', 'llama-cpp-mtp', 'llamacpp-mtp', 'mtp'):
        # Audit finding #7: MTP merged into upstream llama.cpp via
        # --spec-type mtp / draft-mtp, so a "llama.cpp-mtp" session now
        # produces a regular llama.cpp engine profile. The legacy
        # LLAMA_CPP_MTP_PATH env / sibling paths still resolve the
        # binary in case the user built the fork to a non-standard
        # location; MTP itself is detected at the capability layer
        # from --help output.
        command = llama_cpp_mtp_server_from_env() or default_llama_server
        return EngineProfile(
            engine_id='llama.cpp',
            label='llama.cpp',
            server_bin=command,
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
    if normalized == 'turboquant':
        return EngineCapabilities(
            flash_attn_syntax='value',
            flash_attn_flag='--flash-attn',
            supports_ctk_ctv=True,
            supports_cache_type_kv=True,
            supports_parallel=True,
            supports_fit=True,
            supports_fit_ctx=True,
            supports_no_warmup=False,
            gpu_layers_flag='-ngl',
            supported_kv_modes=TURBOQUANT_KV_MODES,
        )
    if normalized in ('llama.cpp-mtp', 'llama-cpp-mtp', 'llamacpp-mtp', 'mtp'):
        return EngineCapabilities(supported_kv_modes=('f16', 'q8_0', 'q4_0'))
    return EngineCapabilities(supported_kv_modes=('f16', 'q8_0', 'q4_0'))


MTP_SPEC_TYPE_VALUES = ('mtp', 'draft-mtp')


def _spec_type_segments(help_text: str) -> Tuple[str, ...]:
    lines = str(help_text or '').lower().splitlines()
    segments: List[str] = []
    for index, line in enumerate(lines):
        if '--spec-type' not in line:
            continue
        segment = line
        next_index = index + 1
        while next_index < len(lines):
            continuation = lines[next_index].strip()
            if not continuation or continuation.startswith('-'):
                break
            segment = f'{segment} {continuation}'
            next_index += 1
        segments.append(segment)
    return tuple(segments)


def parse_spec_type_values(help_text: str, defaults: Optional[EngineCapabilities] = None) -> Tuple[str, ...]:
    default_values = tuple(str(item or '').strip().lower() for item in (getattr(defaults, 'spec_type_values', ()) or ())) if defaults is not None else ()
    segments = _spec_type_segments(help_text)
    values: List[str] = []
    for segment in segments:
        for token in re.split(r'[\s,|<>\[\]{}()]+', segment):
            cleaned = token.strip(' .;:').lower()
            if not cleaned:
                continue
            if cleaned in (
                '--spec-type',
                'spec-type',
                'speculative',
                'type',
                'value',
                'values',
                'allowed',
                'usage',
                'llama-server',
                'n',
                'file',
                'arg',
                'args',
            ):
                continue
            if cleaned.startswith('-') or '=' in cleaned:
                continue
            if not re.match(r'^[a-z0-9][a-z0-9_-]*$', cleaned):
                continue
            if cleaned not in values:
                values.append(cleaned)
    return tuple(values) or default_values


def select_mtp_spec_type_value(spec_type_values: Sequence[str], default_value: str = '') -> str:
    # Upstream llama.cpp advertises `draft-mtp`; the older fork dialect used `mtp`.
    # Prefer `draft-mtp` whenever the binary advertises it, and only fall back to
    # the legacy `mtp` value when that is the sole speculative MTP dialect offered.
    values = {str(item or '').strip().lower() for item in (spec_type_values or ())}
    if 'draft-mtp' in values:
        return 'draft-mtp'
    if 'mtp' in values:
        return 'mtp'
    default_text = str(default_value or '').strip().lower()
    return default_text if default_text in MTP_SPEC_TYPE_VALUES else ''


def mtp_spec_type_value(capabilities: Optional[EngineCapabilities]) -> str:
    value = str(
        getattr(capabilities, 'mtp_spec_type_value', '')
        or getattr(capabilities, 'mtp_spec_type', '')
        or ''
    ).strip().lower() if capabilities is not None else ''
    return value if value in MTP_SPEC_TYPE_VALUES else ''


def command_spec_type_value(command_tokens: Sequence[object]) -> str:
    """Return the last --spec-type value present in a built command, lowercased."""
    tokens = [str(item or '') for item in (command_tokens or [])]
    value = ''
    for index, token in enumerate(tokens):
        if token == '--spec-type' and index + 1 < len(tokens):
            value = tokens[index + 1].strip().lower()
        elif token.startswith('--spec-type='):
            value = token.split('=', 1)[1].strip().lower()
    return value


def validate_mtp_command(
    command_tokens: Sequence[object],
    capabilities: Optional[EngineCapabilities],
) -> Tuple[bool, str]:
    """Reject a launch command whose --spec-type value the binary does not advertise.

    Returns (ok, error). When the binary's speculative types are unknown (help
    could not be parsed) the command is allowed through so a missing --help does
    not block otherwise-valid launches.
    """
    requested = command_spec_type_value(command_tokens)
    if not requested or requested == 'none':
        return True, ''
    advertised = {
        str(item or '').strip().lower()
        for item in (getattr(capabilities, 'spec_type_values', ()) or ())
        if str(item or '').strip()
    }
    if not advertised:
        return True, ''
    if requested in advertised:
        return True, ''
    supported = ', '.join(sorted(advertised))
    return False, f'INVALID_MTP_SPEC_TYPE: requested {requested}, binary supports {supported}'


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
    normalized_engine = (engine_id or '').strip().lower()
    if normalized_engine == 'turboquant' and defaults.supports_ctk_ctv:
        return TURBOQUANT_KV_MODES
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
    supports_fit = bool(re.search(r'(^|\s)--?fit(\s|,|$)', low) or defaults.supports_fit)
    supports_fit_ctx = ('-fitc' in low or '--fit-ctx' in low or defaults.supports_fit_ctx)
    supports_fit_target = ('-fitt' in low or '--fit-target' in low or defaults.supports_fit_target)
    supports_cache_ram = ('--cache-ram' in low or defaults.supports_cache_ram)
    supports_no_mmap = ('--no-mmap' in low or defaults.supports_no_mmap)
    supports_mlock = ('--mlock' in low or defaults.supports_mlock)
    supports_no_warmup = ('--no-warmup' in low or defaults.supports_no_warmup)
    supports_context_shift = ('--context-shift' in low or '--no-context-shift' in low or defaults.supports_context_shift)
    supports_chat_template_kwargs = ('--chat-template-kwargs' in low or defaults.supports_chat_template_kwargs)
    supports_reasoning = ('--reasoning' in low or re.search(r'(^|\s)-rea(\s|,|$)', low) is not None or defaults.supports_reasoning)
    supports_reasoning_budget = ('--reasoning-budget' in low or defaults.supports_reasoning_budget)
    supports_reasoning_format = ('--reasoning-format' in low or defaults.supports_reasoning_format)
    supports_cache_prompt = ('--cache-prompt' in low or '--no-cache-prompt' in low or defaults.supports_cache_prompt)
    supports_cache_reuse = ('--cache-reuse' in low or defaults.supports_cache_reuse)
    supports_top_p = ('--top-p' in low or defaults.supports_top_p)
    supports_top_k = ('--top-k' in low or defaults.supports_top_k)
    supports_min_p = ('--min-p' in low or defaults.supports_min_p)
    supports_repeat_penalty = ('--repeat-penalty' in low or defaults.supports_repeat_penalty)
    supports_presence_penalty = ('--presence-penalty' in low or defaults.supports_presence_penalty)
    supports_samplers = ('--samplers' in low or defaults.supports_samplers)
    supports_seed = (re.search(r'(^|\s)-s(\s|,|$)', low) is not None or '--seed' in low or defaults.supports_seed)
    has_short_cpu_moe = re.search(r'(^|\s|,)-cmoe(\s|,|$)', low) is not None
    has_long_cpu_moe = '--cpu-moe' in low
    supports_cpu_moe = has_short_cpu_moe or has_long_cpu_moe or defaults.supports_cpu_moe
    cpu_moe_flag = '-cmoe' if has_short_cpu_moe else '--cpu-moe' if has_long_cpu_moe else defaults.cpu_moe_flag
    has_short_n_cpu_moe = re.search(r'(^|\s|,)-ncmoe(\s|,|$)', low) is not None
    has_long_n_cpu_moe = '--n-cpu-moe' in low
    supports_n_cpu_moe = has_short_n_cpu_moe or has_long_n_cpu_moe or defaults.supports_n_cpu_moe
    n_cpu_moe_flag = '-ncmoe' if has_short_n_cpu_moe else '--n-cpu-moe' if has_long_n_cpu_moe else defaults.n_cpu_moe_flag
    has_short_override_tensor = re.search(r'(^|\s|,)-ot(\s|,|$)', low) is not None
    has_long_override_tensor = re.search(r'(^|\s|,)--override-tensor(\s|,|$)', low) is not None
    has_long_override_tensors = re.search(r'(^|\s|,)--override-tensors(\s|,|$)', low) is not None
    supports_override_tensor = (
        has_short_override_tensor
        or has_long_override_tensor
        or has_long_override_tensors
        or defaults.supports_override_tensor
    )
    supports_spec_type = '--spec-type' in low or defaults.supports_spec_type
    supports_spec_draft_n_max = '--spec-draft-n-max' in low or defaults.supports_spec_draft_n_max
    has_draft_type_k = '--spec-draft-type-k' in low or '--cache-type-k-draft' in low
    has_draft_type_v = '--spec-draft-type-v' in low or '--cache-type-v-draft' in low
    supports_spec_draft_type_kv = (has_draft_type_k and has_draft_type_v) or defaults.supports_spec_draft_type_kv
    spec_draft_type_k_flag = (
        '--spec-draft-type-k' if '--spec-draft-type-k' in low
        else '--cache-type-k-draft' if '--cache-type-k-draft' in low
        else defaults.spec_draft_type_k_flag
    )
    spec_draft_type_v_flag = (
        '--spec-draft-type-v' if '--spec-draft-type-v' in low
        else '--cache-type-v-draft' if '--cache-type-v-draft' in low
        else defaults.spec_draft_type_v_flag
    )
    spec_type_values = parse_spec_type_values(text, defaults)
    mtp_spec_type = select_mtp_spec_type_value(spec_type_values, mtp_spec_type_value(defaults))
    supports_mtp = bool(supports_spec_type and mtp_spec_type) or defaults.supports_mtp
    override_tensor_flag = (
        '-ot' if has_short_override_tensor
        else '--override-tensor' if has_long_override_tensor
        else '--override-tensors' if has_long_override_tensors
        else defaults.override_tensor_flag
    )
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
        supports_fit=supports_fit,
        supports_fit_ctx=supports_fit_ctx,
        supports_fit_target=supports_fit_target,
        supports_cache_ram=supports_cache_ram,
        supports_no_mmap=supports_no_mmap,
        supports_mlock=supports_mlock,
        supports_no_warmup=supports_no_warmup,
        supports_context_shift=supports_context_shift,
        supports_chat_template_kwargs=supports_chat_template_kwargs,
        supports_reasoning=supports_reasoning,
        supports_reasoning_budget=supports_reasoning_budget,
        supports_reasoning_format=supports_reasoning_format,
        supports_cache_prompt=supports_cache_prompt,
        supports_cache_reuse=supports_cache_reuse,
        supports_top_p=supports_top_p,
        supports_top_k=supports_top_k,
        supports_min_p=supports_min_p,
        supports_repeat_penalty=supports_repeat_penalty,
        supports_presence_penalty=supports_presence_penalty,
        supports_samplers=supports_samplers,
        supports_seed=supports_seed,
        supports_cpu_moe=supports_cpu_moe,
        supports_n_cpu_moe=supports_n_cpu_moe,
        supports_override_tensor=supports_override_tensor,
        supports_spec_type=supports_spec_type,
        supports_mtp=supports_mtp,
        spec_type_values=spec_type_values,
        mtp_spec_type=mtp_spec_type,
        mtp_spec_type_value=mtp_spec_type,
        supports_spec_draft_n_max=supports_spec_draft_n_max,
        supports_spec_draft_type_kv=supports_spec_draft_type_kv,
        cpu_moe_flag=cpu_moe_flag,
        n_cpu_moe_flag=n_cpu_moe_flag,
        override_tensor_flag=override_tensor_flag,
        spec_type_flag='--spec-type',
        spec_draft_n_max_flag='--spec-draft-n-max',
        spec_draft_type_k_flag=spec_draft_type_k_flag,
        spec_draft_type_v_flag=spec_draft_type_v_flag,
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


def clamp_mtp_draft_n_max(value: object, default: int = 3) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, min(3, parsed))


def build_mtp_args(
    model,
    runtime_profile: Optional[RuntimeProfile],
    capabilities: EngineCapabilities,
) -> Tuple[List[str], MtpArgDiagnostics]:
    if runtime_profile is None:
        return [], MtpArgDiagnostics(skipped_flags=('runtime_profile_missing',))
    # MTP is a binary capability, not a dedicated engine: any llama.cpp-compatible
    # binary that advertises the speculative MTP flags can run it. Gate purely on
    # the runtime profile opting in and on the resolved binary capabilities below.
    if not bool(getattr(runtime_profile, 'mtp_enabled', False)):
        return [], MtpArgDiagnostics(skipped_flags=('mtp_disabled',))
    if not bool(getattr(capabilities, 'supports_spec_type', False)):
        return [], MtpArgDiagnostics(
            enabled=True,
            skipped_flags=(getattr(capabilities, 'spec_type_flag', '--spec-type') or '--spec-type',),
            blocked_reason='missing --spec-type',
        )
    profile_spec_type = str(getattr(runtime_profile, 'mtp_spec_type', '') or '').strip().lower()
    capability_spec_type = mtp_spec_type_value(capabilities)
    capability_values = {
        str(item or '').strip().lower()
        for item in tuple(getattr(capabilities, 'spec_type_values', ()) or ())
    }
    profile_spec_allowed = bool(
        profile_spec_type in MTP_SPEC_TYPE_VALUES
        and (
            profile_spec_type == capability_spec_type
            or (capability_values and profile_spec_type in capability_values)
        )
    )
    spec_type = profile_spec_type if profile_spec_allowed else capability_spec_type
    if not bool(getattr(capabilities, 'supports_mtp', False)) or not spec_type:
        return [], MtpArgDiagnostics(
            enabled=True,
            selected_spec_type=spec_type,
            skipped_flags=(getattr(capabilities, 'spec_type_flag', '--spec-type') or '--spec-type',),
            blocked_reason='missing mtp/draft-mtp value',
        )
    if not bool(getattr(capabilities, 'supports_spec_draft_n_max', False)):
        return [], MtpArgDiagnostics(
            enabled=True,
            selected_spec_type=spec_type,
            skipped_flags=(getattr(capabilities, 'spec_draft_n_max_flag', '--spec-draft-n-max') or '--spec-draft-n-max',),
            blocked_reason='missing --spec-draft-n-max',
        )
    try:
        draft_source = int(getattr(runtime_profile, 'mtp_draft_n_max', 0) or 0)
    except (TypeError, ValueError):
        draft_source = 0
    if draft_source <= 0 and model is not None:
        try:
            draft_source = int(getattr(model, 'mtp_draft_n_max', 0) or 0)
        except (TypeError, ValueError):
            draft_source = 0
    draft = clamp_mtp_draft_n_max(draft_source or 3, default=3)
    args = [
        getattr(capabilities, 'spec_type_flag', '--spec-type') or '--spec-type',
        spec_type,
        getattr(capabilities, 'spec_draft_n_max_flag', '--spec-draft-n-max') or '--spec-draft-n-max',
        str(draft),
    ]
    draft_kv_key, draft_kv_value = kv_modes_from_preset(str(getattr(runtime_profile, 'mtp_draft_kv_preset', '') or ''))
    if draft_kv_key and draft_kv_value:
        if getattr(capabilities, 'supports_spec_draft_type_kv', False):
            args += [
                getattr(capabilities, 'spec_draft_type_k_flag', '--spec-draft-type-k') or '--spec-draft-type-k',
                draft_kv_key,
                getattr(capabilities, 'spec_draft_type_v_flag', '--spec-draft-type-v') or '--spec-draft-type-v',
                draft_kv_value,
            ]
        else:
            return [], MtpArgDiagnostics(
                enabled=True,
                selected_spec_type=spec_type,
                draft_n_max=draft,
                skipped_flags=(
                    getattr(capabilities, 'spec_draft_type_k_flag', '--spec-draft-type-k') or '--spec-draft-type-k',
                    getattr(capabilities, 'spec_draft_type_v_flag', '--spec-draft-type-v') or '--spec-draft-type-v',
                ),
                blocked_reason='missing MTP draft KV cache flags',
            )
    return args, MtpArgDiagnostics(
        enabled=True,
        selected_spec_type=spec_type,
        draft_n_max=draft,
        added_flags=tuple(args),
    )


def runtime_profile_extra_args(
    engine: EngineProfile,
    runtime_profile: RuntimeProfile,
    capabilities: EngineCapabilities,
    existing_args: Sequence[str] = (),
) -> List[str]:
    args = strip_runtime_tuning_args(existing_args)
    profile_extra_args = strip_runtime_tuning_args(runtime_profile.extra_args)
    mtp_enabled = bool(getattr(runtime_profile, 'mtp_enabled', False))
    args.extend(build_flash_attn_args(runtime_profile.flash_attn, capabilities))
    kv_key, kv_value = kv_modes_from_preset(runtime_profile.kv_preset)
    if (engine.is_llama_cpp_mtp or mtp_enabled) and kv_key and kv_value and capabilities.supports_ctk_ctv:
        args += ['-ctk', kv_key, '-ctv', kv_value]
    elif engine.is_turboquant and kv_key and kv_value and capabilities.supports_ctk_ctv:
        args += ['-ctk', kv_key, '-ctv', kv_value]
    elif engine.supports_turbo_kv and is_turbo_kv_preset(runtime_profile.kv_preset):
        if capabilities.supports_ctk_ctv:
            args += ['-ctk', kv_key or 'turbo4', '-ctv', kv_value or 'turbo4']
    elif kv_key and kv_value and capabilities.supports_cache_type_kv:
        args += ['--cache-type-k', kv_key, '--cache-type-v', kv_value]
    if runtime_profile.batch_size > 0:
        args += ['--batch-size', str(int(runtime_profile.batch_size))]
    if runtime_profile.ubatch_size > 0:
        args += ['--ubatch-size', str(int(runtime_profile.ubatch_size))]
    if runtime_profile.fit and capabilities.supports_fit:
        args += ['-fit', 'on']
        if runtime_profile.fit_context > 0 and capabilities.supports_fit_ctx:
            args += ['-fitc', str(int(runtime_profile.fit_context))]
    if bool(getattr(runtime_profile, 'no_mmap', False)):
        if capabilities.supports_no_mmap:
            args.append('--no-mmap')
    if runtime_profile.no_warmup and capabilities.supports_no_warmup:
        args.append('--no-warmup')
    if bool(getattr(runtime_profile, 'cpu_moe', False)) and capabilities.supports_cpu_moe:
        args.append(capabilities.cpu_moe_flag or '-cmoe')
    elif int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0) > 0 and capabilities.supports_n_cpu_moe:
        args += [capabilities.n_cpu_moe_flag or '-ncmoe', str(int(getattr(runtime_profile, 'n_cpu_moe', 0) or 0))]
    if capabilities.supports_override_tensor:
        for override in tuple(getattr(runtime_profile, 'tensor_overrides', ()) or ()):
            value = str(override or '').strip()
            if value:
                args += [capabilities.override_tensor_flag or '-ot', value]
    if mtp_enabled:
        mtp_args, _mtp_diagnostics = build_mtp_args(None, runtime_profile, capabilities)
        args += mtp_args
    args.extend(str(item) for item in profile_extra_args)
    return args
