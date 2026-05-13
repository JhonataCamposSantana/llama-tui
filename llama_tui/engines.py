import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .constants import DEFAULT_LLAMA_SERVER, DEFAULT_VLLM_COMMAND
from .runtime_profiles import (
    DEFAULT_BUUN_LLAMA_SERVER,
    DEFAULT_LLAMA_CPP_MTP_SERVER,
    DEFAULT_TQ3_LLAMA_SERVER,
    DEFAULT_TURBOQUANT_LLAMA_SERVER,
    EngineCapabilities,
    default_engine_capabilities,
    detect_engine_capabilities as detect_runtime_engine_capabilities,
)

ENGINE_LLAMA_CPP = 'llama.cpp'
ENGINE_LLAMA_CPP_MTP = 'llama.cpp-mtp'
ENGINE_TURBOQUANT = 'turboquant'
ENGINE_TQ3 = 'tq3'
ENGINE_BUUN = 'buun'
ENGINE_VLLM = 'vllm'


@dataclass(frozen=True)
class EngineDefinition:
    id: str
    display_name: str
    runtime_family: str
    default_binary_env: Optional[str]
    default_paths: List[str]
    path_config_key: Optional[str]
    supports_gguf: bool
    supports_hf_ref: bool
    supports_openai_api: bool
    notes: str = ''


@dataclass(frozen=True)
class EngineInstall:
    id: str
    resolved_command: Optional[str]
    source: str
    exists: bool
    version: Optional[str] = None
    commit: Optional[str] = None


@dataclass(frozen=True)
class EngineHealth:
    id: str
    status: str
    summary: str
    warnings: List[str] = field(default_factory=list)


def get_engine_definitions() -> Dict[str, EngineDefinition]:
    return {
        ENGINE_LLAMA_CPP: EngineDefinition(
            id=ENGINE_LLAMA_CPP,
            display_name='llama.cpp',
            runtime_family='llama.cpp',
            default_binary_env='LLAMA_SERVER',
            default_paths=[
                str(Path.home() / 'llama.cpp' / 'build' / 'bin' / 'llama-server'),
                str(Path.home() / 'llama.cpp' / 'build' / 'bin' / 'server'),
                '/usr/local/bin/llama-server',
                '/usr/bin/llama-server',
            ],
            path_config_key='llama_server',
            supports_gguf=True,
            supports_hf_ref=False,
            supports_openai_api=True,
        ),
        ENGINE_LLAMA_CPP_MTP: EngineDefinition(
            id=ENGINE_LLAMA_CPP_MTP,
            display_name='llama.cpp MTP',
            runtime_family='llama.cpp',
            default_binary_env='LLAMA_CPP_MTP_PATH',
            default_paths=[
                str(Path.home() / 'src' / 'llama.cpp-mtp' / 'build-mtp' / 'bin' / 'llama-server'),
            ],
            path_config_key=None,
            supports_gguf=True,
            supports_hf_ref=False,
            supports_openai_api=True,
            notes='Experimental llama.cpp MTP speculative-decoding branch.',
        ),
        ENGINE_TURBOQUANT: EngineDefinition(
            id=ENGINE_TURBOQUANT,
            display_name='TurboQuant+',
            runtime_family='llama.cpp',
            default_binary_env='TURBOQUANT_LLAMA_SERVER_BIN',
            default_paths=[
                str(Path.home() / 'llama-cpp-turboquant' / 'build' / 'bin' / 'llama-server'),
                'turboquant-llama-server',
            ],
            path_config_key=None,
            supports_gguf=True,
            supports_hf_ref=False,
            supports_openai_api=True,
            notes='Experimental TurboKV-capable llama.cpp fork.',
        ),
        ENGINE_BUUN: EngineDefinition(
            id=ENGINE_BUUN,
            display_name='Buun',
            runtime_family='llama.cpp',
            default_binary_env='BUUN_LLAMA_SERVER_BIN',
            default_paths=['buun-llama-server'],
            path_config_key=None,
            supports_gguf=True,
            supports_hf_ref=False,
            supports_openai_api=True,
            notes='Buun-compatible llama-server command.',
        ),
        ENGINE_TQ3: EngineDefinition(
            id=ENGINE_TQ3,
            display_name='llama.cpp-tq3',
            runtime_family='llama.cpp',
            default_binary_env='TQ3_LLAMA_SERVER_BIN',
            default_paths=[
                str(Path.home() / 'llama.cpp-tq3' / 'build' / 'bin' / 'llama-server'),
                'tq3-llama-server',
            ],
            path_config_key=None,
            supports_gguf=True,
            supports_hf_ref=False,
            supports_openai_api=True,
            notes='Experimental TQ3-native llama.cpp fork.',
        ),
        ENGINE_VLLM: EngineDefinition(
            id=ENGINE_VLLM,
            display_name='vLLM',
            runtime_family='vllm',
            default_binary_env='VLLM_COMMAND',
            default_paths=['vllm'],
            path_config_key='vllm_command',
            supports_gguf=False,
            supports_hf_ref=True,
            supports_openai_api=True,
        ),
    }


def engine_display_name(engine_id: str) -> str:
    engine = normalize_engine_id(engine_id)
    definition = get_engine_definitions().get(engine)
    return definition.display_name if definition else (engine_id or 'Unknown')


def normalize_engine_id(engine_id: str) -> str:
    normalized = (engine_id or ENGINE_LLAMA_CPP).strip().lower()
    if normalized in ('llamacpp', 'llama_cpp', 'llama-cpp'):
        return ENGINE_LLAMA_CPP
    if normalized in ('llama.cpp-mtp', 'llama-cpp-mtp', 'llamacpp-mtp', 'mtp'):
        return ENGINE_LLAMA_CPP_MTP
    if normalized in ('tq', 'turboquant+', 'turboquant'):
        return ENGINE_TURBOQUANT
    if normalized in ('tq3', 'llama.cpp-tq3', 'llama-cpp-tq3', 'llamacpp-tq3'):
        return ENGINE_TQ3
    if normalized == 'buun':
        return ENGINE_BUUN
    if normalized == 'vllm':
        return ENGINE_VLLM
    return normalized or ENGINE_LLAMA_CPP


def command_exists(command: Optional[str]) -> bool:
    parts = shlex.split(command or '')
    if not parts:
        return False
    first = os.path.expanduser(parts[0])
    if '/' in first or first.startswith('.') or first.startswith('~'):
        return Path(first).expanduser().exists()
    return shutil.which(first) is not None


def resolve_llama_cpp_mtp_env(value: str) -> str:
    path = Path(str(value or '')).expanduser()
    if path.name == 'llama-server' or path.is_file():
        return str(path)
    return str(path / 'llama-server')


def resolve_engine_install(config, engine_id: str) -> EngineInstall:
    engine = normalize_engine_id(engine_id)
    definition = get_engine_definitions().get(engine)
    if not definition:
        return EngineInstall(engine, None, 'unknown engine', False)

    env_name = definition.default_binary_env or ''
    env_value = os.environ.get(env_name) if env_name else ''
    runtime_profile = getattr(config, 'runtime_profile', None)
    profile_engine = normalize_engine_id(str(getattr(runtime_profile, 'engine_id', '') or getattr(runtime_profile, 'engine', '') or ''))
    profile_command = str(getattr(runtime_profile, 'server_command', '') or getattr(runtime_profile, 'server_bin', '') or '')
    if profile_engine == engine and profile_command and engine in (ENGINE_BUUN, ENGINE_TURBOQUANT, ENGINE_TQ3, ENGINE_LLAMA_CPP_MTP, ENGINE_LLAMA_CPP):
        command = profile_command
        source = 'runtime_profile'
    elif env_value:
        command = resolve_llama_cpp_mtp_env(env_value) if engine == ENGINE_LLAMA_CPP_MTP else env_value
        source = f'env:{env_name}'
    elif engine == ENGINE_BUUN:
        command = DEFAULT_BUUN_LLAMA_SERVER
        source = 'default'
    elif engine == ENGINE_TURBOQUANT:
        command = DEFAULT_TURBOQUANT_LLAMA_SERVER
        source = 'default'
    elif engine == ENGINE_TQ3:
        command = DEFAULT_TQ3_LLAMA_SERVER
        source = 'default'
    elif engine == ENGINE_LLAMA_CPP_MTP:
        command = DEFAULT_LLAMA_CPP_MTP_SERVER
        source = 'default'
    elif engine == ENGINE_VLLM:
        command = str(getattr(config, 'vllm_command', '') or DEFAULT_VLLM_COMMAND)
        source = 'config:vllm_command' if getattr(config, 'vllm_command', '') else 'default'
    else:
        command = str(getattr(config, 'llama_server', '') or DEFAULT_LLAMA_SERVER)
        source = 'config:llama_server' if getattr(config, 'llama_server', '') else 'default'

    return EngineInstall(
        id=engine,
        resolved_command=command,
        source=source,
        exists=command_exists(command),
    )


def detect_engine_capabilities(install: EngineInstall) -> EngineCapabilities:
    if install.id == ENGINE_VLLM:
        return default_engine_capabilities(ENGINE_LLAMA_CPP)
    if not install.resolved_command:
        return default_engine_capabilities(install.id)
    return detect_runtime_engine_capabilities(install.resolved_command, install.id)


def turboquant_binary_warning(command: str, capabilities: EngineCapabilities) -> str:
    help_text = str(getattr(capabilities, 'help_text', '') or '')
    if not help_text:
        return ''
    if re.search(r'\bturbo(?:2|3|4)(?:_tcq)?\b', help_text.lower()):
        return ''
    path_hint = ''
    parts = shlex.split(command or '')
    if parts:
        low = parts[0].lower()
        if 'llama.cpp' in low and 'llama-cpp-turboquant' not in low:
            path_hint = ' The path looks like a vanilla llama.cpp checkout.'
    return (
        f'TurboQuant+ binary warning: {command} does not advertise turbo cache types in --help.'
        f'{path_hint}'
    )


def tq3_binary_warning(command: str, capabilities: EngineCapabilities) -> str:
    help_text = str(getattr(capabilities, 'help_text', '') or '')
    if not help_text:
        return ''
    if re.search(r'\btq3_0\b', help_text.lower()):
        return ''
    path_hint = ''
    parts = shlex.split(command or '')
    if parts:
        low = parts[0].lower()
        if 'llama.cpp' in low and 'llama.cpp-tq3' not in low and 'llama-cpp-tq3' not in low:
            path_hint = ' The path looks like a vanilla llama.cpp checkout.'
    return (
        f'llama.cpp-tq3 binary warning: {command} does not advertise tq3_0 cache type in --help.'
        f'{path_hint}'
    )


def mtp_binary_warning(command: str, capabilities: EngineCapabilities) -> str:
    help_text = str(getattr(capabilities, 'help_text', '') or '')
    if not help_text:
        return 'llama.cpp MTP binary warning: unable to inspect --help output for MTP flags.'
    if (
        bool(getattr(capabilities, 'supports_spec_type', False))
        and bool(getattr(capabilities, 'supports_mtp', False))
        and bool(getattr(capabilities, 'supports_spec_draft_n_max', False))
    ):
        return ''
    path_hint = ''
    parts = shlex.split(command or '')
    if parts:
        low = parts[0].lower()
        if 'llama.cpp' in low and 'llama.cpp-mtp' not in low and 'llama-cpp-mtp' not in low:
            path_hint = ' The path looks like a stable llama.cpp checkout.'
    return (
        f'llama.cpp MTP binary warning: {command} does not advertise '
        '--spec-type mtp and --spec-draft-n-max in --help.'
        f'{path_hint}'
    )


def get_engine_health(config, engine_id: str) -> EngineHealth:
    install = resolve_engine_install(config, engine_id)
    if not install.exists:
        if install.id == ENGINE_LLAMA_CPP_MTP:
            return EngineHealth(
                id=install.id,
                status='BINARY_NOT_FOUND',
                summary='llama.cpp MTP binary missing',
                warnings=[f'command not found: {install.resolved_command or "-"}'],
            )
        return EngineHealth(
            id=install.id,
            status='FAIL',
            summary=f'{engine_display_name(install.id)} binary missing',
            warnings=[f'command not found: {install.resolved_command or "-"}'],
        )
    warnings: List[str] = []
    capabilities = detect_engine_capabilities(install)
    if install.id != ENGINE_VLLM and not str(getattr(capabilities, 'help_text', '') or '').strip():
        warnings.append('engine capabilities unknown; using built-in defaults')
    if install.id == ENGINE_TURBOQUANT:
        warning = turboquant_binary_warning(
            str(install.resolved_command or ''),
            capabilities,
        )
        if warning:
            warnings.append(warning)
    if install.id == ENGINE_TQ3:
        warning = tq3_binary_warning(str(install.resolved_command or ''), capabilities)
        if warning:
            warnings.append(warning)
    if install.id == ENGINE_LLAMA_CPP_MTP:
        warning = mtp_binary_warning(str(install.resolved_command or ''), capabilities)
        if warning:
            status = 'BUILD_REQUIRED' if not str(getattr(capabilities, 'help_text', '') or '').strip() else 'MTP_FLAGS_NOT_FOUND'
            return EngineHealth(install.id, status, warning, [warning])
        return EngineHealth(install.id, 'READY', f'{engine_display_name(install.id)} ready (Experimental)', [])
    if warnings:
        return EngineHealth(install.id, 'WARN', warnings[0], warnings)
    return EngineHealth(install.id, 'OK', f'{engine_display_name(install.id)} ready', [])
