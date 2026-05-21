import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import DEFAULT_LLAMA_SERVER, DEFAULT_VLLM_COMMAND
from .runtime_profiles import (
    DEFAULT_BUUN_LLAMA_SERVER,
    DEFAULT_TQ3_LLAMA_SERVER,
    DEFAULT_TURBOQUANT_LLAMA_SERVER,
    EngineCapabilities,
    command_file_status,
    default_engine_capabilities,
    detect_engine_capabilities as detect_runtime_engine_capabilities,
    llama_cpp_mtp_default_candidates,
    mtp_spec_type_value,
    resolve_llama_cpp_mtp_binary,
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
    executable: bool = False
    resolved_path: str = ''
    checked_paths: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EngineHealth:
    id: str
    status: str
    summary: str
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeEngineContext:
    """Resolved view of which binary a launch/benchmark will actually use.

    MTP is modelled as a capability of the resolved binary rather than a
    dedicated engine, so callers should consult ``capabilities`` /
    ``selected_mtp_spec_type`` instead of comparing ``engine_id`` against
    ``llama.cpp-mtp``.
    """

    engine_id: str
    install: EngineInstall
    command: str
    source: str
    exists: bool
    executable: bool
    capabilities: EngineCapabilities
    selected_mtp_spec_type: str
    supports_mtp: bool
    diagnostics: Tuple[str, ...] = ()


def get_engine_definitions() -> Dict[str, EngineDefinition]:
    return {
        ENGINE_LLAMA_CPP: EngineDefinition(
            id=ENGINE_LLAMA_CPP,
            display_name='llama.cpp',
            runtime_family='llama.cpp',
            default_binary_env='LLAMA_SERVER',
            # Audit finding #7: MTP is now upstream (--spec-type mtp /
            # draft-mtp). The legacy LLAMA_CPP_MTP_PATH env + sibling
            # paths fold into llama.cpp's discovery so users who built
            # the experimental fork still find their binary; capability
            # detection then decides whether MTP is offered.
            default_paths=[
                str(Path.home() / 'llama.cpp' / 'build' / 'bin' / 'llama-server'),
                str(Path.home() / 'llama.cpp' / 'build' / 'bin' / 'server'),
                '/usr/local/bin/llama-server',
                '/usr/bin/llama-server',
                *llama_cpp_mtp_default_candidates(),
                'llama-server-mtp',
            ],
            path_config_key='llama_server',
            supports_gguf=True,
            supports_hf_ref=False,
            supports_openai_api=True,
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
        # Audit finding #7: legacy MTP-engine alias now collapses to
        # plain llama.cpp; MTP is detected via EngineCapabilities.
        return ENGINE_LLAMA_CPP
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
    return command_file_status(command or '').exists


def resolve_llama_cpp_mtp_env(value: str) -> str:
    return resolve_llama_cpp_mtp_binary(env_value=value).command


def resolve_engine_install(config, engine_id: str) -> EngineInstall:
    engine = normalize_engine_id(engine_id)
    definition = get_engine_definitions().get(engine)
    if not definition:
        return EngineInstall(engine, None, 'unknown engine', False)

    env_name = definition.default_binary_env or ''
    env_value = os.environ.get(env_name) if env_name else ''
    # Audit #7: legacy LLAMA_CPP_MTP_PATH is honoured as a fallback for
    # the llama.cpp engine when LLAMA_SERVER is unset, so users who
    # built the MTP fork and pointed LLAMA_CPP_MTP_PATH at its bin
    # directory still get auto-resolved.
    if engine == ENGINE_LLAMA_CPP and not env_value:
        mtp_env_value = os.environ.get('LLAMA_CPP_MTP_PATH', '')
        if mtp_env_value:
            mtp = resolve_llama_cpp_mtp_binary(env_value=mtp_env_value)
            return EngineInstall(
                id=engine,
                resolved_command=mtp.command,
                source=mtp.source,
                exists=mtp.exists,
                executable=mtp.executable,
                resolved_path=mtp.resolved_path,
                checked_paths=list(mtp.checked_paths),
            )
    runtime_profile = getattr(config, 'runtime_profile', None)
    profile_engine = normalize_engine_id(str(getattr(runtime_profile, 'engine_id', '') or getattr(runtime_profile, 'engine', '') or ''))
    profile_command = str(getattr(runtime_profile, 'server_command', '') or getattr(runtime_profile, 'server_bin', '') or '')
    if profile_engine == engine and profile_command and engine in (ENGINE_BUUN, ENGINE_TURBOQUANT, ENGINE_TQ3, ENGINE_LLAMA_CPP):
        command = profile_command
        source = 'runtime_profile'
    elif env_value:
        command = env_value
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
    elif engine == ENGINE_VLLM:
        command = str(getattr(config, 'vllm_command', '') or DEFAULT_VLLM_COMMAND)
        source = 'config:vllm_command' if getattr(config, 'vllm_command', '') else 'default'
    else:
        command = str(getattr(config, 'llama_server', '') or DEFAULT_LLAMA_SERVER)
        source = 'config:llama_server' if getattr(config, 'llama_server', '') else 'default'

    status = command_file_status(command)
    return EngineInstall(
        id=engine,
        resolved_command=command,
        source=source,
        exists=status.exists,
        executable=status.executable,
        resolved_path=status.resolved_path,
        checked_paths=[command] if command else [],
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


def _yes_no(value: bool) -> str:
    return 'yes' if bool(value) else 'no'


def mtp_binary_warning(
    command: str,
    capabilities: EngineCapabilities,
    source: str = '',
    exists: Optional[bool] = None,
    executable: Optional[bool] = None,
) -> str:
    help_text = str(getattr(capabilities, 'help_text', '') or '')
    status = command_file_status(command or '')
    exists_value = status.exists if exists is None else bool(exists)
    executable_value = status.executable if executable is None else bool(executable)
    supports_spec_type = bool(getattr(capabilities, 'supports_spec_type', False))
    supports_mtp = bool(getattr(capabilities, 'supports_mtp', False))
    mtp_spec_type = mtp_spec_type_value(capabilities)
    supports_draft = bool(getattr(capabilities, 'supports_spec_draft_n_max', False))
    detail = (
        f'resolved={command or "-"}; source={source or "unknown"}; '
        f'exists={_yes_no(exists_value)}; executable={_yes_no(executable_value)}; '
        f'spec_type={_yes_no(supports_spec_type)}; mtp_value={mtp_spec_type or "none"}; mtp={_yes_no(supports_mtp)}; '
        f'spec_draft_n_max={_yes_no(supports_draft)}'
    )
    if not help_text:
        return f'llama.cpp MTP binary warning: unable to inspect --help output for MTP flags ({detail}).'
    if (
        supports_spec_type
        and supports_mtp
        and mtp_spec_type
        and supports_draft
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
        '--spec-type mtp/draft-mtp and --spec-draft-n-max in --help '
        f'({detail}).'
        f'{path_hint}'
    )


def get_engine_health(config, engine_id: str) -> EngineHealth:
    install = resolve_engine_install(config, engine_id)
    if not install.exists:
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
        return EngineHealth(
            install.id,
            'READY',
            (
                f'{engine_display_name(install.id)} ready (Experimental; '
                f'resolved={install.resolved_command}; source={install.source}; '
                f'exists=yes; executable=yes; spec_type=yes; mtp_value={mtp_spec_type_value(capabilities)}; '
                f'mtp=yes; spec_draft_n_max=yes)'
            ),
            [],
        )
    if warnings:
        return EngineHealth(install.id, 'WARN', warnings[0], warnings)
    return EngineHealth(install.id, 'OK', f'{engine_display_name(install.id)} ready', [])


def _engine_capabilities_for_install(app, engine_id: str, install: EngineInstall) -> EngineCapabilities:
    command = str(getattr(install, 'resolved_command', '') or '')
    getter = getattr(app, 'engine_capabilities', None)
    if callable(getter):
        try:
            return getter(server_bin=command, engine_id=engine_id)
        except TypeError:
            try:
                return getter()
            except Exception:
                pass
        except Exception:
            pass
    return detect_engine_capabilities(install)


def resolve_runtime_engine_context(app, model=None, runtime_profile=None) -> RuntimeEngineContext:
    """Resolve the engine, binary and capabilities for a launch/benchmark.

    Resolution order for the engine id:
      1. ``runtime_profile.engine_id`` when a runtime profile is supplied;
      2. ``app.active_engine_key_for_model(model)`` when a model is supplied;
      3. the app's global runtime profile as a last resort.

    Capabilities always come from the actually-resolved binary, never from a
    different engine's global state.
    """
    diagnostics: List[str] = []
    engine_id = ''
    if runtime_profile is not None:
        raw = str(
            getattr(runtime_profile, 'engine_id', '')
            or getattr(runtime_profile, 'engine', '')
            or ''
        )
        if raw:
            engine_id = normalize_engine_id(raw)
            diagnostics.append(f'engine from runtime_profile: {engine_id}')
    if not engine_id and model is not None and hasattr(app, 'active_engine_key_for_model'):
        try:
            resolved = str(app.active_engine_key_for_model(model) or '')
        except Exception as exc:  # pragma: no cover - defensive
            resolved = ''
            diagnostics.append(f'active_engine_key_for_model failed: {exc}')
        if resolved:
            engine_id = normalize_engine_id(resolved)
            diagnostics.append(f'engine from active_engine_key_for_model: {engine_id}')
    if not engine_id:
        global_profile = getattr(app, 'runtime_profile', None)
        engine_id = normalize_engine_id(
            str(
                getattr(global_profile, 'engine_id', '')
                or getattr(global_profile, 'engine', '')
                or ENGINE_LLAMA_CPP
            )
        )
        diagnostics.append(f'engine fallback to app.runtime_profile: {engine_id}')

    install = resolve_engine_install(app, engine_id)
    capabilities = _engine_capabilities_for_install(app, engine_id, install)
    spec_type = mtp_spec_type_value(capabilities)
    supports_mtp = bool(
        getattr(capabilities, 'supports_spec_type', False)
        and spec_type
        and getattr(capabilities, 'supports_spec_draft_n_max', False)
    )
    if spec_type:
        diagnostics.append(f'selected mtp spec type: {spec_type}')
    elif bool(getattr(capabilities, 'supports_spec_type', False)):
        diagnostics.append('binary advertises --spec-type but no mtp/draft-mtp value')
    else:
        diagnostics.append('binary does not advertise --spec-type')

    return RuntimeEngineContext(
        engine_id=engine_id,
        install=install,
        command=str(getattr(install, 'resolved_command', '') or ''),
        source=str(getattr(install, 'source', '') or ''),
        exists=bool(getattr(install, 'exists', False)),
        executable=bool(getattr(install, 'executable', False)),
        capabilities=capabilities,
        selected_mtp_spec_type=spec_type,
        supports_mtp=supports_mtp,
        diagnostics=tuple(diagnostics),
    )
