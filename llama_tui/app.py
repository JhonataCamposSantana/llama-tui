import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error, request

from .constants import (
    APP_NAME,
    CACHE_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DEFAULT_HF_CACHE,
    DEFAULT_LM_STUDIO_MODEL_ROOTS,
    DEFAULT_LLMFIT_CACHE,
    DEFAULT_LLM_MODELS_CACHE,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_VLLM_COMMAND,
)
from .discovery import (
    detected_model_from_path,
    display_runtime,
    is_real_model_file,
    looks_like_model_reference,
)
from .control import CancelToken, check_cancelled, sleep_with_cancel
from .gguf import estimate_kv_bytes_per_token, extra_arg_value, read_gguf_metadata
from .gguf import (
    TURBOQUANT_STATUSES,
    apply_architecture_info,
    apply_turboquant_info,
    detect_architecture_info,
    detect_turboquant_info,
    turboquant_detail,
)
from .hardware import HardwareProfile, benchmark_current_hardware, read_meminfo_bytes
from .models import ContinueSettings, HermesSettings, ModelConfig, OpencodeSettings, UiSettings
from .optimize import choose_gpu_layers_for_profile, effective_gpu_reserve_percent, estimate_safe_context_for_profile
from .runtime_profiles import (
    EngineCapabilities,
    EngineProfile,
    RuntimeProfile,
    detect_engine_capabilities,
    make_runtime_profile,
    runtime_profile_extra_args,
)
from .textutil import compact_message, important_log_excerpt


TERMINAL_LAUNCHER_ORDER = (
    'xdg-terminal-exec',
    'ptyxis',
    'gnome-console',
    'ghostty',
    'konsole',
    'gnome-terminal',
    'kgx',
    'kitty',
    'alacritty',
    'wezterm',
    'foot',
    'tilix',
    'terminator',
    'xfce4-terminal',
    'qterminal',
    'lxterminal',
    'mate-terminal',
    'deepin-terminal',
    'terminology',
    'xterm',
)

FLATPAK_TERMINAL_APP_IDS = (
    'com.mitchellh.ghostty',
    'org.gnome.Console',
    'org.gnome.Terminal',
    'org.wezfurlong.wezterm',
    'com.raggesilver.BlackBox',
)

HOST_TERMINAL_BRIDGES = (
    'host-spawn',
    'distrobox-host-exec',
)

CONTINUE_MANAGED_BEGIN = '  # BEGIN llama-tui managed models'
CONTINUE_MANAGED_END = '  # END llama-tui managed models'
CONTINUE_MERGE_MODES = ('preserve_sections', 'managed_file')
VERIFICATION_STATUSES = ('unknown', 'running', 'passed', 'warning', 'failed', 'needs_benchmark')
ENGINE_BENCHMARK_FIELDS = (
    'last_benchmark_tokens_per_sec',
    'last_benchmark_seconds',
    'last_benchmark_profile',
    'last_benchmark_results',
    'measured_profiles',
    'benchmark_runs',
    'benchmark_fingerprint',
    'default_benchmark_status',
    'default_benchmark_at',
)


def terminal_command_for_launcher(launcher: str, title: str, cwd: Path, shell_cmd: str) -> List[str]:
    if launcher.startswith('flatpak:'):
        app_id = launcher.split(':', 1)[1]
        cwd_text = str(cwd)
        if app_id == 'com.mitchellh.ghostty':
            return ['flatpak', 'run', app_id, '--title', title, '--working-directory', cwd_text, '-e', 'bash', '-lc', shell_cmd]
        if app_id in ('org.gnome.Console', 'org.gnome.Terminal'):
            return ['flatpak', 'run', app_id, '--working-directory', cwd_text, '--', 'bash', '-lc', shell_cmd]
        if app_id == 'org.wezfurlong.wezterm':
            return ['flatpak', 'run', app_id, 'start', '--cwd', cwd_text, '--', 'bash', '-lc', shell_cmd]
        if app_id == 'com.raggesilver.BlackBox':
            return ['flatpak', 'run', app_id, '--working-directory', cwd_text, '--command', shlex.join(['bash', '-lc', shell_cmd])]
        return ['flatpak', 'run', app_id, 'bash', '-lc', shell_cmd]
    launcher_name = Path(launcher).name
    cwd_text = str(cwd)
    cd_shell_cmd = f'cd {shlex.quote(cwd_text)} && {shell_cmd}'
    if launcher_name == 'xdg-terminal-exec':
        return [launcher, 'bash', '-lc', cd_shell_cmd]
    if launcher_name in ('ptyxis', 'gnome-console'):
        return [launcher, '--working-directory', cwd_text, '--title', title, '--', 'bash', '-lc', shell_cmd]
    if launcher_name == 'konsole':
        return [launcher, '--workdir', cwd_text, '-p', f'tabtitle={title}', '-e', 'bash', '-lc', shell_cmd]
    if launcher_name in ('gnome-terminal', 'mate-terminal'):
        return [launcher, '--title', title, '--working-directory', cwd_text, '--', 'bash', '-lc', shell_cmd]
    if launcher_name == 'ghostty':
        return [launcher, '--title', title, '--working-directory', cwd_text, '-e', 'bash', '-lc', shell_cmd]
    if launcher_name == 'kgx':
        return [launcher, '--title', title, '--working-directory', cwd_text, '--', 'bash', '-lc', shell_cmd]
    if launcher_name == 'kitty':
        return [launcher, '--title', title, '--directory', cwd_text, 'bash', '-lc', shell_cmd]
    if launcher_name == 'alacritty':
        return [launcher, '--title', title, '--working-directory', cwd_text, '-e', 'bash', '-lc', shell_cmd]
    if launcher_name == 'wezterm':
        return [launcher, 'start', '--cwd', cwd_text, '--', 'bash', '-lc', shell_cmd]
    if launcher_name == 'foot':
        return [launcher, '--title', title, '--working-directory', cwd_text, 'bash', '-lc', shell_cmd]
    if launcher_name == 'tilix':
        return [launcher, '--working-directory', cwd_text, '--title', title, '-e', 'bash', '-lc', shell_cmd]
    if launcher_name == 'terminator':
        return [launcher, '--working-directory', cwd_text, '-T', title, '-x', 'bash', '-lc', shell_cmd]
    if launcher_name == 'xfce4-terminal':
        return [
            launcher,
            '--title',
            title,
            '--working-directory',
            cwd_text,
            '--command',
            shlex.join(['bash', '-lc', shell_cmd]),
        ]
    if launcher_name == 'qterminal':
        return [launcher, '--workdir', cwd_text, '-T', title, '-e', 'bash', '-lc', shell_cmd]
    if launcher_name == 'lxterminal':
        return [launcher, '--title', title, '--working-directory', cwd_text, '-e', shlex.join(['bash', '-lc', shell_cmd])]
    if launcher_name == 'deepin-terminal':
        return [launcher, '--workdir', cwd_text, '-e', 'bash', '-lc', shell_cmd]
    if launcher_name == 'terminology':
        return [launcher, '-T', title, '-d', cwd_text, '-e', 'bash', '-lc', shell_cmd]
    if launcher_name == 'xterm':
        return [launcher, '-T', title, '-e', 'bash', '-lc', cd_shell_cmd]
    return [launcher, '-e', 'bash', '-lc', shell_cmd]


def render_terminal_template(template: str, title: str, cwd: Path, shell_cmd: str) -> List[str]:
    rendered = template.format(
        title=shlex.quote(title),
        cwd=shlex.quote(str(cwd)),
        cmd=shlex.quote(shell_cmd),
    )
    return shlex.split(rendered)


def shell_env_prefix(env: Dict[str, str]) -> str:
    clean = {key: str(value) for key, value in env.items() if str(value) != ''}
    return (' '.join(f'{key}={shlex.quote(value)}' for key, value in clean.items()) + ' ') if clean else ''


def yaml_quote(value: object) -> str:
    return json.dumps(str(value))


def yaml_list(values: List[str]) -> str:
    return '[' + ', '.join(yaml_quote(value) for value in values) + ']'


def dataclass_payload(cls, raw: Dict[str, object]) -> Dict[str, object]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in dict(raw).items() if key in allowed}


def context_per_slot(model: ModelConfig) -> int:
    ctx = max(0, int(getattr(model, 'ctx', 0) or 0))
    parallel = max(1, int(getattr(model, 'parallel', 1) or 1))
    return ctx // parallel


def terminal_launcher_label(launcher: str) -> str:
    if launcher.startswith('flatpak:'):
        return launcher
    return Path(launcher).name


def host_bridge_command(bridge: str, command: List[str]) -> List[str]:
    bridge_name = Path(bridge).name
    if bridge_name == 'host-spawn':
        return [bridge, '-no-pty', *command]
    return [bridge, *command]


def container_environment_detected() -> bool:
    if os.environ.get('container') or os.environ.get('DISTROBOX_ENTER_PATH'):
        return True
    return Path('/run/.containerenv').exists() or Path('/.dockerenv').exists()


def current_container_name() -> str:
    for key in ('DISTROBOX_CONTAINER_NAME', 'CONTAINER_NAME'):
        value = (os.environ.get(key) or '').strip()
        if value:
            return value
    containerenv = Path('/run/.containerenv')
    if containerenv.exists():
        try:
            for raw_line in containerenv.read_text(errors='replace').splitlines():
                key, sep, value = raw_line.partition('=')
                if sep and key.strip() == 'name':
                    return value.strip().strip('"').strip("'")
        except Exception:
            return ''
    return ''


def desktop_terminal_guess() -> str:
    desktop_text = ' '.join(
        value.lower()
        for value in (
            os.environ.get('XDG_CURRENT_DESKTOP', ''),
            os.environ.get('DESKTOP_SESSION', ''),
        )
        if value
    )
    if 'kde' in desktop_text or 'plasma' in desktop_text:
        return 'konsole'
    if 'gnome' in desktop_text:
        return 'ptyxis'
    if 'xfce' in desktop_text:
        return 'xfce4-terminal'
    return ''


class AppConfig:
    def __init__(self, config_path: Path, runtime_profile: Optional[EngineProfile] = None):
        self.config_path = config_path
        self.llama_server = os.environ.get('LLAMA_SERVER', DEFAULT_LLAMA_SERVER)
        self.vllm_command = DEFAULT_VLLM_COMMAND
        self.runtime_profile = runtime_profile or make_runtime_profile('llama.cpp', self.llama_server)
        self.hf_cache_root = str(DEFAULT_HF_CACHE)
        self.llmfit_cache_root = str(DEFAULT_LLMFIT_CACHE)
        self.llm_models_cache_root = str(DEFAULT_LLM_MODELS_CACHE)
        self.lm_studio_model_roots = ', '.join(str(path) for path in DEFAULT_LM_STUDIO_MODEL_ROOTS)
        self.opencode = OpencodeSettings(
            path='',
            backup_dir=str(CONFIG_DIR / 'backups'),
        )
        self.continue_settings = ContinueSettings(
            path='',
            backup_dir=str(CONFIG_DIR / 'backups'),
        )
        self.hermes = HermesSettings(
            home_root=str(CACHE_DIR / 'hermes'),
        )
        self.ui = UiSettings()
        self.models: List[ModelConfig] = []
        self.load_warnings: List[str] = []
        self._hardware_profile: Optional[HardwareProfile] = None
        self._hardware_profile_at = 0.0
        self._owned_pids: set[int] = set()
        self._runtime_check_cache: Dict[str, Tuple[bool, str]] = {}
        self._runtime_version_cache: Dict[str, str] = {}
        self._engine_capability_cache: Dict[str, EngineCapabilities] = {}
        self._shutdown_cleanup_done = False
        self._benchmark_views_active = False
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.load()
        chosen_profile = runtime_profile or self.runtime_profile
        profile_server = self.llama_server
        if (
            runtime_profile is not None
            and chosen_profile.server_command
            and chosen_profile.server_command != DEFAULT_LLAMA_SERVER
        ):
            profile_server = chosen_profile.server_command
        self.runtime_profile = make_runtime_profile(
            chosen_profile.engine,
            profile_server,
            ctx_override=chosen_profile.context_override,
            kv_mode=chosen_profile.kv_mode,
            kv_key_mode=chosen_profile.kv_key_mode,
            kv_value_mode=chosen_profile.kv_value_mode,
        )
        self._activate_engine_benchmark_views()

    def load(self):
        if not self.config_path.exists():
            self.save()
            return
        self.load_warnings = []
        try:
            data = json.loads(self.config_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            archived = self._archive_broken_config_file()
            detail = f'Config recovery: {compact_message(str(exc))}'
            if archived:
                detail += f' | original saved to {archived}'
            detail += ' | defaults were restored; review settings and re-add any missing manual edits.'
            self.load_warnings.append(detail)
            self.save()
            return
        if not isinstance(data, dict):
            self.load_warnings.append('Config recovery: top-level config must be a JSON object; defaults were restored.')
            self.save()
            return
        self.llama_server = data.get('llama_server', self.llama_server)
        self.vllm_command = data.get('vllm_command', self.vllm_command)
        self.hf_cache_root = data.get('hf_cache_root', self.hf_cache_root)
        self.llmfit_cache_root = data.get('llmfit_cache_root', self.llmfit_cache_root)
        self.llm_models_cache_root = data.get('llm_models_cache_root', self.llm_models_cache_root)
        self.lm_studio_model_roots = data.get('lm_studio_model_roots', self.lm_studio_model_roots)
        self.opencode = self._load_settings(OpencodeSettings, data.get('opencode', {}), self.opencode, 'opencode')
        self.continue_settings = self._load_settings(ContinueSettings, data.get('continue', {}), self.continue_settings, 'continue')
        if not self.continue_settings.backup_dir:
            self.continue_settings.backup_dir = str(CONFIG_DIR / 'backups')
        if getattr(self.continue_settings, 'merge_mode', '') not in CONTINUE_MERGE_MODES:
            self.continue_settings.merge_mode = 'preserve_sections'
        self.hermes = self._load_settings(HermesSettings, data.get('hermes', {}), self.hermes, 'hermes')
        if not self.hermes.home_root:
            self.hermes.home_root = str(CACHE_DIR / 'hermes')
        self.ui = self._load_settings(UiSettings, data.get('ui', {}), self.ui, 'ui')
        loaded_models: List[ModelConfig] = []
        raw_models = data.get('models', [])
        if raw_models and not isinstance(raw_models, list):
            self.load_warnings.append('Config recovery: models must be a list; model entries were ignored.')
            raw_models = []
        for index, item in enumerate(raw_models):
            try:
                loaded_models.append(self._load_model(item, index))
            except Exception as exc:
                self.load_warnings.append(
                    f'Config recovery: skipped model row {index + 1}: {compact_message(str(exc))}'
                )
                continue
        self.models = loaded_models
        roots_changed = False
        for m in self.models:
            if not getattr(m, 'sort_rank', 0):
                m.sort_rank = self.next_sort_rank()
                roots_changed = True
            if self.enrich_model_architecture(m):
                roots_changed = True
            if self.enrich_model_turboquant(m):
                roots_changed = True
            inferred = self.infer_model_source(m)
            if m.source != inferred:
                m.source = inferred
                roots_changed = True
            if not getattr(m, 'benchmark_fingerprint', '') and float(getattr(m, 'last_benchmark_tokens_per_sec', 0.0) or 0.0) > 0.0:
                m.benchmark_fingerprint = self.model_fingerprint(m)
                if not getattr(m, 'default_benchmark_status', ''):
                    m.default_benchmark_status = 'done'
                if not getattr(m, 'default_benchmark_at', ''):
                    m.default_benchmark_at = datetime.now().isoformat(timespec='seconds')
                roots_changed = True
        if self._normalize_model_ranks():
            roots_changed = True
        if len(loaded_models) != len(raw_models) or roots_changed:
            self.save()

    def save(self):
        if self._benchmark_views_active:
            self._persist_engine_benchmark_views()
        self._normalize_model_ranks()
        data = {
            'llama_server': self.llama_server,
            'vllm_command': self.vllm_command,
            'hf_cache_root': self.hf_cache_root,
            'llmfit_cache_root': self.llmfit_cache_root,
            'llm_models_cache_root': self.llm_models_cache_root,
            'lm_studio_model_roots': self.lm_studio_model_roots,
            'opencode': asdict(self.opencode),
            'continue': asdict(self.continue_settings),
            'hermes': asdict(self.hermes),
            'ui': asdict(self.ui),
            'models': [asdict(m) for m in self.models],
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

    def _default_engine_benchmark_payload(self) -> Dict[str, object]:
        return {
            'last_benchmark_tokens_per_sec': 0.0,
            'last_benchmark_seconds': 0.0,
            'last_benchmark_profile': '',
            'last_benchmark_results': [],
            'measured_profiles': {},
            'benchmark_runs': [],
            'benchmark_fingerprint': '',
            'default_benchmark_status': '',
            'default_benchmark_at': '',
        }

    def active_engine_key_for_model(self, model: ModelConfig) -> str:
        runtime = (getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp').strip().lower()
        if runtime == 'vllm':
            return 'vllm'
        if self.runtime_profile.engine == 'buun':
            return 'buun'
        return 'llama.cpp'

    def _benchmark_payload_for_model(self, model: ModelConfig) -> Dict[str, object]:
        return {
            field: self._copy_benchmark_value(getattr(model, field))
            for field in ENGINE_BENCHMARK_FIELDS
        }

    def _copy_benchmark_value(self, value: object) -> object:
        if isinstance(value, list):
            return [dict(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):
            return {
                str(key): dict(item) if isinstance(item, dict) else item
                for key, item in value.items()
            }
        return value

    def _apply_benchmark_payload(self, model: ModelConfig, payload: Dict[str, object]):
        defaults = self._default_engine_benchmark_payload()
        for field in ENGINE_BENCHMARK_FIELDS:
            value = payload.get(field, defaults[field])
            if isinstance(defaults[field], list):
                value = list(value) if isinstance(value, list) else []
            elif isinstance(defaults[field], dict):
                value = dict(value) if isinstance(value, dict) else {}
            setattr(model, field, value)

    def _persist_engine_benchmark_views(self):
        for model in self.models:
            store = dict(getattr(model, 'engine_benchmark_store', {}) or {})
            store[self.active_engine_key_for_model(model)] = self._benchmark_payload_for_model(model)
            model.engine_benchmark_store = store

    def _canonical_legacy_engine_key(self, model: ModelConfig) -> str:
        runtime = (getattr(model, 'runtime', 'llama.cpp') or '').strip().lower()
        return 'vllm' if runtime == 'vllm' else 'llama.cpp'

    def _has_benchmark_payload(self, model: ModelConfig) -> bool:
        if float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0) > 0.0:
            return True
        if (getattr(model, 'default_benchmark_status', '') or '').strip():
            return True
        if getattr(model, 'last_benchmark_results', None):
            return True
        if getattr(model, 'measured_profiles', None):
            return True
        if getattr(model, 'benchmark_runs', None):
            return True
        if (getattr(model, 'benchmark_fingerprint', '') or '').strip():
            return True
        return False

    def _activate_engine_benchmark_views(self):
        changed = False
        for model in self.models:
            store = dict(getattr(model, 'engine_benchmark_store', {}) or {})
            # One-time legacy migration: seed historical values under canonical runtime key.
            runtime_key = self._canonical_legacy_engine_key(model)
            if runtime_key not in store and self._has_benchmark_payload(model):
                store[runtime_key] = self._benchmark_payload_for_model(model)
                changed = True
            model.engine_benchmark_store = store
            active_key = self.active_engine_key_for_model(model)
            self._apply_benchmark_payload(model, store.get(active_key, {}))
        self._benchmark_views_active = True
        if changed:
            self.save()

    def _archive_broken_config_file(self) -> Optional[Path]:
        if not self.config_path.exists():
            return None
        backup_dir = CONFIG_DIR / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_path = backup_dir / f'{self.config_path.stem}.broken.{stamp}{self.config_path.suffix}'
        try:
            shutil.copy2(self.config_path, backup_path)
        except OSError:
            return None
        return backup_path

    def _load_settings(self, cls, raw: object, current, label: str):
        payload = dataclass_payload(cls, raw) if isinstance(raw, dict) else {}
        if raw not in ({}, None) and not isinstance(raw, dict):
            self.load_warnings.append(f'Config recovery: {label} settings were invalid and were reset to defaults.')
            return current
        try:
            return cls(**payload)
        except Exception as exc:
            self.load_warnings.append(
                f'Config recovery: {label} settings were invalid ({compact_message(str(exc))}); defaults were kept.'
            )
            return current

    def _load_model(self, raw: object, index: int) -> ModelConfig:
        if not isinstance(raw, dict):
            raise ValueError('entry is not an object')
        payload = dict(raw)
        required = ('id', 'name', 'path', 'alias', 'port')
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(f'missing fields: {", ".join(missing)}')
        payload['port'] = int(payload.get('port', 0) or 0)
        payload['ctx'] = int(payload.get('ctx', 8192) or 8192)
        payload['ctx_min'] = int(payload.get('ctx_min', 2048) or 2048)
        payload['ctx_max'] = int(payload.get('ctx_max', 131072) or 131072)
        payload['threads'] = int(payload.get('threads', 6) or 6)
        payload['ngl'] = int(payload.get('ngl', 999) or 999)
        payload['parallel'] = int(payload.get('parallel', 1) or 1)
        payload['memory_reserve_percent'] = int(payload.get('memory_reserve_percent', 25) or 25)
        payload['cache_ram'] = int(payload.get('cache_ram', 0) or 0)
        payload['output'] = int(payload.get('output', 4096) or 4096)
        payload['temp'] = float(payload.get('temp', 0.7) or 0.7)
        payload['architecture'] = str(payload.get('architecture', '') or '')
        payload['architecture_type'] = str(payload.get('architecture_type', 'unknown') or 'unknown').strip().lower()
        if payload['architecture_type'] not in ('dense', 'moe', 'unknown'):
            payload['architecture_type'] = 'unknown'
        payload['model_family'] = str(payload.get('model_family', '') or '')
        for key in (
            'expert_count',
            'expert_used_count',
            'expert_shared_count',
            'expert_group_count',
            'expert_group_used_count',
            'moe_every_n_layers',
            'leading_dense_block_count',
        ):
            payload[key] = int(payload.get(key, 0) or 0)
        payload['active_expert_ratio'] = float(payload.get('active_expert_ratio', 0.0) or 0.0)
        payload['classification_confidence'] = float(payload.get('classification_confidence', 0.0) or 0.0)
        payload['classification_source'] = str(payload.get('classification_source', '') or '')
        payload['classification_reason'] = str(payload.get('classification_reason', '') or '')
        payload['turboquant_status'] = str(payload.get('turboquant_status', 'unknown') or 'unknown').strip().lower()
        if payload['turboquant_status'] not in TURBOQUANT_STATUSES:
            payload['turboquant_status'] = 'unknown'
        for key in ('turboquant_head_dim', 'turboquant_key_dim', 'turboquant_value_dim'):
            payload[key] = int(payload.get(key, 0) or 0)
        payload['turboquant_source'] = str(payload.get('turboquant_source', '') or '')
        payload['turboquant_reason'] = str(payload.get('turboquant_reason', '') or '')
        payload['favorite'] = bool(payload.get('favorite', False))
        payload['last_used_at'] = str(payload.get('last_used_at', '') or '')
        payload['sort_rank'] = int(payload.get('sort_rank', index + 1) or (index + 1))
        extra_args = payload.get('extra_args', [])
        payload['extra_args'] = [str(item) for item in extra_args] if isinstance(extra_args, list) else []
        tags = payload.get('tags', [])
        payload['tags'] = [str(item).strip() for item in tags if str(item).strip()] if isinstance(tags, list) else []
        payload['verification_status'] = str(payload.get('verification_status', 'unknown') or 'unknown')
        if payload['verification_status'] not in VERIFICATION_STATUSES:
            payload['verification_status'] = 'unknown'
        payload['verification_at'] = str(payload.get('verification_at', '') or '')
        payload['verification_fingerprint'] = str(payload.get('verification_fingerprint', '') or '')
        payload['verification_summary'] = str(payload.get('verification_summary', '') or '')
        verification_results = payload.get('verification_results', {})
        payload['verification_results'] = verification_results if isinstance(verification_results, dict) else {}
        return ModelConfig(**dataclass_payload(ModelConfig, payload))

    def enrich_model_architecture(self, model: ModelConfig) -> bool:
        if (getattr(model, 'classification_source', '') or '') == 'manual':
            return False
        current_type = (getattr(model, 'architecture_type', '') or 'unknown').strip().lower()
        current_conf = float(getattr(model, 'classification_confidence', 0.0) or 0.0)
        try:
            detected = detect_architecture_info(model)
        except Exception:
            return False
        should_update = (
            current_type not in ('dense', 'moe')
            or not getattr(model, 'classification_source', '')
            or float(detected.confidence or 0.0) > current_conf
            or (
                current_type == 'moe'
                and int(getattr(model, 'expert_count', 0) or 0) <= 0
                and int(detected.expert_count or 0) > 0
            )
        )
        if not should_update:
            return False
        before = asdict(model)
        apply_architecture_info(model, detected)
        return asdict(model) != before

    def enrich_model_turboquant(self, model: ModelConfig) -> bool:
        if (getattr(model, 'turboquant_source', '') or '') == 'manual':
            return False
        try:
            detected = detect_turboquant_info(model)
        except Exception:
            return False
        before = asdict(model)
        current_status = (getattr(model, 'turboquant_status', '') or 'unknown').strip().lower()
        if current_status not in TURBOQUANT_STATUSES:
            current_status = 'unknown'
        should_update = (
            current_status != detected.status
            or not getattr(model, 'turboquant_source', '')
            or int(getattr(model, 'turboquant_head_dim', 0) or 0) != int(detected.head_dim or 0)
            or int(getattr(model, 'turboquant_key_dim', 0) or 0) != int(detected.key_dim or 0)
            or int(getattr(model, 'turboquant_value_dim', 0) or 0) != int(detected.value_dim or 0)
            or (getattr(model, 'turboquant_reason', '') or '') != (detected.reason or '')
        )
        if not should_update:
            return False
        apply_turboquant_info(model, detected)
        return asdict(model) != before

    def _normalize_model_ranks(self) -> bool:
        changed = False
        ordered = sorted(
            self.models,
            key=lambda model: (
                int(getattr(model, 'sort_rank', 0) or 0) if int(getattr(model, 'sort_rank', 0) or 0) > 0 else 10**9,
                int(getattr(model, 'port', 0) or 0),
                str(getattr(model, 'id', '') or ''),
            ),
        )
        for index, model in enumerate(ordered, 1):
            if int(getattr(model, 'sort_rank', 0) or 0) != index:
                model.sort_rank = index
                changed = True
        if ordered != self.models:
            self.models = ordered
            changed = True
        return changed

    def pop_load_warnings(self) -> List[str]:
        warnings = list(self.load_warnings)
        self.load_warnings = []
        return warnings

    def pidfile(self, model_id: str) -> Path:
        return CACHE_DIR / f'{model_id}.pid'

    def pid_metadata_file(self, model_id: str) -> Path:
        return CACHE_DIR / f'{model_id}.pid.json'

    def logfile(self, model_id: str) -> Path:
        return CACHE_DIR / f'{model_id}.log'

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        return next((m for m in self.models if m.id == model_id), None)

    def next_sort_rank(self) -> int:
        values = [int(getattr(model, 'sort_rank', 0) or 0) for model in self.models]
        return (max(values) if values else 0) + 1

    def append_log(self, model_id: str, text: str):
        log_path = self.logfile(model_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%H:%M:%S')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{stamp}] llama-tui: {compact_message(text)}\n')

    def hardware_profile(self, refresh: bool = False) -> HardwareProfile:
        now = time.time()
        if refresh or self._hardware_profile is None or now - self._hardware_profile_at > 30:
            self._hardware_profile = benchmark_current_hardware()
            self._hardware_profile_at = now
        return self._hardware_profile

    def runtime_binary_version(self, model: ModelConfig) -> str:
        runtime = getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp'
        command = self.runtime_server_command(runtime)
        key = f'{runtime}:{command}'
        if key in self._runtime_version_cache:
            return self._runtime_version_cache[key]
        parts = shlex.split(command) if command else []
        if not parts:
            self._runtime_version_cache[key] = ''
            return ''
        try:
            result = subprocess.run(
                [*parts, '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=2,
                check=False,
            )
            line = compact_message((result.stdout or '').splitlines()[0] if result.stdout else '')
        except Exception:
            line = ''
        self._runtime_version_cache[key] = line
        return line

    def runtime_server_command(self, runtime: str) -> str:
        if runtime == 'vllm':
            return self.vllm_command
        if runtime == 'llama.cpp':
            return self.runtime_profile.server_command or self.llama_server
        return runtime

    def engine_capabilities(self, engine_profile: Optional[EngineProfile] = None) -> EngineCapabilities:
        profile = engine_profile or self.runtime_profile
        key = f'{profile.engine_id}:{profile.server_bin}'
        if key not in self._engine_capability_cache:
            self._engine_capability_cache[key] = detect_engine_capabilities(profile.server_bin, profile.engine_id)
        return self._engine_capability_cache[key]

    def runtime_profile_from_model(
        self,
        model: ModelConfig,
        ctx_value: int,
        parallel_value: int,
        ngl_value: int,
        runtime_profile: Optional[RuntimeProfile] = None,
    ) -> RuntimeProfile:
        if runtime_profile is not None:
            return runtime_profile
        args = list(getattr(model, 'extra_args', []) or [])
        engine_id = self.active_engine_key_for_model(model)
        if engine_id == 'buun':
            key_mode, value_mode = self.runtime_profile.buun_kv_pair()
            kv_preset = f'{key_mode}/{value_mode}'
        else:
            key_mode = (
                extra_arg_value(args, '--cache-type-k')
                or extra_arg_value(args, '--cache-type')
                or ''
            )
            value_mode = (
                extra_arg_value(args, '--cache-type-v')
                or extra_arg_value(args, '--cache-type')
                or ''
            )
            kv_preset = f'{key_mode or value_mode}/{value_mode or key_mode}' if (key_mode or value_mode) else 'default'

        def int_extra(*flags: str) -> int:
            try:
                return int(extra_arg_value(args, *flags) or 0)
            except Exception:
                return 0

        return RuntimeProfile(
            engine_id=engine_id,
            ctx_size=max(1, int(ctx_value or 1)),
            gpu_layers=int(ngl_value or 0),
            parallel=max(1, int(parallel_value or 1)),
            kv_preset=kv_preset,
            flash_attn='on' if bool(getattr(model, 'flash_attn', True)) else 'off',
            batch_size=int_extra('--batch-size', '-b'),
            ubatch_size=int_extra('--ubatch-size', '-ub'),
            extra_args=(),
            name='manual',
        )

    def runtime_indicator(self) -> str:
        return self.runtime_profile.header_indicator()

    def turboquant_session_advisory(self, model: ModelConfig) -> str:
        if self.active_engine_key_for_model(model) != 'buun':
            return ''
        status = (getattr(model, 'turboquant_status', '') or 'unknown').strip().lower()
        if status in ('native', 'padded'):
            return ''
        return f'TurboQuant advisory: {turboquant_detail(model)}'

    def model_fingerprint(self, model: ModelConfig) -> str:
        target = (getattr(model, 'path', '') or '').strip()
        path = Path(target).expanduser()
        stat_data: Dict[str, object] = {}
        if path.exists():
            try:
                stat = path.stat()
                stat_data = {
                    'path': str(path.resolve(strict=False)),
                    'size': stat.st_size,
                    'mtime_ns': stat.st_mtime_ns,
                }
            except OSError:
                stat_data = {'path': str(path.resolve(strict=False))}
        else:
            stat_data = {'ref': target}

        metadata = read_gguf_metadata(path) if path.exists() and path.suffix.lower() == '.gguf' else {}
        arch = str(metadata.get('general.architecture') or '')
        metadata_keys = [
            'general.architecture',
            'general.name',
            'general.file_type',
        ]
        if arch:
            metadata_keys.extend([
                f'{arch}.block_count',
                f'{arch}.context_length',
                f'{arch}.embedding_length',
                f'{arch}.attention.head_count',
                f'{arch}.attention.head_count_kv',
                f'{arch}.attention.key_length',
                f'{arch}.attention.value_length',
                f'{arch}.expert_count',
                f'{arch}.expert_used_count',
                f'{arch}.expert_shared_count',
                f'{arch}.expert_group_count',
                f'{arch}.expert_group_used_count',
                f'{arch}.moe_every_n_layers',
                f'{arch}.leading_dense_block_count',
            ])
        payload = {
            'runtime': getattr(model, 'runtime', 'llama.cpp'),
            'target': stat_data,
            'metadata': {key: metadata.get(key) for key in metadata_keys if key in metadata},
            'architecture': {
                'architecture': getattr(model, 'architecture', ''),
                'architecture_type': getattr(model, 'architecture_type', 'unknown'),
                'expert_count': int(getattr(model, 'expert_count', 0) or 0),
                'expert_used_count': int(getattr(model, 'expert_used_count', 0) or 0),
                'active_expert_ratio': float(getattr(model, 'active_expert_ratio', 0.0) or 0.0),
                'classification_source': getattr(model, 'classification_source', ''),
                'classification_confidence': float(getattr(model, 'classification_confidence', 0.0) or 0.0),
            },
            'runtime_config': {
                'extra_args': list(getattr(model, 'extra_args', []) or []),
                'engine_key': self.active_engine_key_for_model(model),
                'kv_key_mode': self.runtime_profile.kv_key_mode if self.active_engine_key_for_model(model) == 'buun' else '',
                'kv_value_mode': self.runtime_profile.kv_value_mode if self.active_engine_key_for_model(model) == 'buun' else '',
                'llama_server': self.runtime_server_command('llama.cpp') if getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp' else '',
                'vllm_command': self.vllm_command if getattr(model, 'runtime', 'llama.cpp') == 'vllm' else '',
                'binary_version': self.runtime_binary_version(model),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()[:24]

    def models_needing_default_benchmark(self) -> List[ModelConfig]:
        needed: List[ModelConfig] = []
        for model in self.models:
            if not getattr(model, 'enabled', True):
                continue
            status = (getattr(model, 'default_benchmark_status', '') or '').strip().lower()
            current_fingerprint = self.model_fingerprint(model)
            saved_fingerprint = getattr(model, 'benchmark_fingerprint', '') or ''
            has_benchmark = float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0) > 0.0
            if status == 'pending':
                needed.append(model)
            elif saved_fingerprint and saved_fingerprint != current_fingerprint:
                needed.append(model)
            elif not saved_fingerprint and not has_benchmark and status not in ('failed', 'aborted', 'running'):
                needed.append(model)
        return needed

    def _metadata_native_context(self, metadata: Dict[str, object]) -> int:
        arch = str(metadata.get('general.architecture') or '')
        keys = ['general.context_length']
        if arch:
            keys.append(f'{arch}.context_length')
        for key in keys:
            try:
                value = int(metadata.get(key) or 0)
            except Exception:
                value = 0
            if value > 0:
                return value
        return 0

    def static_model_diagnostics(self, model: ModelConfig) -> Dict[str, object]:
        runtime = getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp'
        target = (getattr(model, 'path', '') or '').strip()
        result: Dict[str, object] = {
            'runtime': runtime,
            'target': target,
            'status': 'passed',
            'reason': '',
            'exists': False,
            'file_size': 0,
            'gguf_magic': '',
            'metadata_ok': False,
            'native_context': 0,
            'kv_bytes_per_token': 0,
            'architecture_type': getattr(model, 'architecture_type', 'unknown'),
            'architecture': getattr(model, 'architecture', ''),
            'classification_source': getattr(model, 'classification_source', ''),
            'classification_confidence': float(getattr(model, 'classification_confidence', 0.0) or 0.0),
            'turboquant_status': getattr(model, 'turboquant_status', 'unknown'),
            'turboquant_detail': turboquant_detail(model),
        }
        if not target:
            result.update({'status': 'failed', 'reason': 'model target is empty'})
            return result
        path = Path(target).expanduser()
        if runtime == 'vllm':
            exists = path.exists()
            result['exists'] = exists
            if exists:
                try:
                    result['file_size'] = path.stat().st_size if path.is_file() else 0
                except OSError:
                    result['file_size'] = 0
            if exists or looks_like_model_reference(target):
                result['reason'] = 'vLLM target is a local path or offline-valid repo reference'
                result['kv_bytes_per_token'] = estimate_kv_bytes_per_token(model)
                return result
            result.update({'status': 'failed', 'reason': f'vLLM target is not a path or repo id: {target}'})
            return result

        if not path.exists():
            result.update({'status': 'failed', 'reason': f'model path missing: {target}'})
            return result
        result['exists'] = True
        try:
            result['file_size'] = path.stat().st_size
        except OSError:
            result['file_size'] = 0
        if path.suffix.lower() != '.gguf':
            result.update({'status': 'failed', 'reason': f'not a GGUF file: {target}'})
            return result
        if 'mmproj' in path.name.lower():
            result.update({'status': 'failed', 'reason': 'mmproj files are projection files, not chat models'})
            return result
        try:
            with open(path, 'rb') as file_obj:
                magic = file_obj.read(4)
        except OSError as exc:
            result.update({'status': 'failed', 'reason': f'failed to read model file: {exc}'})
            return result
        result['gguf_magic'] = magic.decode('ascii', errors='replace')
        if magic != b'GGUF':
            result.update({'status': 'failed', 'reason': 'bad GGUF magic header'})
            return result
        if int(result.get('file_size', 0) or 0) < 32:
            result.update({'status': 'failed', 'reason': 'truncated GGUF header'})
            return result
        metadata = read_gguf_metadata(path)
        result['metadata_ok'] = bool(metadata)
        result['native_context'] = self._metadata_native_context(metadata)
        result['kv_bytes_per_token'] = estimate_kv_bytes_per_token(model)
        if not metadata:
            result.update({'status': 'warning', 'reason': 'GGUF header is valid, but metadata could not be parsed'})
        elif not result['native_context']:
            result.update({'status': 'warning', 'reason': 'GGUF metadata parsed, but native context was not found'})
        else:
            result['reason'] = 'GGUF metadata parsed'
        return result

    def model_cap_diagnosis(self, model: ModelConfig) -> Dict[str, object]:
        requested_ctx = max(0, int(getattr(model, 'ctx', 0) or 0))
        parallel = max(1, int(getattr(model, 'parallel', 1) or 1))
        per_slot = requested_ctx // parallel
        ctx_max = max(0, int(getattr(model, 'ctx_max', 0) or 0))
        ctx_min = max(0, int(getattr(model, 'ctx_min', 0) or 0))
        static = self.static_model_diagnostics(model)
        native_context = int(static.get('native_context', 0) or 0)
        profile = self.hardware_profile(refresh=True)
        estimated_safe = estimate_safe_context_for_profile(
            model,
            profile,
            max(5, min(70, int(getattr(model, 'memory_reserve_percent', 25) or 25))),
            parallel,
            max(256, ctx_min or 2048),
            max(ctx_min or 2048, ctx_max or requested_ctx or 2048),
        )
        measured_values = []
        for item in (getattr(model, 'measured_profiles', {}) or {}).values():
            if isinstance(item, dict) and str(item.get('status', 'ok') or 'ok') == 'ok':
                measured_values.append(int(item.get('ctx_per_slot', item.get('ctx', 0)) or 0))
        measured_max = max(measured_values or [0])
        candidates: List[Tuple[str, int]] = []
        if ctx_max and requested_ctx > ctx_max:
            candidates.append(('user_ctx_max', ctx_max))
        if native_context and requested_ctx > native_context:
            candidates.append(('model_native_context', native_context))
        if int(estimated_safe or 0) > 0 and requested_ctx > int(estimated_safe or 0):
            candidates.append(('hardware_safe_context', int(estimated_safe or 0)))
        if parallel > 1 and per_slot < requested_ctx:
            candidates.append(('parallel_split', per_slot))
        if measured_max and requested_ctx > measured_max:
            candidates.append(('benchmark_proof', measured_max))
        limiting_factor = 'configured_request'
        effective_limit = requested_ctx
        if candidates:
            priority = {
                'user_ctx_max': 0,
                'model_native_context': 1,
                'hardware_safe_context': 2,
                'parallel_split': 3,
                'benchmark_proof': 4,
            }
            limiting_factor, effective_limit = min(
                candidates,
                key=lambda item: (max(0, int(item[1] or 0)), priority.get(item[0], 99)),
            )
        return {
            'configured_ctx': requested_ctx,
            'parallel': parallel,
            'ctx_per_slot': per_slot,
            'ctx_min': ctx_min,
            'ctx_max': ctx_max,
            'native_context': native_context,
            'estimated_safe_context': int(estimated_safe or 0),
            'measured_max_context': measured_max,
            'limiting_factor': limiting_factor,
            'effective_limit': int(effective_limit or 0),
            'hardware': profile.short_summary(),
        }

    def verify_model(self, model: ModelConfig, save: bool = True) -> Dict[str, object]:
        now = datetime.now().isoformat(timespec='seconds')
        fingerprint = ''
        try:
            fingerprint = self.model_fingerprint(model)
        except Exception:
            fingerprint = ''
        static = self.static_model_diagnostics(model)
        cap = self.model_cap_diagnosis(model)
        status = 'needs_benchmark'
        summary = 'Benchmark proof needed.'
        fresh = False
        try:
            from .benchmark import benchmark_profile_is_fresh
            fresh = benchmark_profile_is_fresh(self, model)
        except Exception:
            fresh = False
        if static.get('status') == 'failed':
            status = 'failed'
            summary = str(static.get('reason') or 'Static model validation failed.')
        elif fresh:
            status = 'passed' if static.get('status') == 'passed' else 'warning'
            summary = 'Fresh benchmark proof exists.' if status == 'passed' else f'Fresh benchmark proof exists, but static check warns: {static.get("reason")}'
        elif static.get('status') == 'warning':
            status = 'warning'
            summary = f'{static.get("reason")}; benchmark proof needed.'
        results = {
            'status': status,
            'summary': summary,
            'fingerprint': fingerprint,
            'static': static,
            'cap': cap,
            'fresh_benchmark': fresh,
        }
        model.verification_status = status
        model.verification_at = now
        model.verification_fingerprint = fingerprint
        model.verification_summary = summary
        model.verification_results = results
        if save:
            self.add_or_update(model)
        return results

    def benchmark_proof_model_ids(self, force: bool = False) -> List[str]:
        ids: List[str] = []
        try:
            from .benchmark import deep_benchmark_model_decision
        except Exception:
            deep_benchmark_model_decision = None
        for model in self.models:
            if not getattr(model, 'enabled', True):
                continue
            if deep_benchmark_model_decision is not None:
                should_run, _reason = deep_benchmark_model_decision(self, model, force=force)
                if should_run:
                    ids.append(model.id)
                continue
            result = self.verify_model(model, save=False)
            if force or result.get('status') in ('needs_benchmark', 'warning'):
                ids.append(model.id)
        return ids

    def command_prefix(self, command: str) -> List[str]:
        return shlex.split(command) if command else []

    def command_exists(self, command: str) -> bool:
        parts = self.command_prefix(command)
        if not parts:
            return False
        first = os.path.expanduser(parts[0])
        if '/' in first or first.startswith('.') or first.startswith('~'):
            return Path(first).exists()
        return shutil.which(first) is not None

    def validate_workspace_path(self, workspace: str | Path) -> Tuple[bool, Optional[Path], str]:
        raw = str(workspace or '').strip()
        if not raw:
            return False, None, 'workspace path is empty'
        path = Path(raw).expanduser().resolve(strict=False)
        if not path.exists():
            return False, None, f'workspace does not exist: {path}'
        if not path.is_dir():
            return False, None, f'workspace is not a directory: {path}'
        return True, path, ''

    def workspace_settings(self, runtime: str = 'opencode'):
        return self.hermes if runtime == 'hermes' else self.opencode

    def workspace_presets(self, runtime: str = 'opencode') -> List[str]:
        settings = self.workspace_settings(runtime)
        values = list(getattr(settings, 'workspace_presets', []) or [])
        if getattr(settings, 'last_workspace_path', ''):
            values.insert(0, str(getattr(settings, 'last_workspace_path', '') or ''))
        deduped: List[str] = []
        seen = set()
        for value in values:
            clean = str(value or '').strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        return deduped[:8]

    def remember_workspace_preset(self, runtime: str, workspace: str):
        clean = str(workspace or '').strip()
        if not clean:
            return
        settings = self.workspace_settings(runtime)
        current = self.workspace_presets(runtime)
        merged = [clean] + [value for value in current if value != clean]
        settings.last_workspace_path = clean
        settings.workspace_presets = merged[:8]
        self.save()

    def mark_model_used(self, model_id: str):
        model = self.get_model(model_id)
        if not model:
            return
        model.last_used_at = datetime.now().isoformat(timespec='seconds')
        self.save()

    def toggle_favorite(self, model_id: str) -> Tuple[bool, str]:
        model = self.get_model(model_id)
        if not model:
            return False, 'model not found'
        model.favorite = not bool(getattr(model, 'favorite', False))
        self.save()
        return bool(model.favorite), ('favorited' if model.favorite else 'unfavorited')

    def opencode_provider_key(self, model: ModelConfig) -> str:
        return f'local-{model.id}'

    def opencode_model_ref(self, model: ModelConfig) -> str:
        return f'{self.opencode_provider_key(model)}/{model.alias}'

    def continue_model_ref(self, model: ModelConfig) -> str:
        return model.alias or model.id

    def continue_base_url(self, model: ModelConfig) -> str:
        return f'http://{model.host}:{model.port}/v1'

    def hermes_provider_key(self, model: ModelConfig) -> str:
        return f'local-{model.id}'

    def hermes_model_ref(self, model: ModelConfig) -> str:
        return model.alias

    def hermes_base_url(self, model: ModelConfig) -> str:
        return f'http://{model.host}:{model.port}/v1'

    def detect_terminal_launcher(self) -> Optional[str]:
        for launcher in TERMINAL_LAUNCHER_ORDER:
            resolved = shutil.which(launcher)
            if resolved:
                return resolved
        if shutil.which('flatpak'):
            for app_id in FLATPAK_TERMINAL_APP_IDS:
                try:
                    result = subprocess.run(
                        ['flatpak', 'info', app_id],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                        check=False,
                    )
                except Exception:
                    continue
                if result.returncode == 0:
                    return f'flatpak:{app_id}'
        return None

    def detect_host_terminal_bridge(self) -> Optional[str]:
        for bridge in HOST_TERMINAL_BRIDGES:
            resolved = shutil.which(bridge)
            if resolved:
                return resolved
        return None

    def host_command_available(self, bridge: str, shell_check: str, timeout: float = 2.0) -> bool:
        try:
            result = subprocess.run(
                host_bridge_command(bridge, ['sh', '-lc', shell_check]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except Exception:
            return False
        return result.returncode == 0

    def detect_host_terminal_launcher(self) -> Tuple[Optional[str], Optional[str]]:
        bridge = self.detect_host_terminal_bridge()
        if not bridge:
            return None, None
        for launcher in TERMINAL_LAUNCHER_ORDER:
            if self.host_command_available(bridge, f'command -v {shlex.quote(launcher)} >/dev/null 2>&1'):
                return bridge, launcher
        for app_id in FLATPAK_TERMINAL_APP_IDS:
            if self.host_command_available(bridge, f'command -v flatpak >/dev/null 2>&1 && flatpak info {shlex.quote(app_id)} >/dev/null 2>&1'):
                return bridge, f'flatpak:{app_id}'
        guessed = desktop_terminal_guess()
        if guessed:
            return bridge, guessed
        return bridge, None

    def build_container_reentry_shell_command(self, shell_cmd: str) -> Tuple[bool, str, str]:
        container_name = current_container_name()
        if not container_name:
            return False, '', (
                'Detected a container session, but could not determine its distrobox/container name. '
                'Set opencode.terminal_command to a terminal command that re-enters this environment.'
            )
        return True, f'distrobox enter {shlex.quote(container_name)} -- bash -lc {shlex.quote(shell_cmd)}', ''

    def terminal_setup_error(self, shell_cmd: str, host_detail: str = '') -> str:
        container_note = 'detected container session' if container_environment_detected() else 'no container session detected'
        detail = f'No terminal launcher was visible from llama-tui ({container_note}).'
        if host_detail:
            detail += f' {host_detail}'
        detail += (
            ' Set opencode.terminal_command in settings using {title}, {cwd}, and {cmd}. '
            'Example: konsole --workdir {cwd} -p tabtitle={title} -e bash -lc {cmd}. '
            f'Manual OpenCode command: {shell_cmd}'
        )
        return detail

    def build_opencode_shell_command(self, model: ModelConfig, workspace: Path) -> str:
        env = {
            'OPENCODE_DISABLE_AUTOUPDATE': 'true',
            'OPENCODE_DISABLE_PRUNE': 'true',
            'OPENCODE_DISABLE_MODELS_FETCH': 'true',
            'OPENCODE_CLIENT': 'llama-tui-stack',
        }
        if self.opencode.path:
            env['OPENCODE_CONFIG'] = str(Path(self.opencode.path).expanduser())
        env_prefix = shell_env_prefix(env)
        command = [
            'opencode',
            str(workspace),
            '--model', self.opencode_model_ref(model),
            '--agent', 'build',
        ]
        opencode_cmd = env_prefix + ' '.join(shlex.quote(part) for part in command)
        return (
            f'cd {shlex.quote(str(workspace))} && {opencode_cmd}; status=$?; '
            'printf "\\nOpenCode exited with status %s\\n" "$status"; '
            'printf "Press Enter to close..."; read -r _; exit "$status"'
        )

    def hermes_home_for_model(self, model: ModelConfig) -> Path:
        root = Path(getattr(self.hermes, 'home_root', '') or str(CACHE_DIR / 'hermes')).expanduser()
        return root / model.id

    def hermes_config_path(self, model: ModelConfig) -> Path:
        return self.hermes_home_for_model(model) / 'config.yaml'

    def hermes_context_policy(self, model: ModelConfig) -> Dict[str, object]:
        parallel = max(1, int(getattr(model, 'parallel', 1) or 1))
        actual_ctx = max(0, int(getattr(model, 'ctx', 0) or 0) // parallel)
        required = max(1, int(getattr(self.hermes, 'min_context_tokens', 64000) or 64000))
        override = max(0, int(getattr(self.hermes, 'experimental_context_override_tokens', 0) or 0))
        experimental = bool(getattr(self.hermes, 'allow_experimental_context_override', False)) and override > 0
        configured = override if experimental else actual_ctx
        detail = ''
        if experimental and configured != actual_ctx:
            detail = (
                f'experimental Hermes context override: config={configured} '
                f'actual ctx/slot={actual_ctx}'
            )
        return {
            'required_context': required,
            'actual_ctx_per_slot': actual_ctx,
            'configured_context_length': configured,
            'experimental_context_override': experimental,
            'context_detail': detail,
        }

    def generate_hermes_config(self, model: ModelConfig) -> Tuple[bool, str]:
        if model is None:
            return False, 'No model selected for Hermes config.'
        home = self.hermes_home_for_model(model)
        config_path = self.hermes_config_path(model)
        toolsets = list(getattr(self.hermes, 'toolsets', []) or ['terminal', 'file', 'todo'])
        max_turns = int(getattr(self.hermes, 'max_turns', 20) or 20)
        context_policy = self.hermes_context_policy(model)
        lines = [
            '# Generated by llama-tui. Safe to delete; it will be recreated.',
            'model:',
            f'  default: {yaml_quote(self.hermes_model_ref(model))}',
            '  provider: "custom"',
            f'  base_url: {yaml_quote(self.hermes_base_url(model))}',
            f'  context_length: {int(context_policy["configured_context_length"] or 0)}',
            f'  max_tokens: {int(getattr(model, "output", 0) or 0)}',
            'terminal:',
            '  backend: "local"',
            '  cwd: "."',
            '  timeout: 180',
            '  lifetime_seconds: 300',
            'platform_toolsets:',
            f'  cli: {yaml_list(toolsets)}',
            'agent:',
            f'  max_turns: {max_turns}',
            '  verbose: false',
            'memory:',
            '  memory_enabled: false',
            '  user_profile_enabled: false',
            'skills:',
            '  creation_nudge_interval: 0',
            '',
        ]
        home.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            self._backup_export_file(config_path, str(config_path.parent / 'backups'))
        config_path.write_text('\n'.join(lines), encoding='utf-8')
        return True, f'Generated Hermes config {config_path}'

    def build_hermes_env(self, model: ModelConfig, workspace: Path, benchmark: bool = False) -> Dict[str, str]:
        home = self.hermes_home_for_model(model)
        env = {
            'HERMES_HOME': str(home),
            'HERMES_INFERENCE_PROVIDER': 'custom',
            'OPENAI_BASE_URL': self.hermes_base_url(model),
            'OPENAI_API_KEY': 'no-key-required',
            'HERMES_MAX_ITERATIONS': str(int(getattr(self.hermes, 'max_turns', 20) or 20)),
        }
        if benchmark:
            env.update({
                'HERMES_CLIENT': 'llama-tui-benchmark',
                'HERMES_YOLO_MODE': '1',
                'NO_COLOR': '1',
            })
        return env

    def hermes_command_prefix(self) -> List[str]:
        return self.command_prefix(getattr(self.hermes, 'command', '') or 'hermes') or ['hermes']

    def build_hermes_cli_command(self, model: ModelConfig, workspace: Path, prompt: str = '', benchmark: bool = False) -> List[str]:
        toolsets = ','.join(list(getattr(self.hermes, 'toolsets', []) or ['terminal', 'file', 'todo']))
        command = self.hermes_command_prefix() + [
            'chat',
            '-m', self.hermes_model_ref(model),
            '-t', toolsets,
            '--max-turns', str(int(getattr(self.hermes, 'max_turns', 20) or 20)),
        ]
        if benchmark:
            command.extend(['--yolo'])
            if bool(getattr(self.hermes, 'quiet', True)):
                command.extend(['--quiet'])
        if prompt:
            command.extend(['-q', prompt])
        return command

    def build_hermes_shell_command(self, model: ModelConfig, workspace: Path) -> str:
        self.generate_hermes_config(model)
        env_prefix = shell_env_prefix(self.build_hermes_env(model, workspace, benchmark=False))
        command = self.build_hermes_cli_command(model, workspace)
        hermes_cmd = env_prefix + ' '.join(shlex.quote(part) for part in command)
        return (
            f'cd {shlex.quote(str(workspace))} && {hermes_cmd}; status=$?; '
            'printf "\\nHermes exited with status %s\\n" "$status"; '
            'printf "Press Enter to close..."; read -r _; exit "$status"'
        )

    def build_terminal_command(
        self,
        title: str,
        workspace: Path,
        shell_cmd: str,
        terminal_template: str = '',
    ) -> Tuple[bool, List[str], str]:
        template = terminal_template.strip() if terminal_template else getattr(self.opencode, 'terminal_command', '').strip()
        if template:
            try:
                return True, render_terminal_template(template, title, workspace, shell_cmd), 'custom'
            except Exception as exc:
                return False, [], f'invalid opencode.terminal_command: {exc}'
        launcher = self.detect_terminal_launcher()
        if launcher:
            return True, terminal_command_for_launcher(launcher, title, workspace, shell_cmd), f'local:{terminal_launcher_label(launcher)}'
        if container_environment_detected():
            bridge, host_launcher = self.detect_host_terminal_launcher()
            if bridge and host_launcher:
                reentry_ok, reentry_cmd, reentry_msg = self.build_container_reentry_shell_command(shell_cmd)
                if not reentry_ok:
                    return False, [], reentry_msg
                terminal_cmd = terminal_command_for_launcher(host_launcher, title, workspace, reentry_cmd)
                label = f'host:{Path(bridge).name}/{terminal_launcher_label(host_launcher)}'
                return True, host_bridge_command(bridge, terminal_cmd), label
            if bridge:
                return False, [], self.terminal_setup_error(
                    shell_cmd,
                    f'Host bridge {Path(bridge).name} is available, but no host terminal launcher was detected.',
                )
        return False, [], self.terminal_setup_error(shell_cmd)

    def launch_opencode_terminal(self, model: ModelConfig, workspace: str | Path) -> Tuple[bool, str]:
        if not shutil.which('opencode'):
            return False, 'opencode command not found in PATH'
        valid, workspace_path, reason = self.validate_workspace_path(workspace)
        if not valid or workspace_path is None:
            return False, reason
        shell_cmd = self.build_opencode_shell_command(model, workspace_path)
        ok, terminal_cmd, terminal_label = self.build_terminal_command(
            f'OpenCode {model.id}',
            workspace_path,
            shell_cmd,
        )
        if not ok:
            return False, terminal_label
        self.append_log(model.id, f'OpenCode terminal command ({terminal_label}): {shlex.join(terminal_cmd)}')
        try:
            proc = subprocess.Popen(
                terminal_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return False, f'failed to launch OpenCode terminal: {exc}'
        time.sleep(0.15)
        returncode = proc.poll()
        if returncode not in (None, 0):
            return False, (
                f'OpenCode terminal launcher exited immediately with status {returncode}. '
                f'Command: {shlex.join(terminal_cmd)}'
            )
        self.append_log(model.id, f'OpenCode terminal launched pid={proc.pid} workspace={workspace_path} via {terminal_label}')
        return True, f'OpenCode terminal launched for {model.id} in {workspace_path}'

    def launch_hermes_terminal(self, model: ModelConfig, workspace: str | Path) -> Tuple[bool, str]:
        command_prefix = self.hermes_command_prefix()
        if not command_prefix or not self.command_exists(command_prefix[0]):
            return False, f'Hermes command not found: {getattr(self.hermes, "command", "hermes") or "hermes"}'
        valid, workspace_path, reason = self.validate_workspace_path(workspace)
        if not valid or workspace_path is None:
            return False, reason
        config_ok, config_msg = self.generate_hermes_config(model)
        if not config_ok:
            return False, config_msg
        shell_cmd = self.build_hermes_shell_command(model, workspace_path)
        ok, terminal_cmd, terminal_label = self.build_terminal_command(
            f'Hermes {model.id}',
            workspace_path,
            shell_cmd,
            terminal_template=getattr(self.hermes, 'terminal_command', ''),
        )
        if not ok:
            return False, terminal_label
        self.append_log(model.id, f'Hermes config: {config_msg}')
        self.append_log(model.id, f'Hermes terminal command ({terminal_label}): {shlex.join(terminal_cmd)}')
        try:
            proc = subprocess.Popen(
                terminal_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return False, f'failed to launch Hermes terminal: {exc}'
        time.sleep(0.15)
        returncode = proc.poll()
        if returncode not in (None, 0):
            return False, (
                f'Hermes terminal launcher exited immediately with status {returncode}. '
                f'Command: {shlex.join(terminal_cmd)}'
            )
        self.append_log(model.id, f'Hermes terminal launched pid={proc.pid} workspace={workspace_path} via {terminal_label}')
        return True, f'Hermes terminal launched for {model.id} in {workspace_path}'

    def launch_vscode_workspace(self, workspace: str | Path) -> Tuple[bool, str]:
        resolved_code = shutil.which('code')
        if not resolved_code:
            return False, 'VS Code command not found; skipped code --new-window'
        valid, workspace_path, reason = self.validate_workspace_path(workspace)
        if not valid or workspace_path is None:
            return False, reason
        try:
            proc = subprocess.Popen(
                [resolved_code, '--new-window', str(workspace_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return False, f'failed to launch VS Code: {exc}'
        return True, f'VS Code launched pid={proc.pid} workspace={workspace_path}'

    def runtime_command_ready(self, runtime: str, command: str) -> Tuple[bool, str]:
        if runtime != 'llama.cpp':
            return True, ''
        parts = self.command_prefix(command)
        if not parts:
            return False, 'empty llama-server command'
        cache_key = '\0'.join(parts)
        if cache_key in self._runtime_check_cache:
            return self._runtime_check_cache[cache_key]

        binary = os.path.expanduser(parts[0])
        if Path(binary).exists() and shutil.which('ldd'):
            try:
                ldd_result = subprocess.run(
                    ['ldd', binary],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=4,
                    check=False,
                )
                ldd_output = compact_message(f'{ldd_result.stdout}\n{ldd_result.stderr}')
                ldd_low = ldd_output.lower()
                if 'not found' in ldd_low and any(
                    marker in ldd_low
                    for marker in ('libcuda', 'libcudart', 'libcublas', 'libggml', 'libllama')
                ):
                    outcome = (False, ldd_output[:800])
                    self._runtime_check_cache[cache_key] = outcome
                    return outcome
            except (OSError, subprocess.TimeoutExpired):
                pass

        try:
            result = subprocess.run(
                parts + ['--help'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=4,
                check=False,
            )
        except subprocess.TimeoutExpired:
            outcome = (True, '')
        except OSError as exc:
            outcome = (False, compact_message(str(exc)))
        else:
            output = compact_message(f'{result.stderr}\n{result.stdout}')
            low = output.lower()
            fatal_markers = (
                'error while loading shared libraries',
                'cannot open shared object file',
                'libcudart.so',
                'libcuda.so',
            )
            if any(marker in low for marker in fatal_markers):
                outcome = (False, output[:800])
            else:
                outcome = (True, '')

        self._runtime_check_cache[cache_key] = outcome
        return outcome

    def normalize_model_ref(self, path: str | Path) -> str:
        raw = str(path).strip()
        p = Path(raw).expanduser()

        # Keep real local file paths and discovered cache files in the same format,
        # otherwise detect() will keep re-adding the same model because one side is
        # stored as "path:/abs/file.gguf" and the other as "/abs/file.gguf".
        if p.exists() or raw.startswith('~') or '/' in raw or raw.startswith('.'):
            return str(p.resolve(strict=False))

        # Non-path references (mainly vLLM / HF repo IDs) should remain symbolic.
        return f'ref:{raw}'

    def validate_model_target(self, model: ModelConfig) -> Tuple[bool, str]:
        target = (model.path or '').strip()
        if not target:
            return False, 'model target is empty'
        if getattr(model, 'runtime', 'llama.cpp') == 'vllm':
            p = Path(target).expanduser()
            if p.exists() or looks_like_model_reference(target):
                return True, ''
            return False, f'vLLM model target not found / not a repo id: {target}'
        p = Path(target).expanduser()
        if not p.exists():
            return False, f'model path missing: {target}'
        if not is_real_model_file(p):
            return False, f'not a supported GGUF model: {target}'
        return True, ''

    def _command_matches_runtime(self, parts: List[str], runtime: str) -> bool:
        if not parts:
            return False
        runtime = getattr(runtime, 'strip', lambda: runtime)()
        if runtime == 'vllm':
            prefixes = self.command_prefix(self.vllm_command)
            target = os.path.basename(prefixes[0]) if prefixes else 'vllm'
            return (
                any(
                    os.path.basename(part) == target
                    or os.path.basename(part).startswith('vllm')
                    or 'vllm.entrypoints' in part
                    for part in parts
                )
                and ('serve' in parts or any('api_server' in part for part in parts))
            )
        candidate_commands = [self.llama_server, self.runtime_server_command('llama.cpp')]
        targets = set()
        for command in candidate_commands:
            prefixes = self.command_prefix(command)
            target = os.path.basename(prefixes[0]) if prefixes else os.path.basename(command or '')
            if target:
                targets.add(target)
        return any(os.path.basename(part) in targets for part in parts)

    def available_memory_bytes(self) -> int:
        return read_meminfo_bytes().get('MemAvailable', 0)

    def _estimate_kv_bytes_per_token(self, model: ModelConfig) -> int:
        return estimate_kv_bytes_per_token(model)

    def requested_context_for_launch(self, model: ModelConfig) -> int:
        runtime = (getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp').strip().lower()
        if runtime == 'llama.cpp' and self.runtime_profile.context_override is not None:
            return max(1, int(self.runtime_profile.context_override))
        return max(1, int(getattr(model, 'ctx', 8192)))

    def safe_launch_profile(self, model: ModelConfig) -> Tuple[bool, Dict[str, int], str]:
        mode = (getattr(model, 'optimize_mode', 'max_context_safe') or 'max_context_safe').strip().lower()
        requested_ctx = self.requested_context_for_launch(model)
        requested_parallel = max(1, int(getattr(model, 'parallel', 1)))
        if mode == 'manual' or mode.startswith('measured_'):
            label = 'measured profile' if mode.startswith('measured_') else 'manual mode'
            return True, {'ctx': requested_ctx, 'parallel': requested_parallel}, label

        profile = self.hardware_profile(refresh=True)
        if (profile.memory_available or self.available_memory_bytes()) <= 0 and not profile.has_usable_gpu():
            return True, {'ctx': requested_ctx, 'parallel': requested_parallel}, 'safe mode (memory probe unavailable)'

        reserve_pct = max(5, min(60, int(getattr(model, 'memory_reserve_percent', 25))))
        cap_parallel = max(1, requested_parallel)
        min_ctx = max(256, int(getattr(model, 'ctx_min', 2048)))
        max_ctx = max(min_ctx, int(getattr(model, 'ctx_max', 131072)))
        launch_model = ModelConfig(**asdict(model))
        launch_profile: Dict[str, int] = {'parallel': cap_parallel}
        if getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp':
            tier = (getattr(model, 'optimize_tier', 'moderate') or 'moderate').strip().lower()
            if tier not in ('safe', 'moderate', 'extreme'):
                tier = 'moderate'
            launch_model.ngl = choose_gpu_layers_for_profile(model, profile, tier)
            launch_profile['ngl'] = launch_model.ngl
        safe_ctx = estimate_safe_context_for_profile(launch_model, profile, reserve_pct, cap_parallel, min_ctx, max_ctx)
        if safe_ctx < min_ctx:
            kv_mib = self._estimate_kv_bytes_per_token(launch_model) / 1024**2
            return False, {}, (
                f'not enough memory for minimum ctx={min_ctx} '
                f'(safe_ctx={safe_ctx}, kv≈{kv_mib:.2f} MiB/token, {profile.short_summary()})'
            )
        applied_ctx = max(min_ctx, min(requested_ctx, max_ctx, safe_ctx))

        notes = [f'safe mode ram reserve={reserve_pct}%']
        if profile.has_usable_gpu() and getattr(model, 'runtime', 'llama.cpp') == 'llama.cpp':
            notes.append(f'gpu reserve≈{effective_gpu_reserve_percent(reserve_pct, tier)}%')
        if applied_ctx != requested_ctx:
            notes.append(f'ctx {requested_ctx}→{applied_ctx}')
        if 'ngl' in launch_profile and launch_profile['ngl'] != int(getattr(model, 'ngl', 0) or 0):
            notes.append(f'ngl {getattr(model, "ngl", 0)}→{launch_profile["ngl"]}')
        launch_profile['ctx'] = applied_ctx
        return True, launch_profile, ', '.join(notes)

    def managed_roots(self) -> Dict[str, Path]:
        return {
            'huggingface': Path(self.hf_cache_root).expanduser(),
            'llmfit': Path(self.llmfit_cache_root).expanduser(),
            'llm-models': Path(self.llm_models_cache_root).expanduser(),
        }

    def lm_studio_roots(self) -> List[Path]:
        roots: List[Path] = []
        seen = set()
        for raw in str(getattr(self, 'lm_studio_model_roots', '') or '').split(','):
            value = raw.strip()
            if not value:
                continue
            path = Path(value).expanduser()
            key = str(path.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            roots.append(path)
        return roots

    def managed_source_roots(self) -> List[Tuple[str, Path]]:
        roots = list(self.managed_roots().items())
        roots.extend(('lm-studio', root) for root in self.lm_studio_roots())
        return roots

    def normalize_model_path(self, path: str | Path) -> Path:
        return Path(path).expanduser().resolve(strict=False)

    def infer_model_source(self, model: ModelConfig) -> str:
        if getattr(model, 'runtime', 'llama.cpp') == 'vllm':
            return 'manual'
        if getattr(model, 'source', '') in ('manual', 'huggingface', 'llmfit', 'llm-models', 'lm-studio'):
            existing = getattr(model, 'source', '')
            if existing and existing != 'manual':
                return existing
        p = self.normalize_model_path(model.path)
        for source, root in self.managed_source_roots():
            try:
                p.relative_to(root.resolve(strict=False))
                return source
            except Exception:
                continue
        return 'manual'

    def discover_source_files(self) -> Tuple[Dict[str, Tuple[Path, str]], List[str]]:
        discovered: Dict[str, Tuple[Path, str]] = {}
        notes: List[str] = []

        for source, root in self.managed_source_roots():
            if not root.exists():
                notes.append(f'{source} cache not found: {root}')
                continue

            if source == 'huggingface':
                candidates = root.glob('models--*/snapshots/*/*.gguf')
            else:
                candidates = root.rglob('*.gguf')

            for gguf in sorted(candidates):
                if not is_real_model_file(gguf):
                    continue
                discovered[str(gguf.resolve(strict=False))] = (gguf, source)

        return discovered, notes

    def _proc_state(self, pid: int) -> Optional[str]:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            end = stat.rfind(")")
            if end == -1:
                return None
            rest = stat[end + 2:].split()
            return rest[0] if rest else None
        except Exception:
            return None

    def _pid_looks_like_runtime(self, pid: int, runtime: str) -> bool:
        return self._command_matches_runtime(self._pid_cmdline_parts(pid), runtime)

    def _pid_alive(self, pid: int, include_zombie: bool = False) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        state = self._proc_state(pid)
        if not include_zombie and state in ('Z', 'X'):
            return False
        return True

    def _pid_cmdline_parts(self, pid: int) -> List[str]:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            return [p.decode(errors='ignore') for p in raw.split(b'\x00') if p]
        except Exception:
            return []

    def _pid_looks_like_any_runtime(self, pid: int) -> bool:
        parts = self._pid_cmdline_parts(pid)
        return self._command_matches_runtime(parts, 'llama.cpp') or self._command_matches_runtime(parts, 'vllm')

    def _pid_matches_model(self, pid: int, model: ModelConfig) -> bool:
        parts = self._pid_cmdline_parts(pid)
        runtime = getattr(model, 'runtime', 'llama.cpp')
        if not parts or not self._command_matches_runtime(parts, runtime):
            return False
        port = str(model.port)
        alias = model.alias
        path = model.path
        joined = '\x00'.join(parts)
        return (
            f'\x00--port\x00{port}\x00' in joined
            or f'\x00--alias\x00{alias}\x00' in joined
            or f'\x00--served-model-name\x00{alias}\x00' in joined
            or path in parts
            or path in joined
        )

    def _read_pidfile(self, model_id: str) -> Optional[int]:
        try:
            return int(self.pidfile(model_id).read_text().strip())
        except Exception:
            return None

    def _read_pid_metadata(self, model_id: str) -> Dict:
        try:
            return json.loads(self.pid_metadata_file(model_id).read_text())
        except Exception:
            return {}

    def _tracked_pgid(self, model_id: str, pid: Optional[int]) -> Optional[int]:
        metadata = self._read_pid_metadata(model_id)
        try:
            pgid = int(metadata.get('pgid') or 0)
            if pgid > 0:
                return pgid
        except Exception:
            pass
        if pid:
            try:
                return os.getpgid(pid)
            except OSError:
                return None
        return None

    def _clear_pid_tracking(self, model_id: str, pid: Optional[int] = None):
        self.pidfile(model_id).unlink(missing_ok=True)
        self.pid_metadata_file(model_id).unlink(missing_ok=True)
        if pid is not None:
            self._owned_pids.discard(pid)

    def _write_pid_tracking(self, model: ModelConfig, pid: int, command: List[str]):
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = pid
        self._owned_pids.add(pid)
        self.pidfile(model.id).write_text(str(pid))
        metadata = {
            'app': APP_NAME,
            'owner_pid': os.getpid(),
            'model_id': model.id,
            'pid': pid,
            'pgid': pgid,
            'runtime': getattr(model, 'runtime', 'llama.cpp'),
            'alias': model.alias,
            'port': model.port,
            'path': model.path,
            'started_at': datetime.now().isoformat(timespec='seconds'),
            'command': command,
        }
        try:
            self.pid_metadata_file(model.id).write_text(json.dumps(metadata, indent=2) + '\n')
        except Exception:
            pass

    def _pid_is_tracked(self, model_id: str, pid: int) -> bool:
        if pid in self._owned_pids:
            return True
        tracked_pid = self._read_pidfile(model_id)
        if tracked_pid == pid:
            return True
        pgid = self._tracked_pgid(model_id, tracked_pid)
        if pgid is None:
            return False
        try:
            return os.getpgid(pid) == pgid
        except OSError:
            return False

    def _process_group_pids(self, pgid: int) -> List[int]:
        pids = []
        for proc_dir in Path('/proc').iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                pid = int(proc_dir.name)
                if os.getpgid(pid) != pgid:
                    continue
                if self._proc_state(pid) in ('Z', 'X'):
                    continue
                pids.append(pid)
            except Exception:
                continue
        return pids

    def _reap_pid(self, pid: int):
        try:
            while True:
                reaped, _status = os.waitpid(pid, os.WNOHANG)
                if reaped == 0:
                    break
                if reaped == pid:
                    break
        except ChildProcessError:
            pass
        except OSError:
            pass

    def _process_gone(self, pid: int, pgid: Optional[int]) -> bool:
        self._reap_pid(pid)
        if pgid is not None:
            return not self._process_group_pids(pgid)
        return not self._pid_alive(pid)

    def _send_signal(self, pid: int, sig: signal.Signals | int, use_group: bool) -> Tuple[Optional[int], bool]:
        pgid: Optional[int] = None
        if use_group:
            try:
                pgid = os.getpgid(pid)
            except OSError:
                pgid = None
            if pgid is not None and pgid != os.getpgrp():
                os.killpg(pgid, sig)
                return pgid, True
        os.kill(pid, sig)
        return pgid, False

    def terminate_process_group(self, pid: int, grace_seconds: float = 3.0) -> Tuple[bool, str]:
        if not pid:
            return True, 'no pid'
        if not self._pid_alive(pid, include_zombie=True):
            self._reap_pid(pid)
            return True, 'already stopped'
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = None
        use_group = pgid is not None and pgid != os.getpgrp()

        try:
            if use_group and pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            if not self._pid_alive(pid, include_zombie=True):
                self._reap_pid(pid)
                return True, 'already stopped'
            return False, str(exc)

        deadline = time.time() + max(0.1, grace_seconds)
        while time.time() < deadline:
            if self._process_gone(pid, pgid if use_group else None):
                return True, 'stopped'
            time.sleep(0.1)

        try:
            if use_group and pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

        deadline = time.time() + 1.5
        while time.time() < deadline:
            if self._process_gone(pid, pgid if use_group else None):
                return True, 'stopped (forced)'
            time.sleep(0.1)
        return False, 'did not stop cleanly'

    def _stop_pid(self, model_id: str, pid: int, use_group: bool) -> Tuple[bool, str]:
        if not self._pid_alive(pid):
            self._clear_pid_tracking(model_id, pid)
            self._reap_pid(pid)
            return True, 'already stopped'
        try:
            pgid, signaled_group = self._send_signal(pid, signal.SIGTERM, use_group)
        except OSError as e:
            if not self._pid_alive(pid):
                self._clear_pid_tracking(model_id, pid)
                self._reap_pid(pid)
                return True, 'already stopped'
            return False, str(e)
        watched_pgid = pgid if signaled_group else None
        for _ in range(40):
            time.sleep(0.2)
            if self._process_gone(pid, watched_pgid):
                self._clear_pid_tracking(model_id, pid)
                return True, 'stopped'
        force_pgid = pgid
        force_signaled_group = signaled_group
        try:
            force_pgid, force_signaled_group = self._send_signal(pid, signal.SIGKILL, use_group)
        except OSError:
            if not self._pid_alive(pid):
                self._clear_pid_tracking(model_id, pid)
                self._reap_pid(pid)
                return True, 'already stopped'
        watched_pgid = force_pgid if force_signaled_group else None
        for _ in range(20):
            time.sleep(0.1)
            if self._process_gone(pid, watched_pgid):
                self._clear_pid_tracking(model_id, pid)
                return True, 'stopped (forced)'
        return False, 'did not stop cleanly'

    def _find_model_pid(self, model: ModelConfig) -> Optional[int]:
        for proc_dir in Path('/proc').iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                pid = int(proc_dir.name)
                if self._pid_matches_model(pid, model) and self._pid_alive(pid):
                    return pid
            except Exception:
                continue
        return None

    def get_pid(self, model: ModelConfig, discover: bool = True, managed_only: bool = False) -> Optional[int]:
        pidfile = self.pidfile(model.id)
        if pidfile.exists():
            pid = self._read_pidfile(model.id)
            if pid and self._pid_alive(pid) and self._pid_matches_model(pid, model):
                return pid
            pgid = self._tracked_pgid(model.id, pid)
            if pgid is not None:
                for group_pid in self._process_group_pids(pgid):
                    if self._pid_matches_model(group_pid, model):
                        return group_pid
                if self._read_pid_metadata(model.id):
                    group_pids = self._process_group_pids(pgid)
                    if group_pids:
                        return group_pids[0]
            self._clear_pid_tracking(model.id, pid)

        if managed_only or not discover:
            return None
        return self._find_model_pid(model)

    def health(self, model: ModelConfig) -> Tuple[str, str]:
        pid = self.get_pid(model)
        url = f'http://{model.host}:{model.port}/v1/models'
        try:
            with request.urlopen(url, timeout=1.2) as resp:
                if 200 <= resp.status < 300:
                    return 'READY', 'responding'
                return 'UNKNOWN', f'http {resp.status}'
        except error.HTTPError as e:
            if e.code == 503:
                return 'LOADING', 'warming up'
            return 'ERROR', f'http {e.code}'
        except Exception as e:
            if pid:
                return 'STARTING', 'process alive, endpoint not ready'
            return 'STOPPED', str(e).split(':')[0]

    def build_command(
        self,
        model: ModelConfig,
        ctx_override: Optional[int] = None,
        parallel_override: Optional[int] = None,
        ngl_override: Optional[int] = None,
        runtime_profile: Optional[RuntimeProfile] = None,
    ) -> List[str]:
        runtime = getattr(model, 'runtime', 'llama.cpp')
        ctx_value = int(ctx_override if ctx_override is not None else model.ctx)
        parallel_value = int(parallel_override if parallel_override is not None else model.parallel)
        ngl_value = int(ngl_override if ngl_override is not None else model.ngl)
        if runtime_profile is not None:
            if ctx_override is None:
                ctx_value = int(runtime_profile.ctx_size or ctx_value)
            if parallel_override is None:
                parallel_value = int(runtime_profile.parallel or parallel_value)
            if ngl_override is None:
                ngl_value = int(runtime_profile.gpu_layers if runtime_profile.gpu_layers is not None else ngl_value)
        if runtime == 'vllm':
            cmd = self.command_prefix(self.vllm_command) + [
                'serve',
                model.path,
                '--host', model.host,
                '--port', str(model.port),
                '--served-model-name', model.alias,
            ]
            if ctx_value > 0:
                cmd += ['--max-model-len', str(ctx_value)]
            cmd += model.extra_args
            return cmd

        engine_profile = self.runtime_profile
        capabilities = self.engine_capabilities(engine_profile)
        measured_runtime = self.runtime_profile_from_model(
            model,
            ctx_value,
            parallel_value,
            ngl_value,
            runtime_profile=runtime_profile,
        )
        cmd = self.command_prefix(self.runtime_server_command('llama.cpp')) + [
            '-m', model.path,
            '--alias', model.alias,
            '--host', model.host,
            '--port', str(model.port),
            '--ctx-size', str(ctx_value),
            '--threads', str(model.threads),
        ]
        cmd += list(engine_profile.default_args)
        cmd += [capabilities.gpu_layers_flag, str(ngl_value)]
        if capabilities.supports_parallel:
            cmd += ['--parallel', str(parallel_value)]
        cmd += [
            '--cache-ram', str(model.cache_ram),
            '--temp', str(model.temp),
        ]
        if model.jinja:
            cmd += ['--jinja']
        cmd += runtime_profile_extra_args(
            engine_profile,
            measured_runtime,
            capabilities,
            existing_args=list(getattr(model, 'extra_args', []) or []),
        )
        return cmd

    def start(self, model: ModelConfig, runtime_profile: Optional[RuntimeProfile] = None) -> Tuple[bool, str]:
        runtime = getattr(model, 'runtime', 'llama.cpp')
        engine_key = (
            runtime_profile.engine_id
            if runtime_profile is not None and getattr(runtime_profile, 'engine_id', '')
            else self.active_engine_key_for_model(model)
        )
        command = self.runtime_server_command(runtime)
        if runtime == 'vllm':
            label = 'vLLM command'
        elif engine_key == 'buun':
            label = 'buun server'
        else:
            label = 'llama-server'
        if not self.command_exists(command):
            return False, f'{label} not found: {command}'
        runtime_ok, runtime_msg = self.runtime_command_ready(runtime, command)
        if not runtime_ok:
            self.append_log(model.id, f'{label} runtime check failed: {runtime_msg}')
            return False, f'{label} cannot run: {runtime_msg}'
        valid, reason = self.validate_model_target(model)
        if not valid:
            return False, reason
        tq_advisory = self.turboquant_session_advisory(model)
        if runtime_profile is not None:
            profile = {
                'ctx': int(runtime_profile.ctx_size or getattr(model, 'ctx', 0) or 0),
                'parallel': max(1, int(runtime_profile.parallel or getattr(model, 'parallel', 1) or 1)),
                'ngl': int(runtime_profile.gpu_layers if runtime_profile.gpu_layers is not None else getattr(model, 'ngl', 0) or 0),
            }
            profile_msg = f'runtime profile {runtime_profile.name or runtime_profile.engine_id}'
        else:
            profile_ok, profile, profile_msg = self.safe_launch_profile(model)
            if not profile_ok:
                return False, profile_msg
        if self.get_pid(model):
            return True, f'{model.id} already running'
        log_path = self.logfile(model.id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.build_command(
            model,
            ctx_override=profile.get('ctx'),
            parallel_override=profile.get('parallel'),
            ngl_override=profile.get('ngl'),
            runtime_profile=runtime_profile,
        )
        env = os.environ.copy()
        env['LLAMA_TUI_MODEL_ID'] = model.id
        env['LLAMA_TUI_OWNER_PID'] = str(os.getpid())
        self.append_log(model.id, f'launch profile: {profile_msg}')
        if tq_advisory:
            self.append_log(model.id, tq_advisory)
        self.append_log(model.id, f'launch command: {shlex.join(command)}')
        try:
            with open(log_path, 'ab') as log_file:
                proc = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
        except OSError as exc:
            self.append_log(model.id, f'launch failed before PID: {exc}')
            return False, f'failed to launch {label}: {exc}'
        self._write_pid_tracking(model, proc.pid, command)
        model.last_good_ctx = profile.get('ctx', model.ctx)
        model.last_good_parallel = profile.get('parallel', model.parallel)
        self.save()
        detail = f'started PID {proc.pid} ({profile_msg})'
        if tq_advisory:
            detail += f' | {tq_advisory}'
        return True, detail

    def _runtime_log_after_last_launch(self, model: ModelConfig, max_lines: int = 400) -> List[str]:
        path = self.logfile(model.id)
        if not path.exists():
            return []
        try:
            lines = path.read_text(errors='replace').splitlines()
        except Exception:
            return []
        for idx in range(len(lines) - 1, -1, -1):
            if '] llama-tui:' in lines[idx] and 'launch command:' in lines[idx]:
                lines = lines[idx + 1:]
                break
        return [line for line in lines[-max_lines:] if '] llama-tui:' not in line]

    def _runtime_log_indicates_ready(self, model: ModelConfig) -> bool:
        text = '\n'.join(self._runtime_log_after_last_launch(model)).lower()
        if not text:
            return False
        loaded = 'model loaded' in text or 'llama model loaded' in text or 'model is loaded' in text
        listening = 'server is listening' in text or 'listening on' in text or 'http server listening' in text
        return loaded and listening

    def wait_until_ready(
        self,
        model: ModelConfig,
        timeout: int = 180,
        cancel_token: Optional[CancelToken] = None,
    ) -> Tuple[bool, str]:
        start = time.time()
        while time.time() - start < timeout:
            check_cancelled(cancel_token)
            status, detail = self.health(model)
            if status == 'READY':
                return True, f'✅ {model.id} is ready on http://{model.host}:{model.port}'
            if self._runtime_log_indicates_ready(model):
                return True, f'✅ {model.id} is ready from server log on http://{model.host}:{model.port}'
            if status == 'STOPPED' and not self.get_pid(model):
                excerpt = '\n'.join(important_log_excerpt(self.logfile(model.id), 24, after_last_launch=True))
                return False, f'❌ {model.id} crashed during startup (log: {self.logfile(model.id)})\n{excerpt}'
            sleep_with_cancel(0.5, cancel_token)
        excerpt = '\n'.join(important_log_excerpt(self.logfile(model.id), 24, after_last_launch=True))
        detail = f'⏳ {model.id} is still loading'
        if excerpt:
            detail += f' (log: {self.logfile(model.id)})\n{excerpt}'
        return False, detail

    def stop(self, model: ModelConfig, managed_only: bool = False) -> Tuple[bool, str]:
        pid = self.get_pid(model, discover=not managed_only, managed_only=managed_only)
        if not pid:
            status = 'STOPPED'
            if not managed_only:
                status, _ = self.health(model)
            self._clear_pid_tracking(model.id)
            if status == 'READY':
                return False, 'running but unmanaged; could not find PID'
            return True, 'already stopped'
        use_group = self._pid_is_tracked(model.id, pid)
        return self._stop_pid(model.id, pid, use_group=use_group)

    def stop_all(self, managed_only: bool = False) -> List[str]:
        msgs = []
        for model in self.models:
            ok, msg = self.stop(model, managed_only=managed_only)
            msgs.append(f'{model.id}: {msg}')
        return msgs

    def cleanup_managed_processes(self) -> List[str]:
        if self._shutdown_cleanup_done:
            return []
        self._shutdown_cleanup_done = True
        msgs = self.stop_all(managed_only=True)
        known_model_ids = {model.id for model in self.models}
        for pidfile in CACHE_DIR.glob('*.pid'):
            model_id = pidfile.stem
            if model_id in known_model_ids:
                continue
            pid = self._read_pidfile(model_id)
            pgid = self._tracked_pgid(model_id, pid)
            group_pids = self._process_group_pids(pgid) if pgid is not None else []
            runtime_pids = [group_pid for group_pid in group_pids if self._pid_looks_like_any_runtime(group_pid)]
            target_pid = pid if pid and self._pid_alive(pid) else None
            if not target_pid:
                target_pid = runtime_pids[0] if runtime_pids else (group_pids[0] if group_pids else None)
            if not target_pid:
                self._clear_pid_tracking(model_id, pid)
                continue
            if not self._read_pid_metadata(model_id) and not self._pid_looks_like_any_runtime(target_pid):
                self._clear_pid_tracking(model_id, pid)
                continue
            ok, msg = self._stop_pid(model_id, target_pid, use_group=True)
            msgs.append(f'{model_id}: {msg}')
        return msgs

    def leave_managed_processes_running(self):
        self._shutdown_cleanup_done = True

    def add_or_update(self, model: ModelConfig):
        self.enrich_model_architecture(model)
        self.enrich_model_turboquant(model)
        for idx, existing in enumerate(self.models):
            if existing.id == model.id:
                if not getattr(model, 'sort_rank', 0):
                    model.sort_rank = int(getattr(existing, 'sort_rank', 0) or idx + 1)
                if not getattr(model, 'last_used_at', ''):
                    model.last_used_at = str(getattr(existing, 'last_used_at', '') or '')
                if not getattr(model, 'favorite', False):
                    model.favorite = bool(getattr(existing, 'favorite', False))
                self.models[idx] = model
                self._normalize_model_ranks()
                self.save()
                return
        if not getattr(model, 'sort_rank', 0):
            model.sort_rank = self.next_sort_rank()
        self.models.append(model)
        self._normalize_model_ranks()
        self.save()

    def delete(self, model_id: str) -> Tuple[bool, str]:
        for i, model in enumerate(self.models):
            if model.id == model_id:
                self.stop(model)
                del self.models[i]
                self._clear_pid_tracking(model_id)
                self._clear_roles(model_id)
                self.save()
                return True, 'deleted'
        return False, 'not found'

    def prune_missing_models(self) -> Tuple[int, List[str]]:
        discovered, _ = self.discover_source_files()
        removed = []
        changed = False
        for model in list(self.models):
            source = self.infer_model_source(model)
            if model.source != source:
                model.source = source
                changed = True
            normalized = str(self.normalize_model_path(model.path))
            path_exists = Path(model.path).expanduser().exists()

            runtime = getattr(model, 'runtime', 'llama.cpp')
            if source == 'manual':
                if runtime == 'vllm':
                    target = (model.path or '').strip()
                    should_remove = not target or (not Path(target).expanduser().exists() and not looks_like_model_reference(target))
                else:
                    should_remove = (not path_exists) or (not is_real_model_file(Path(model.path)))
            else:
                should_remove = normalized not in discovered

            if should_remove:
                self.delete(model.id)
                removed.append(model.id)
                changed = True

        if changed:
            self.save()
        return len(removed), removed

    def detect_models(self) -> Tuple[int, List[str]]:
        discovered, notes = self.discover_source_files()
        existing_paths = {self.normalize_model_ref(m.path): m for m in self.models}
        added = []
        changed = False

        for resolved, (gguf, source) in discovered.items():
            if resolved in existing_paths:
                model = existing_paths[resolved]
                if model.source != source:
                    model.source = source
                    changed = True
                continue

            model = detected_model_from_path(gguf, self.models, source=source)
            self.add_or_update(model)
            existing_paths[resolved] = model
            added.append(model.id)
            changed = True

        removed_count, removed = self.prune_missing_models()
        if changed:
            self.save()

        parts = []
        if added:
            parts.append('added: ' + ', '.join(added[:6]))
        if removed:
            parts.append('pruned: ' + ', '.join(removed[:6]))
        if parts:
            summary = ' | '.join(parts)
            if len(added) > 6 or len(removed) > 6:
                summary += ' ...'
            return len(added), [summary]
        if notes:
            return 0, notes
        return 0, ['No new GGUFs found.']

    def next_port(self, start: int = 8080) -> int:
        used = {m.port for m in self.models}
        port = start
        while port in used:
            port += 1
        return port

    def _backup_export_file(self, path: Path, backup_dir_raw: str) -> Optional[Path]:
        if not path.exists():
            return None
        backup_dir = Path(backup_dir_raw or (path.parent / 'backups')).expanduser()
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_path = backup_dir / f'{path.stem}.{stamp}{path.suffix}'
        shutil.copy2(path, backup_path)
        return backup_path

    def _clear_roles(self, model_id: str):
        for attr in ('default_model_id', 'small_model_id', 'build_model_id', 'plan_model_id'):
            if getattr(self.opencode, attr) == model_id:
                setattr(self.opencode, attr, '')
        for attr in ('default_model_id', 'code_model_id'):
            if getattr(self.hermes, attr) == model_id:
                setattr(self.hermes, attr, '')

    def set_role(self, role: str, model_id: str):
        mapping = {
            'main': 'default_model_id',
            'small': 'small_model_id',
            'build': 'build_model_id',
            'plan': 'plan_model_id',
        }
        attr = mapping[role]
        setattr(self.opencode, attr, model_id)
        self.save()

    def role_badges(self, model_id: str) -> str:
        badges = []
        if self.opencode.default_model_id == model_id:
            badges.append('M')
        if self.opencode.small_model_id == model_id:
            badges.append('S')
        if self.opencode.build_model_id == model_id:
            badges.append('B')
        if self.opencode.plan_model_id == model_id:
            badges.append('P')
        if self.hermes.default_model_id == model_id:
            badges.append('H')
        if self.hermes.code_model_id == model_id:
            badges.append('C')
        return ''.join(badges) or '-'

    def generate_opencode(self) -> Tuple[bool, str]:
        path = Path(self.opencode.path).expanduser() if self.opencode.path else None
        if not path:
            return False, 'Set opencode.path first in settings.'

        enabled_models = [m for m in self.models if m.enabled]
        if not enabled_models:
            return False, 'No enabled models to export.'

        default_model = self.get_model(self.opencode.default_model_id) or (enabled_models[0] if enabled_models else None)
        small_model = self.get_model(self.opencode.small_model_id) or (enabled_models[1] if len(enabled_models) > 1 else default_model)
        build_model = self.get_model(self.opencode.build_model_id) or default_model
        plan_model = self.get_model(self.opencode.plan_model_id) or build_model

        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = {}
            self._backup_export_file(path, self.opencode.backup_dir)

        instructions = existing.get('instructions', self.opencode.instructions)
        build_prompt = existing.get('agent', {}).get('build', {}).get('prompt', self.opencode.build_prompt)
        plan_prompt = existing.get('agent', {}).get('plan', {}).get('prompt', self.opencode.plan_prompt)

        provider = {}
        for model in enabled_models:
            provider_key = self.opencode_provider_key(model)
            runtime_label = 'vLLM' if getattr(model, 'runtime', 'llama.cpp') == 'vllm' else 'llama.cpp'
            provider[provider_key] = {
                'npm': '@ai-sdk/openai-compatible',
                'name': f'{runtime_label} {model.name}',
                'options': {
                    'baseURL': f'http://{model.host}:{model.port}/v1',
                    'timeout': self.opencode.timeout,
                    'chunkTimeout': self.opencode.chunk_timeout,
                },
                'models': {
                    model.alias: {
                        'name': model.name,
                        'limit': {
                            'context': model.ctx,
                            'output': model.output,
                        }
                    }
                }
            }

        def ref(model: ModelConfig) -> str:
            return self.opencode_model_ref(model)

        config = {
            '$schema': existing.get('$schema', 'https://opencode.ai/config.json'),
            'instructions': instructions,
            'agent': {
                'build': {
                    'model': ref(build_model),
                    'prompt': build_prompt,
                },
                'plan': {
                    'model': ref(plan_model),
                    'prompt': plan_prompt,
                },
            },
            'provider': provider,
            'model': ref(default_model),
            'small_model': ref(small_model),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + '\n')
        self.save()
        return True, f'Generated {path}'

    def continue_role_models(self, enabled_models: List[ModelConfig]) -> Tuple[ModelConfig, ModelConfig, ModelConfig]:
        default_model = (
            self.get_model(getattr(self.continue_settings, 'default_model_id', ''))
            or self.get_model(self.opencode.default_model_id)
            or enabled_models[0]
        )
        edit_model = (
            self.get_model(getattr(self.continue_settings, 'edit_model_id', ''))
            or self.get_model(self.opencode.build_model_id)
            or default_model
        )
        autocomplete_model = (
            self.get_model(getattr(self.continue_settings, 'autocomplete_model_id', ''))
            or self.get_model(self.opencode.small_model_id)
            or self.get_model(self.hermes.code_model_id)
            or (enabled_models[1] if len(enabled_models) > 1 else default_model)
        )
        return default_model, edit_model, autocomplete_model

    def _continue_managed_model_lines(self, enabled_models: List[ModelConfig]) -> List[str]:
        default_model, edit_model, autocomplete_model = self.continue_role_models(enabled_models)
        ordered_models: List[ModelConfig] = []
        seen_ids = set()
        for model in (default_model, edit_model, autocomplete_model, *enabled_models):
            if model.id in seen_ids:
                continue
            seen_ids.add(model.id)
            ordered_models.append(model)

        used_names: set[str] = set()

        def model_display_name(model: ModelConfig) -> str:
            base = (model.name or model.id or model.alias or 'Local Model').strip() or 'Local Model'
            candidate = base
            suffix = 2
            while candidate in used_names:
                candidate = f'{base} ({suffix})'
                suffix += 1
            used_names.add(candidate)
            return candidate

        def model_roles(model: ModelConfig) -> List[str]:
            roles: List[str] = []
            if model.id == default_model.id:
                roles.append('chat')
            if model.id == edit_model.id:
                roles.extend(['edit', 'apply'])
            if model.id == autocomplete_model.id:
                roles.append('autocomplete')
            if not roles:
                roles.append('chat')
            deduped: List[str] = []
            for role in roles:
                if role not in deduped:
                    deduped.append(role)
            return deduped

        def autocomplete_prompt_tokens(model: ModelConfig) -> int:
            return min(2048, max(256, context_per_slot(model)))

        lines = [
            CONTINUE_MANAGED_BEGIN,
            '  # Generated by llama-tui. Edit models in llama-tui, then regenerate.',
        ]
        for model in ordered_models:
            roles = model_roles(model)
            lines.extend([
                f'  - name: {yaml_quote(model_display_name(model))}',
                '    provider: "openai"',
                f'    model: {yaml_quote(self.continue_model_ref(model))}',
                f'    apiBase: {yaml_quote(self.continue_base_url(model))}',
                '    apiKey: "no-key-required"',
                '    roles:',
            ])
            lines.extend(f'      - {role}' for role in roles)
            lines.extend([
                '    defaultCompletionOptions:',
                f'      contextLength: {max(1, context_per_slot(model))}',
                f'      maxTokens: {max(1, int(getattr(model, "output", 0) or 0))}',
                f'      temperature: {float(getattr(model, "temp", 0.7) or 0.7)}',
            ])
            if 'autocomplete' in roles:
                lines.extend([
                    '    autocompleteOptions:',
                    '      debounceDelay: 250',
                    f'      maxPromptTokens: {autocomplete_prompt_tokens(model)}',
                    '      onlyMyCode: true',
                ])
        lines.append(CONTINUE_MANAGED_END)
        return lines

    def _render_continue_full_config(self, managed_model_lines: List[str]) -> str:
        lines = [
            '# Generated by llama-tui. Existing files are backed up before overwrite.',
            f'name: {yaml_quote("llama-tui Local Models")}',
            f'version: {yaml_quote("1.0.0")}',
            'schema: "v1"',
            'models:',
            *managed_model_lines,
        ]
        return '\n'.join(lines) + '\n'

    def _merge_continue_config_text(self, existing_text: str, managed_model_lines: List[str]) -> str:
        if not existing_text.strip():
            return self._render_continue_full_config(managed_model_lines)
        lines = existing_text.splitlines()
        try:
            begin = lines.index(CONTINUE_MANAGED_BEGIN)
            end = lines.index(CONTINUE_MANAGED_END, begin + 1)
        except ValueError:
            begin = -1
            end = -1
        if begin >= 0 and end >= begin:
            merged = lines[:begin] + managed_model_lines + lines[end + 1:]
            return '\n'.join(merged).rstrip() + '\n'

        models_index = next((idx for idx, line in enumerate(lines) if line.strip() == 'models:' and not line.startswith(' ')), -1)
        if models_index >= 0:
            merged = lines[:models_index + 1] + managed_model_lines + lines[models_index + 1:]
            return '\n'.join(merged).rstrip() + '\n'

        merged = lines
        if merged and merged[-1].strip():
            merged.append('')
        merged.extend(['models:', *managed_model_lines])
        return '\n'.join(merged).rstrip() + '\n'

    def generate_continue_config(self) -> Tuple[bool, str]:
        path = Path(self.continue_settings.path).expanduser() if self.continue_settings.path else None
        if not path:
            return False, 'Set continue.path first in settings.'

        enabled_models = [m for m in self.models if m.enabled]
        if not enabled_models:
            return False, 'No enabled models to export.'

        managed_lines = self._continue_managed_model_lines(enabled_models)
        existing_text = ''
        if path.exists():
            existing_text = path.read_text(encoding='utf-8')
            self._backup_export_file(path, self.continue_settings.backup_dir)

        merge_mode = getattr(self.continue_settings, 'merge_mode', 'preserve_sections') or 'preserve_sections'
        if merge_mode == 'managed_file' or not path.exists():
            output = self._render_continue_full_config(managed_lines)
        else:
            output = self._merge_continue_config_text(existing_text, managed_lines)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding='utf-8')
        self.save()
        return True, f'Generated {path}'
