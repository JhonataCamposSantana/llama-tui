#!/usr/bin/env python3
import curses
import json
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request, error

SCRIPT_DIR = Path(__file__).resolve().parent
APP_NAME = 'llama-tui'
CONFIG_DIR = Path.home() / '.config' / APP_NAME
DATA_DIR = Path.home() / '.local' / 'share' / APP_NAME
CACHE_DIR = Path.home() / '.cache' / APP_NAME
DEFAULT_CONFIG_PATH = Path(os.environ['LLAMA_TUI_CONFIG']).expanduser() if os.environ.get('LLAMA_TUI_CONFIG') else (CONFIG_DIR / 'models.json')
DEFAULT_HOST = '127.0.0.1'
DEFAULT_HF_CACHE = Path('/var/home/jcampos/.cache/huggingface/hub') if Path('/var/home/jcampos/.cache/huggingface/hub').exists() else Path.home() / '.cache' / 'huggingface' / 'hub'
DEFAULT_LLMFIT_CACHE = Path(os.environ.get('LLMFIT_CACHE_ROOT', Path.home() / '.cache' / 'llmfit' / 'models')).expanduser()
DEFAULT_LLM_MODELS_CACHE = Path(os.environ.get('LLM_MODELS_CACHE_ROOT', Path.home() / '.cache' / 'llm-models')).expanduser()
DEFAULT_LLAMA_SERVER = str(Path('/var/home/jcampos/llama.cpp/build/bin/llama-server') if Path('/var/home/jcampos/llama.cpp/build/bin/llama-server').exists() else Path.home() / 'llama.cpp' / 'build' / 'bin' / 'llama-server')
DEFAULT_VLLM_COMMAND = os.environ.get('VLLM_COMMAND', 'vllm')
REFRESH_SECONDS = 2.0
LOGO = [
    "  _ _                         _         _       _ ",
    " | | | __ _ _ __ ___   __ _  | |_ _   _(_)_   _| |",
    " | | |/ _` | '_ ` _ \\ / _` | | __| | | | | | | |",
    " | | | (_| | | | | | | (_| | | |_| |_| | | |_| |_|",
    " |_|_|\\__,_|_| |_| |_|\\__,_|  \\__|\\__,_|_|\\__,_(_)",
]


@dataclass
class ModelConfig:
    id: str
    name: str
    path: str
    alias: str
    port: int
    host: str = DEFAULT_HOST
    ctx: int = 8192
    threads: int = 6
    ngl: int = 999
    temp: float = 0.7
    parallel: int = 1
    cache_ram: int = 0
    flash_attn: bool = True
    jinja: bool = True
    output: int = 4096
    optimize_mode: str = 'max_context_safe'
    optimize_tier: str = 'moderate'
    ctx_min: int = 2048
    ctx_max: int = 131072
    memory_reserve_percent: int = 25
    last_good_ctx: int = 0
    last_good_parallel: int = 0
    last_benchmark_tokens_per_sec: float = 0.0
    last_benchmark_seconds: float = 0.0
    last_benchmark_profile: str = ''
    enabled: bool = True
    runtime: str = 'llama.cpp'
    source: str = 'manual'
    extra_args: List[str] = field(default_factory=list)


@dataclass
class OpencodeSettings:
    path: str = ''
    backup_dir: str = ''
    timeout: int = 600000
    chunk_timeout: int = 60000
    instructions: List[str] = field(default_factory=lambda: ['./instructions/warm.md'])
    build_prompt: str = '{file:./prompts/friendly-build.txt}'
    plan_prompt: str = '{file:./prompts/friendly-build.txt}'
    default_model_id: str = ''
    small_model_id: str = ''
    build_model_id: str = ''
    plan_model_id: str = ''


@dataclass
class HardwareProfile:
    cpu_logical: int = 0
    cpu_physical: int = 0
    memory_total: int = 0
    memory_available: int = 0
    gpu_name: str = ''
    gpu_memory_total: int = 0
    gpu_memory_free: int = 0
    gpu_error: str = ''

    def has_usable_gpu(self) -> bool:
        return self.gpu_memory_free > 0

    def short_summary(self) -> str:
        total_gib = bytes_to_gib(self.memory_total)
        avail_gib = bytes_to_gib(self.memory_available)
        cpu = f'cpu={self.cpu_physical or "?"}c/{self.cpu_logical or "?"}t'
        ram = f'ram={avail_gib:.1f}/{total_gib:.1f}GiB'
        if self.has_usable_gpu():
            gpu_free = bytes_to_gib(self.gpu_memory_free)
            gpu_total = bytes_to_gib(self.gpu_memory_total)
            gpu = f'gpu={self.gpu_name or "detected"} {gpu_free:.1f}/{gpu_total:.1f}GiB'
        elif self.gpu_error:
            gpu = 'gpu=unavailable'
        else:
            gpu = 'gpu=none'
        return f'{cpu} {ram} {gpu}'


class AppConfig:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.llama_server = os.environ.get('LLAMA_SERVER', DEFAULT_LLAMA_SERVER)
        self.vllm_command = DEFAULT_VLLM_COMMAND
        self.hf_cache_root = str(DEFAULT_HF_CACHE)
        self.llmfit_cache_root = str(DEFAULT_LLMFIT_CACHE)
        self.llm_models_cache_root = str(DEFAULT_LLM_MODELS_CACHE)
        self.opencode = OpencodeSettings(
            path='',
            backup_dir=str(CONFIG_DIR / 'backups'),
        )
        self.models: List[ModelConfig] = []
        self._hardware_profile: Optional[HardwareProfile] = None
        self._hardware_profile_at = 0.0
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        if not self.config_path.exists():
            self.save()
            return
        data = json.loads(self.config_path.read_text())
        self.llama_server = data.get('llama_server', self.llama_server)
        self.vllm_command = data.get('vllm_command', self.vllm_command)
        self.hf_cache_root = data.get('hf_cache_root', self.hf_cache_root)
        self.llmfit_cache_root = data.get('llmfit_cache_root', self.llmfit_cache_root)
        self.llm_models_cache_root = data.get('llm_models_cache_root', self.llm_models_cache_root)
        self.opencode = OpencodeSettings(**data.get('opencode', {}))
        self.models = [ModelConfig(**item) for item in data.get('models', [])]
        filtered_models = [m for m in self.models if is_registered_model_entry(m)]
        roots_changed = False
        for m in filtered_models:
            inferred = self.infer_model_source(m)
            if m.source != inferred:
                m.source = inferred
                roots_changed = True
        if len(filtered_models) != len(self.models) or roots_changed:
            self.models = filtered_models
            self.save()
        else:
            self.models = filtered_models

    def save(self):
        data = {
            'llama_server': self.llama_server,
            'vllm_command': self.vllm_command,
            'hf_cache_root': self.hf_cache_root,
            'llmfit_cache_root': self.llmfit_cache_root,
            'llm_models_cache_root': self.llm_models_cache_root,
            'opencode': asdict(self.opencode),
            'models': [asdict(m) for m in self.models],
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(data, indent=2) + '\n')

    def pidfile(self, model_id: str) -> Path:
        return CACHE_DIR / f'{model_id}.pid'

    def logfile(self, model_id: str) -> Path:
        return CACHE_DIR / f'{model_id}.log'

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        return next((m for m in self.models if m.id == model_id), None)


    def hardware_profile(self, refresh: bool = False) -> HardwareProfile:
        now = time.time()
        if refresh or self._hardware_profile is None or now - self._hardware_profile_at > 30:
            self._hardware_profile = benchmark_current_hardware()
            self._hardware_profile_at = now
        return self._hardware_profile

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
        prefixes = self.command_prefix(self.llama_server)
        target = os.path.basename(prefixes[0]) if prefixes else os.path.basename(self.llama_server)
        return any(os.path.basename(part) == target for part in parts)

    def available_memory_bytes(self) -> int:
        return read_meminfo_bytes().get('MemAvailable', 0)

    def _estimate_kv_bytes_per_token(self, model: ModelConfig) -> int:
        return estimate_kv_bytes_per_token(model)

    def safe_launch_profile(self, model: ModelConfig) -> Tuple[bool, Dict[str, int], str]:
        mode = (getattr(model, 'optimize_mode', 'max_context_safe') or 'max_context_safe').strip().lower()
        requested_ctx = max(1, int(getattr(model, 'ctx', 8192)))
        requested_parallel = max(1, int(getattr(model, 'parallel', 1)))
        if mode == 'manual':
            return True, {'ctx': requested_ctx, 'parallel': requested_parallel}, 'manual mode'

        profile = self.hardware_profile(refresh=True)
        mem_available = profile.memory_available or self.available_memory_bytes()
        if mem_available <= 0:
            return True, {'ctx': requested_ctx, 'parallel': requested_parallel}, 'safe mode (memory probe unavailable)'

        reserve_pct = max(5, min(60, int(getattr(model, 'memory_reserve_percent', 25))))
        usable = int(mem_available * ((100 - reserve_pct) / 100.0))
        if usable <= 0:
            return False, {}, f'not enough free memory (reserve={reserve_pct}%)'

        kv_per_token = self._estimate_kv_bytes_per_token(model)
        cap_parallel = max(1, requested_parallel)
        safe_ctx_by_mem = max(1, usable // (kv_per_token * cap_parallel))
        min_ctx = max(256, int(getattr(model, 'ctx_min', 2048)))
        max_ctx = max(min_ctx, int(getattr(model, 'ctx_max', 131072)))
        applied_ctx = max(min_ctx, min(requested_ctx, max_ctx, safe_ctx_by_mem))
        if applied_ctx < min_ctx:
            return False, {}, f'not enough memory for minimum ctx={min_ctx} (available={mem_available // 1024**2} MiB)'

        notes = [f'safe mode reserve={reserve_pct}%']
        if applied_ctx != requested_ctx:
            notes.append(f'ctx {requested_ctx}→{applied_ctx}')
        return True, {'ctx': applied_ctx, 'parallel': cap_parallel}, ', '.join(notes)

    def managed_roots(self) -> Dict[str, Path]:
        return {
            'huggingface': Path(self.hf_cache_root).expanduser(),
            'llmfit': Path(self.llmfit_cache_root).expanduser(),
            'llm-models': Path(self.llm_models_cache_root).expanduser(),
        }

    def normalize_model_path(self, path: str | Path) -> Path:
        return Path(path).expanduser().resolve(strict=False)

    def infer_model_source(self, model: ModelConfig) -> str:
        if getattr(model, 'runtime', 'llama.cpp') == 'vllm':
            return 'manual'
        if getattr(model, 'source', '') in ('manual', 'huggingface', 'llmfit', 'llm-models'):
            existing = getattr(model, 'source', '')
            if existing and existing != 'manual':
                return existing
        p = self.normalize_model_path(model.path)
        for source, root in self.managed_roots().items():
            try:
                p.relative_to(root.resolve(strict=False))
                return source
            except Exception:
                continue
        return 'manual'

    def discover_source_files(self) -> Tuple[Dict[str, Tuple[Path, str]], List[str]]:
        discovered: Dict[str, Tuple[Path, str]] = {}
        notes: List[str] = []

        for source, root in self.managed_roots().items():
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
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            parts = [p.decode(errors='ignore') for p in raw.split(b'\x00') if p]
            return self._command_matches_runtime(parts, runtime)
        except Exception:
            return False

    def _find_model_pid(self, model: ModelConfig) -> Optional[int]:
        port = str(model.port)
        alias = model.alias
        path = model.path
        runtime = getattr(model, 'runtime', 'llama.cpp')
        for proc_dir in Path('/proc').iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                pid = int(proc_dir.name)
                raw = (proc_dir / 'cmdline').read_bytes()
                parts = [p.decode(errors='ignore') for p in raw.split(b'\x00') if p]
                if not parts or not self._command_matches_runtime(parts, runtime):
                    continue
                joined = '\x00'.join(parts)
                path_match = path in parts or path in joined
                if (
                    f'\x00--port\x00{port}\x00' in joined
                    or f'\x00--alias\x00{alias}\x00' in joined
                    or f'\x00--served-model-name\x00{alias}\x00' in joined
                    or path_match
                ):
                    state = self._proc_state(pid)
                    if state not in ('Z', 'X'):
                        return pid
            except Exception:
                continue
        return None

    def get_pid(self, model: ModelConfig) -> Optional[int]:
        pidfile = self.pidfile(model.id)
        if pidfile.exists():
            try:
                pid = int(pidfile.read_text().strip())
                os.kill(pid, 0)
                state = self._proc_state(pid)
                if state in ('Z', 'X'):
                    pidfile.unlink(missing_ok=True)
                elif self._pid_looks_like_runtime(pid, getattr(model, 'runtime', 'llama.cpp')):
                    return pid
                else:
                    pidfile.unlink(missing_ok=True)
            except Exception:
                pidfile.unlink(missing_ok=True)

        pid = self._find_model_pid(model)
        if pid:
            pidfile.write_text(str(pid))
            return pid
        return None

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

    def build_command(self, model: ModelConfig, ctx_override: Optional[int] = None, parallel_override: Optional[int] = None) -> List[str]:
        runtime = getattr(model, 'runtime', 'llama.cpp')
        ctx_value = int(ctx_override if ctx_override is not None else model.ctx)
        parallel_value = int(parallel_override if parallel_override is not None else model.parallel)
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

        cmd = self.command_prefix(self.llama_server) + [
            '-m', model.path,
            '--alias', model.alias,
            '--host', model.host,
            '--port', str(model.port),
            '--ctx-size', str(ctx_value),
            '--threads', str(model.threads),
            '--n-gpu-layers', str(model.ngl),
            '--parallel', str(parallel_value),
            '--cache-ram', str(model.cache_ram),
            '--temp', str(model.temp),
        ]
        if model.flash_attn:
            cmd += ['--flash-attn', 'on']
        if model.jinja:
            cmd += ['--jinja']
        cmd += model.extra_args
        return cmd

    def start(self, model: ModelConfig) -> Tuple[bool, str]:
        runtime = getattr(model, 'runtime', 'llama.cpp')
        command = self.vllm_command if runtime == 'vllm' else self.llama_server
        if not self.command_exists(command):
            label = 'vLLM command' if runtime == 'vllm' else 'llama-server'
            return False, f'{label} not found: {command}'
        valid, reason = self.validate_model_target(model)
        if not valid:
            return False, reason
        profile_ok, profile, profile_msg = self.safe_launch_profile(model)
        if not profile_ok:
            return False, profile_msg
        if self.get_pid(model):
            return True, f'{model.id} already running'
        log_path = self.logfile(model.id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'ab') as log_file:
            proc = subprocess.Popen(
                self.build_command(
                    model,
                    ctx_override=profile.get('ctx'),
                    parallel_override=profile.get('parallel'),
                ),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.pidfile(model.id).write_text(str(proc.pid))
        model.last_good_ctx = profile.get('ctx', model.ctx)
        model.last_good_parallel = profile.get('parallel', model.parallel)
        self.save()
        return True, f'started PID {proc.pid} ({profile_msg})'

    def wait_until_ready(self, model: ModelConfig, timeout: int = 180) -> Tuple[bool, str]:
        start = time.time()
        while time.time() - start < timeout:
            status, detail = self.health(model)
            if status == 'READY':
                return True, f'✅ {model.id} is ready on http://{model.host}:{model.port}'
            if status == 'STOPPED' and not self.get_pid(model):
                tail = '\n'.join(tail_text(self.logfile(model.id), 8))
                return False, f'❌ {model.id} crashed during startup\n{tail}'
            time.sleep(0.5)
        return False, f'⏳ {model.id} is still loading'

    def stop(self, model: ModelConfig) -> Tuple[bool, str]:
        pid = self.get_pid(model)
        if not pid:
            status, _ = self.health(model)
            self.pidfile(model.id).unlink(missing_ok=True)
            if status == 'READY':
                return False, 'running but unmanaged; could not find PID'
            return True, 'already stopped'
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            self.pidfile(model.id).unlink(missing_ok=True)
            return False, str(e)
        for _ in range(25):
            time.sleep(0.2)
            if not self.get_pid(model):
                self.pidfile(model.id).unlink(missing_ok=True)
                return True, 'stopped'
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        for _ in range(10):
            time.sleep(0.1)
            if not self.get_pid(model):
                self.pidfile(model.id).unlink(missing_ok=True)
                return True, 'stopped (forced)'
        return False, 'did not stop cleanly'

    def stop_all(self) -> List[str]:
        msgs = []
        for model in self.models:
            ok, msg = self.stop(model)
            msgs.append(f'{model.id}: {msg}')
        return msgs

    def add_or_update(self, model: ModelConfig):
        for idx, existing in enumerate(self.models):
            if existing.id == model.id:
                self.models[idx] = model
                self.save()
                return
        self.models.append(model)
        self.models.sort(key=lambda m: m.port)
        self.save()

    def delete(self, model_id: str) -> Tuple[bool, str]:
        for i, model in enumerate(self.models):
            if model.id == model_id:
                self.stop(model)
                del self.models[i]
                self.pidfile(model_id).unlink(missing_ok=True)
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

    def _clear_roles(self, model_id: str):
        for attr in ('default_model_id', 'small_model_id', 'build_model_id', 'plan_model_id'):
            if getattr(self.opencode, attr) == model_id:
                setattr(self.opencode, attr, '')

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
            backup_dir = Path(self.opencode.backup_dir or (path.parent / 'backups')).expanduser()
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            backup_path = backup_dir / f'{path.stem}.{stamp}{path.suffix}'
            shutil.copy2(path, backup_path)

        instructions = existing.get('instructions', self.opencode.instructions)
        build_prompt = existing.get('agent', {}).get('build', {}).get('prompt', self.opencode.build_prompt)
        plan_prompt = existing.get('agent', {}).get('plan', {}).get('prompt', self.opencode.plan_prompt)

        provider = {}
        for model in enabled_models:
            provider_key = f'local-{model.id}'
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
            return f'local-{model.id}/{model.alias}'

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


# ---------- hardware helpers ----------

def bytes_to_gib(value: int) -> float:
    return float(value or 0) / float(1024**3)


def read_meminfo_bytes() -> Dict[str, int]:
    meminfo = Path('/proc/meminfo')
    values: Dict[str, int] = {}
    if not meminfo.exists():
        return values
    try:
        for line in meminfo.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                values[parts[0].rstrip(':')] = int(parts[1]) * 1024
    except Exception:
        return {}
    return values


def detect_cpu_counts() -> Tuple[int, int]:
    logical = os.cpu_count() or 1
    cpuinfo = Path('/proc/cpuinfo')
    if not cpuinfo.exists():
        return logical, max(1, logical // 2)

    physical_cores = set()
    current_physical = ''
    current_core = ''
    try:
        for line in cpuinfo.read_text(errors='replace').splitlines() + ['']:
            if not line.strip():
                if current_core:
                    physical_cores.add((current_physical or '0', current_core))
                current_physical = ''
                current_core = ''
                continue
            if ':' not in line:
                continue
            key, value = [part.strip() for part in line.split(':', 1)]
            if key == 'physical id':
                current_physical = value
            elif key == 'core id':
                current_core = value
    except Exception:
        return logical, max(1, logical // 2)

    physical = len(physical_cores) if physical_cores else max(1, logical // 2)
    return logical, max(1, min(physical, logical))


def probe_nvidia_gpu() -> Tuple[str, int, int, str]:
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return '', 0, 0, ''
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                '--query-gpu=name,memory.total,memory.free',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        return '', 0, 0, f'nvidia-smi failed: {exc}'

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or '').strip().splitlines()
        return '', 0, 0, detail[0] if detail else 'nvidia-smi failed'

    best_name = ''
    best_total = 0
    best_free = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            total = int(float(parts[1])) * 1024**2
            free = int(float(parts[2])) * 1024**2
        except ValueError:
            continue
        if free > best_free:
            best_name = parts[0]
            best_total = total
            best_free = free
    return best_name, best_total, best_free, ''


def benchmark_current_hardware() -> HardwareProfile:
    logical, physical = detect_cpu_counts()
    mem = read_meminfo_bytes()
    gpu_name, gpu_total, gpu_free, gpu_error = probe_nvidia_gpu()
    return HardwareProfile(
        cpu_logical=logical,
        cpu_physical=physical,
        memory_total=mem.get('MemTotal', 0),
        memory_available=mem.get('MemAvailable', 0),
        gpu_name=gpu_name,
        gpu_memory_total=gpu_total,
        gpu_memory_free=gpu_free,
        gpu_error=gpu_error,
    )


def model_file_size(model: ModelConfig) -> int:
    target = Path(getattr(model, 'path', '') or '').expanduser()
    try:
        return target.stat().st_size if target.exists() else 0
    except Exception:
        return 0


def estimate_kv_bytes_per_token(model: ModelConfig) -> int:
    runtime = getattr(model, 'runtime', 'llama.cpp')
    if runtime == 'vllm':
        return 32768
    size = model_file_size(model)
    if size <= 0:
        return 32768
    if size < 8 * 1024**3:
        return 16384
    if size < 20 * 1024**3:
        return 32768
    return 65536


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
    if not profile or profile.memory_available <= 0:
        return ctx_max

    reserve_pct = max(5, min(70, reserve_pct))
    usable = int(profile.memory_available * ((100 - reserve_pct) / 100.0))
    if getattr(model, 'runtime', 'llama.cpp') != 'vllm' and not model_likely_fits_gpu(model, profile, getattr(model, 'optimize_tier', 'moderate')):
        usable -= model_file_size(model)
    if usable <= 0:
        return ctx_min

    kv_per_token = max(1, estimate_kv_bytes_per_token(model))
    safe_ctx = usable // (kv_per_token * max(1, parallel))
    return max(ctx_min, min(ctx_max, int(safe_ctx)))


def select_best_tier(model: ModelConfig, profile: Optional[HardwareProfile]) -> str:
    if not profile:
        return 'moderate'

    size = model_file_size(model)
    available = profile.memory_available
    total = profile.memory_total
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
    if profile and profile.has_usable_gpu() and model_likely_fits_gpu(model, profile, 'moderate'):
        return 'tokens_per_sec'
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
        elif not profile.has_usable_gpu():
            model.ngl = 0


# ---------- detection helpers ----------

def slugify(text: str) -> str:
    text = text.lower().replace('.', '-').replace('_', '-')
    text = re.sub(r'[^a-z0-9-]+', '-', text)
    text = re.sub(r'-+', '-', text).strip('-')
    return text or 'model'


def pretty_name_from_filename(stem: str) -> str:
    text = stem.replace('_', ' ').replace('-', ' ')
    text = re.sub(r'\bq(\d(?:\.\d+)?)\b', lambda m: f'Q{m.group(1)}', text, flags=re.I)
    text = re.sub(r'\biq(\d(?:\.\d+)?)\b', lambda m: f'IQ{m.group(1)}', text, flags=re.I)
    return ' '.join(part if part.isupper() else part.capitalize() for part in text.split())


def choose_defaults(name: str) -> Tuple[int, int, float]:
    low = name.lower()
    if '30b-a3b' in low or 'a3b' in low:
        return 8192, 16, 0.45
    if 'gemma' in low and 'e4b' in low:
        return 8192, 0, 0.70
    if '9b' in low or 'opus' in low:
        return 32768, 999, 0.55
    if 'coder' in low:
        return 16384, 999, 0.45
    return 8192, 999, 0.65


def is_real_model_file(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() != '.gguf':
        return False
    if 'mmproj' in name:
        return False
    return True


def looks_like_model_reference(value: str) -> bool:
    value = (value or '').strip()
    if not value or ' ' in value:
        return False
    if value.lower().endswith('.gguf'):
        return False
    if value.startswith('hf://'):
        return True
    return bool(re.match(r'^[^/\s]+/[^\s]+$', value))


def is_registered_model_entry(model: ModelConfig) -> bool:
    runtime = getattr(model, 'runtime', 'llama.cpp')
    target = (getattr(model, 'path', '') or '').strip()
    if runtime == 'vllm':
        if not target:
            return False
        p = Path(target).expanduser()
        return p.exists() or looks_like_model_reference(target)
    return is_real_model_file(Path(target))


def display_runtime(model: ModelConfig) -> str:
    runtime = (getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp').strip().lower()
    mapping = {
        'llama.cpp': 'llama.cpp',
        'vllm': 'vLLM',
        'ollama': 'Ollama',
    }
    return mapping.get(runtime, runtime or 'unknown')


def extract_quant(model: ModelConfig) -> str:
    text = ' '.join([
        getattr(model, 'name', '') or '',
        getattr(model, 'alias', '') or '',
        getattr(model, 'path', '') or '',
    ])
    pattern = re.compile(r'(?i)(iq\d+(?:_[a-z0-9]+)+|iq\d+(?:\.\d+)?|q\d+(?:_[a-z0-9]+)+|q\d+(?:\.\d+)?(?:_[a-z0-9]+)*|bf16|fp16|f16|bf8|fp8|f32|fp32|int8|int4)')
    match = pattern.search(text)
    if not match:
        return '-'
    return match.group(1).upper()


def classify_model_type(model: ModelConfig) -> str:
    text = ' '.join([
        getattr(model, 'name', '') or '',
        getattr(model, 'alias', '') or '',
        getattr(model, 'path', '') or '',
    ]).lower()

    if (
        re.search(r'\bmoe\b', text)
        or re.search(r'\ba\d+b\b', text)
        or any(token in text for token in ('mixtral', 'switch', 'mixture-of-experts', 'mixture of experts'))
    ):
        return 'MoE'

    runtime = (getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp').strip().lower()
    if runtime == 'vllm':
        return 'GPU'
    if runtime == 'ollama':
        return 'GPU'

    try:
        return 'CPU' if int(getattr(model, 'ngl', 0)) == 0 else 'GPU'
    except Exception:
        return 'Dense'


def detected_model_from_path(path: Path, existing_models: List[ModelConfig], source: str = 'manual') -> ModelConfig:
    stem = path.stem
    name = pretty_name_from_filename(stem)
    repo_dir = path.parts[-4] if len(path.parts) >= 4 else ''
    repo_hint = repo_dir.replace('models--', '').replace('--', '-')
    base_id = slugify(name)
    if 'qwen3-30b-a3b' in stem.lower():
        base_id = 'qwen30b'
    elif 'gemma' in stem.lower() and 'e4b' in stem.lower():
        base_id = 'gemma_e4b'
    elif 'opus' in stem.lower() or 'claude-4-6-opus' in repo_hint.lower():
        base_id = 'opus'
    ids = {m.id for m in existing_models}
    model_id = base_id
    i = 2
    while model_id in ids:
        model_id = f'{base_id}_{i}'
        i += 1
    ports = {m.port for m in existing_models}
    port = 8080
    while port in ports:
        port += 1
    ctx, ngl, temp = choose_defaults(stem)
    return ModelConfig(
        id=model_id,
        name=name,
        path=str(path),
        alias=slugify(stem),
        port=port,
        ctx=ctx,
        ngl=ngl,
        temp=temp,
        threads=6,
        parallel=1,
        cache_ram=0,
        flash_attn=True,
        jinja=True,
        output=4096,
        extra_args=[],
    )


# ---------- UI helpers ----------

def tail_text(path: Path, max_lines: int = 40) -> List[str]:
    if not path.exists():
        return ['<no log file yet>']
    try:
        lines = path.read_text(errors='replace').splitlines()
        return lines[-max_lines:] or ['<empty log>']
    except Exception as e:
        return [f'<failed to read log: {e}>']


def prompt_value(stdscr, title: str, fields: List[Tuple[str, str]]) -> Optional[Dict[str, str]]:
    curses.endwin()
    print(f'\n{title}')
    print('-' * len(title))
    print('Leave blank to keep current value. Ctrl+C to cancel.\n')
    answers = {}
    try:
        for label, default in fields:
            suffix = f' [{default}]' if default else ''
            value = input(f'{label}{suffix}: ').strip()
            answers[label] = value if value else default
    except KeyboardInterrupt:
        answers = None
    print()
    input('Press Enter to return to the TUI...')
    stdscr.clear()
    stdscr.refresh()
    return answers


def prompt_model(stdscr, title: str, initial: Optional[ModelConfig] = None) -> Optional[ModelConfig]:
    initial = initial or ModelConfig(id='', name='', path='', alias='', port=8080)
    answers = prompt_value(stdscr, title, [
        ('id', initial.id),
        ('name', initial.name),
        ('runtime (llama.cpp/vllm)', getattr(initial, 'runtime', 'llama.cpp')),
        ('optimize_mode (max_context_safe/manual)', getattr(initial, 'optimize_mode', 'max_context_safe')),
        ('optimize_tier (safe/moderate/extreme)', getattr(initial, 'optimize_tier', 'moderate')),
        ('path', initial.path),
        ('alias', initial.alias or initial.id),
        ('port', str(initial.port)),
        ('host', initial.host),
        ('ctx', str(initial.ctx)),
        ('ctx_min', str(getattr(initial, 'ctx_min', 2048))),
        ('ctx_max', str(getattr(initial, 'ctx_max', 131072))),
        ('threads', str(initial.threads)),
        ('ngl', str(initial.ngl)),
        ('temp', str(initial.temp)),
        ('parallel', str(initial.parallel)),
        ('memory_reserve_percent', str(getattr(initial, 'memory_reserve_percent', 25))),
        ('cache_ram', str(initial.cache_ram)),
        ('output', str(initial.output)),
        ('enabled true/false', str(initial.enabled).lower()),
        ('flash_attn true/false', str(initial.flash_attn).lower()),
        ('jinja true/false', str(initial.jinja).lower()),
        ('extra_args (space-separated)', ' '.join(initial.extra_args)),
    ])
    if not answers:
        return None
    try:
        return ModelConfig(
            id=answers['id'],
            name=answers['name'],
            path=answers['path'],
            alias=answers['alias'],
            port=int(answers['port']),
            host=answers['host'],
            ctx=int(answers['ctx']),
            ctx_min=int(answers['ctx_min']),
            ctx_max=int(answers['ctx_max']),
            threads=int(answers['threads']),
            ngl=int(answers['ngl']),
            temp=float(answers['temp']),
            parallel=int(answers['parallel']),
            optimize_mode=(answers['optimize_mode (max_context_safe/manual)'].strip() or 'max_context_safe'),
            optimize_tier=(answers['optimize_tier (safe/moderate/extreme)'].strip() or 'moderate'),
            memory_reserve_percent=int(answers['memory_reserve_percent']),
            cache_ram=int(answers['cache_ram']),
            output=int(answers['output']),
            enabled=answers['enabled true/false'].lower() == 'true',
            runtime=(answers['runtime (llama.cpp/vllm)'].strip().lower() or 'llama.cpp'),
            flash_attn=answers['flash_attn true/false'].lower() == 'true',
            jinja=answers['jinja true/false'].lower() == 'true',
            source=getattr(initial, 'source', 'manual'),
            extra_args=answers['extra_args (space-separated)'].split() if answers['extra_args (space-separated)'] else [],
        )
    except Exception:
        return None


def prompt_settings(stdscr, app: AppConfig) -> bool:
    o = app.opencode
    answers = prompt_value(stdscr, 'Settings', [
        ('llama_server', app.llama_server),
        ('vllm_command', app.vllm_command),
        ('hf_cache_root', app.hf_cache_root),
        ('llm_models_cache_root', app.llm_models_cache_root),
        ('llmfit_cache_root', app.llmfit_cache_root),
        ('opencode_path', o.path),
        ('opencode_backup_dir', o.backup_dir),
        ('default_model_id', o.default_model_id),
        ('small_model_id', o.small_model_id),
        ('build_model_id', o.build_model_id),
        ('plan_model_id', o.plan_model_id),
        ('instructions (comma-separated)', ', '.join(o.instructions)),
        ('build_prompt', o.build_prompt),
        ('plan_prompt', o.plan_prompt),
        ('timeout', str(o.timeout)),
        ('chunk_timeout', str(o.chunk_timeout)),
    ])
    if not answers:
        return False
    try:
        app.llama_server = answers['llama_server']
        app.vllm_command = answers['vllm_command']
        app.hf_cache_root = answers['hf_cache_root']
        app.llm_models_cache_root = answers['llm_models_cache_root']
        app.llmfit_cache_root = answers['llmfit_cache_root']
        o.path = answers['opencode_path']
        o.backup_dir = answers['opencode_backup_dir']
        o.default_model_id = answers['default_model_id']
        o.small_model_id = answers['small_model_id']
        o.build_model_id = answers['build_model_id']
        o.plan_model_id = answers['plan_model_id']
        o.instructions = [s.strip() for s in answers['instructions (comma-separated)'].split(',') if s.strip()]
        o.build_prompt = answers['build_prompt']
        o.plan_prompt = answers['plan_prompt']
        o.timeout = int(answers['timeout'])
        o.chunk_timeout = int(answers['chunk_timeout'])
        app.save()
        return True
    except Exception:
        return False


def prompt_modal_choice(stdscr, colors, title: str, options: List[Tuple[str, str, str]]) -> str:
    h, w = stdscr.getmaxyx()
    box_w = min(68, max(48, w - 8))
    box_h = max(8, len(options) + 6)
    if h < box_h + 4 or w < box_w + 4:
        return 'cancel'
    box_x = max(2, (w - box_w) // 2)
    box_y = max(2, (h - box_h) // 2)
    modal = curses.newwin(box_h + 1, box_w, box_y, box_x)
    modal.keypad(True)
    stdscr.nodelay(False)
    while True:
        modal.erase()
        draw_box(modal, 0, 0, box_h - 1, box_w, title, colors['accent'] | curses.A_BOLD, colors['accent'])
        modal.addstr(2, 2, 'Choose an option:', colors['panel'] | curses.A_BOLD)
        for idx, (key, label, _val) in enumerate(options):
            modal.addstr(3 + idx, 4, f'[{key}] {label}'[: box_w - 8], colors['panel'])
        modal.addstr(box_h - 1, 2, 'Press key to continue...'[: box_w - 6], colors['muted'])
        modal.refresh()
        key = modal.getch()
        if key == -1:
            continue
        key_str = chr(key).lower() if 0 <= key <= 255 else ''
        for option_key, _label, value in options:
            if key_str == option_key:
                stdscr.touchwin()
                stdscr.nodelay(True)
                return value


def prompt_launch_optimization(stdscr, model: ModelConfig, colors) -> str:
    return prompt_modal_choice(stdscr, colors, f'Launch {model.id}', [
        ('1', 'Best optimization for this PC', 'best'),
        ('2', 'Optimize for max context', 'max_context'),
        ('3', 'Optimize for tokens/sec', 'tokens_per_sec'),
        ('4', 'Keep current settings', 'keep'),
        ('q', 'Cancel', 'cancel'),
    ])


def prompt_optimization_tier(stdscr, colors) -> str:
    return prompt_modal_choice(stdscr, colors, 'Optimization tier', [
        ('1', 'Safe (most stable)', 'safe'),
        ('2', 'Moderate (balanced)', 'moderate'),
        ('3', 'Extreme (aggressive)', 'extreme'),
        ('q', 'Cancel', 'cancel'),
    ])


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
        model.ctx = min(model.ctx, estimate_safe_context_for_profile(model, profile, model.memory_reserve_percent, model.parallel, ctx_min, ctx_max))
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
        model.ctx = min(model.ctx, estimate_safe_context_for_profile(model, profile, model.memory_reserve_percent, model.parallel, ctx_min, ctx_max))
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
        hw_note = f' | {profile.short_summary()}' if profile else ''
        return f'{model.id}: preset=tokens_per_sec/{tier} ({runtime}) ctx={model.ctx} parallel={model.parallel} threads={model.threads} ngl={model.ngl}{hw_note}'

    return f'{model.id}: keeping current settings'


def sync_opencode_after_tuning(app: AppConfig) -> str:
    if not app.opencode.path:
        return 'opencode.path unset; skipped opencode sync'
    ok, msg = app.generate_opencode()
    return msg if ok else f'opencode sync failed: {msg}'


def apply_best_optimization(
    model: ModelConfig,
    tier: str = 'moderate',
    profile: Optional[HardwareProfile] = None,
) -> str:
    if tier == 'auto':
        tier = select_best_tier(model, profile)
    preset = choose_best_preset(model, profile)
    return apply_optimization_preset(model, preset, tier=tier, profile=profile)


def fallback_tiers(selected_tier: str) -> List[str]:
    order = ['extreme', 'moderate', 'safe']
    selected = (selected_tier or 'moderate').strip().lower()
    if selected == 'auto':
        selected = 'moderate'
    if selected not in order:
        selected = 'moderate'
    return order[order.index(selected):]


def launch_with_failsafe(app: AppConfig, model: ModelConfig, mode: str, tier: str) -> Tuple[bool, str]:
    attempts = []
    profile = app.hardware_profile(refresh=True)
    if tier == 'auto':
        tier = select_best_tier(model, profile)
    for current_tier in fallback_tiers(tier):
        if mode == 'best':
            tune_msg = apply_best_optimization(model, tier=current_tier, profile=profile)
        else:
            tune_msg = apply_optimization_preset(model, mode, tier=current_tier, profile=profile)
        app.add_or_update(model)
        sync_msg = sync_opencode_after_tuning(app)
        ok, msg = app.start(model)
        if not ok:
            attempts.append(f'{current_tier}: start failed ({msg})')
            continue
        ready_ok, ready_msg = app.wait_until_ready(model, timeout=120)
        if ready_ok:
            return True, f'{ready_msg} [{current_tier}] | {tune_msg} | {sync_msg}'
        app.stop(model)
        attempts.append(f'{current_tier}: not ready')
    return False, '❌ optimization failed; fallback exhausted -> ' + '; '.join(attempts[:3])


def clone_model_config(model: ModelConfig) -> ModelConfig:
    return ModelConfig(**asdict(model))


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\s\w]", text)))


def completion_text_from_response(data: Dict) -> str:
    choices = data.get('choices') or []
    if not choices:
        return ''
    first = choices[0] or {}
    message = first.get('message') or {}
    content = message.get('content')
    if isinstance(content, list):
        return ' '.join(str(item.get('text', item)) if isinstance(item, dict) else str(item) for item in content)
    if content is not None:
        return str(content)
    return str(first.get('text', ''))


def post_json(url: str, payload: Dict, timeout: int) -> Dict:
    body = json.dumps(payload).encode('utf-8')
    req = request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8', errors='replace'))


def benchmark_completion(model: ModelConfig, max_tokens: int = 64, timeout: int = 180) -> Tuple[bool, Dict]:
    prompt = (
        'Write a concise technical checklist for keeping a local language model '
        'server fast and stable. Use short bullet points.'
    )
    payload = {
        'model': model.alias,
        'messages': [
            {'role': 'system', 'content': 'You are a concise local model benchmark assistant.'},
            {'role': 'user', 'content': prompt},
        ],
        'max_tokens': max_tokens,
        'temperature': 0,
        'stream': False,
    }
    url = f'http://{model.host}:{model.port}/v1/chat/completions'
    started = time.time()
    try:
        data = post_json(url, payload, timeout=timeout)
    except Exception as exc:
        return False, {'error': str(exc)}
    elapsed = max(0.001, time.time() - started)
    usage = data.get('usage') or {}
    text = completion_text_from_response(data)
    completion_tokens = int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
    prompt_tokens = int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
    if completion_tokens <= 0:
        completion_tokens = estimate_text_tokens(text)
    if prompt_tokens <= 0:
        prompt_tokens = estimate_text_tokens(prompt)
    return True, {
        'elapsed': elapsed,
        'completion_tokens': completion_tokens,
        'prompt_tokens': prompt_tokens,
        'tokens_per_sec': completion_tokens / elapsed,
        'text': text,
    }


def benchmark_candidate_models(model: ModelConfig, profile: HardwareProfile) -> List[Tuple[str, str, ModelConfig, str]]:
    selected_tier = select_best_tier(model, profile)
    selected_preset = choose_best_preset(model, profile)
    alternate_preset = 'max_context' if selected_preset == 'tokens_per_sec' else 'tokens_per_sec'
    tier_order = ['safe', 'moderate', 'extreme']
    selected_idx = tier_order.index(selected_tier)
    neighbor_tiers = [selected_tier]
    if selected_idx > 0:
        neighbor_tiers.append(tier_order[selected_idx - 1])
    if selected_idx < len(tier_order) - 1:
        neighbor_tiers.append(tier_order[selected_idx + 1])

    requested: List[Tuple[str, str]] = []
    for tier in neighbor_tiers:
        requested.append((selected_preset, tier))
    requested.append((alternate_preset, selected_tier))
    if selected_tier != 'safe':
        requested.append((alternate_preset, 'safe'))

    candidates = []
    seen = set()
    for preset, tier in requested:
        key = (preset, tier)
        if key in seen:
            continue
        seen.add(key)
        candidate = clone_model_config(model)
        tune_msg = apply_optimization_preset(candidate, preset, tier=tier, profile=profile)
        candidates.append((preset, tier, candidate, tune_msg))
        if len(candidates) >= 5:
            break
    return candidates


def benchmark_best_optimization(app: AppConfig, model: ModelConfig) -> Tuple[bool, str]:
    status, _detail = app.health(model)
    if status in ('READY', 'LOADING', 'STARTING') or app.get_pid(model):
        return False, '❌ Stop the model before running benchmark optimization.'

    profile = app.hardware_profile(refresh=True)
    candidates = benchmark_candidate_models(model, profile)
    results = []
    failures = []

    for preset, tier, candidate, tune_msg in candidates:
        ok, msg = app.start(candidate)
        if not ok:
            failures.append(f'{preset}/{tier}: start failed ({msg})')
            continue

        try:
            ready_ok, ready_msg = app.wait_until_ready(candidate, timeout=180)
            if not ready_ok:
                failures.append(f'{preset}/{tier}: {compact_message(ready_msg)}')
                continue

            benchmark_completion(candidate, max_tokens=16, timeout=120)
            bench_ok, bench = benchmark_completion(candidate, max_tokens=64, timeout=180)
            if not bench_ok:
                failures.append(f'{preset}/{tier}: benchmark failed ({bench.get("error", "unknown error")})')
                continue

            score = float(bench['tokens_per_sec'])
            results.append({
                'score': score,
                'preset': preset,
                'tier': tier,
                'model': candidate,
                'elapsed': float(bench['elapsed']),
                'completion_tokens': int(bench['completion_tokens']),
                'prompt_tokens': int(bench['prompt_tokens']),
                'tune_msg': tune_msg,
            })
        finally:
            app.stop(candidate)
            time.sleep(0.5)

    if not results:
        details = '; '.join(failures[:3]) if failures else 'no candidates completed'
        return False, f'❌ benchmark failed: {details}'

    best = max(results, key=lambda item: item['score'])
    best_model = best['model']
    best_model.last_benchmark_tokens_per_sec = round(best['score'], 2)
    best_model.last_benchmark_seconds = round(best['elapsed'], 2)
    best_model.last_benchmark_profile = (
        f'{best["preset"]}/{best["tier"]} '
        f'{best["score"]:.2f} tok/s '
        f'{profile.short_summary()}'
    )
    app.add_or_update(best_model)
    sync_msg = sync_opencode_after_tuning(app)
    return True, (
        f'✅ benchmark winner: {best_model.id} {best["preset"]}/{best["tier"]} '
        f'{best["score"]:.2f} tok/s ctx={best_model.ctx} parallel={best_model.parallel} '
        f'threads={best_model.threads} ngl={best_model.ngl} | {sync_msg}'
    )


def draw_box(stdscr, y: int, x: int, h: int, w: int, title: str, title_attr: int = curses.A_BOLD, border_attr: int = 0):
    if h < 2 or w < 4:
        return
    stdscr.addstr(y, x + 2, f' {title} ', title_attr)
    for i in range(x, x + w):
        stdscr.addch(y + 1, i, curses.ACS_HLINE, border_attr)
    for i in range(y + 1, y + h):
        stdscr.addch(i, x, curses.ACS_VLINE, border_attr)
        stdscr.addch(i, x + w - 1, curses.ACS_VLINE, border_attr)
    stdscr.addch(y + 1, x, curses.ACS_ULCORNER, border_attr)
    stdscr.addch(y + 1, x + w - 1, curses.ACS_URCORNER, border_attr)
    stdscr.addch(y + h, x, curses.ACS_LLCORNER, border_attr)
    stdscr.addch(y + h, x + w - 1, curses.ACS_LRCORNER, border_attr)
    for i in range(x + 1, x + w - 1):
        stdscr.addch(y + h, i, curses.ACS_HLINE, border_attr)



def init_colors():
    palette = {
        'default': 0,
        'accent': 0,
        'success': 0,
        'warning': 0,
        'error': 0,
        'muted': 0,
        'selection': 0,
        'banner': 0,
        'panel': 0,
        'chip_ready': 0,
        'chip_loading': 0,
        'chip_stopped': 0,
    }
    if not curses.has_colors():
        return palette
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    pairs = [
        ('accent', curses.COLOR_CYAN, -1),
        ('success', curses.COLOR_GREEN, -1),
        ('warning', curses.COLOR_YELLOW, -1),
        ('error', curses.COLOR_RED, -1),
        ('muted', curses.COLOR_BLUE, -1),
        ('selection', curses.COLOR_BLACK, curses.COLOR_CYAN),
        ('banner', curses.COLOR_MAGENTA, -1),
        ('panel', curses.COLOR_WHITE, -1),
        ('chip_ready', curses.COLOR_BLACK, curses.COLOR_GREEN),
        ('chip_loading', curses.COLOR_BLACK, curses.COLOR_YELLOW),
        ('chip_stopped', curses.COLOR_WHITE, curses.COLOR_BLUE),
    ]
    pair_id = 1
    for name, fg, bg in pairs:
        try:
            curses.init_pair(pair_id, fg, bg)
            palette[name] = curses.color_pair(pair_id)
            pair_id += 1
        except curses.error:
            palette[name] = curses.A_BOLD if name in ('accent', 'success', 'warning', 'error', 'banner') else 0
    return palette


def status_attr(colors, status: str):
    mapping = {
        'READY': colors['success'] | curses.A_BOLD,
        'LOADING': colors['warning'] | curses.A_BOLD,
        'STARTING': colors['warning'],
        'STOPPED': colors['muted'],
        'ERROR': colors['error'] | curses.A_BOLD,
    }
    return mapping.get(status, colors['accent'])


def status_symbol(status: str) -> str:
    symbols = {
        'READY': '●',
        'LOADING': '◐',
        'STARTING': '◔',
        'STOPPED': '○',
        'ERROR': '✖',
    }
    return symbols.get(status, '·')


def chip_attr(colors, label: str):
    mapping = {
        'READY': colors['chip_ready'] | curses.A_BOLD,
        'LOADING': colors['chip_loading'] | curses.A_BOLD,
        'STARTING': colors['chip_loading'] | curses.A_BOLD,
        'STOPPED': colors['chip_stopped'] | curses.A_BOLD,
    }
    return mapping.get(label, colors['accent'] | curses.A_BOLD)


def ellipsize(text: str, width: int) -> str:
    if width <= 0:
        return ''
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + '...'


def compact_message(text: str) -> str:
    return ' | '.join(part.strip() for part in str(text).splitlines() if part.strip())


def tui(stdscr, app: AppConfig):
    colors = init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    selected = 0
    message = 'Ready.'
    last_refresh = 0.0
    statuses: Dict[str, Tuple[str, str]] = {}

    while True:
        now = time.time()
        if now - last_refresh > REFRESH_SECONDS:
            statuses = {m.id: app.health(m) for m in app.models}
            last_refresh = now

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 18 or w < 88:
            stdscr.addstr(1, 2, 'Window too small for llama-tui. Stretch it a bit.', colors['warning'] | curses.A_BOLD)
            stdscr.addstr(3, 2, f'Current size: {w}x{h}')
            stdscr.addstr(5, 2, '[q] quit', curses.A_BOLD)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord('q'), 27):
                break
            time.sleep(0.05)
            continue

        y = 0
        if w >= 100:
            for line in LOGO:
                stdscr.addstr(y, 2, line[:w-4], colors['banner'] | curses.A_BOLD)
                y += 1
            stdscr.addstr(1, min(w - 28, 60), 'local model control plane', colors['accent'] | curses.A_BOLD)
            header_y = y + 1
        else:
            stdscr.addstr(0, 2, 'llama-tui', colors['banner'] | curses.A_BOLD)
            stdscr.addstr(0, 14, 'local model control plane', colors['accent'] | curses.A_BOLD)
            header_y = 2

        stdscr.addstr(header_y, 2, f'config: {app.config_path}', colors['muted'])
        stdscr.addstr(header_y + 1, 2, f'llama-server: {app.llama_server}', colors['muted'])
        stdscr.addstr(header_y + 2, 2, f'vllm-command: {app.vllm_command}', colors['muted'])
        stdscr.addstr(header_y + 3, 2, f'hf-cache: {app.hf_cache_root}', colors['muted'])
        stdscr.addstr(header_y + 4, 2, f'llm-models-cache: {app.llm_models_cache_root}', colors['muted'])
        stdscr.addstr(header_y + 5, 2, f'llmfit-cache: {app.llmfit_cache_root}', colors['muted'])
        stdscr.addstr(header_y + 6, 2, f'opencode: {app.opencode.path or "<unset>"}', colors['muted'])

        counts = {'READY': 0, 'LOADING': 0, 'STARTING': 0, 'STOPPED': 0, 'ERROR': 0}
        for _mid, (st, _detail) in statuses.items():
            if st in counts:
                counts[st] += 1

        msg_attr = colors['accent'] | curses.A_BOLD
        if message.startswith('✅'):
            msg_attr = colors['success'] | curses.A_BOLD
        elif message.startswith('❌'):
            msg_attr = colors['error'] | curses.A_BOLD
        elif message.startswith('⏳'):
            msg_attr = colors['warning'] | curses.A_BOLD
        header_message = compact_message(message)
        msg_line = ellipsize(header_message, max(10, w - 4))
        stdscr.addstr(header_y + 7, 2, msg_line, msg_attr)

        chip_y = header_y + 7
        chip_x = min(max(40, len(msg_line) + 6), max(40, w - 34))
        chips = [
            ('READY', counts['READY']),
            ('LOADING', counts['LOADING'] + counts['STARTING']),
            ('STOPPED', counts['STOPPED']),
        ]
        for label, value in chips:
            text = f' {label}:{value} '
            if chip_x + len(text) < w - 2:
                stdscr.addstr(chip_y, chip_x, text, chip_attr(colors, label))
                chip_x += len(text) + 1

        box_top = header_y + 8
        left_w = max(76, min(112, (w // 2) + 8))
        right_x = left_w + 2
        right_w = max(38, w - right_x - 2)
        visible_rows = max(8, h - box_top - 6)

        draw_box(stdscr, box_top, 1, h - box_top - 4, left_w, 'Models', colors['accent'] | curses.A_BOLD, colors['accent'])
        draw_box(stdscr, box_top, right_x, h - box_top - 4, right_w, 'Details / Logs / Roles', colors['accent'] | curses.A_BOLD, colors['accent'])

        header = ' ID              PRT  ST        RLS  ENG        QNT      TYPE   NAME'
        stdscr.addstr(box_top + 2, 3, header, colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD)

        if app.models:
            start_idx = max(0, selected - visible_rows + 3)
            end_idx = min(len(app.models), start_idx + visible_rows)
            for idx in range(start_idx, end_idx):
                model = app.models[idx]
                status, _ = statuses.get(model.id, ('?', ''))
                roles = app.role_badges(model.id)
                engine = display_runtime(model)[:10]
                quant = extract_quant(model)[:8]
                model_type = classify_model_type(model)[:6]
                name_col_width = max(10, left_w - 70)
                line = f' {model.id[:14]:14} {model.port:4}  {status_symbol(status)} {status[:6]:6}  {roles:3}  {engine:10} {quant:8} {model_type:6} {model.name[:name_col_width]}'
                row_y = box_top + 3 + idx - start_idx
                if idx == selected:
                    try:
                        stdscr.addstr(row_y, 3, line[: left_w - 3], colors['selection'] | curses.A_BOLD)
                    except curses.error:
                        stdscr.addstr(row_y, 3, line[: left_w - 3], curses.A_REVERSE)
                else:
                    stdscr.addstr(row_y, 3, line[: left_w - 3])
                    status_x = 3 + 1 + 14 + 1 + 4 + 2
                    stdscr.addstr(row_y, status_x, f'{status_symbol(status)} {status[:6]:6}', status_attr(colors, status))
            if len(app.models) > visible_rows:
                bar_h = max(1, visible_rows)
                track_x = left_w - 1
                for i in range(bar_h):
                    stdscr.addch(box_top + 3 + i, track_x, '│', colors['muted'])
                thumb_h = max(1, int(bar_h * (visible_rows / max(1, len(app.models)))))
                thumb_top = int((start_idx / max(1, len(app.models) - visible_rows)) * max(0, bar_h - thumb_h))
                for i in range(thumb_h):
                    stdscr.addch(box_top + 3 + thumb_top + i, track_x, '█', colors['accent'] | curses.A_BOLD)
        else:
            stdscr.addstr(box_top + 3, 3, 'No models yet. Press x to detect GGUFs or a to add a llama.cpp/vLLM model.', colors['warning'])

        if app.models:
            model = app.models[selected]
            status, detail = statuses.get(model.id, ('?', ''))
            show_alert = message.startswith('❌') or status == 'ERROR'
            alert_lines = [ellipsize(line.strip(), right_w - 6) for line in str(message).splitlines() if line.strip()][:4] if show_alert else []
            lines = [
                f'name: {model.name}',
                f'id: {model.id}',
                f'path: {ellipsize(model.path, right_w - 12)}',
                f'alias: {model.alias}',
                f'runtime/source: {display_runtime(model)} / {getattr(model, "source", "manual")}',
                f'quant/type: {extract_quant(model)} / {classify_model_type(model)}',
                f'bind: {model.host}:{model.port}',
                f'ctx={model.ctx} threads={model.threads} ngl={model.ngl} temp={model.temp}',
                f'parallel={model.parallel} cache_ram={model.cache_ram} output={model.output}',
                f'optimize={getattr(model, "optimize_mode", "max_context_safe")}/{getattr(model, "optimize_tier", "moderate")} ctx_range={getattr(model, "ctx_min", 2048)}..{getattr(model, "ctx_max", 131072)} reserve={getattr(model, "memory_reserve_percent", 25)}%',
                f'benchmark={getattr(model, "last_benchmark_tokens_per_sec", 0.0):.2f} tok/s {getattr(model, "last_benchmark_profile", "")}',
                f'flags: enabled={model.enabled} flash_attn={model.flash_attn} jinja={model.jinja}',
                f'status: {status} ({detail})',
                f'pid: {app.get_pid(model) or "-"}',
                f'roles: {app.role_badges(model.id)}  [m main] [s small] [b build] [p plan]',
                f'log: {app.logfile(model.id)}',
                'command preview:',
                ellipsize(' '.join(app.build_command(model)), right_w - 6),
                '',
                'last log lines:',
            ]
            if alert_lines:
                lines = ['error alert:', *alert_lines, ''] + lines
            lines.extend(tail_text(app.logfile(model.id), max_lines=max(8, h - box_top - 22)))
            for i, line in enumerate(lines[: h - box_top - 7]):
                attr = curses.A_NORMAL
                if alert_lines and i == 0:
                    attr = colors['error'] | curses.A_BOLD
                elif alert_lines and i <= len(alert_lines):
                    attr = colors['error']
                elif line.startswith('status:'):
                    attr = status_attr(colors, status)
                elif line in ('command preview:', 'last log lines:'):
                    attr = colors['accent'] | curses.A_BOLD
                stdscr.addstr(box_top + 2 + i, right_x + 2, line[: right_w - 4], attr)

        footer = '[Enter] start/stop  [z] best optimization  [B] benchmark best  [x] detect  [X] prune'
        footer2 = '[a/e/d] models  [m/s/b/p] set roles  [r] sync inventory  [S] stop-all  [q] quit'
        stdscr.addstr(h - 2, 2, footer[: w - 4], colors['accent'] | curses.A_BOLD)
        stdscr.addstr(h - 1, 2, footer2[: w - 4], colors['muted'] | curses.A_BOLD)
        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break

        if key == -1:
            time.sleep(0.05)
            continue
        if key in (ord('q'), 27):
            break
        if key in (curses.KEY_UP, ord('k')) and app.models:
            selected = max(0, selected - 1)
        elif key in (curses.KEY_DOWN, ord('j')) and app.models:
            selected = min(len(app.models) - 1, selected + 1)
        elif key == ord('r'):
            count, items = app.detect_models()
            statuses = {m.id: app.health(m) for m in app.models}
            message = items[0] if items else (f'Synced {count} model(s)' if count else 'Synced.')
        elif key == ord('S'):
            message = '; '.join(app.stop_all())[: max(20, w - 4)]
        elif key in (10, 13, curses.KEY_ENTER) and app.models:
            model = app.models[selected]
            status, _ = app.health(model)
            if status in ('READY', 'LOADING', 'STARTING') or app.get_pid(model):
                ok, msg = app.stop(model)
                message = f'{model.id}: {msg}'
            else:
                launch_mode = prompt_launch_optimization(stdscr, model, colors)
                if launch_mode == 'cancel':
                    message = 'Launch cancelled.'
                    continue
                if launch_mode == 'best':
                    ok, msg = launch_with_failsafe(app, model, launch_mode, 'auto')
                    message = msg
                elif launch_mode in ('max_context', 'tokens_per_sec'):
                    tier = prompt_optimization_tier(stdscr, colors)
                    if tier == 'cancel':
                        message = 'Launch cancelled.'
                        continue
                    ok, msg = launch_with_failsafe(app, model, launch_mode, tier)
                    message = msg
                else:
                    ok, msg = app.start(model)
                    if ok:
                        ready_ok, ready_msg = app.wait_until_ready(model, timeout=120)
                        message = ready_msg
                    else:
                        message = msg
        elif key == ord('z') and app.models:
            model = app.models[selected]
            profile = app.hardware_profile(refresh=True)
            tier = select_best_tier(model, profile)
            tune_msg = apply_best_optimization(model, tier=tier, profile=profile)
            app.add_or_update(model)
            sync_msg = sync_opencode_after_tuning(app)
            message = f'{tune_msg} | {sync_msg}'
        elif key == ord('B') and app.models:
            model = app.models[selected]
            message = f'⏳ benchmarking {model.id}; this can take a few minutes...'
            stdscr.addstr(header_y + 7, 2, ellipsize(message, max(10, w - 4)), colors['warning'] | curses.A_BOLD)
            stdscr.refresh()
            ok, msg = benchmark_best_optimization(app, model)
            message = msg
        elif key == ord('a'):
            model = prompt_model(stdscr, 'Add model')
            if model:
                app.add_or_update(model)
                selected = len(app.models) - 1
                message = f'Added {model.id}.'
        elif key == ord('e') and app.models:
            current = app.models[selected]
            updated = prompt_model(stdscr, f'Edit {current.id}', current)
            if updated:
                if updated.id != current.id:
                    app.delete(current.id)
                app.add_or_update(updated)
                selected = min(selected, len(app.models) - 1)
                message = f'Updated {updated.id}.'
        elif key == ord('d') and app.models:
            curses.endwin()
            ans = input(f'Delete {app.models[selected].id} from llama-tui config? [y/N]: ').strip().lower()
            stdscr.clear(); stdscr.refresh()
            if ans == 'y':
                target_id = app.models[selected].id
                ok, msg = app.delete(target_id)
                selected = max(0, min(selected, len(app.models) - 1))
                message = f'{target_id}: {msg}'
            else:
                message = 'Delete cancelled.'
        elif key == ord('x'):
            count, items = app.detect_models()
            message = items[0] if items else (f'Detected {count} new model(s)' if count else 'No new GGUFs found.')
            selected = min(selected, len(app.models) - 1 if app.models else 0)
        elif key == ord('X'):
            count, removed = app.prune_missing_models()
            message = f'Pruned {count}: {", ".join(removed[:5])}' if count else 'No missing models to prune.'
            selected = max(0, min(selected, len(app.models) - 1))
        elif key == ord('g'):
            ok, msg = app.generate_opencode()
            message = msg
        elif key == ord('o'):
            if prompt_settings(stdscr, app):
                message = 'Settings saved.'
            else:
                message = 'Settings unchanged.'
        elif key == ord('m') and app.models:
            app.set_role('main', app.models[selected].id)
            message = f'{app.models[selected].id} set as main model.'
        elif key == ord('s') and app.models:
            app.set_role('small', app.models[selected].id)
            message = f'{app.models[selected].id} set as small model.'
        elif key == ord('b') and app.models:
            app.set_role('build', app.models[selected].id)
            message = f'{app.models[selected].id} set as build model.'
        elif key == ord('p') and app.models:
            app.set_role('plan', app.models[selected].id)
            message = f'{app.models[selected].id} set as plan model.'


def ensure_bootstrap_files(config_path: Path) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        return config_path
    sample_candidates = [
        DATA_DIR / 'llama_models.sample.json',
        Path(__file__).with_name('llama_models.sample.json'),
        Path(__file__).with_name('llama_models.v2.sample.json'),
        SCRIPT_DIR / 'models.json',
    ]
    for sample_path in sample_candidates:
        if sample_path.exists():
            config_path.write_text(sample_path.read_text())
            return config_path
    app = AppConfig(config_path)
    app.save()
    return config_path


def main():
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_bootstrap_files(config_path)
    app = AppConfig(config_path)
    curses.wrapper(tui, app)


if __name__ == '__main__':
    main()
