import json
import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import asdict
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
    DEFAULT_LLMFIT_CACHE,
    DEFAULT_LLM_MODELS_CACHE,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_VLLM_COMMAND,
)
from .discovery import (
    detected_model_from_path,
    display_runtime,
    is_real_model_file,
    is_registered_model_entry,
    looks_like_model_reference,
)
from .control import CancelToken, check_cancelled, sleep_with_cancel
from .gguf import estimate_kv_bytes_per_token, read_gguf_metadata
from .hardware import HardwareProfile, benchmark_current_hardware, read_meminfo_bytes
from .models import ModelConfig, OpencodeSettings
from .optimize import choose_gpu_layers_for_profile, effective_gpu_reserve_percent, estimate_safe_context_for_profile
from .textutil import compact_message, important_log_excerpt


TERMINAL_LAUNCHER_ORDER = (
    'xdg-terminal-exec',
    'ptyxis',
    'gnome-console',
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
    'xterm',
)


def terminal_command_for_launcher(launcher: str, title: str, cwd: Path, shell_cmd: str) -> List[str]:
    launcher_name = Path(launcher).name
    cwd_text = str(cwd)
    cd_shell_cmd = f'cd {shlex.quote(cwd_text)} && {shell_cmd}'
    if launcher_name == 'xdg-terminal-exec':
        return [launcher, 'bash', '-lc', cd_shell_cmd]
    if launcher_name in ('ptyxis', 'gnome-console'):
        return [launcher, '--working-directory', cwd_text, '--title', title, '--', 'bash', '-lc', shell_cmd]
    if launcher_name == 'konsole':
        return [launcher, '--workdir', cwd_text, '-p', f'tabtitle={title}', '-e', 'bash', '-lc', shell_cmd]
    if launcher_name == 'gnome-terminal':
        return [launcher, '--title', title, '--working-directory', cwd_text, '--', 'bash', '-lc', shell_cmd]
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
        self._owned_pids: set[int] = set()
        self._runtime_check_cache: Dict[str, Tuple[bool, str]] = {}
        self._shutdown_cleanup_done = False
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
            if not getattr(m, 'benchmark_fingerprint', '') and float(getattr(m, 'last_benchmark_tokens_per_sec', 0.0) or 0.0) > 0.0:
                m.benchmark_fingerprint = self.model_fingerprint(m)
                if not getattr(m, 'default_benchmark_status', ''):
                    m.default_benchmark_status = 'done'
                if not getattr(m, 'default_benchmark_at', ''):
                    m.default_benchmark_at = datetime.now().isoformat(timespec='seconds')
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

    def pid_metadata_file(self, model_id: str) -> Path:
        return CACHE_DIR / f'{model_id}.pid.json'

    def logfile(self, model_id: str) -> Path:
        return CACHE_DIR / f'{model_id}.log'

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        return next((m for m in self.models if m.id == model_id), None)

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
            ])
        payload = {
            'runtime': getattr(model, 'runtime', 'llama.cpp'),
            'target': stat_data,
            'metadata': {key: metadata.get(key) for key in metadata_keys if key in metadata},
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

    def opencode_provider_key(self, model: ModelConfig) -> str:
        return f'local-{model.id}'

    def opencode_model_ref(self, model: ModelConfig) -> str:
        return f'{self.opencode_provider_key(model)}/{model.alias}'

    def detect_terminal_launcher(self) -> Optional[str]:
        for launcher in TERMINAL_LAUNCHER_ORDER:
            resolved = shutil.which(launcher)
            if resolved:
                return resolved
        return None

    def build_opencode_shell_command(self, model: ModelConfig, workspace: Path) -> str:
        env = {
            'OPENCODE_DISABLE_AUTOUPDATE': 'true',
            'OPENCODE_DISABLE_PRUNE': 'true',
            'OPENCODE_DISABLE_MODELS_FETCH': 'true',
            'OPENCODE_CLIENT': 'llama-tui-stack',
        }
        if self.opencode.path:
            env['OPENCODE_CONFIG'] = str(Path(self.opencode.path).expanduser())
        env_prefix = ' '.join(f'{key}={shlex.quote(str(value))}' for key, value in env.items()) + ' '
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

    def build_terminal_command(self, title: str, workspace: Path, shell_cmd: str) -> Tuple[bool, List[str], str]:
        template = getattr(self.opencode, 'terminal_command', '').strip()
        if template:
            try:
                return True, render_terminal_template(template, title, workspace, shell_cmd), 'custom terminal command'
            except Exception as exc:
                return False, [], f'invalid opencode.terminal_command: {exc}'
        launcher = self.detect_terminal_launcher()
        if not launcher:
            return (
                False,
                [],
                'No terminal launcher found. Set opencode.terminal_command in settings '
                f'using {{title}}, {{cwd}}, and {{cmd}}. Manual command: {shell_cmd}'
            )
        return True, terminal_command_for_launcher(launcher, title, workspace, shell_cmd), Path(launcher).name

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
        try:
            proc = subprocess.Popen(
                terminal_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return False, f'failed to launch OpenCode terminal: {exc}'
        self.append_log(model.id, f'OpenCode terminal launched pid={proc.pid} workspace={workspace_path} via {terminal_label}')
        return True, f'OpenCode terminal launched for {model.id} in {workspace_path}'

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
        try:
            watched_pgid, signaled_group = self._send_signal(pid, signal.SIGKILL, use_group)
        except OSError:
            pass
        watched_pgid = watched_pgid if signaled_group else None
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
    ) -> List[str]:
        runtime = getattr(model, 'runtime', 'llama.cpp')
        ctx_value = int(ctx_override if ctx_override is not None else model.ctx)
        parallel_value = int(parallel_override if parallel_override is not None else model.parallel)
        ngl_value = int(ngl_override if ngl_override is not None else model.ngl)
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
            '--n-gpu-layers', str(ngl_value),
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
        label = 'vLLM command' if runtime == 'vllm' else 'llama-server'
        if not self.command_exists(command):
            return False, f'{label} not found: {command}'
        runtime_ok, runtime_msg = self.runtime_command_ready(runtime, command)
        if not runtime_ok:
            self.append_log(model.id, f'{label} runtime check failed: {runtime_msg}')
            return False, f'{label} cannot run: {runtime_msg}'
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
        command = self.build_command(
            model,
            ctx_override=profile.get('ctx'),
            parallel_override=profile.get('parallel'),
            ngl_override=profile.get('ngl'),
        )
        env = os.environ.copy()
        env['LLAMA_TUI_MODEL_ID'] = model.id
        env['LLAMA_TUI_OWNER_PID'] = str(os.getpid())
        self.append_log(model.id, f'launch profile: {profile_msg}')
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
        return True, f'started PID {proc.pid} ({profile_msg})'

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
            if status == 'STOPPED' and not self.get_pid(model):
                excerpt = '\n'.join(important_log_excerpt(self.logfile(model.id), 24, after_last_launch=True))
                return False, f'❌ {model.id} crashed during startup (log: {self.logfile(model.id)})\n{excerpt}'
            sleep_with_cancel(0.5, cancel_token)
        return False, f'⏳ {model.id} is still loading'

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
