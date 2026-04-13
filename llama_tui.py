#!/usr/bin/env python3
import curses
import json
import os
import re
import shutil
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
DEFAULT_LLAMA_SERVER = str(Path('/var/home/jcampos/llama.cpp/build/bin/llama-server') if Path('/var/home/jcampos/llama.cpp/build/bin/llama-server').exists() else Path.home() / 'llama.cpp' / 'build' / 'bin' / 'llama-server')
REFRESH_SECONDS = 2.0
LOGO = [
    r'  _ _                         _         _       _ ',
    r' | | | __ _ _ __ ___   __ _  | |_ _   _(_)_   _| |',
    r' | | |/ _` | \'_ ` _ \\ / _` | | __| | | | | | | |',
    r' | | | (_| | | | | | | (_| | | |_| |_| | | |_| |_|',
    r' |_|_|\\__,_|_| |_| |_|\\__,_|  \\__|\\__,_|_|\\__,_(_)',
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
    enabled: bool = True
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


class AppConfig:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.llama_server = os.environ.get('LLAMA_SERVER', DEFAULT_LLAMA_SERVER)
        self.hf_cache_root = str(DEFAULT_HF_CACHE)
        self.llmfit_cache_root = str(DEFAULT_LLMFIT_CACHE)
        self.opencode = OpencodeSettings(
            path='',
            backup_dir=str(CONFIG_DIR / 'backups'),
        )
        self.models: List[ModelConfig] = []
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
        self.hf_cache_root = data.get('hf_cache_root', self.hf_cache_root)
        self.llmfit_cache_root = data.get('llmfit_cache_root', self.llmfit_cache_root)
        self.opencode = OpencodeSettings(**data.get('opencode', {}))
        self.models = [ModelConfig(**item) for item in data.get('models', [])]

    def save(self):
        data = {
            'llama_server': self.llama_server,
            'hf_cache_root': self.hf_cache_root,
            'llmfit_cache_root': self.llmfit_cache_root,
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

    def _pid_looks_like_llama(self, pid: int) -> bool:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            parts = [p.decode(errors='ignore') for p in raw.split(b'\x00') if p]
            if not parts:
                return False
            target = os.path.basename(self.llama_server)
            return any(os.path.basename(part) == target for part in parts)
        except Exception:
            return False

    def _find_model_pid(self, model: ModelConfig) -> Optional[int]:
        target = os.path.basename(self.llama_server)
        port = str(model.port)
        alias = model.alias
        path = model.path
        for proc_dir in Path('/proc').iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                pid = int(proc_dir.name)
                raw = (proc_dir / 'cmdline').read_bytes()
                parts = [p.decode(errors='ignore') for p in raw.split(b'\x00') if p]
                if not parts:
                    continue
                if not any(os.path.basename(part) == target for part in parts):
                    continue
                joined = '\x00'.join(parts)
                if f'\x00--port\x00{port}\x00' in joined or f'\x00--alias\x00{alias}\x00' in joined or path in parts:
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
                elif self._pid_looks_like_llama(pid):
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

    def build_command(self, model: ModelConfig) -> List[str]:
        cmd = [
            self.llama_server,
            '-m', model.path,
            '--alias', model.alias,
            '--host', model.host,
            '--port', str(model.port),
            '--ctx-size', str(model.ctx),
            '--threads', str(model.threads),
            '--n-gpu-layers', str(model.ngl),
            '--parallel', str(model.parallel),
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
        if not Path(self.llama_server).exists():
            return False, f'llama-server not found: {self.llama_server}'
        if not Path(model.path).exists():
            return False, f'model path missing: {model.path}'
        if self.get_pid(model):
            return True, f'{model.id} already running'
        log_path = self.logfile(model.id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'ab') as log_file:
            proc = subprocess.Popen(
                self.build_command(model),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.pidfile(model.id).write_text(str(proc.pid))
        return True, f'started PID {proc.pid}'

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
        removed = []
        for model in list(self.models):
            if not Path(model.path).exists():
                self.delete(model.id)
                removed.append(model.id)
        return len(removed), removed

    def detect_models(self) -> Tuple[int, List[str]]:
        hf_root = Path(self.hf_cache_root).expanduser()
        llmfit_root = Path(self.llmfit_cache_root).expanduser()

        existing_paths = {str(Path(m.path).resolve()) for m in self.models if Path(m.path).exists()}
        added = []
        notes = []
        candidates: List[Path] = []

        if hf_root.exists():
            candidates.extend(sorted(hf_root.glob('models--*/snapshots/*/*.gguf')))
        else:
            notes.append(f'HF cache not found: {hf_root}')

        if llmfit_root.exists():
            candidates.extend(sorted(llmfit_root.glob('*.gguf')))
        else:
            notes.append(f'llmfit cache not found: {llmfit_root}')

        for gguf in candidates:
            resolved = str(gguf.resolve())
            if resolved in existing_paths:
                continue
            model = detected_model_from_path(gguf, self.models)
            self.add_or_update(model)
            existing_paths.add(resolved)
            added.append(model.id)

        if added:
            return len(added), added
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
            provider_key = f'llama-{model.id}'
            provider[provider_key] = {
                'npm': '@ai-sdk/openai-compatible',
                'name': f'llama.cpp {model.name}',
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
            return f'llama-{model.id}/{model.alias}'

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


def detected_model_from_path(path: Path, existing_models: List[ModelConfig]) -> ModelConfig:
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
        ('path', initial.path),
        ('alias', initial.alias or initial.id),
        ('port', str(initial.port)),
        ('host', initial.host),
        ('ctx', str(initial.ctx)),
        ('threads', str(initial.threads)),
        ('ngl', str(initial.ngl)),
        ('temp', str(initial.temp)),
        ('parallel', str(initial.parallel)),
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
            threads=int(answers['threads']),
            ngl=int(answers['ngl']),
            temp=float(answers['temp']),
            parallel=int(answers['parallel']),
            cache_ram=int(answers['cache_ram']),
            output=int(answers['output']),
            enabled=answers['enabled true/false'].lower() == 'true',
            flash_attn=answers['flash_attn true/false'].lower() == 'true',
            jinja=answers['jinja true/false'].lower() == 'true',
            extra_args=answers['extra_args (space-separated)'].split() if answers['extra_args (space-separated)'] else [],
        )
    except Exception:
        return None


def prompt_settings(stdscr, app: AppConfig) -> bool:
    o = app.opencode
    answers = prompt_value(stdscr, 'Settings', [
        ('llama_server', app.llama_server),
        ('hf_cache_root', app.hf_cache_root),
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
        app.hf_cache_root = answers['hf_cache_root']
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
        stdscr.addstr(header_y + 2, 2, f'hf-cache: {app.hf_cache_root}', colors['muted'])
        stdscr.addstr(header_y + 3, 2, f'llmfit-cache: {app.llmfit_cache_root}', colors['muted'])
        stdscr.addstr(header_y + 4, 2, f'opencode: {app.opencode.path or "<unset>"}', colors['muted'])

        msg_attr = colors['accent'] | curses.A_BOLD
        if message.startswith('✅'):
            msg_attr = colors['success'] | curses.A_BOLD
        elif message.startswith('❌'):
            msg_attr = colors['error'] | curses.A_BOLD
        elif message.startswith('⏳'):
            msg_attr = colors['warning'] | curses.A_BOLD
        stdscr.addstr(header_y + 5, 2, message[: max(10, w - 4)], msg_attr)

        box_top = header_y + 6
        left_w = max(56, min(84, w // 2))
        right_x = left_w + 2
        right_w = max(38, w - right_x - 2)
        visible_rows = max(8, h - box_top - 6)

        draw_box(stdscr, box_top, 1, h - box_top - 4, left_w, 'Models', colors['accent'] | curses.A_BOLD, colors['accent'])
        draw_box(stdscr, box_top, right_x, h - box_top - 4, right_w, 'Details / Logs / Roles', colors['accent'] | curses.A_BOLD, colors['accent'])

        header = ' ID              PRT  ST        RLS  NAME'
        stdscr.addstr(box_top + 2, 3, header, colors['accent'] | curses.A_UNDERLINE | curses.A_BOLD)

        if app.models:
            start_idx = max(0, selected - visible_rows + 3)
            end_idx = min(len(app.models), start_idx + visible_rows)
            for idx in range(start_idx, end_idx):
                model = app.models[idx]
                status, _ = statuses.get(model.id, ('?', ''))
                roles = app.role_badges(model.id)
                name_col_width = max(10, left_w - 41)
                line = f' {model.id[:14]:14} {model.port:4}  {status[:8]:8}  {roles:3}  {model.name[:name_col_width]}'
                row_y = box_top + 3 + idx - start_idx
                if idx == selected:
                    try:
                        stdscr.addstr(row_y, 3, line[: left_w - 3], colors['selection'] | curses.A_BOLD)
                    except curses.error:
                        stdscr.addstr(row_y, 3, line[: left_w - 3], curses.A_REVERSE)
                else:
                    stdscr.addstr(row_y, 3, line[: left_w - 3])
                    status_x = 3 + 1 + 14 + 1 + 4 + 2
                    stdscr.addstr(row_y, status_x, f'{status[:8]:8}', status_attr(colors, status))
        else:
            stdscr.addstr(box_top + 3, 3, 'No models yet. Press x to detect from Hugging Face cache or a to add manually.', colors['warning'])

        if app.models:
            model = app.models[selected]
            status, detail = statuses.get(model.id, ('?', ''))
            lines = [
                f'name: {model.name}',
                f'id: {model.id}',
                f'path: {model.path}',
                f'alias: {model.alias}',
                f'bind: {model.host}:{model.port}',
                f'ctx={model.ctx} threads={model.threads} ngl={model.ngl} temp={model.temp}',
                f'parallel={model.parallel} cache_ram={model.cache_ram} output={model.output}',
                f'flags: enabled={model.enabled} flash_attn={model.flash_attn} jinja={model.jinja}',
                f'status: {status} ({detail})',
                f'pid: {app.get_pid(model) or "-"}',
                f'roles: {app.role_badges(model.id)}  [m main] [s small] [b build] [p plan]',
                f'log: {app.logfile(model.id)}',
                'command preview:',
                ' '.join(app.build_command(model))[: right_w - 6],
                '',
                'last log lines:',
            ]
            lines.extend(tail_text(app.logfile(model.id), max_lines=max(8, h - box_top - 22)))
            for i, line in enumerate(lines[: h - box_top - 7]):
                attr = curses.A_NORMAL
                if i == 8:
                    attr = status_attr(colors, status)
                elif i in (12, 15):
                    attr = colors['accent'] | curses.A_BOLD
                stdscr.addstr(box_top + 2 + i, right_x + 2, line[: right_w - 4], attr)

        footer = '[Enter] start/stop  [x] detect  [X] prune  [g] gen opencode  [o] settings  [a/e/d] models  [S] stop-all  [q] quit'
        footer2 = '[m] set main  [s] set small  [b] set build  [p] set plan  [r] refresh'
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
            statuses = {m.id: app.health(m) for m in app.models}
            message = 'Refreshed.'
        elif key == ord('S'):
            message = '; '.join(app.stop_all())[: max(20, w - 4)]
        elif key in (10, 13, curses.KEY_ENTER) and app.models:
            model = app.models[selected]
            status, _ = app.health(model)
            if status in ('READY', 'LOADING', 'STARTING') or app.get_pid(model):
                ok, msg = app.stop(model)
                message = f'{model.id}: {msg}'
            else:
                ok, msg = app.start(model)
                if ok:
                    ready_ok, ready_msg = app.wait_until_ready(model, timeout=120)
                    message = ready_msg
                else:
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
            message = f'Detected {count} new model(s): {", ".join(items[:5])}' if count else (items[0] if items else 'No new GGUFs found.')
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
