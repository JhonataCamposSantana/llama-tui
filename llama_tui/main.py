import atexit
import argparse
import curses
import json
import os
import signal
from pathlib import Path

from .app import AppConfig
from .constants import CONFIG_DIR, DATA_DIR, CACHE_DIR, DEFAULT_CONFIG_PATH, SCRIPT_DIR, DEFAULT_LLAMA_SERVER
from .runtime_profiles import BUUN_KV_MODES, make_runtime_profile
from .ui import tui


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
        SCRIPT_DIR / 'examples' / 'models.sample.json',
        SCRIPT_DIR / 'models.json',
    ]
    for sample_path in sample_candidates:
        if sample_path.exists():
            config_path.write_text(sample_path.read_text())
            return config_path
    app = AppConfig(config_path)
    app.save()
    return config_path

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='llama-tui')
    parser.add_argument('config_path', nargs='?', default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument('--engine', choices=('llama.cpp', 'buun'), default='llama.cpp')
    parser.add_argument('--ctx', type=int, default=None)
    parser.add_argument('--kv', default='')
    parser.add_argument('--kv-key', default='')
    parser.add_argument('--kv-value', default='')
    return parser


def engine_session_lock_path() -> Path:
    return CACHE_DIR / 'runtime_engine_session.lock'


def engine_session_dir() -> Path:
    return CACHE_DIR / 'runtime_engine_sessions'


def engine_session_path(pid: int | None = None) -> Path:
    return engine_session_dir() / f'{int(pid or os.getpid())}.json'


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_engine_session(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def prune_dead_engine_sessions() -> list[dict]:
    sessions: list[dict] = []
    session_dir = engine_session_dir()
    session_dir.mkdir(parents=True, exist_ok=True)
    for path in session_dir.glob('*.json'):
        payload = read_engine_session(path)
        running_pid = int(payload.get('pid', 0) or 0)
        running_engine = str(payload.get('engine', '') or '').strip() or 'llama.cpp'
        if pid_is_alive(running_pid):
            sessions.append({'pid': running_pid, 'engine': running_engine, 'path': path})
        else:
            try:
                path.unlink()
            except OSError:
                pass
    return sessions


def legacy_engine_session() -> dict:
    lock_path = engine_session_lock_path()
    if not lock_path.exists():
        return {}
    payload = read_engine_session(lock_path)
    running_pid = int(payload.get('pid', 0) or 0)
    running_engine = str(payload.get('engine', '') or '').strip() or 'llama.cpp'
    if pid_is_alive(running_pid):
        return {'pid': running_pid, 'engine': running_engine, 'path': lock_path}
    try:
        lock_path.unlink()
    except OSError:
        pass
    return {}


def ensure_engine_session_lock(engine: str) -> Path:
    sessions = prune_dead_engine_sessions()
    legacy_session = legacy_engine_session()
    if legacy_session:
        sessions.append(legacy_session)
    for session in sessions:
        running_engine = str(session.get('engine', '') or '').strip() or 'llama.cpp'
        running_pid = int(session.get('pid', 0) or 0)
        if running_engine != engine:
            raise SystemExit(
                f'Engine switch blocked: llama-tui PID {running_pid} is running with engine "{running_engine}". '
                f'Stop it before launching with "{engine}".'
            )
    lock_path = engine_session_path()
    lock_path.write_text(
        json.dumps({'pid': os.getpid(), 'engine': engine}, indent=2) + '\n',
        encoding='utf-8',
    )
    return lock_path


def release_engine_session_lock(lock_path: Path):
    payload = read_engine_session(lock_path)
    if int(payload.get('pid', 0) or 0) == os.getpid():
        try:
            lock_path.unlink()
        except OSError:
            pass


def validate_buun_kv_args(args):
    if args.engine != 'buun':
        return
    for flag, value in (
        ('--kv', args.kv),
        ('--kv-key', args.kv_key),
        ('--kv-value', args.kv_value),
    ):
        if value and value not in BUUN_KV_MODES:
            raise SystemExit(f'Unsupported {flag} "{value}". Supported buun modes: {", ".join(BUUN_KV_MODES)}')


def main():
    args = build_cli_parser().parse_args()
    validate_buun_kv_args(args)
    config_path = Path(args.config_path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_bootstrap_files(config_path)
    lock_path = ensure_engine_session_lock(args.engine)
    app = AppConfig(
        config_path,
        runtime_profile=make_runtime_profile(
            args.engine,
            default_llama_server=DEFAULT_LLAMA_SERVER,
            ctx_override=args.ctx,
            kv_mode=args.kv,
            kv_key_mode=args.kv_key,
            kv_value_mode=args.kv_value,
        ),
    )
    cleanup = app.cleanup_managed_processes
    atexit.register(cleanup)
    atexit.register(release_engine_session_lock, lock_path)
    previous_handlers = {}

    def raise_keyboard_interrupt(_signum, _frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, raise_keyboard_interrupt)
    try:
        try:
            curses.wrapper(tui, app)
        except KeyboardInterrupt:
            pass
    finally:
        cleanup()
        try:
            atexit.unregister(cleanup)
        except Exception:
            pass
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
