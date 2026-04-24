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
    return parser


def engine_session_lock_path() -> Path:
    return CACHE_DIR / 'runtime_engine_session.lock'


def ensure_engine_session_lock(engine: str) -> Path:
    lock_path = engine_session_lock_path()
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding='utf-8'))
        except Exception:
            payload = {}
        running_pid = int(payload.get('pid', 0) or 0)
        running_engine = str(payload.get('engine', '') or '').strip() or 'llama.cpp'
        if running_pid > 0:
            try:
                os.kill(running_pid, 0)
            except OSError:
                running_pid = 0
        if running_pid > 0 and running_engine != engine:
            raise SystemExit(
                f'Engine switch blocked: llama-tui PID {running_pid} is running with engine "{running_engine}". '
                f'Stop it before launching with "{engine}".'
            )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({'pid': os.getpid(), 'engine': engine}, indent=2) + '\n',
        encoding='utf-8',
    )
    return lock_path


def release_engine_session_lock(lock_path: Path):
    try:
        payload = json.loads(lock_path.read_text(encoding='utf-8'))
    except Exception:
        payload = {}
    if int(payload.get('pid', 0) or 0) == os.getpid():
        try:
            lock_path.unlink()
        except OSError:
            pass


def main():
    args = build_cli_parser().parse_args()
    if args.engine == 'buun' and args.kv and args.kv not in BUUN_KV_MODES:
        raise SystemExit(f'Unsupported --kv "{args.kv}". Supported buun modes: {", ".join(BUUN_KV_MODES)}')
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
