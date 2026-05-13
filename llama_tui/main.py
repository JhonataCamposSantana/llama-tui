import atexit
import argparse
import curses
import json
import os
import signal
import sys
import time
from pathlib import Path

from .app import AppConfig
from .constants import CONFIG_DIR, DATA_DIR, CACHE_DIR, DEFAULT_CONFIG_PATH, SCRIPT_DIR, DEFAULT_LLAMA_SERVER
from .runtime_profiles import BUUN_KV_MODES, TQ3_KV_MODES, TURBOQUANT_KV_MODES, make_runtime_profile
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
    parser = argparse.ArgumentParser(
        description='llama-tui: local LLM control plane for llama.cpp, llama.cpp MTP, TurboQuant+, llama.cpp-tq3, Buun, and vLLM models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''examples:
  llama-tui
  llama-tui {DEFAULT_CONFIG_PATH}
  llama-tui --engine turboquant --kv-key q8_0 --kv-value turbo4
  llama-tui --engine llama.cpp-mtp
  llama-tui --engine tq3
  llama-tui --engine buun --kill-existing

defaults:
  supported runtimes: llama.cpp, llama.cpp-mtp, turboquant, tq3, buun, vLLM saved model entries
  config path: {DEFAULT_CONFIG_PATH}
  llama.cpp binary: LLAMA_SERVER or config llama_server
  llama.cpp MTP binary: LLAMA_CPP_MTP_PATH
  TurboQuant+ binary: TURBOQUANT_LLAMA_SERVER_BIN
  llama.cpp-tq3 binary: TQ3_LLAMA_SERVER_BIN
  Buun binary: BUUN_LLAMA_SERVER_BIN
  vLLM command: VLLM_COMMAND or config vllm_command

notes:
  --help exits before curses starts.
  --engine switches the active GGUF engine for this TUI session.
  vLLM model entries still use their saved runtime and vllm_command.
''',
    )
    parser.add_argument('config_path', nargs='?', default=str(DEFAULT_CONFIG_PATH), help='models.json config path (default: %(default)s)')
    parser.add_argument('--engine', choices=('llama.cpp', 'llama.cpp-mtp', 'buun', 'turboquant', 'tq3'), default='llama.cpp', help='active GGUF engine for this session (default: %(default)s)')
    parser.add_argument('--ctx', type=int, default=None, help='optional session context override for the active GGUF engine')
    parser.add_argument('--kv', default='', help='KV cache mode shorthand for Buun/TurboQuant+/TQ3 sessions')
    parser.add_argument('--kv-key', default='', help='key-cache mode for Buun/TurboQuant+/TQ3 sessions')
    parser.add_argument('--kv-value', default='', help='value-cache mode for Buun/TurboQuant+/TQ3 sessions')
    parser.add_argument(
        '--kill-existing',
        action='store_true',
        help='stop known blocking llama-tui engine sessions before startup',
    )
    return parser


def engine_session_lock_path() -> Path:
    return CACHE_DIR / 'runtime_engine_session.lock'


def engine_session_dir() -> Path:
    return CACHE_DIR / 'runtime_engine_sessions'


def engine_session_path(pid: int | None = None) -> Path:
    return engine_session_dir() / f'{int(pid or os.getpid())}.json'


def pid_state(pid: int) -> str:
    try:
        text = (Path('/proc') / str(int(pid)) / 'stat').read_text(encoding='utf-8', errors='replace')
    except Exception:
        return ''
    close_paren = text.rfind(')')
    if close_paren < 0:
        return ''
    fields = text[close_paren + 2:].split()
    return fields[0] if fields else ''


def pid_status(pid: int) -> str:
    try:
        lines = (Path('/proc') / str(int(pid)) / 'status').read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        return pid_state(pid)
    for line in lines:
        if line.startswith('State:'):
            return line.split(':', 1)[1].strip()
    return pid_state(pid)


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return pid_state(pid) not in {'Z', 'X'}


def describe_pid(pid: int) -> dict:
    detail = {
        'pid': int(pid),
        'command': '',
        'cwd': '',
        'status': '',
        'state': '',
    }
    if pid <= 0:
        return detail
    proc_dir = Path('/proc') / str(int(pid))
    try:
        raw_cmdline = (proc_dir / 'cmdline').read_bytes()
        parts = [part.decode('utf-8', errors='replace') for part in raw_cmdline.split(b'\0') if part]
        detail['command'] = ' '.join(parts)
    except Exception:
        pass
    if not detail['command']:
        try:
            detail['command'] = (proc_dir / 'comm').read_text(encoding='utf-8', errors='replace').strip()
        except Exception:
            pass
    try:
        detail['cwd'] = os.readlink(proc_dir / 'cwd')
    except Exception:
        pass
    detail['state'] = pid_state(pid)
    detail['status'] = pid_status(pid)
    return detail


def read_engine_session(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def session_process_detail(payload: dict, pid: int) -> dict:
    detail = describe_pid(pid)
    for key in ('command', 'cwd'):
        if not detail.get(key) and payload.get(key):
            detail[key] = str(payload.get(key) or '')
    return detail


def prune_dead_engine_sessions() -> list[dict]:
    sessions: list[dict] = []
    session_dir = engine_session_dir()
    session_dir.mkdir(parents=True, exist_ok=True)
    for path in session_dir.glob('*.json'):
        payload = read_engine_session(path)
        running_pid = int(payload.get('pid', 0) or 0)
        running_engine = str(payload.get('engine', '') or '').strip() or 'llama.cpp'
        if pid_is_alive(running_pid):
            sessions.append(
                {
                    'pid': running_pid,
                    'engine': running_engine,
                    'path': path,
                    'process': session_process_detail(payload, running_pid),
                }
            )
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
        return {
            'pid': running_pid,
            'engine': running_engine,
            'path': lock_path,
            'process': session_process_detail(payload, running_pid),
        }
    try:
        lock_path.unlink()
    except OSError:
        pass
    return {}


def _dedupe_engine_sessions(sessions: list[dict]) -> list[dict]:
    seen = set()
    deduped: list[dict] = []
    for session in sessions:
        key = (int(session.get('pid', 0) or 0), str(session.get('path', '') or ''))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(session)
    return deduped


def collect_engine_sessions() -> list[dict]:
    sessions = prune_dead_engine_sessions()
    legacy_session = legacy_engine_session()
    if legacy_session:
        sessions.append(legacy_session)
    return _dedupe_engine_sessions(sessions)


def blocking_engine_sessions(engine: str) -> list[dict]:
    blockers = []
    for session in collect_engine_sessions():
        running_engine = str(session.get('engine', '') or '').strip() or 'llama.cpp'
        if running_engine != engine:
            blockers.append(session)
    return blockers


def format_engine_block_message(engine: str, sessions: list[dict]) -> str:
    count = len(sessions)
    plural = 'session' if count == 1 else 'sessions'
    lines = [
        f'Engine switch blocked: {count} llama-tui {plural} running with a different engine.',
        f'Requested engine: "{engine}"',
    ]
    for session in sessions:
        running_engine = str(session.get('engine', '') or '').strip() or 'llama.cpp'
        running_pid = int(session.get('pid', 0) or 0)
        process = session.get('process') or describe_pid(running_pid)
        lines.append(f'- PID {running_pid} engine "{running_engine}"')
        status = str(process.get('status', '') or process.get('state', '') or '').strip()
        command = str(process.get('command', '') or '').strip()
        cwd = str(process.get('cwd', '') or '').strip()
        if status:
            lines.append(f'  status: {status}')
        if command:
            lines.append(f'  command: {command}')
        if cwd:
            lines.append(f'  cwd: {cwd}')
        lines.append(f'  session: {session.get("path", "")}')
    lines.append('Re-run with --kill-existing to stop known llama-tui sessions automatically.')
    return '\n'.join(lines)


def _pid_group_is_alive(pid: int, pgid: int | None, use_group: bool) -> bool:
    if not use_group or pgid is None:
        return pid_is_alive(pid)
    proc_root = Path('/proc')
    try:
        entries = list(proc_root.iterdir())
    except Exception:
        return pid_is_alive(pid)
    for entry in entries:
        if not entry.name.isdigit():
            continue
        candidate = int(entry.name)
        try:
            if os.getpgid(candidate) == pgid and pid_is_alive(candidate):
                return True
        except OSError:
            continue
    return False


def terminate_pid_group(pid: int, grace_seconds: float = 2.0) -> tuple[bool, str]:
    pid = int(pid)
    if pid <= 0:
        return True, f'PID {pid} is not valid'
    if pid == os.getpid():
        return False, f'refusing to terminate current llama-tui PID {pid}'
    if not pid_is_alive(pid):
        return True, f'PID {pid} is already stopped'
    pgid = None
    use_group = False
    try:
        pgid = os.getpgid(pid)
        use_group = pgid > 0 and pgid != os.getpgrp()
    except OSError:
        pass
    target = f'process group {pgid}' if use_group else f'PID {pid}'

    def signal_target(sig: int):
        if use_group and pgid is not None:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)

    try:
        signal_target(signal.SIGTERM)
    except ProcessLookupError:
        return True, f'{target} is already stopped'
    except OSError as exc:
        return False, f'failed to terminate {target}: {exc}'

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if not _pid_group_is_alive(pid, pgid, use_group):
            return True, f'terminated {target}'
        time.sleep(0.05)

    try:
        signal_target(signal.SIGKILL)
    except ProcessLookupError:
        return True, f'{target} is already stopped'
    except OSError as exc:
        return False, f'failed to kill {target}: {exc}'

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _pid_group_is_alive(pid, pgid, use_group):
            return True, f'killed {target}'
        time.sleep(0.05)
    return False, f'{target} did not stop'


def terminate_engine_sessions(sessions: list[dict]) -> tuple[bool, list[str], int]:
    ok = True
    messages: list[str] = []
    stopped_count = 0
    for session in sessions:
        pid = int(session.get('pid', 0) or 0)
        stopped, message = terminate_pid_group(pid)
        messages.append(message)
        if stopped:
            stopped_count += 1
            path = session.get('path')
            if isinstance(path, Path):
                try:
                    path.unlink()
                except OSError:
                    pass
        else:
            ok = False
    return ok, messages, stopped_count


def prompt_kill_engine_sessions(engine: str, sessions: list[dict]) -> bool:
    print(format_engine_block_message(engine, sessions), file=sys.stderr)
    try:
        answer = input('Kill blocking llama-tui session(s) and continue? [y/N] ')
    except EOFError:
        return False
    return answer.strip().lower() in {'y', 'yes'}


_LAST_ENGINE_SESSION_STOP_COUNT = 0


def last_engine_session_stop_count() -> int:
    return _LAST_ENGINE_SESSION_STOP_COUNT


def ensure_engine_session_lock(
    engine: str,
    *,
    kill_existing: bool = False,
    interactive: bool = False,
    prompt_fn=None,
) -> Path:
    global _LAST_ENGINE_SESSION_STOP_COUNT
    _LAST_ENGINE_SESSION_STOP_COUNT = 0
    blockers = blocking_engine_sessions(engine)
    if blockers:
        should_kill = kill_existing
        if not should_kill and interactive:
            should_kill = (prompt_fn or prompt_kill_engine_sessions)(engine, blockers)
            if not should_kill:
                raise SystemExit('Startup canceled. Existing llama-tui session was left running.')
        if not should_kill:
            raise SystemExit(format_engine_block_message(engine, blockers))
        ok, messages, stopped_count = terminate_engine_sessions(blockers)
        _LAST_ENGINE_SESSION_STOP_COUNT = stopped_count
        if not ok:
            detail = '\n'.join(messages)
            raise SystemExit(f'{format_engine_block_message(engine, blockers)}\n{detail}')
        blockers = blocking_engine_sessions(engine)
        if blockers:
            raise SystemExit(
                f'{format_engine_block_message(engine, blockers)}\n'
                'Some blocking llama-tui sessions are still alive after the kill attempt.'
            )
    lock_path = engine_session_path()
    lock_path.write_text(
        json.dumps(
            {
                'pid': os.getpid(),
                'engine': engine,
                'command': ' '.join(sys.argv),
                'cwd': os.getcwd(),
            },
            indent=2,
        ) + '\n',
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


def validate_turboquant_kv_args(args):
    if args.engine != 'turboquant':
        return
    for flag, value in (
        ('--kv', args.kv),
        ('--kv-key', args.kv_key),
        ('--kv-value', args.kv_value),
    ):
        if value and value not in TURBOQUANT_KV_MODES:
            raise SystemExit(
                f'Unsupported {flag} "{value}". Supported TurboQuant+ modes: {", ".join(TURBOQUANT_KV_MODES)}'
            )


def validate_tq3_kv_args(args):
    if args.engine != 'tq3':
        return
    for flag, value in (
        ('--kv', args.kv),
        ('--kv-key', args.kv_key),
        ('--kv-value', args.kv_value),
    ):
        if value and value not in TQ3_KV_MODES:
            raise SystemExit(
                f'Unsupported {flag} "{value}". Supported llama.cpp-tq3 modes: {", ".join(TQ3_KV_MODES)}'
            )


def main():
    args = build_cli_parser().parse_args()
    validate_buun_kv_args(args)
    validate_turboquant_kv_args(args)
    validate_tq3_kv_args(args)
    config_path = Path(args.config_path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_bootstrap_files(config_path)
    lock_path = ensure_engine_session_lock(
        args.engine,
        kill_existing=args.kill_existing,
        interactive=sys.stdin.isatty() and sys.stdout.isatty(),
    )
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
    if last_engine_session_stop_count():
        cleanup()
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
