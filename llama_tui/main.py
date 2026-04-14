import atexit
import curses
import signal
import sys
from pathlib import Path

from .app import AppConfig
from .constants import CONFIG_DIR, DATA_DIR, CACHE_DIR, DEFAULT_CONFIG_PATH, SCRIPT_DIR
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

def main():
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_bootstrap_files(config_path)
    app = AppConfig(config_path)
    cleanup = app.cleanup_managed_processes
    atexit.register(cleanup)
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
