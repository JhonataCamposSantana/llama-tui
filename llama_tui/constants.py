import os
from pathlib import Path


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback.expanduser()


def _first_existing(paths: list[Path], fallback: Path) -> Path:
    for path in paths:
        expanded = path.expanduser()
        if expanded.exists():
            return expanded
    return fallback.expanduser()


def default_lm_studio_home() -> Path:
    env_home = os.environ.get('LM_STUDIO_HOME')
    if env_home:
        return Path(env_home).expanduser()
    pointer = Path.home() / '.lmstudio-home-pointer'
    try:
        pointed = pointer.read_text(encoding='utf-8').strip()
    except OSError:
        pointed = ''
    if pointed:
        return Path(pointed).expanduser()
    return Path.home() / '.lmstudio'


def default_lm_studio_model_roots() -> list[Path]:
    home = default_lm_studio_home()
    return [
        home / 'models',
        home / 'hub' / 'models',
    ]

SCRIPT_DIR = Path(__file__).resolve().parent.parent
APP_NAME = 'llama-tui'
CONFIG_DIR = Path.home() / '.config' / APP_NAME
DATA_DIR = Path.home() / '.local' / 'share' / APP_NAME
CACHE_DIR = Path.home() / '.cache' / APP_NAME
DEFAULT_CONFIG_PATH = Path(os.environ['LLAMA_TUI_CONFIG']).expanduser() if os.environ.get('LLAMA_TUI_CONFIG') else (CONFIG_DIR / 'models.json')
DEFAULT_HOST = '127.0.0.1'
DEFAULT_HF_CACHE = _env_path(
    'HF_CACHE_ROOT',
    Path(os.environ['HF_HOME']).expanduser() / 'hub'
    if os.environ.get('HF_HOME')
    else _env_path('HF_HUB_CACHE', Path.home() / '.cache' / 'huggingface' / 'hub'),
)
DEFAULT_LLMFIT_CACHE = Path(os.environ.get('LLMFIT_CACHE_ROOT', Path.home() / '.cache' / 'llmfit' / 'models')).expanduser()
DEFAULT_LLM_MODELS_CACHE = Path(os.environ.get('LLM_MODELS_CACHE_ROOT', Path.home() / '.cache' / 'llm-models')).expanduser()
DEFAULT_LM_STUDIO_MODEL_ROOTS = default_lm_studio_model_roots()
DEFAULT_LLAMA_SERVER = os.environ.get('LLAMA_SERVER') or str(_first_existing(
    [
        Path.home() / 'llama.cpp' / 'build' / 'bin' / 'llama-server',
        Path.home() / 'llama.cpp' / 'build' / 'bin' / 'server',
        Path('/usr/local/bin/llama-server'),
        Path('/usr/bin/llama-server'),
    ],
    Path.home() / 'llama.cpp' / 'build' / 'bin' / 'llama-server',
))
DEFAULT_VLLM_COMMAND = os.environ.get('VLLM_COMMAND', 'vllm')
REFRESH_SECONDS = 2.0
LOGO = [
    "  _ _                         _         _       _ ",
    " | | | __ _ _ __ ___   __ _  | |_ _   _(_)_   _| |",
    " | | |/ _` | '_ ` _ \\ / _` | | __| | | | | | | |",
    " | | | (_| | | | | | | (_| | | |_| |_| | | |_| |_|",
    " |_|_|\\__,_|_| |_| |_|\\__,_|  \\__|\\__,_|_|\\__,_(_)",
]
