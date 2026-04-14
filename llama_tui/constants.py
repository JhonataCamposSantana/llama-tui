from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent.parent
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
