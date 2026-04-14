import re
from pathlib import Path
from typing import List, Tuple

from .models import ModelConfig


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
