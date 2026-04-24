import re
from dataclasses import asdict
from pathlib import Path
from typing import List

from .gguf import (
    apply_architecture_info,
    architecture_label,
    detect_architecture_info,
    read_gguf_metadata,
)
from .models import ModelConfig

GENERIC_DISCOVERY_CTX = 2048
GENERIC_DISCOVERY_CTX_MAX = 131072
GENERIC_DISCOVERY_MEMORY_RESERVE = 40


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
def gguf_context_max(path: Path) -> int:
    metadata = read_gguf_metadata(path)
    arch = str(metadata.get('general.architecture') or '')
    keys = ['general.context_length']
    if arch:
        keys.append(f'{arch}.context_length')
    for key in keys:
        try:
            value = int(metadata.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return max(GENERIC_DISCOVERY_CTX, value)
    return GENERIC_DISCOVERY_CTX_MAX
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
def display_offload(model: ModelConfig) -> str:
    runtime = (getattr(model, 'runtime', 'llama.cpp') or 'llama.cpp').strip().lower()
    if runtime in ('vllm', 'ollama'):
        return 'GPU'
    try:
        ngl = int(getattr(model, 'ngl', 0) or 0)
    except Exception:
        ngl = 0
    if ngl <= 0:
        return 'CPU'
    if ngl >= 999:
        return 'GPU full'
    return 'GPU partial'
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
    if (getattr(model, 'architecture_type', '') or '').strip().lower() in ('dense', 'moe'):
        return architecture_label(model)
    return architecture_label(apply_architecture_info(ModelConfig(**asdict(model)), detect_architecture_info(model)))
def detected_model_from_path(path: Path, existing_models: List[ModelConfig], source: str = 'manual') -> ModelConfig:
    stem = path.stem
    name = pretty_name_from_filename(stem)
    base_id = slugify(name)
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
    ctx_max = gguf_context_max(path)
    model = ModelConfig(
        id=model_id,
        name=name,
        path=str(path),
        alias=slugify(stem),
        port=port,
        ctx=GENERIC_DISCOVERY_CTX,
        ctx_min=GENERIC_DISCOVERY_CTX,
        ctx_max=ctx_max,
        ngl=0,
        temp=0.7,
        threads=6,
        parallel=1,
        cache_ram=0,
        flash_attn=True,
        jinja=True,
        output=2048,
        optimize_mode='max_context_safe',
        optimize_tier='safe',
        memory_reserve_percent=GENERIC_DISCOVERY_MEMORY_RESERVE,
        default_benchmark_status='pending',
        source=source,
        extra_args=[],
    )
    return apply_architecture_info(model, detect_architecture_info(model))
