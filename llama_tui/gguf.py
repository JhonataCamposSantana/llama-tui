import os
import struct
from pathlib import Path
from typing import Dict, List, Optional

from .models import ModelConfig


def model_file_size(model: ModelConfig) -> int:
    target = Path(getattr(model, 'path', '') or '').expanduser()
    try:
        return target.stat().st_size if target.exists() else 0
    except Exception:
        return 0

GGUF_VALUE_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}

GGUF_METADATA_CACHE: Dict[str, Dict[str, object]] = {}

def _read_exact(file_obj, size: int) -> bytes:
    data = file_obj.read(size)
    if len(data) != size:
        raise EOFError('unexpected EOF while reading GGUF metadata')
    return data
def _read_gguf_string(file_obj) -> str:
    length = struct.unpack('<Q', _read_exact(file_obj, 8))[0]
    return _read_exact(file_obj, length).decode('utf-8', errors='replace')
def _read_gguf_scalar(file_obj, value_type: int):
    if value_type == 0:
        return struct.unpack('<B', _read_exact(file_obj, 1))[0]
    if value_type == 1:
        return struct.unpack('<b', _read_exact(file_obj, 1))[0]
    if value_type == 2:
        return struct.unpack('<H', _read_exact(file_obj, 2))[0]
    if value_type == 3:
        return struct.unpack('<h', _read_exact(file_obj, 2))[0]
    if value_type == 4:
        return struct.unpack('<I', _read_exact(file_obj, 4))[0]
    if value_type == 5:
        return struct.unpack('<i', _read_exact(file_obj, 4))[0]
    if value_type == 6:
        return struct.unpack('<f', _read_exact(file_obj, 4))[0]
    if value_type == 7:
        return bool(struct.unpack('<?', _read_exact(file_obj, 1))[0])
    if value_type == 8:
        return _read_gguf_string(file_obj)
    if value_type == 10:
        return struct.unpack('<Q', _read_exact(file_obj, 8))[0]
    if value_type == 11:
        return struct.unpack('<q', _read_exact(file_obj, 8))[0]
    if value_type == 12:
        return struct.unpack('<d', _read_exact(file_obj, 8))[0]
    raise ValueError(f'unsupported GGUF metadata type: {value_type}')
def _skip_gguf_value(file_obj, value_type: int):
    if value_type in GGUF_VALUE_SIZES:
        file_obj.seek(GGUF_VALUE_SIZES[value_type], os.SEEK_CUR)
        return
    if value_type == 8:
        length = struct.unpack('<Q', _read_exact(file_obj, 8))[0]
        file_obj.seek(length, os.SEEK_CUR)
        return
    if value_type == 9:
        item_type = struct.unpack('<I', _read_exact(file_obj, 4))[0]
        item_count = struct.unpack('<Q', _read_exact(file_obj, 8))[0]
        if item_type in GGUF_VALUE_SIZES:
            file_obj.seek(GGUF_VALUE_SIZES[item_type] * item_count, os.SEEK_CUR)
            return
        for _ in range(item_count):
            _skip_gguf_value(file_obj, item_type)
        return
    raise ValueError(f'unsupported GGUF metadata type: {value_type}')
def _gguf_has_kv_fields(metadata: Dict[str, object]) -> bool:
    arch = str(metadata.get('general.architecture') or '')
    if not arch:
        return False
    prefix = f'{arch}.'
    has_layers = f'{prefix}block_count' in metadata
    has_heads = f'{prefix}attention.head_count' in metadata
    has_embd = f'{prefix}embedding_length' in metadata or f'{prefix}attention.key_length' in metadata
    return has_layers and has_heads and has_embd
def read_gguf_metadata(path: str | Path) -> Dict[str, object]:
    target = str(Path(path).expanduser().resolve(strict=False))
    cached = GGUF_METADATA_CACHE.get(target)
    if cached is not None:
        return cached
    metadata: Dict[str, object] = {}
    p = Path(target)
    if not p.exists() or p.suffix.lower() != '.gguf':
        GGUF_METADATA_CACHE[target] = metadata
        return metadata
    try:
        with open(p, 'rb') as file_obj:
            if _read_exact(file_obj, 4) != b'GGUF':
                GGUF_METADATA_CACHE[target] = metadata
                return metadata
            _version = struct.unpack('<I', _read_exact(file_obj, 4))[0]
            _tensor_count = struct.unpack('<Q', _read_exact(file_obj, 8))[0]
            metadata_count = struct.unpack('<Q', _read_exact(file_obj, 8))[0]
            for _ in range(metadata_count):
                key = _read_gguf_string(file_obj)
                value_type = struct.unpack('<I', _read_exact(file_obj, 4))[0]
                if key.startswith('tokenizer.') and _gguf_has_kv_fields(metadata):
                    _skip_gguf_value(file_obj, value_type)
                    break
                if value_type == 9:
                    _skip_gguf_value(file_obj, value_type)
                else:
                    metadata[key] = _read_gguf_scalar(file_obj, value_type)
    except Exception:
        metadata = {}
    GGUF_METADATA_CACHE[target] = metadata
    return metadata
def extra_arg_value(args: List[str], *flags: str) -> Optional[str]:
    flag_set = set(flags)
    for idx, token in enumerate(args):
        if token in flag_set:
            if idx + 1 < len(args):
                return args[idx + 1]
            return None
        for flag in flag_set:
            prefix = f'{flag}='
            if token.startswith(prefix):
                return token[len(prefix):]
    return None
def has_extra_flag(args: List[str], *flags: str) -> bool:
    flag_set = set(flags)
    return any(token in flag_set or any(token.startswith(f'{flag}=') for flag in flag_set) for token in args)
def strip_extra_args(args: List[str], *flags: str) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    flag_set = set(flags)
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token in flag_set:
            skip_next = True
            continue
        if any(token.startswith(f'{flag}=') for flag in flag_set):
            continue
        cleaned.append(token)
    return cleaned
def set_model_extra_arg(model: ModelConfig, flag: str, value: str):
    model.extra_args = strip_extra_args(list(getattr(model, 'extra_args', []) or []), flag)
    model.extra_args += [flag, value]
def cache_type_bytes(cache_type: str) -> float:
    normalized = (cache_type or 'f16').strip().lower()
    return {
        'f32': 4.0,
        'fp32': 4.0,
        'f16': 2.0,
        'fp16': 2.0,
        'bf16': 2.0,
        'q8_0': 1.0625,
        'q6_k': 0.8125,
        'q5_0': 0.6875,
        'q5_1': 0.75,
        'q4_0': 0.5625,
        'q4_1': 0.625,
        'iq4_nl': 0.5625,
    }.get(normalized, 2.0)
def selected_cache_type(model: ModelConfig, side: str) -> str:
    args = list(getattr(model, 'extra_args', []) or [])
    if side == 'k':
        return extra_arg_value(args, '--cache-type-k', '-ctk') or extra_arg_value(args, '--cache-type', '-ct') or 'f16'
    return extra_arg_value(args, '--cache-type-v', '-ctv') or extra_arg_value(args, '--cache-type', '-ct') or 'f16'
def gguf_architecture(model: ModelConfig) -> str:
    metadata = read_gguf_metadata(getattr(model, 'path', '') or '')
    return str(metadata.get('general.architecture') or '')
def gguf_metadata_int(model: ModelConfig, suffix: str, default: int = 0) -> int:
    metadata = read_gguf_metadata(getattr(model, 'path', '') or '')
    arch = str(metadata.get('general.architecture') or '')
    if not arch:
        return default
    try:
        return int(metadata.get(f'{arch}.{suffix}') or default)
    except Exception:
        return default
def gguf_layer_count(model: ModelConfig) -> int:
    return gguf_metadata_int(model, 'block_count', 0)
def estimate_kv_bytes_per_token(model: ModelConfig) -> int:
    runtime = getattr(model, 'runtime', 'llama.cpp')
    if runtime == 'vllm':
        return 32768
    metadata = read_gguf_metadata(getattr(model, 'path', '') or '')
    arch = str(metadata.get('general.architecture') or '')
    if arch:
        prefix = f'{arch}.'
        try:
            layers = int(metadata.get(f'{prefix}block_count') or 0)
            heads = int(metadata.get(f'{prefix}attention.head_count') or 0)
            kv_heads = int(metadata.get(f'{prefix}attention.head_count_kv') or heads)
            embedding = int(metadata.get(f'{prefix}embedding_length') or 0)
            key_length = int(metadata.get(f'{prefix}attention.key_length') or (embedding // heads if heads else 0))
            value_length = int(metadata.get(f'{prefix}attention.value_length') or key_length)
            if layers > 0 and kv_heads > 0 and key_length > 0 and value_length > 0:
                k_bytes = cache_type_bytes(selected_cache_type(model, 'k'))
                v_bytes = cache_type_bytes(selected_cache_type(model, 'v'))
                estimated = layers * kv_heads * ((key_length * k_bytes) + (value_length * v_bytes))
                return max(1, int(estimated * 1.08))
        except Exception:
            pass
    size = model_file_size(model)
    if size <= 0:
        return 32768
    if size < 8 * 1024**3:
        return 16384
    if size < 20 * 1024**3:
        return 32768
    return 65536
