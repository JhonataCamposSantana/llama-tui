import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
GGUF_TENSOR_DESCRIPTOR_CACHE: Dict[str, List[Dict[str, object]]] = {}


@dataclass
class ArchitectureInfo:
    architecture: str = ''
    architecture_type: str = 'unknown'
    model_family: str = ''
    expert_count: int = 0
    expert_used_count: int = 0
    expert_shared_count: int = 0
    expert_group_count: int = 0
    expert_group_used_count: int = 0
    moe_every_n_layers: int = 0
    leading_dense_block_count: int = 0
    active_expert_ratio: float = 0.0
    confidence: float = 0.0
    source: str = ''
    reason: str = ''


@dataclass
class TurboQuantInfo:
    status: str = 'unknown'
    head_dim: int = 0
    key_dim: int = 0
    value_dim: int = 0
    source: str = ''
    reason: str = ''


TURBOQUANT_STATUSES = ('native', 'padded', 'incompatible', 'unknown', 'not_applicable')

GGUF_EXPERT_SUFFIXES = (
    'expert_count',
    'expert_used_count',
    'expert_shared_count',
    'expert_group_count',
    'expert_group_used_count',
    'moe_every_n_layers',
    'leading_dense_block_count',
    'expert_feed_forward_length',
    'expert_shared_feed_forward_length',
)

STRONG_MOE_ARCHITECTURES = (
    'qwen2moe',
    'qwen3moe',
    'mixtral',
    'deepseek2',
    'deepseek-v2',
    'deepseek-v3',
    'gpt-oss',
    'glm-moe',
    'glm4moe',
    'switch',
)

TENSOR_MOE_PATTERNS = (
    re.compile(r'ffn_(?:gate|down|up)_exps', re.IGNORECASE),
    re.compile(r'(?:^|\.)experts?(?:\.|$|_)', re.IGNORECASE),
    re.compile(r'(?:^|\.|_)router(?:\.|_|$)', re.IGNORECASE),
    re.compile(r'(?:^|\.|_)gate_inp(?:\.|_|$)', re.IGNORECASE),
)

GGML_TYPE_SIZE = {
    0: 4.0,       # F32
    1: 2.0,       # F16
    2: 0.5625,    # Q4_0
    3: 0.625,     # Q4_1
    6: 1.0625,    # Q8_0
    7: 0.6875,    # Q5_0
    8: 0.75,      # Q5_1
    9: 0.8125,    # Q6_K
    10: 0.5625,   # Q4_K
    11: 0.625,    # Q5_K
    12: 0.4375,   # Q3_K
    13: 0.3125,   # Q2_K
    14: 0.5625,   # IQ4_NL
    15: 0.5,      # IQ4_XS
    16: 1.0,      # I8
    17: 0.5,      # I4
    18: 0.25,     # I2
    19: 0.125,    # I1
    20: 2.0,      # BF16
}

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
    source_path = Path(path).expanduser()
    target = str(source_path.resolve(strict=False))
    cached = GGUF_METADATA_CACHE.get(target)
    if cached is not None:
        return cached
    metadata: Dict[str, object] = {}
    if not source_path.exists() or source_path.suffix.lower() != '.gguf':
        return metadata
    try:
        with open(source_path, 'rb') as file_obj:
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


def _target_path(model_or_path) -> Path:
    if isinstance(model_or_path, ModelConfig):
        return Path(getattr(model_or_path, 'path', '') or '').expanduser()
    return Path(str(model_or_path or '')).expanduser()


def _metadata_int_value(metadata: Dict[str, object], arch: str, suffix: str) -> int:
    keys = []
    if arch:
        keys.append(f'{arch}.{suffix}')
    keys.extend(key for key in metadata if key.endswith(f'.{suffix}') and key not in keys)
    for key in keys:
        try:
            value = int(metadata.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _strong_moe_arch_signal(text: str) -> bool:
    normalized = (text or '').strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in STRONG_MOE_ARCHITECTURES) or bool(re.search(r'(^|[-_./])moe($|[-_./])', normalized))


def _filename_moe_signal(text: str) -> Tuple[bool, str, float]:
    normalized = (text or '').strip().lower()
    if not normalized:
        return False, '', 0.0
    patterns = (
        (r'(^|[^a-z0-9])moe([^a-z0-9]|$)', 'moe token', 0.65),
        (r'(^|[^a-z0-9])mixture[-_ ]of[-_ ]experts([^a-z0-9]|$)', 'mixture-of-experts token', 0.70),
        (r'(^|[^a-z0-9])(mixtral|switch|qwen2moe|qwen3moe|deepseek2|deepseek-v2|deepseek-v3|gpt-oss|glm[-_]?moe)([^a-z0-9]|$)', 'known MoE family token', 0.70),
        (r'(^|[^a-z0-9])\d+b[-_]?a\d+b([^a-z0-9]|$)', 'active-parameter name token', 0.60),
    )
    for pattern, reason, confidence in patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True, reason, confidence
    return False, '', 0.0


def _tensor_name_says_moe(name: str) -> bool:
    return any(pattern.search(name or '') for pattern in TENSOR_MOE_PATTERNS)


def _tensor_descriptor_cache_key(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def read_gguf_tensor_descriptors(path: str | Path) -> List[Dict[str, object]]:
    target_path = Path(path).expanduser()
    target = _tensor_descriptor_cache_key(target_path)
    cached = GGUF_TENSOR_DESCRIPTOR_CACHE.get(target)
    if cached is not None:
        return cached
    descriptors: List[Dict[str, object]] = []
    if not target_path.exists() or target_path.suffix.lower() != '.gguf':
        GGUF_TENSOR_DESCRIPTOR_CACHE[target] = descriptors
        return descriptors
    try:
        with open(target_path, 'rb') as file_obj:
            if _read_exact(file_obj, 4) != b'GGUF':
                GGUF_TENSOR_DESCRIPTOR_CACHE[target] = descriptors
                return descriptors
            _version = struct.unpack('<I', _read_exact(file_obj, 4))[0]
            tensor_count = struct.unpack('<Q', _read_exact(file_obj, 8))[0]
            metadata_count = struct.unpack('<Q', _read_exact(file_obj, 8))[0]
            for _ in range(metadata_count):
                _key = _read_gguf_string(file_obj)
                value_type = struct.unpack('<I', _read_exact(file_obj, 4))[0]
                _skip_gguf_value(file_obj, value_type)
            for _ in range(tensor_count):
                name = _read_gguf_string(file_obj)
                n_dims = struct.unpack('<I', _read_exact(file_obj, 4))[0]
                dims = [struct.unpack('<Q', _read_exact(file_obj, 8))[0] for _ in range(n_dims)]
                tensor_type = struct.unpack('<I', _read_exact(file_obj, 4))[0]
                offset = struct.unpack('<Q', _read_exact(file_obj, 8))[0]
                descriptors.append({
                    'name': name,
                    'dimensions': dims,
                    'type': tensor_type,
                    'offset': offset,
                })
    except Exception:
        descriptors = []
    GGUF_TENSOR_DESCRIPTOR_CACHE[target] = descriptors
    return descriptors


def _estimated_tensor_payload_bytes(descriptor: Dict[str, object]) -> int:
    dims = descriptor.get('dimensions', []) or []
    elements = 1
    try:
        for dim in dims:
            elements *= max(1, int(dim))
    except Exception:
        elements = 0
    try:
        type_size = GGML_TYPE_SIZE.get(int(descriptor.get('type', 0) or 0), 2.0)
    except Exception:
        type_size = 2.0
    return max(0, int(elements * type_size))


def _descriptor_payload_bytes(descriptors: List[Dict[str, object]]) -> List[int]:
    if not descriptors:
        return []
    ordered = sorted(enumerate(descriptors), key=lambda item: int(item[1].get('offset', 0) or 0))
    bytes_by_index = [0 for _ in descriptors]
    for ordered_idx, (original_idx, descriptor) in enumerate(ordered):
        current_offset = int(descriptor.get('offset', 0) or 0)
        if ordered_idx + 1 < len(ordered):
            next_offset = int(ordered[ordered_idx + 1][1].get('offset', current_offset) or current_offset)
            size = max(0, next_offset - current_offset)
        else:
            size = _estimated_tensor_payload_bytes(descriptor)
        bytes_by_index[original_idx] = size
    return bytes_by_index


def estimate_layer_weight_bytes_from_tensor_descriptors(path: str | Path) -> List[int]:
    descriptors = read_gguf_tensor_descriptors(path)
    if not descriptors:
        return []
    payload_sizes = _descriptor_payload_bytes(descriptors)
    layer_bytes: Dict[int, int] = {}
    layer_pattern = re.compile(r'(?:^|\.)(?:blk|block|layers?)\.(\d+)\.', re.IGNORECASE)
    for descriptor, size in zip(descriptors, payload_sizes):
        name = str(descriptor.get('name', '') or '')
        match = layer_pattern.search(name)
        if not match:
            continue
        try:
            layer = int(match.group(1))
        except Exception:
            continue
        layer_bytes[layer] = layer_bytes.get(layer, 0) + int(size or 0)
    if not layer_bytes:
        return []
    max_layer = max(layer_bytes)
    return [int(layer_bytes.get(idx, 0) or 0) for idx in range(max_layer + 1)]


def detect_architecture_info(model_or_path) -> ArchitectureInfo:
    path = _target_path(model_or_path)
    metadata = read_gguf_metadata(path)
    arch = str(metadata.get('general.architecture') or '').strip()
    info = ArchitectureInfo(
        architecture=arch,
        architecture_type='unknown',
        model_family=arch,
        source='unknown',
    )
    expert_values = {
        suffix: _metadata_int_value(metadata, arch, suffix)
        for suffix in GGUF_EXPERT_SUFFIXES
    }
    info.expert_count = expert_values.get('expert_count', 0)
    info.expert_used_count = expert_values.get('expert_used_count', 0)
    info.expert_shared_count = expert_values.get('expert_shared_count', 0)
    info.expert_group_count = expert_values.get('expert_group_count', 0)
    info.expert_group_used_count = expert_values.get('expert_group_used_count', 0)
    info.moe_every_n_layers = expert_values.get('moe_every_n_layers', 0)
    info.leading_dense_block_count = expert_values.get('leading_dense_block_count', 0)
    if info.expert_count > 0 and info.expert_used_count > 0:
        info.active_expert_ratio = info.expert_used_count / max(1, info.expert_count)

    expert_moe = (
        info.expert_count > 1
        or info.expert_used_count > 0
        or info.expert_shared_count > 0
        or info.expert_group_count > 0
        or info.expert_group_used_count > 0
        or info.moe_every_n_layers > 0
    )
    if expert_moe:
        info.architecture_type = 'moe'
        info.confidence = 1.0
        info.source = 'gguf_metadata'
        info.reason = 'GGUF expert metadata'
        return info

    if arch and _strong_moe_arch_signal(arch):
        info.architecture_type = 'moe'
        info.confidence = 0.9
        info.source = 'gguf_metadata'
        info.reason = f'GGUF architecture {arch}'
        return info

    descriptors = read_gguf_tensor_descriptors(path)
    moe_tensor = next((str(item.get('name', '') or '') for item in descriptors if _tensor_name_says_moe(str(item.get('name', '') or ''))), '')
    if moe_tensor:
        info.architecture_type = 'moe'
        info.confidence = 0.9
        info.source = 'tensor_names'
        info.reason = f'expert tensor {moe_tensor}'
        return info

    if metadata and arch:
        info.architecture_type = 'dense'
        info.confidence = 0.85
        info.source = 'gguf_metadata'
        info.reason = 'GGUF architecture metadata with no MoE expert keys'
        return info

    text_parts = []
    if isinstance(model_or_path, ModelConfig):
        text_parts.extend([
            getattr(model_or_path, 'name', '') or '',
            getattr(model_or_path, 'alias', '') or '',
            getattr(model_or_path, 'path', '') or '',
        ])
    else:
        text_parts.append(str(path))
    has_filename_signal, reason, confidence = _filename_moe_signal(' '.join(text_parts))
    if has_filename_signal:
        info.architecture_type = 'moe'
        info.confidence = confidence
        info.source = 'filename_heuristic'
        info.reason = reason
        return info

    info.confidence = 0.0
    info.source = 'unknown'
    info.reason = 'no GGUF architecture metadata or MoE signal'
    return info


def detect_turboquant_info(model_or_path) -> TurboQuantInfo:
    if isinstance(model_or_path, ModelConfig):
        runtime = (getattr(model_or_path, 'runtime', 'llama.cpp') or 'llama.cpp').strip().lower()
        if runtime != 'llama.cpp':
            return TurboQuantInfo(
                status='not_applicable',
                source='runtime',
                reason=f'TurboQuant applies to llama.cpp GGUF sessions, not {runtime}',
            )

    path = _target_path(model_or_path)
    if not str(path):
        return TurboQuantInfo(status='not_applicable', source='path', reason='model path is empty')
    if not path.exists():
        return TurboQuantInfo(status='not_applicable', source='path', reason='model path is missing')
    if path.suffix.lower() != '.gguf':
        return TurboQuantInfo(status='not_applicable', source='path', reason='model is not a GGUF file')
    if 'mmproj' in path.name.lower():
        return TurboQuantInfo(status='not_applicable', source='path', reason='projection files are not TurboQuant targets')

    metadata = read_gguf_metadata(path)
    if not metadata:
        return TurboQuantInfo(status='unknown', source='gguf_metadata', reason='GGUF metadata missing or unreadable')

    arch = str(metadata.get('general.architecture') or '').strip()
    key_dim = _metadata_int_value(metadata, arch, 'attention.key_length')
    value_dim = _metadata_int_value(metadata, arch, 'attention.value_length')
    source = 'gguf_metadata'
    fallback_dim = 0
    embedding = _metadata_int_value(metadata, arch, 'embedding_length')
    head_count = _metadata_int_value(metadata, arch, 'attention.head_count')
    if embedding > 0 and head_count > 0 and embedding >= head_count and embedding % head_count == 0:
        fallback_dim = embedding // head_count
    if key_dim <= 0 and fallback_dim > 0:
        key_dim = fallback_dim
        source = 'gguf_metadata_fallback'
    if value_dim <= 0 and fallback_dim > 0:
        value_dim = fallback_dim
        source = 'gguf_metadata_fallback'
    if key_dim <= 0 or value_dim <= 0:
        return TurboQuantInfo(
            status='unknown',
            key_dim=max(0, key_dim),
            value_dim=max(0, value_dim),
            head_dim=max(0, key_dim, value_dim),
            source=source,
            reason='attention key/value dimensions were not found',
        )

    head_dim = key_dim if key_dim == value_dim else max(key_dim, value_dim)
    native = key_dim % 128 == 0 and value_dim % 128 == 0
    if key_dim < 128 or value_dim < 128:
        reason = 'TurboKV block size 128 does not fit key/value head dims below 128'
        status = 'incompatible'
    elif native:
        reason = 'key/value head dims are multiples of 128'
        status = 'native'
    else:
        reason = 'buun zero-padding handles non-128 head dims'
        status = 'padded'
    if source == 'gguf_metadata_fallback':
        reason += ' from embedding_length / attention.head_count'
    return TurboQuantInfo(
        status=status,
        head_dim=head_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        source=source,
        reason=reason,
    )


def architecture_label(model: ModelConfig) -> str:
    architecture_type = (getattr(model, 'architecture_type', '') or 'unknown').strip().lower()
    if architecture_type == 'moe':
        expert_count = int(getattr(model, 'expert_count', 0) or 0)
        expert_used = int(getattr(model, 'expert_used_count', 0) or 0)
        if expert_count > 0 and expert_used > 0:
            return f'MoE {expert_count}x{expert_used}'
        return 'MoE'
    if architecture_type == 'dense':
        return 'Dense'
    return 'Unknown'


def architecture_detail(model: ModelConfig) -> str:
    label = architecture_label(model)
    source = getattr(model, 'classification_source', '') or 'unknown'
    confidence = float(getattr(model, 'classification_confidence', 0.0) or 0.0)
    reason = getattr(model, 'classification_reason', '') or ''
    arch = getattr(model, 'architecture', '') or getattr(model, 'model_family', '') or ''
    if label.startswith('MoE'):
        parts = [
            f'MoE: {int(getattr(model, "expert_count", 0) or 0)} total experts',
            f'{int(getattr(model, "expert_used_count", 0) or 0)} active',
        ]
        shared = int(getattr(model, 'expert_shared_count', 0) or 0)
        every = int(getattr(model, 'moe_every_n_layers', 0) or 0)
        ratio = float(getattr(model, 'active_expert_ratio', 0.0) or 0.0)
        if shared:
            parts.append(f'shared={shared}')
        if every:
            parts.append(f'every {every} layer')
        if ratio:
            parts.append(f'active_ratio={ratio:.3f}')
        parts.append(f'confidence={confidence:.2f} from {source}')
        if arch:
            parts.append(f'arch={arch}')
        if reason:
            parts.append(reason)
        return ', '.join(parts)
    if label == 'Dense':
        detail = f'Dense: confidence={confidence:.2f} from {source}'
        if arch:
            detail += f', arch={arch}'
        if reason:
            detail += f', {reason}'
        return detail
    detail = f'Unknown: confidence={confidence:.2f} from {source}'
    if reason:
        detail += f', {reason}'
    return detail


def turboquant_short(model: ModelConfig) -> str:
    status = (getattr(model, 'turboquant_status', '') or 'unknown').strip().lower()
    return {
        'native': 'NAT',
        'padded': 'PAD',
        'incompatible': 'INC',
        'unknown': 'UNK',
        'not_applicable': 'N/A',
    }.get(status, 'UNK')


def turboquant_detail(model: ModelConfig) -> str:
    status = (getattr(model, 'turboquant_status', '') or 'unknown').strip().lower()
    if status not in TURBOQUANT_STATUSES:
        status = 'unknown'
    label = 'not applicable' if status == 'not_applicable' else status
    parts = [label]
    key_dim = int(getattr(model, 'turboquant_key_dim', 0) or 0)
    value_dim = int(getattr(model, 'turboquant_value_dim', 0) or 0)
    if key_dim or value_dim:
        parts.append(f'key={key_dim or "-"} value={value_dim or "-"}')
    source = getattr(model, 'turboquant_source', '') or ''
    if source:
        parts.append(f'from {source}')
    reason = getattr(model, 'turboquant_reason', '') or ''
    if reason:
        parts.append(reason)
    return ' '.join(parts)


def apply_turboquant_info(model: ModelConfig, info: TurboQuantInfo) -> ModelConfig:
    status = (info.status or 'unknown').strip().lower()
    model.turboquant_status = status if status in TURBOQUANT_STATUSES else 'unknown'
    model.turboquant_head_dim = int(info.head_dim or 0)
    model.turboquant_key_dim = int(info.key_dim or 0)
    model.turboquant_value_dim = int(info.value_dim or 0)
    model.turboquant_source = info.source or ''
    model.turboquant_reason = info.reason or ''
    return model


def apply_architecture_info(model: ModelConfig, info: ArchitectureInfo) -> ModelConfig:
    model.architecture = info.architecture
    model.architecture_type = info.architecture_type if info.architecture_type in ('dense', 'moe', 'unknown') else 'unknown'
    model.model_family = info.model_family
    model.expert_count = int(info.expert_count or 0)
    model.expert_used_count = int(info.expert_used_count or 0)
    model.expert_shared_count = int(info.expert_shared_count or 0)
    model.expert_group_count = int(info.expert_group_count or 0)
    model.expert_group_used_count = int(info.expert_group_used_count or 0)
    model.moe_every_n_layers = int(info.moe_every_n_layers or 0)
    model.leading_dense_block_count = int(info.leading_dense_block_count or 0)
    model.active_expert_ratio = float(info.active_expert_ratio or 0.0)
    model.classification_confidence = float(info.confidence or 0.0)
    model.classification_source = info.source
    model.classification_reason = info.reason
    return model
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
