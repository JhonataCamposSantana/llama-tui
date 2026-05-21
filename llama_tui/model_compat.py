import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from .engines import (
    ENGINE_BUUN,
    ENGINE_LLAMA_CPP,
    ENGINE_TQ3,
    ENGINE_TURBOQUANT,
    ENGINE_VLLM,
    normalize_engine_id,
)
from .gguf import read_gguf_metadata
from .models import ModelConfig
from .mtp import model_has_mmproj_config, mtp_support_auto_hint, normalize_mtp_support
from .provenance import parse_hf_cache_provenance
from .runtime_profiles import EngineCapabilities, mtp_spec_type_value


@dataclass(frozen=True)
class ModelEngineCompatibility:
    compatible: bool
    status: str = 'unknown'
    reason: str = ''
    severity: str = 'info'
    detected_features: Tuple[str, ...] = ()


TQ3_TEXT_RE = re.compile(r'(?i)(?:^|[^a-z0-9])tq3(?:[-_]?(?:1s|4s|native))?(?:[^a-z0-9]|$)')
MTP_TEXT_RE = re.compile(r'(?i)(?:^|[^a-z0-9])(?:gguf[-_]?mtp|native[-_]?mtp|mtp|nextn|next[-_]?n)(?:[^a-z0-9]|$)')
TURBOQUANT_TEXT_RE = re.compile(r'(?i)(?:^|[^a-z0-9])(?:turboquant|tq4(?:[-_]?(?:1s|native))?)(?:[^a-z0-9]|$)')
QUANT_TEXT_RE = re.compile(
    r'(?i)(?:^|[^a-z0-9])(?:q[2-8](?:_[a-z0-9]+)*|iq[1-4](?:_[a-z0-9]+)*|f16|bf16|f32|tq3(?:_[14]s)?)(?:[^a-z0-9]|$)'
)
COMPATIBLE_STATUSES = {'preferred', 'compatible', 'compatible_with_warning'}


def _path_for_model(model: ModelConfig) -> Path:
    return Path(str(getattr(model, 'path', '') or '')).expanduser()


def _looks_like_hf_ref(value: str) -> bool:
    value = (value or '').strip()
    if not value or ' ' in value:
        return False
    if value.lower().endswith('.gguf'):
        return False
    if value.startswith('hf://'):
        return True
    return bool(re.match(r'^[^/\s]+/[^\s]+$', value))


def _metadata_text(path: Path) -> str:
    if not path.exists() or path.suffix.lower() != '.gguf':
        return ''
    try:
        metadata = read_gguf_metadata(path)
    except Exception:
        metadata = {}
    chunks = []
    for key, value in metadata.items():
        chunks.append(str(key))
        if isinstance(value, (str, int, float, bool)):
            chunks.append(str(value))
        elif isinstance(value, (list, tuple)):
            chunks.extend(str(item) for item in value[:16])
    return ' '.join(chunks).lower()


def _model_text(model: ModelConfig) -> str:
    path = _path_for_model(model)
    provenance = parse_hf_cache_provenance(path)
    return ' '.join(
        str(item or '')
        for item in (
            getattr(model, 'id', ''),
            getattr(model, 'name', ''),
            getattr(model, 'alias', ''),
            getattr(model, 'path', ''),
            path.name,
            getattr(model, 'source', ''),
            getattr(model, 'source_path', ''),
            getattr(model, 'source_root', ''),
            getattr(model, 'source_repo_id', ''),
            getattr(model, 'source_snapshot', ''),
            ' '.join(getattr(model, 'source_labels', []) or []),
            provenance.get('repo_folder', ''),
            provenance.get('repo_id', ''),
            provenance.get('snapshot', ''),
            getattr(model, 'model_family', ''),
            getattr(model, 'architecture', ''),
            getattr(model, 'tq3_weight_format', ''),
        )
    ).lower()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _has_known_quant_hint(model: ModelConfig, combined: str) -> bool:
    quant_fields = (
        getattr(model, 'quant', ''),
        getattr(model, 'tq3_weight_format', ''),
        getattr(model, 'turboquant_quant', ''),
    )
    if any(str(item or '').strip() for item in quant_fields):
        return True
    return bool(QUANT_TEXT_RE.search(combined))


def detect_model_runtime_features(
    model: ModelConfig,
    hardware_profile=None,
    model_size_bytes: int = 0,
) -> Set[str]:
    features: Set[str] = set()
    path = _path_for_model(model)
    text = _model_text(model)
    filename_text = ' '.join(
        str(item or '')
        for item in (
            getattr(model, 'id', ''),
            getattr(model, 'name', ''),
            getattr(model, 'alias', ''),
            path.name,
            getattr(model, 'tq3_weight_format', ''),
        )
    ).lower()
    metadata_text = _metadata_text(path)
    combined = f'{metadata_text} {text}'.strip()

    target = str(getattr(model, 'path', '') or '').strip()
    runtime = str(getattr(model, 'runtime', '') or '').strip().lower()
    if runtime == ENGINE_VLLM or _looks_like_hf_ref(target):
        features.add('hf_ref')

    if path.suffix.lower() == '.gguf':
        features.add('local_gguf')

    if 'mmproj' in combined or model_has_mmproj_config(model):
        features.add('mmproj')
    if _contains_any(combined, ('vision', 'mmproj', 'clip', 'multimodal', 'vl-')):
        features.add('vision')

    architecture_type = str(getattr(model, 'architecture_type', '') or '').strip().lower()
    if (
        architecture_type == 'moe'
        or int(getattr(model, 'expert_count', 0) or 0) > 0
        or int(getattr(model, 'expert_used_count', 0) or 0) > 0
        or 'feed_forward_expert_count' in metadata_text
        or 'expert' in metadata_text
    ):
        features.add('moe')
    elif architecture_type == 'dense' or path.suffix.lower() == '.gguf' or 'hf_ref' in features:
        features.add('dense')

    tq3_status = str(getattr(model, 'tq3_status', '') or '').strip().lower()
    tq3_source = str(getattr(model, 'tq3_source', '') or '').strip().lower()
    if tq3_status == 'native':
        features.add('tq3_native')
    elif tq3_source != 'manual' and (
        'tq3_1s' in combined
        or 'tq3_4s' in combined
        or TQ3_TEXT_RE.search(combined)
        or TQ3_TEXT_RE.search(filename_text)
    ):
        features.add('tq3_native')

    mtp_support = normalize_mtp_support(getattr(model, 'supports_mtp', 'auto'))
    if mtp_support == 'yes':
        features.add('mtp_native')
    elif mtp_support != 'no' and (
        mtp_support_auto_hint(model)
        or 'speculative' in metadata_text and 'mtp' in metadata_text
        or MTP_TEXT_RE.search(combined)
    ):
        features.add('mtp_native')

    if (
        'nextn' in combined
        or 'next-n' in combined
        or 'next_n' in combined
    ) and mtp_support != 'no':
        features.add('nextn_native')
        features.add('mtp_native')

    turboquant_status = str(getattr(model, 'turboquant_status', '') or '').strip().lower()
    if turboquant_status in ('native', 'padded') or TURBOQUANT_TEXT_RE.search(combined):
        features.add('turboquant_weight_native')

    try:
        ctx_max = int(getattr(model, 'ctx_max', 0) or 0)
    except Exception:
        ctx_max = 0
    if ctx_max >= 65_536 or 'context_length' in metadata_text and any(marker in metadata_text for marker in ('65536', '131072', '262144')):
        features.add('long_context')

    try:
        gpu_total = int(getattr(hardware_profile, 'gpu_memory_total', 0) or 0)
    except Exception:
        gpu_total = 0
    try:
        gpu_free = int(getattr(hardware_profile, 'gpu_memory_free', 0) or 0)
    except Exception:
        gpu_free = 0
    try:
        model_size = int(model_size_bytes or 0)
    except Exception:
        model_size = 0
    effective_vram = gpu_free or gpu_total
    if gpu_total and gpu_total <= 9 * 1024**3:
        if model_size <= 0 or (effective_vram and model_size >= int(effective_vram * 0.65)):
            features.add('small_vram_risk')

    if 'local_gguf' in features and not _has_known_quant_hint(model, combined):
        features.add('unknown_quant')

    if 'local_gguf' in features and not {'tq3_native', 'mtp_native', 'mmproj'} & features:
        features.add('normal_gguf')

    return features


def _result(status: str, reason: str, severity: str, features: Set[str]) -> ModelEngineCompatibility:
    normalized = str(status or 'unknown').strip().lower()
    if normalized not in {'preferred', 'compatible', 'compatible_with_warning', 'unsupported', 'unknown'}:
        normalized = 'unknown'
    return ModelEngineCompatibility(
        compatible=normalized in COMPATIBLE_STATUSES,
        status=normalized,
        reason=reason,
        severity=severity,
        detected_features=tuple(sorted(features)),
    )


def _mtp_binary_block_reason(capabilities: Optional[EngineCapabilities]) -> str:
    if capabilities is not None:
        supports_spec_type = bool(getattr(capabilities, 'supports_spec_type', False))
        supports_draft = bool(getattr(capabilities, 'supports_spec_draft_n_max', False))
        supports_mtp = bool(getattr(capabilities, 'supports_mtp', False))
        spec_type = mtp_spec_type_value(capabilities)
        if supports_spec_type and supports_draft and not spec_type:
            return 'MTP_FLAGS_NOT_FOUND: selected binary has --spec-type but does not list mtp or draft-mtp'
        if supports_spec_type and supports_draft and supports_mtp and spec_type:
            return ''
    return 'Selected MTP binary does not advertise --spec-type mtp/draft-mtp'


def engine_supports_model(
    engine_id: str,
    model: ModelConfig,
    capabilities: Optional[EngineCapabilities] = None,
) -> ModelEngineCompatibility:
    engine = normalize_engine_id(engine_id)
    features = detect_model_runtime_features(model)

    if 'mmproj' in features:
        # Audit #7: the legacy ENGINE_LLAMA_CPP_MTP-specific branch
        # collapsed into the standard mmproj block — normalize_engine_id
        # routes the alias to ENGINE_LLAMA_CPP, and MTP+mmproj is
        # unsupported on every llama.cpp variant regardless.
        return _result('unsupported', 'mmproj files are not standalone language models', 'block', features)

    if 'tq3_native' in features and engine != ENGINE_TQ3:
        return _result('unsupported', f'TQ3-native GGUFs require the tq3 engine (selected engine: {engine})', 'block', features)

    if engine == ENGINE_TQ3:
        if 'tq3_native' in features:
            return _result('preferred', 'TQ3-native GGUF', 'info', features)
        return _result('unknown', 'TQ3-native GGUFs require detected TQ3 tensor format; this model is uncertain', 'warn', features)

    if engine in (ENGINE_LLAMA_CPP, ENGINE_TURBOQUANT, ENGINE_BUUN):
        if 'mtp_native' in features:
            # MTP is a binary capability: an MTP-native GGUF runs accelerated on
            # any llama.cpp-compatible binary that advertises the speculative MTP
            # flags, not only on the legacy llama.cpp-mtp engine alias.
            if capabilities is not None and bool(getattr(capabilities, 'supports_mtp', False)) and mtp_spec_type_value(capabilities):
                return _result('preferred', 'MTP-capable GGUF on a binary that advertises MTP', 'info', features)
            return _result(
                'compatible_with_warning',
                f'MTP-capable GGUF will load on {engine}; MTP acceleration needs a binary that advertises --spec-type draft-mtp',
                'warn',
                features,
            )
        if 'local_gguf' in features:
            if 'unknown_quant' in features:
                return _result('compatible_with_warning', 'Local GGUF quantization is unknown; launch may require validation', 'warn', features)
            return _result('compatible', 'GGUF compatible with selected llama.cpp-family engine', 'info', features)
        return _result('unknown', 'This engine expects a local GGUF model', 'warn', features)

    if engine == ENGINE_VLLM:
        if 'hf_ref' in features:
            return _result('preferred', 'HF model reference', 'info', features)
        if 'local_gguf' in features:
            return _result('unsupported', 'vLLM local GGUF support is not implemented in llama-tui', 'block', features)
        return _result('unknown', 'vLLM model entries should be HF refs', 'warn', features)

    return _result('unknown', f'Unknown engine {engine_id}', 'warn', features)


def engine_shows_model(
    engine_id: str,
    model: ModelConfig,
    capabilities: Optional[EngineCapabilities] = None,
) -> ModelEngineCompatibility:
    """Return whether the engine browser should show a model.

    This is intentionally less strict than launch compatibility for cases where
    a model belongs under an engine but the current binary cannot launch it.
    """
    engine = normalize_engine_id(engine_id)
    features = detect_model_runtime_features(model)

    if 'mmproj' in features:
        # Same collapse as engine_supports_model above; the legacy
        # ENGINE_LLAMA_CPP_MTP-only branch is dead after audit #7.
        return _result('unsupported', 'mmproj files are not standalone language models', 'block', features)

    if 'tq3_native' in features and engine != ENGINE_TQ3:
        return _result('unsupported', f'TQ3-native GGUFs require the tq3 engine (selected engine: {engine})', 'block', features)

    return engine_supports_model(engine, model, capabilities)


def compatible_engine_ids_for_model(
    model: ModelConfig,
    capabilities_by_engine: Optional[dict[str, EngineCapabilities]] = None,
) -> Tuple[str, ...]:
    compatible = []
    # Audit #7: ENGINE_LLAMA_CPP_MTP collapsed into ENGINE_LLAMA_CPP, so
    # only iterate the five canonical engines now.
    for engine in (ENGINE_LLAMA_CPP, ENGINE_TURBOQUANT, ENGINE_BUUN, ENGINE_TQ3, ENGINE_VLLM):
        caps = (capabilities_by_engine or {}).get(engine)
        if engine_supports_model(engine, model, caps).compatible:
            compatible.append(engine)
    return tuple(compatible)
