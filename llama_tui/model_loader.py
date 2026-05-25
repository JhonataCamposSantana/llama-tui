"""``ModelConfig`` deserialisation from raw JSON dicts.

Third extraction of audit finding #9 (split ``AppConfig``). The
``load_model_from_payload`` function takes a raw dict (as read from
``models.json``) plus an index and returns a fully-defaulted, type-
coerced ``ModelConfig`` instance — or raises ``ValueError`` if the
entry is missing required fields. Pure: no AppConfig dependency, no
filesystem access; the caller decides whether to log warnings or
substitute defaults on failure.
"""

from typing import List

from .gguf import TURBOQUANT_STATUSES
from .models import ModelConfig, dataclass_payload
from .mtp import clamp_mtp_draft, normalize_mtp_support
from .provenance import normalize_source_labels, source_labels_text


VERIFICATION_STATUSES = ('unknown', 'running', 'passed', 'warning', 'failed', 'needs_benchmark')


def load_model_from_payload(raw: object, index: int) -> ModelConfig:
    """Construct a :class:`ModelConfig` from a raw config-file entry.

    Raises ``ValueError`` for entries that are not dicts or that omit
    one of the required identity fields. Every other field gets a
    deterministic default + type coercion so a partial / hand-edited
    config never produces a torn instance.
    """
    if not isinstance(raw, dict):
        raise ValueError('entry is not an object')
    payload = dict(raw)
    required = ('id', 'name', 'path', 'alias', 'port')
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f'missing fields: {", ".join(missing)}')
    payload['port'] = int(payload.get('port', 0) or 0)
    payload['ctx'] = int(payload.get('ctx', 8192) or 8192)
    payload['ctx_min'] = int(payload.get('ctx_min', 2048) or 2048)
    payload['ctx_max'] = int(payload.get('ctx_max', 131072) or 131072)
    payload['threads'] = int(payload.get('threads', 6) or 6)
    payload['ngl'] = int(payload.get('ngl', 999) or 999)
    payload['parallel'] = int(payload.get('parallel', 1) or 1)
    payload['memory_reserve_percent'] = int(payload.get('memory_reserve_percent', 25) or 25)
    payload['cache_ram'] = int(payload.get('cache_ram', 0) or 0)
    payload['output'] = int(payload.get('output', 4096) or 4096)
    payload['temp'] = float(payload.get('temp', 0.7) or 0.7)
    payload['top_p'] = float(payload.get('top_p', 0.95) if payload.get('top_p', 0.95) is not None else 0.95)
    payload['top_k'] = int(payload.get('top_k', 40) or 0)
    payload['repeat_penalty'] = float(payload.get('repeat_penalty', 1.0) if payload.get('repeat_penalty', 1.0) is not None else 1.0)
    payload['presence_penalty'] = float(payload.get('presence_penalty', 0.0) if payload.get('presence_penalty', 0.0) is not None else 0.0)
    payload['no_context_shift'] = bool(payload.get('no_context_shift', False))
    preserve_thinking = str(payload.get('preserve_thinking', 'auto') or 'auto').strip().lower()
    payload['preserve_thinking'] = preserve_thinking if preserve_thinking in ('auto', 'on', 'off') else 'auto'
    payload['source'] = source_labels_text(payload.get('source', 'manual'))
    payload['source_path'] = str(payload.get('source_path', '') or '')
    payload['source_root'] = str(payload.get('source_root', '') or '')
    payload['source_repo_id'] = str(payload.get('source_repo_id', '') or '')
    payload['source_snapshot'] = str(payload.get('source_snapshot', '') or '')
    payload['source_labels'] = normalize_source_labels(payload.get('source_labels', []), payload['source'])
    payload['architecture'] = str(payload.get('architecture', '') or '')
    payload['architecture_type'] = str(payload.get('architecture_type', 'unknown') or 'unknown').strip().lower()
    if payload['architecture_type'] not in ('dense', 'moe', 'unknown'):
        payload['architecture_type'] = 'unknown'
    payload['model_family'] = str(payload.get('model_family', '') or '')
    for key in (
        'expert_count',
        'expert_used_count',
        'expert_shared_count',
        'expert_group_count',
        'expert_group_used_count',
        'moe_every_n_layers',
        'leading_dense_block_count',
    ):
        payload[key] = int(payload.get(key, 0) or 0)
    payload['active_expert_ratio'] = float(payload.get('active_expert_ratio', 0.0) or 0.0)
    payload['classification_confidence'] = float(payload.get('classification_confidence', 0.0) or 0.0)
    payload['classification_source'] = str(payload.get('classification_source', '') or '')
    payload['classification_reason'] = str(payload.get('classification_reason', '') or '')
    payload['turboquant_status'] = str(payload.get('turboquant_status', 'unknown') or 'unknown').strip().lower()
    if payload['turboquant_status'] not in TURBOQUANT_STATUSES:
        payload['turboquant_status'] = 'unknown'
    for key in ('turboquant_head_dim', 'turboquant_key_dim', 'turboquant_value_dim'):
        payload[key] = int(payload.get(key, 0) or 0)
    payload['turboquant_source'] = str(payload.get('turboquant_source', '') or '')
    payload['turboquant_reason'] = str(payload.get('turboquant_reason', '') or '')
    payload['supports_mtp'] = normalize_mtp_support(payload.get('supports_mtp', 'auto'))
    payload['mtp_enabled'] = bool(payload.get('mtp_enabled', False))
    payload['mtp_draft_n_max'] = clamp_mtp_draft(payload.get('mtp_draft_n_max', 3), default=3)
    payload['moe_placement_strategy'] = str(payload.get('moe_placement_strategy', '') or '').strip()
    payload['cpu_moe'] = bool(payload.get('cpu_moe', False))
    payload['n_cpu_moe'] = max(0, int(payload.get('n_cpu_moe', 0) or 0))
    tensor_overrides = payload.get('tensor_overrides', [])
    payload['tensor_overrides'] = (
        [str(item).strip() for item in tensor_overrides if str(item).strip()]
        if isinstance(tensor_overrides, list)
        else []
    )
    payload['favorite'] = bool(payload.get('favorite', False))
    payload['last_used_at'] = str(payload.get('last_used_at', '') or '')
    payload['sort_rank'] = int(payload.get('sort_rank', index + 1) or (index + 1))
    extra_args = payload.get('extra_args', [])
    payload['extra_args'] = [str(item) for item in extra_args] if isinstance(extra_args, list) else []
    launch_overrides = payload.get('launch_overrides', {})
    payload['launch_overrides'] = dict(launch_overrides) if isinstance(launch_overrides, dict) else {}
    tags = payload.get('tags', [])
    payload['tags'] = [str(item).strip() for item in tags if str(item).strip()] if isinstance(tags, list) else []
    payload['verification_status'] = str(payload.get('verification_status', 'unknown') or 'unknown')
    if payload['verification_status'] not in VERIFICATION_STATUSES:
        payload['verification_status'] = 'unknown'
    payload['verification_at'] = str(payload.get('verification_at', '') or '')
    payload['verification_fingerprint'] = str(payload.get('verification_fingerprint', '') or '')
    payload['verification_summary'] = str(payload.get('verification_summary', '') or '')
    verification_results = payload.get('verification_results', {})
    payload['verification_results'] = verification_results if isinstance(verification_results, dict) else {}
    return ModelConfig(**dataclass_payload(ModelConfig, payload))
