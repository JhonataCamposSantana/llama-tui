"""Per-engine benchmark payload storage helpers.

First extraction of audit finding #9 (split ``AppConfig`` god class).
Owns the bookkeeping for ``ModelConfig.engine_benchmark_store``: each
model carries a dict keyed by engine ID (``llama.cpp``, ``turboquant``,
etc.) holding the last benchmark numbers, measured profiles, and run
history for that engine. When the active engine switches, AppConfig
swaps the active payload's fields onto the flat ``ModelConfig``
attributes so the rest of the UI/benchmark code sees a single
consistent view.

The functions here are pure — they take a ``ModelConfig`` and a dict
payload and return / mutate values without touching the broader
``AppConfig`` state. AppConfig keeps a thin wrapper around
``activate_engine_benchmark_views`` / ``persist_engine_benchmark_views``
because those need to resolve the active engine per model, which is
the AppConfig's responsibility.
"""

from typing import Dict, Tuple

from .models import ModelConfig


ENGINE_BENCHMARK_FIELDS: Tuple[str, ...] = (
    'last_benchmark_tokens_per_sec',
    'last_benchmark_seconds',
    'last_benchmark_profile',
    'last_benchmark_results',
    'measured_profiles',
    'benchmark_runs',
    'benchmark_fingerprint',
    'default_benchmark_status',
    'default_benchmark_at',
)


def default_engine_benchmark_payload() -> Dict[str, object]:
    """Return a fresh payload dict with the standard field defaults."""
    return {
        'last_benchmark_tokens_per_sec': 0.0,
        'last_benchmark_seconds': 0.0,
        'last_benchmark_profile': '',
        'last_benchmark_results': [],
        'measured_profiles': {},
        'benchmark_runs': [],
        'benchmark_fingerprint': '',
        'default_benchmark_status': '',
        'default_benchmark_at': '',
    }


def copy_benchmark_value(value: object) -> object:
    """Deep-ish copy a benchmark field value.

    Lists are duplicated element-wise (cloning any dict elements);
    dicts are shallow-copied with one extra level of cloning for dict
    values. Other types are returned as-is. Matches the original
    ``AppConfig._copy_benchmark_value`` semantics so swap-in is safe.
    """
    if isinstance(value, list):
        return [dict(item) if isinstance(item, dict) else item for item in value]
    if isinstance(value, dict):
        return {
            str(key): dict(item) if isinstance(item, dict) else item
            for key, item in value.items()
        }
    return value


def benchmark_payload_for_model(model: ModelConfig) -> Dict[str, object]:
    """Snapshot the flat benchmark fields off ``model`` into a payload."""
    return {
        field: copy_benchmark_value(getattr(model, field))
        for field in ENGINE_BENCHMARK_FIELDS
    }


def apply_benchmark_payload(model: ModelConfig, payload: Dict[str, object]) -> None:
    """Write payload fields back onto ``model``'s flat benchmark slots.

    Missing or wrong-typed payload entries fall back to the
    ``default_engine_benchmark_payload`` defaults so a partial store
    cannot leave the model in a torn state.
    """
    defaults = default_engine_benchmark_payload()
    for field in ENGINE_BENCHMARK_FIELDS:
        value = payload.get(field, defaults[field])
        if isinstance(defaults[field], list):
            value = list(value) if isinstance(value, list) else []
        elif isinstance(defaults[field], dict):
            value = dict(value) if isinstance(value, dict) else {}
        setattr(model, field, value)


def has_benchmark_payload(model: ModelConfig) -> bool:
    """True when the flat benchmark fields on ``model`` carry any signal.

    Used by the one-time legacy migration that seeds historical values
    under their canonical runtime key when no ``engine_benchmark_store``
    entry exists yet.
    """
    if float(getattr(model, 'last_benchmark_tokens_per_sec', 0.0) or 0.0) > 0.0:
        return True
    if (getattr(model, 'default_benchmark_status', '') or '').strip():
        return True
    if getattr(model, 'last_benchmark_results', None):
        return True
    if getattr(model, 'measured_profiles', None):
        return True
    if getattr(model, 'benchmark_runs', None):
        return True
    if (getattr(model, 'benchmark_fingerprint', '') or '').strip():
        return True
    return False


def canonical_legacy_engine_key(model: ModelConfig) -> str:
    """Return the engine key under which legacy benchmark data was stored.

    Pre-multi-engine builds stored everything under ``'llama.cpp'``.
    """
    return 'llama.cpp'
