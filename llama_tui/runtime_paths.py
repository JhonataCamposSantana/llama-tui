"""Pure path helpers for per-engine runtime PID/log/metadata files.

Second extraction of audit finding #9 (split ``AppConfig``). Owns the
filesystem layout for managed-process artefacts:

  - ``CACHE_DIR/runtime/{engine_slug}/{model_id}{suffix}`` — current
    layout, keyed by engine so concurrent multi-binary sessions
    (e.g. llama.cpp + TurboQuant+) don't collide.
  - ``CACHE_DIR/{model_id}{suffix}`` — legacy flat layout still read by
    ``cleanup_managed_processes`` so PID files left behind by older
    builds get reclaimed on next launch.

The engine-key resolution itself stays in ``AppConfig`` because it
depends on the model registry and the active runtime profile; this
module only handles the file-path arithmetic.
"""

from pathlib import Path

from . import constants


def _cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the explicit cache root or the module-level default."""
    return Path(cache_dir) if cache_dir is not None else constants.CACHE_DIR


def runtime_artifact_dir(engine_key: str, cache_dir: Path | None = None) -> Path:
    """Return the per-engine runtime cache directory.

    The engine key is slugified to a filesystem-safe form so engine
    identifiers like ``llama.cpp-mtp`` produce ``llama.cpp-mtp`` (which
    is already safe) while anything exotic falls back to alnum + dash
    + underscore + dot.
    """
    slug = ''.join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(engine_key or 'llama.cpp')
    )
    return _cache_dir(cache_dir) / 'runtime' / (slug or 'llama.cpp')


def runtime_artifact_path(
    model_id: str,
    suffix: str,
    engine_key: str,
    cache_dir: Path | None = None,
) -> Path:
    """Per-engine ``{model_id}{suffix}`` path under the runtime dir."""
    return runtime_artifact_dir(engine_key, cache_dir=cache_dir) / f'{model_id}{suffix}'


def runtime_pidfile(model_id: str, engine_key: str, cache_dir: Path | None = None) -> Path:
    return runtime_artifact_path(model_id, '.pid', engine_key, cache_dir=cache_dir)


def runtime_pid_metadata_file(model_id: str, engine_key: str, cache_dir: Path | None = None) -> Path:
    return runtime_artifact_path(model_id, '.pid.json', engine_key, cache_dir=cache_dir)


def runtime_logfile(model_id: str, engine_key: str, cache_dir: Path | None = None) -> Path:
    return runtime_artifact_path(model_id, '.log', engine_key, cache_dir=cache_dir)


def legacy_pidfile(model_id: str, cache_dir: Path | None = None) -> Path:
    return _cache_dir(cache_dir) / f'{model_id}.pid'


def legacy_pid_metadata_file(model_id: str, cache_dir: Path | None = None) -> Path:
    return _cache_dir(cache_dir) / f'{model_id}.pid.json'


def legacy_logfile(model_id: str, cache_dir: Path | None = None) -> Path:
    return _cache_dir(cache_dir) / f'{model_id}.log'
