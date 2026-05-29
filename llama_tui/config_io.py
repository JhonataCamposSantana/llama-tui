"""Config-file serialisation and recovery helpers.

Fourth extraction of audit finding #9 (split ``AppConfig``). Owns the
pure JSON-shape work and the broken-config archival path:

  - ``serialize_app_state(app)`` snapshots the settings dataclasses and
    model list off ``AppConfig`` into the dict shape ``models.json``
    expects on disk.
  - ``write_config_dict(path, data)`` writes the dict as pretty JSON,
    creating the parent directory as needed.
  - ``archive_broken_config_file(config_path, backup_dir)`` copies a
    malformed config file aside before recovery so the user can inspect
    or restore it manually.

AppConfig keeps thin ``save()`` / ``_archive_broken_config_file``
wrappers so the orchestration (state-restore between stages, sort-rank
normalisation, atexit hooks) stays in the class. The helpers here have
zero AppConfig dependency.
"""

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def serialize_app_state(app) -> Dict[str, object]:
    """Snapshot ``AppConfig`` fields into a JSON-serialisable dict.

    Field order is preserved so re-saving an unchanged config produces
    an identical file — useful for diff tooling and avoids spurious git
    churn in user-managed ``models.json`` files.
    """
    return {
        'llama_server': app.llama_server,
        'hf_cache_root': app.hf_cache_root,
        'llmfit_cache_root': app.llmfit_cache_root,
        'llm_models_cache_root': app.llm_models_cache_root,
        'lm_studio_model_roots': app.lm_studio_model_roots,
        'opencode': asdict(app.opencode),
        'continue': asdict(app.continue_settings),
        'hermes': asdict(app.hermes),
        'ui': asdict(app.ui),
        'models': [asdict(m) for m in app.models],
    }


def _unique_backup_path(source_path: Path, backup_dir: Path, label: str = '') -> Path:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    infix = f'.{label}' if label else ''
    base_name = f'{source_path.stem}{infix}.{stamp}'
    candidate = backup_dir / f'{base_name}{source_path.suffix}'
    counter = 1
    while candidate.exists():
        candidate = backup_dir / f'{base_name}.{counter}{source_path.suffix}'
        counter += 1
    return candidate


def write_text_atomic(path: Path, text: str, encoding: str = 'utf-8') -> None:
    """Atomically replace ``path`` with ``text`` after writing it in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding=encoding,
            dir=str(path.parent),
            prefix=f'.{path.name}.',
            suffix='.tmp',
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def write_config_dict(config_path: Path, data: Dict[str, object]) -> None:
    """Write the config dict as pretty JSON, creating parents as needed."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(config_path, json.dumps(data, indent=2) + '\n')


def backup_file(source_path: Path, backup_dir: Path, label: str = '') -> Optional[Path]:
    """Copy ``source_path`` into ``backup_dir`` with a collision-safe timestamped name."""
    if not source_path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = _unique_backup_path(source_path, backup_dir, label)
    try:
        shutil.copy2(source_path, backup_path)
    except OSError:
        return None
    return backup_path


def archive_broken_config_file(config_path: Path, backup_dir: Path) -> Optional[Path]:
    """Copy a malformed config file aside under ``backup_dir`` with a stamp.

    Returns the backup path on success, ``None`` when there is nothing to
    archive (missing source) or the copy itself fails. The stamp format
    is ``YYYYMMDD-HHMMSS-microseconds`` so files sort chronologically and
    repeated backups in the same second do not overwrite one another.
    """
    return backup_file(config_path, backup_dir, label='broken')
