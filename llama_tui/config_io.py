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
import shutil
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


def write_config_dict(config_path: Path, data: Dict[str, object]) -> None:
    """Write the config dict as pretty JSON, creating parents as needed."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def archive_broken_config_file(config_path: Path, backup_dir: Path) -> Optional[Path]:
    """Copy a malformed config file aside under ``backup_dir`` with a stamp.

    Returns the backup path on success, ``None`` when there is nothing to
    archive (missing source) or the copy itself fails. The stamp format
    is ``YYYYMMDD-HHMMSS`` so files sort chronologically.
    """
    if not config_path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_path = backup_dir / f'{config_path.stem}.broken.{stamp}{config_path.suffix}'
    try:
        shutil.copy2(config_path, backup_path)
    except OSError:
        return None
    return backup_path
