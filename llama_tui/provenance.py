from pathlib import Path
from typing import Iterable


def parse_hf_cache_provenance(path: Path) -> dict[str, str]:
    parts = path.expanduser().parts

    for idx, part in enumerate(parts):
        if not part.startswith('models--'):
            continue

        raw = part
        repo_id = ''
        try:
            _, owner, repo = raw.split('--', 2)
            repo_id = f'{owner}/{repo}'
        except ValueError:
            repo_id = raw

        snapshot = ''
        if idx + 2 < len(parts) and parts[idx + 1] == 'snapshots':
            snapshot = parts[idx + 2]

        return {
            'repo_folder': raw,
            'repo_id': repo_id,
            'snapshot': snapshot,
        }

    return {}


def normalize_source_labels(*values: object) -> list[str]:
    labels: list[str] = []

    def add(value: object):
        if isinstance(value, str):
            candidates: Iterable[object] = value.split(',')
        elif isinstance(value, (list, tuple, set)):
            candidates = value
        else:
            candidates = (value,)
        for candidate in candidates:
            clean = str(candidate or '').strip()
            if clean and clean not in labels:
                labels.append(clean)

    for value in values:
        add(value)
    return labels


def source_labels_text(*values: object) -> str:
    labels = normalize_source_labels(*values)
    return ','.join(labels) if labels else 'manual'


def model_source_provenance(path: Path, source: object = 'manual', root: Path | None = None) -> dict[str, object]:
    parsed = parse_hf_cache_provenance(path)
    return {
        'source_path': str(path),
        'source_root': str(root.expanduser()) if root is not None else '',
        'source_repo_id': parsed.get('repo_id', ''),
        'source_snapshot': parsed.get('snapshot', ''),
        'source_labels': normalize_source_labels(source),
    }
