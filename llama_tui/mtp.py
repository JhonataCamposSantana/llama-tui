from pathlib import Path
from typing import Iterable, Tuple


MTP_SUPPORT_VALUES = ('auto', 'yes', 'no')
MTP_DRAFT_VALUES = (1, 2, 3)
MTP_AUTO_HINTS = ('gguf-mtp', 'native-mtp', 'mtp', 'nextn', 'next-n', 'next_n')


def normalize_mtp_support(value: object) -> str:
    text = str(value or 'auto').strip().lower()
    return text if text in MTP_SUPPORT_VALUES else 'auto'


def clamp_mtp_draft(value: object, default: int = 3) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(min(parsed, max(MTP_DRAFT_VALUES)), min(MTP_DRAFT_VALUES))


def mtp_support_auto_hint(model) -> bool:
    haystack = ' '.join(
        str(item or '').lower()
        for item in (
            getattr(model, 'id', ''),
            getattr(model, 'name', ''),
            getattr(model, 'alias', ''),
            getattr(model, 'path', ''),
            Path(str(getattr(model, 'path', '') or '')).name,
            getattr(model, 'model_family', ''),
            getattr(model, 'architecture', ''),
        )
    )
    return any(marker in haystack for marker in MTP_AUTO_HINTS)


def mtp_support_label(model) -> str:
    support = normalize_mtp_support(getattr(model, 'supports_mtp', 'auto'))
    if support == 'yes':
        return 'supported'
    if support == 'no':
        return 'unsupported'
    return 'auto hint' if mtp_support_auto_hint(model) else 'auto unknown'


def model_mtp_allowed(model) -> bool:
    return normalize_mtp_support(getattr(model, 'supports_mtp', 'auto')) != 'no'


def model_mtp_enabled(model, runtime_profile=None) -> bool:
    if runtime_profile is not None and hasattr(runtime_profile, 'mtp_enabled'):
        return bool(getattr(runtime_profile, 'mtp_enabled', False))
    return bool(getattr(model, 'mtp_enabled', False))


def model_mtp_draft_n_max(model, runtime_profile=None) -> int:
    if runtime_profile is not None and hasattr(runtime_profile, 'mtp_draft_n_max'):
        value = int(getattr(runtime_profile, 'mtp_draft_n_max', 0) or 0)
        if value > 0:
            return clamp_mtp_draft(value)
    return clamp_mtp_draft(getattr(model, 'mtp_draft_n_max', 3), default=3)


def _tokens(args: Iterable[object]) -> Tuple[str, ...]:
    return tuple(str(item or '').strip() for item in list(args or []) if str(item or '').strip())


def model_has_mmproj_config(model) -> bool:
    args = _tokens(getattr(model, 'extra_args', []) or [])
    for idx, token in enumerate(args):
        low = token.lower()
        if low in ('--mmproj', '-mmproj'):
            return True
        if low.startswith('--mmproj=') or low.startswith('-mmproj='):
            return True
        if 'mmproj' in low and (idx == 0 or args[idx - 1].lower() in ('--mmproj', '-mmproj')):
            return True
    return False


def mtp_label(model, runtime_profile=None) -> str:
    if not model_mtp_enabled(model, runtime_profile):
        return 'off'
    return f'n={model_mtp_draft_n_max(model, runtime_profile)}'
