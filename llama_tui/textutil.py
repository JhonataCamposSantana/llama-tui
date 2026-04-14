import textwrap
from pathlib import Path
from typing import List


def tail_text(path: Path, max_lines: int = 40) -> List[str]:
    if not path.exists():
        return ['<no log file yet>']
    try:
        lines = path.read_text(errors='replace').splitlines()
        return lines[-max_lines:] or ['<empty log>']
    except Exception as e:
        return [f'<failed to read log: {e}>']
def ellipsize(text: str, width: int) -> str:
    if width <= 0:
        return ''
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + '...'
def compact_message(text: str) -> str:
    return ' | '.join(part.strip() for part in str(text).splitlines() if part.strip())
def is_error_message(text: str) -> bool:
    low = compact_message(text).lower()
    return bool(low) and (
        low.startswith('❌')
        or ' error' in low
        or 'failed' in low
        or 'crashed' in low
        or 'traceback' in low
    )
def wrap_display_lines(text: str, width: int) -> List[str]:
    if width <= 0:
        return []
    wrapped: List[str] = []
    for paragraph in str(text).splitlines() or ['']:
        if not paragraph:
            wrapped.append('')
            continue
        wrapped.extend(textwrap.wrap(paragraph, width=width, replace_whitespace=False, break_long_words=True) or [''])
    return wrapped
