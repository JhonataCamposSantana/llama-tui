import re
import textwrap
from pathlib import Path
from typing import List


IMPORTANT_LOG_MARKERS = (
    'error',
    'failed',
    'fatal',
    'abort',
    'exception',
    'traceback',
    'cannot open shared object file',
    'error while loading shared libraries',
    'out of memory',
    'cuda',
    'ggml_abort',
    'llama_model_load',
    'common_init',
)


def tail_text(path: Path, max_lines: int = 40) -> List[str]:
    if not path.exists():
        return ['<no log file yet>']
    try:
        lines = path.read_text(errors='replace').splitlines()
        return lines[-max_lines:] or ['<empty log>']
    except Exception as e:
        return [f'<failed to read log: {e}>']
def _read_log_lines(path: Path) -> List[str]:
    if not path.exists():
        return ['<no log file yet>']
    try:
        return path.read_text(errors='replace').splitlines() or ['<empty log>']
    except Exception as e:
        return [f'<failed to read log: {e}>']
def _is_internal_log_line(line: str) -> bool:
    return '] llama-tui:' in line
def _collapse_repeated_lines(lines: List[str]) -> List[str]:
    collapsed: List[str] = []
    last = ''
    count = 0

    def flush():
        if not last:
            return
        suffix = f' (repeated {count}x)' if count > 1 else ''
        collapsed.append(f'{last}{suffix}')

    for line in lines:
        if line == last:
            count += 1
            continue
        flush()
        last = line
        count = 1
    flush()
    return collapsed
def important_log_excerpt(path: Path, max_lines: int = 24, after_last_launch: bool = False) -> List[str]:
    """Return recent server/runtime diagnostics without recursively echoing UI progress."""
    lines = _read_log_lines(path)
    if not lines or lines[0].startswith('<'):
        return lines
    if after_last_launch:
        for idx in range(len(lines) - 1, -1, -1):
            if _is_internal_log_line(lines[idx]) and 'launch command:' in lines[idx]:
                lines = lines[idx + 1:]
                break

    recent = lines[-400:]
    important: List[str] = []
    for line in recent:
        if _is_internal_log_line(line):
            continue
        low = line.lower()
        if any(marker in low for marker in IMPORTANT_LOG_MARKERS):
            important.append(line)

    if important:
        return _collapse_repeated_lines(important)[-max_lines:]

    runtime_tail = [line for line in recent if not _is_internal_log_line(line)]
    if runtime_tail:
        return _collapse_repeated_lines(runtime_tail)[-max_lines:]
    return _collapse_repeated_lines(recent)[-max_lines:]
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
    if not low:
        return False
    if low.startswith('❌') or low.startswith('[error]') or low.startswith('error:'):
        return True
    if low.startswith(('right tab:', 'focus:', 'no errors captured', 'error captured in')):
        return False
    if re.search(r'\b(failed|failure|fatal|crashed|traceback|exception|aborted)\b', low):
        return True
    if re.search(r'\b(no|zero)\s+errors?\b', low):
        return False
    if re.search(r'\b0\s+errors?\b', low):
        return False
    if re.search(r'\bwithout\s+errors?\b', low):
        return False
    if re.search(r'\berror[-\s]?free\b', low):
        return False
    return bool(re.search(r'\berror\b', low))
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
