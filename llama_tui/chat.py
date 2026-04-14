import json
from typing import Dict, Iterable, Iterator, List, Tuple
from urllib import request

from .control import CancelToken, check_cancelled
from .models import ModelConfig

CHAT_TIMEOUT_SECONDS = 180


def build_chat_payload(model: ModelConfig, messages: List[Dict[str, str]], stream: bool = True) -> Dict[str, object]:
    return {
        'model': model.alias or model.id,
        'messages': messages,
        'temperature': float(getattr(model, 'temp', 0.7) or 0.7),
        'max_tokens': max(1, int(getattr(model, 'output', 512) or 512)),
        'stream': bool(stream),
    }


def chat_completion_url(model: ModelConfig) -> str:
    return f'http://{model.host}:{model.port}/v1/chat/completions'


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    parts.append(str(item.get('text') or ''))
                elif 'text' in item:
                    parts.append(str(item.get('text') or ''))
            elif item is not None:
                parts.append(str(item))
        return ''.join(parts)
    if content is None:
        return ''
    return str(content)


def parse_openai_sse_lines(lines: Iterable[str]) -> Iterator[Tuple[str, str]]:
    for raw_line in lines:
        line = raw_line.decode('utf-8', errors='replace') if isinstance(raw_line, bytes) else str(raw_line)
        line = line.strip()
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        if payload == '[DONE]':
            yield 'done', ''
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = data.get('choices') or []
        if not choices:
            continue
        first = choices[0] or {}
        delta = first.get('delta') or {}
        text = _content_text(delta.get('content'))
        if not text and 'text' in first:
            text = _content_text(first.get('text'))
        if text:
            yield 'chunk', text


def stream_chat_completion(
    model: ModelConfig,
    messages: List[Dict[str, str]],
    cancel_token: CancelToken | None = None,
) -> Iterator[str]:
    check_cancelled(cancel_token)
    body = json.dumps(build_chat_payload(model, messages, stream=True)).encode('utf-8')
    req = request.Request(
        chat_completion_url(model),
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with request.urlopen(req, timeout=CHAT_TIMEOUT_SECONDS) as resp:
        for raw_line in resp:
            check_cancelled(cancel_token)
            for event, text in parse_openai_sse_lines([raw_line]):
                check_cancelled(cancel_token)
                if event == 'done':
                    return
                if text:
                    yield text
