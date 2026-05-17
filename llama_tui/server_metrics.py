"""Scrape llama-server's Prometheus ``/metrics`` endpoint.

llama.cpp-family servers launched with ``--metrics`` expose real prompt-eval
and token-generation rates plus KV-cache usage. Scraping these gives
server-truth prefill vs decode throughput instead of conflated client-side
timing. Every entry point degrades gracefully (returns ``None`` / empty) when
the endpoint is absent, so a build without ``--metrics`` never breaks a run.
"""

import urllib.error
import urllib.request
from typing import Dict, Optional


# Prometheus metric name -> friendly key in the returned dict.
_LLAMA_METRIC_KEYS = {
    'llamacpp:prompt_tokens_seconds': 'prompt_tokens_per_sec',
    'llamacpp:predicted_tokens_seconds': 'decode_tokens_per_sec',
    'llamacpp:kv_cache_usage_ratio': 'kv_cache_usage_ratio',
    'llamacpp:requests_deferred': 'requests_deferred',
    'llamacpp:requests_processing': 'requests_processing',
    'llamacpp:prompt_tokens_total': 'prompt_tokens_total',
    'llamacpp:tokens_predicted_total': 'tokens_predicted_total',
}

# Engines that run a llama-server binary and therefore expose /metrics.
LLAMA_SERVER_METRIC_ENGINES = ('llama.cpp', 'llama.cpp-mtp', 'turboquant', 'tq3', 'buun')


def engine_supports_metrics(engine_id: str) -> bool:
    """True for llama-server-based engines that expose a Prometheus endpoint."""
    return str(engine_id or '').strip().lower() in LLAMA_SERVER_METRIC_ENGINES


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus text exposition into ``{friendly_key: float}``.

    Only the llama.cpp metric names in ``_LLAMA_METRIC_KEYS`` are kept. Comment
    lines (``# HELP`` / ``# TYPE``) are skipped and any label set is dropped.
    Pure/string-only so it is unit testable without a server.
    """
    values: Dict[str, float] = {}
    for raw in str(text or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        brace = name.find('{')
        if brace != -1:
            name = name[:brace]
        key = _LLAMA_METRIC_KEYS.get(name)
        if key is None:
            continue
        try:
            values[key] = float(parts[1])
        except (TypeError, ValueError):
            continue
    return values


def _metrics_host(host: str) -> str:
    normalized = str(host or '').strip()
    if not normalized or normalized in ('0.0.0.0', '::', '[::]'):
        return '127.0.0.1'
    return normalized


def scrape_llama_server_metrics(
    host: str,
    port: int,
    timeout: float = 2.0,
) -> Optional[Dict[str, float]]:
    """GET ``http://{host}:{port}/metrics`` and parse it; ``None`` on any failure.

    A ``None`` result lets every caller degrade gracefully when the server was
    not built/launched with ``--metrics`` or is unreachable.
    """
    try:
        port_num = int(port)
    except (TypeError, ValueError):
        return None
    if port_num <= 0:
        return None
    url = f'http://{_metrics_host(host)}:{port_num}/metrics'
    try:
        with urllib.request.urlopen(url, timeout=max(0.1, float(timeout))) as response:
            if getattr(response, 'status', 200) not in (200, None):
                return None
            body = response.read().decode('utf-8', errors='replace')
    except (urllib.error.URLError, OSError, ValueError):
        return None
    parsed = parse_prometheus_metrics(body)
    return parsed or None
