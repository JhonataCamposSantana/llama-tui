"""Stderr/log → benchmark-failure-category mapping.

Extracted from ``benchmark.py`` as the first step of audit finding #6
(decompose ``benchmark.py``). The audit specifically named this slice as
self-contained: it takes a chunk of stderr/log text and returns a
structured record with a category, a one-line reason, a suggested fix,
and an excerpt. Nothing else here touches subprocess, hardware, or
runtime profiles, which makes it the cleanest first extraction.

The classifier is intentionally imperative — order matters because
several branches share substring matches but have different priorities
(e.g. "fatal error" + "ggml" is a runtime crash). Audit finding #11
covers converting it to a declarative pattern table, but that needs a
captured stderr-fixture set first; until then the imperative form
preserves behaviour bit-for-bit.
"""

import re
from typing import Dict, List

from .textutil import compact_message, concise_failure


FAILURE_CATEGORIES = (
    'ENGINE_BINARY_MISSING',
    'CLI_INVALID',
    'ENGINE_RUNTIME_CRASH',
    'BASELINE_NOT_SUPPORTED_FOR_RECURRENT_NEXTN',
    'blocked_missing_capability',
    'MEMORY_GUARDRAIL',
    'MEMORY_FIT_FAILED',
    'FIXED_GPU_LAYERS_BLOCKED_FIT',
    'FIXED_GPU_LAYERS_FIT_FAILED',
    'CUDA_OOM_WEIGHTS',
    'CUDA_OOM_KV',
    'KV_MODE_INCOMPATIBLE',
    'MODEL_LOAD_FAILED',
    'RAW_ENGINE_TIMEOUT',
    'SERVER_TIMEOUT',
    'API_TIMEOUT',
    'PORT_UNREACHABLE',
    'CHAT_TEMPLATE_ERROR',
)

FAILURE_EXCERPT_MARKERS = (
    'ggml_assert',
    'assertion',
    'fatal error',
    'error while handling argument',
    'unknown speculative type',
    'n_rs_seq',
    'unknown value',
    'invalid argument',
    'unrecognized argument',
    'failed to fit params',
    'cannot meet free memory target',
    'n_gpu_layers already set by user',
    'gpu layers already set by user',
    'cudamalloc failed',
    'failed to allocate cuda',
    'unable to allocate cuda',
    'alloc_tensor_range',
    'failed to allocate buffer for kv cache',
    'failed to create context',
    'llama_model_load',
    'failed to load model',
)


def benchmark_failure_excerpt(text: str, limit: int = 320) -> str:
    lines = [line.strip() for line in str(text or '').splitlines() if line.strip()]
    if not lines:
        return ''
    selected: List[str] = []
    for marker in FAILURE_EXCERPT_MARKERS:
        for line in lines:
            if marker in line.lower() and line not in selected:
                selected.append(line)
                break
        if len(selected) >= 3:
            break
    if not selected:
        selected = lines[-3:]
    return concise_failure(' | '.join(selected), limit=limit)


def classify_benchmark_failure(text: str, default_category: str = 'SERVER_TIMEOUT') -> Dict[str, str]:
    excerpt = benchmark_failure_excerpt(text)
    detail = compact_message(text or '')
    low = detail.lower()
    category = default_category if default_category in FAILURE_CATEGORIES else 'SERVER_TIMEOUT'
    reason = excerpt or detail or category
    suggested = ''
    terminal = False
    if 'ggml_assert' in low or 'assertion' in low or ('fatal error' in low and 'ggml' in low):
        category = 'ENGINE_RUNTIME_CRASH'
        reason = excerpt or detail or 'The engine crashed during startup.'
        suggested = 'Treat this as an engine/runtime crash; try another binary or runtime flag set.'
        terminal = True
    if 'engine_binary_missing' in low:
        category = 'ENGINE_BINARY_MISSING'
        reason = detail or 'The active engine server binary is missing.'
        suggested = 'Point the engine binary env var at a built llama-server.'
        terminal = True
    if (
        re.search(r'(unknown|invalid|unrecognized).{0,80}(argument|option|value|flag|type)', low)
        or 'unknown speculative type' in low
        or 'error while handling argument' in low
        or 'requires an argument' in low
    ):
        category = 'CLI_INVALID'
        suggested = 'Check the generated command and use syntax supported by this server binary.'
        terminal = True
        if 'spec-type' in low or 'speculative type' in low:
            reason = excerpt or detail or 'The binary rejected the generated --spec-type value.'
            suggested = 'Use the speculative decoding type advertised by this server binary.'
        if 'flash-attn' in low or '-fa' in low:
            reason = 'The binary rejected the generated flash-attn syntax.'
            suggested = 'Use "--flash-attn on" or "-fa on" for builds that require a flash-attn value.'
    if 'unknown value for --flash-attn' in low:
        category = 'CLI_INVALID'
        reason = 'The binary requires --flash-attn to receive a valid value.'
        suggested = 'Use "--flash-attn on" or "-fa on"; do not emit bare "-fa".'
        terminal = True
    if 'unsupported cache type' in low and ('cache' in low or 'turbo' in low):
        category = 'KV_MODE_INCOMPATIBLE'
        reason = detail or 'The selected KV mode is not supported by this server/model combination.'
        suggested = 'Try a different TurboKV mode or benchmark default/q8 KV cache.'
        terminal = True
    if 'chat template' in low or 'jinja' in low and 'template' in low:
        category = 'CHAT_TEMPLATE_ERROR'
        suggested = 'Disable Jinja or adjust the model chat template before benchmarking.'
    if 'does not divide' in low and ('cache' in low or 'turbo' in low):
        category = 'KV_MODE_INCOMPATIBLE'
        reason = detail or 'The selected KV mode is incompatible with this model/head dimension.'
        suggested = 'Try a different TurboKV mode or benchmark default/q8 KV cache.'
        terminal = True
    fit_memory_failure = (
        'failed to fit params to free device memory' in low
        or 'cannot meet free memory target' in low
        or ('projected to use' in low and 'device memory' in low and 'free device memory' in low)
        or ('llama_params_fit_impl' in low and 'free device memory' in low)
    )
    fixed_gpu_layer_failure = fit_memory_failure and (
        'n_gpu_layers already set by user' in low
        or 'gpu layers already set by user' in low
        or re.search(r'n_gpu_layers.{0,80}set by user', low) is not None
    )
    if fixed_gpu_layer_failure:
        category = 'FIXED_GPU_LAYERS_BLOCKED_FIT'
        reason = detail or 'MTP fit was blocked because --n-gpu-layers was fixed.'
        suggested = 'Use a fit-assisted MTP profile without fixed --n-gpu-layers.'
        terminal = True
    elif fit_memory_failure:
        category = 'MEMORY_FIT_FAILED'
        reason = detail or 'The runtime fit planner could not meet the current free memory target.'
        suggested = 'Reduce context/offload for this run or retry after freeing RAM/VRAM.'
        terminal = True
    memory_allocation_failure = (
        'cudamalloc failed' in low
        or ('cuda error' in low and 'out of memory' in low)
        or 'out of memory' in low
        or 'failed to allocate cuda' in low
        or 'unable to allocate cuda' in low
        or 'alloc_tensor_range' in low
        or 'failed to allocate buffer for kv cache' in low
        or ('failed to create context' in low and ('memory' in low or 'kv' in low or 'cache' in low))
    )
    if not terminal and memory_allocation_failure:
        weight_allocation_failure = (
            'failed to allocate cuda' in low
            or 'unable to allocate cuda' in low
            or 'alloc_tensor_range' in low
            or 'loading model tensors' in low
            or 'llama_model_load' in low
        )
        if not weight_allocation_failure and ('kv' in low or 'cache' in low or 'context' in low):
            category = 'CUDA_OOM_KV'
            reason = excerpt or detail or 'CUDA memory allocation failed for KV/context.'
            suggested = 'Reduce context, parallelism, or KV cache size; for OpenCode prefer parallel=1.'
        else:
            category = 'CUDA_OOM_WEIGHTS'
            reason = excerpt or detail or 'CUDA memory allocation failed while loading model weights.'
            suggested = 'Reduce GPU layers; for heavy MoE models try partial offload such as -ngl 20 and --parallel 1.'
        terminal = True
    if not terminal and ('failed to load model' in low or ('llama_model_load' in low and 'failed' in low) or 'model load failed' in low):
        category = 'MODEL_LOAD_FAILED'
        suggested = suggested or 'Verify the GGUF file and model path, then retry with a smaller launch profile.'
    if not terminal and ('connection refused' in low or 'failed to establish a new connection' in low or 'port unreachable' in low):
        category = 'PORT_UNREACHABLE'
        suggested = 'The server process did not expose the expected API port.'
    if not terminal and ('timed out' in low or 'timeout' in low):
        if default_category == 'RAW_ENGINE_TIMEOUT':
            category = 'RAW_ENGINE_TIMEOUT'
            suggested = suggested or 'Keep raw engine probes bounded; promote only candidates that pass server validation.'
        else:
            category = 'API_TIMEOUT' if default_category == 'API_TIMEOUT' else 'SERVER_TIMEOUT'
        suggested = suggested or 'Retry with a smaller context or lower GPU layer count.'
    return {
        'failure_category': category,
        'failure_reason': concise_failure(reason, limit=500),
        'suggested_fix': suggested,
        'failure_excerpt': excerpt,
    }


def benchmark_failure_summary(records: List[Dict[str, object]], fallback: str) -> str:
    for record in records:
        category = str(record.get('failure_category', '') or '')
        if not category:
            continue
        profile = str(record.get('runtime_profile', '') or record.get('objective', '') or record.get('preset', '') or 'candidate')
        reason = str(record.get('failure_reason', '') or record.get('detail', '') or category)
        fix = str(record.get('suggested_fix', '') or '')
        summary = f'{profile} failed before benchmark: {category} - {concise_failure(reason)}'
        if fix:
            summary += f' Suggested fix: {fix}'
        return summary
    return fallback
