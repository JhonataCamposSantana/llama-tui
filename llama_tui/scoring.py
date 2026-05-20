"""Per-objective benchmark scoring functions.

Fourth extraction of audit finding #6 (decompose ``benchmark.py``).
Each ``score_*`` function maps a measurement record into a scalar
score for one objective (fast_chat / long_context / opencode_ready /
auto), trading off throughput, context, headroom, stability, and a
KV-quality penalty. The helpers (``_record_headroom_score``,
``_record_stability_score``, ``_tps_curve``, ``_ctx_curve``,
``_record_kv_quality_penalty``) live here too so any tuning happens
inside one module.

Pure: no AppConfig, no subprocess, no hardware probing. Everything
operates on a measurement record dict + a ``ModelConfig``.
"""

from typing import Dict, Optional

from .models import ModelConfig
from .optimize import model_is_moe
from .runtime_profiles import turbo_kv_profile_for_preset
from .textutil import compact_message


def _record_headroom_score(record: Dict[str, object]) -> float:
    ram = int(record.get('ram_available', 0) or 0)
    vram = int(record.get('gpu_memory_free', 0) or 0)
    score = min(1.0, (ram / 1024**3) / 8.0) if ram else 0.35
    if vram:
        score = max(score, min(1.0, (vram / 1024**3) / 2.0))
    pressure = float(record.get('process_pressure_score', 0.0) or 0.0)
    if pressure:
        score *= max(0.45, 1.0 - pressure * 0.35)
    return max(0.0, min(1.0, score))


def _record_stability_score(record: Dict[str, object]) -> float:
    score = 1.0
    if int(record.get('retry_attempt', 1) or 1) > 1:
        score -= 0.15
    status = str(record.get('status', '') or '').lower()
    if status not in ('ok', 'probe ok', 'tests passed'):
        score -= 0.35
    detail = compact_message(str(record.get('detail', '') or '')).lower()
    if detail not in ('', '1 samples', '2 samples', '3 samples', 'all tasks passed'):
        score -= 0.05
    ready = float(record.get('ready_seconds', 0.0) or 0.0)
    if ready > 90:
        score -= 0.10
    return max(0.0, min(1.0, score))


def _tps_curve(record: Dict[str, object]) -> float:
    tps = float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0)
    return max(0.0, min(1.0, tps / (tps + 30.0))) if tps > 0 else 0.0


def _ctx_curve(record: Dict[str, object], cap: int) -> float:
    ctx = int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0)
    return max(0.0, min(1.0, ctx / max(1, cap)))


def _record_kv_quality_penalty(record: Dict[str, object]) -> float:
    try:
        explicit = float(record.get('kv_score_penalty', 0.0) or 0.0)
    except Exception:
        explicit = 0.0
    if explicit:
        return max(0.0, explicit)
    profile = turbo_kv_profile_for_preset(str(record.get('kv_preset', '') or ''))
    return max(0.0, float(profile.score_penalty if profile else 0.0))


def score_fast_chat(record: Dict[str, object], model: ModelConfig) -> float:
    cap = 16384 if model_is_moe(model) else 8192
    tps = float(record.get('decode_tokens_per_sec', record.get('tokens_per_sec', 0.0)) or 0.0)
    return (
        tps
        + 4.0 * _record_headroom_score(record)
        + 2.0 * _record_stability_score(record)
        + 3.0 * _ctx_curve(record, cap)
    )


def score_long_context(record: Dict[str, object], model: ModelConfig) -> float:
    cap = max(32768, int(getattr(model, 'ctx_max', 32768) or 32768))
    return (
        0.55 * _ctx_curve(record, cap)
        + 0.20 * _record_headroom_score(record)
        + 0.15 * _tps_curve(record)
        + 0.10 * _record_stability_score(record)
        - _record_kv_quality_penalty(record)
    )


def score_opencode_ready(record: Dict[str, object], model: ModelConfig) -> float:
    ctx = int(record.get('ctx_per_slot', record.get('ctx', 0)) or 0)
    target = 32768 if model_is_moe(model) else 16384
    score = (
        0.40 * _ctx_curve(record, target)
        + 0.30 * _tps_curve(record)
        + 0.20 * _record_headroom_score(record)
        + 0.10 * _record_stability_score(record)
    )
    if model_is_moe(model) and ctx < 16384:
        score *= 0.35
    if model_is_moe(model) and ctx >= 32768:
        score += 0.15
    return score - _record_kv_quality_penalty(record)


def score_auto(record: Dict[str, object], model: ModelConfig) -> float:
    if model_is_moe(model) and int(record.get('ngl', 0) or 0) != 999:
        return (
            0.35 * score_opencode_ready(record, model)
            + 0.35 * score_long_context(record, model)
            + 0.20 * score_fast_chat(record, model)
            + 0.10 * _record_headroom_score(record)
        )
    return (
        0.50 * _tps_curve(record)
        + 0.30 * _ctx_curve(record, 32768)
        + 0.12 * _record_headroom_score(record)
        + 0.08 * _record_stability_score(record)
        - _record_kv_quality_penalty(record)
    )
