"""Full-suite benchmark orchestration helpers.

Seventh extraction of audit finding #6 (decompose ``benchmark.py``).
Covers the cohesive pure subset of the staged full-suite benchmark code:

  - state-field constants the suite restores between stages
  - state-restore helper that keeps benchmark fields intact while
    rolling back everything else to the suite's starting ``ModelConfig``
  - benchmark-run lookup by kind + strategy id
  - MTP acceptance stage status mapping
  - human-readable suite summary formatter

The heavy orchestrator ``benchmark_full_suite`` stays in ``benchmark.py``
along with ``_write_full_suite_log``, ``_moe_tuning_skip_reason_for_app``,
``_current_model_for_suite``, ``_annotate_latest_stage_run``,
``_mtp_acceptance_best_from_run``, ``full_suite_recommended_profile_key``,
``suite_run_recommended_profile_key``, and
``apply_full_suite_profile_recommendation`` because they depend on
``AppConfig`` methods, ``get_measured_profile``, the moe-overlay flag
helpers, ``CACHE_DIR``, and ``benchmark_config_fingerprint`` — pulling
those along would tangle the module graph again.

The orchestrator imports the pure helpers here and stays a thin
coordinator.
"""

from typing import Dict, Optional, Sequence

from .models import ModelConfig


SUITE_BENCHMARK_STATE_FIELDS = {
    'last_benchmark_tokens_per_sec',
    'last_benchmark_seconds',
    'last_benchmark_profile',
    'last_benchmark_results',
    'measured_profiles',
    'benchmark_runs',
    'benchmark_fingerprint',
    'default_benchmark_status',
    'default_benchmark_at',
    'engine_benchmark_store',
    'last_opencode_benchmark_score',
    'last_opencode_benchmark_seconds',
    'last_opencode_benchmark_profile',
    'last_opencode_benchmark_results',
    'last_hermes_benchmark_score',
    'last_hermes_benchmark_seconds',
    'last_hermes_benchmark_profile',
    'last_hermes_benchmark_results',
}


def _suite_restore_config_fields(
    current: ModelConfig,
    original: ModelConfig,
    prior_profiles: Optional[Dict[str, Dict[str, object]]] = None,
) -> ModelConfig:
    """Restore the suite's starting ModelConfig but keep benchmark state.

    Benchmark stages may mutate launch settings on the saved ``ModelConfig``;
    once the suite finishes we want the user's pre-suite config back, while
    preserving every newly measured profile, benchmark run, and the rolling
    benchmark score fields. Returns a fresh ModelConfig instance.
    """
    from dataclasses import asdict

    restored = ModelConfig(**asdict(current))
    for field in getattr(ModelConfig, '__dataclass_fields__', {}):
        if field in SUITE_BENCHMARK_STATE_FIELDS:
            continue
        setattr(restored, field, getattr(original, field))
    if prior_profiles:
        merged = {
            str(key): (dict(value) if isinstance(value, dict) else value)
            for key, value in prior_profiles.items()
        }
        for key, value in (getattr(current, 'measured_profiles', {}) or {}).items():
            merged[str(key)] = dict(value) if isinstance(value, dict) else value
        restored.measured_profiles = merged
    return restored


def _latest_benchmark_run(
    model: ModelConfig,
    kind: str,
    strategy_id: str = '',
) -> Dict[str, object]:
    for run in list(getattr(model, 'benchmark_runs', []) or []):
        if not isinstance(run, dict):
            continue
        if str(run.get('kind', '') or '') != kind:
            continue
        if strategy_id and str(run.get('benchmark_strategy_id', '') or '') != strategy_id:
            continue
        return run
    return {}


def _mtp_acceptance_stage_status(run: Dict[str, object], best: Dict[str, object]) -> str:
    run_status = str(run.get('status', '') or '').lower()
    if not best:
        return 'failed'
    if run_status in ('complete', 'done'):
        return 'done'
    return 'usable'


def full_suite_summary_text(
    stage_records: Sequence[Dict[str, object]],
    recommendations: Optional[Dict[str, object]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> str:
    recommendations = dict(recommendations or {})
    warnings = list(warnings or [])
    mtp_row = next((item for item in stage_records if str(item.get('stage', '') or '') == 'mtp_acceptance'), None)
    if mtp_row:
        mtp_status = str(mtp_row.get('status', '') or 'unknown')
        moe_row = next((item for item in stage_records if str(item.get('stage', '') or '') == 'moe_placement'), None)
        moe_status = str((moe_row or {}).get('status', '') or 'skipped')
        summary_row = next((item for item in reversed(stage_records) if str(item.get('stage', '') or '') == 'summary'), None)
        suite_status = str((summary_row or {}).get('status', '') or '')
        if not suite_status:
            suite_status = 'failed' if mtp_status in ('failed', 'blocked_missing_capability') else 'partial'
            if mtp_status == 'done' and moe_status in ('done', 'skipped'):
                suite_status = 'done'
        parts = [f'mtp_acceptance={mtp_status}', f'moe_placement={moe_status}']
        mtp_recommendation = recommendations.get('mtp_acceptance')
        if isinstance(mtp_recommendation, dict) and mtp_recommendation.get('draft_n'):
            parts.append(f'best_draft_n={mtp_recommendation.get("draft_n")}')
        if warnings:
            parts.append(f'{len(warnings)} warning(s)')
        return f'MTP Full Suite {suite_status}: ' + ', '.join(parts)

    parts = []
    for name in ('moe_placement', 'model_benchmark', 'hermes', 'opencode'):
        row = next((item for item in stage_records if str(item.get('stage', '') or '') == name), None)
        if not row:
            continue
        status = str(row.get('status', '') or '')
        if name == 'moe_placement' and recommendations.get('moe_placement'):
            parts.append(f'moe={recommendations["moe_placement"]}')
        elif name == 'model_benchmark' and recommendations.get('default_profile'):
            parts.append(f'profile={recommendations["default_profile"]}')
        else:
            parts.append(f'{name}={status}')
    if warnings:
        parts.append(f'{len(warnings)} warning(s)')
    return 'Full Suite Benchmark complete: ' + ', '.join(parts) if parts else 'Full Suite Benchmark complete'
