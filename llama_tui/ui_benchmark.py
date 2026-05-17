from typing import Dict, List, Sequence, Tuple

from .benchmark import record_matches_profile
from .textutil import compact_message, ellipsize, wrap_display_lines


SERVER_WINNER_LABELS = {
    'fast_chat': ('Winner', 'Fastest'),
    'long_context': ('Winner', 'Highest Context'),
    'opencode_ready': ('Winner', 'OpenCode-ready'),
    'auto': ('Winner', 'Ideal'),
    'moe_placement': ('Winner', 'MoE placement'),
}

RANK_ROLE_PRIORITY = {
    'Winner': 0,
    'Runner-up': 1,
    'Fastest': 2,
    'Highest Context': 3,
    'OpenCode-ready': 4,
    'Ideal': 5,
    'Auto': 5,
    'Possible': 6,
    'Measured': 7,
    'Passed': 7,
    'Probe': 8,
    'Skipped': 20,
    'Failed': 30,
    'Break Point': 31,
}

BENCHMARK_WIKI_SECTIONS = [
    (
        'What is a benchmark?',
        'A benchmark is a safe test run. llama-tui starts the model with one set of settings, checks that the server is ready, asks it to write a short answer, records speed and stability, then stops that server before trying the next set.',
    ),
    (
        'What the numbers mean',
        'ctx is how much conversation or code the model can keep in memory. ctx/slot is how much of that memory each simultaneous request gets. parallel is how many requests the server can handle at once. tok/s is how fast the model writes. threads is CPU worker count. ngl is how many llama.cpp layers go to the GPU. Headroom is RAM or VRAM left after the test.',
    ),
    (
        'Extra table labels',
        'variant shows runtime tweaks, usually default or q8 KV. measurement_type tells you whether a row was a quick probe or a full speed measurement. planner_reason explains why llama-tui tested that row, such as frontier, speed_knee, chat_parallel, opencode_floor, or q8_probe.',
    ),
    (
        'Deep benchmark: B',
        'B is the careful benchmark. It first finds the safe edge for context, then fully measures only the settings that could realistically win. This is the smart bounded path: less waste than testing everything, but winners still come from real measurements.',
    ),
    (
        'Fast benchmark: F',
        'F is the quick benchmark. It tests a small set of practical settings and gives you a useful first profile faster. Use it when you want a good starting point; use B when you want higher confidence.',
    ),
    (
        'OpenCode benchmark: O',
        'O is the OpenCode check. It runs headless, meaning no terminal opens. llama-tui uses throwaway test projects, captures logs and exit codes, checks the result with python -m unittest -q, then cleans up OpenCode and the model server.',
    ),
    (
        'Reading results',
        'Winner is the saved setting for that category. Runner-up is the next best measured option. Failed means the server did not start, did not become ready, or could not finish the sample. Break Point means a setting failed twice, so llama-tui stopped trying larger or heavier settings in that direction.',
    ),
    (
        'Which result should I use?',
        'Fast Chat is for snappy replies. Long Context is for large files and long sessions. OpenCode-ready is for coding workflows with OpenCode. Auto is the balanced everyday choice when you just want llama-tui to pick a sensible profile.',
    ),
]

FULL_SUITE_STAGES = (
    ('preflight', 'Preflight'),
    ('moe_placement', 'MoE Placement'),
    ('model_benchmark', 'Smart Benchmark'),
    ('hermes', 'Hermes Benchmark'),
    ('opencode', 'OpenCode Benchmark'),
)
MTP_SUITE_STAGES = (
    ('preflight', 'Preflight'),
    ('mtp_acceptance', 'MTP Acceptance'),
    ('moe_placement', 'MoE Placement'),
    ('summary', 'Summary'),
)


def benchmark_wiki_lines(width: int) -> List[str]:
    width = max(24, int(width or 24))
    lines: List[str] = []
    for title, body in BENCHMARK_WIKI_SECTIONS:
        if lines:
            lines.append('')
        lines.append(title)
        lines.extend(wrap_display_lines(body, width))
    return lines


def full_suite_stage_map(records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stages: Dict[str, Dict[str, object]] = {}
    for row in list(records or []):
        if not isinstance(row, dict):
            continue
        stage = str(row.get('stage', '') or '')
        if stage:
            stages[stage] = dict(row)
    return stages


def full_suite_is_mtp(records: Sequence[Dict[str, object]]) -> bool:
    return any(str(row.get('stage', '') or '') == 'mtp_acceptance' for row in list(records or []) if isinstance(row, dict))


def full_suite_status_symbol(status: str, active: bool = False) -> str:
    value = str(status or '').strip().lower()
    if value in ('ok', 'done', 'passed', 'usable', 'complete', 'partial'):
        return '[x]'
    if value in ('skipped', 'skipped_runtime_assert', 'skipped_missing_baseline'):
        return '[-]'
    if value in ('failed', 'aborted', 'blocked', 'blocked_missing_capability', 'failed_terminal'):
        return '[!]'
    if active:
        return '[>]'
    return '[ ]'


def full_suite_stage_lines(records: Sequence[Dict[str, object]], active_phase: str = '') -> List[str]:
    stages = full_suite_stage_map(records)
    phase_key = str(active_phase or '').strip().lower().replace(' ', '_')
    lines: List[str] = []
    stage_sequence = MTP_SUITE_STAGES if full_suite_is_mtp(records) else FULL_SUITE_STAGES
    for key, label in stage_sequence:
        row = stages.get(key, {})
        status = str(row.get('status', '') or '')
        active = bool(phase_key and (phase_key == key or key.replace('_', ' ') in str(active_phase or '').lower()))
        symbol = full_suite_status_symbol(status, active=active)
        detail = compact_message(str(row.get('detail', '') or ''))
        if key == 'moe_placement':
            rec = str(row.get('winner', '') or row.get('candidate', '') or '')
            if rec:
                detail = rec
        if key in ('hermes', 'opencode', 'model_benchmark') and row.get('profile_used'):
            detail = f'profile {row.get("profile_used")}'
        suffix = f'  {detail}' if detail else ''
        lines.append(f'{symbol} {label:17} {status or "pending"}{suffix}')
    return lines


def benchmark_record_score(record: Dict[str, object]) -> Tuple[str, float]:
    if 'score' in record:
        return 'score', float(record.get('score', 0.0) or 0.0)
    return 'tok/s', float(record.get('tokens_per_sec', 0.0) or 0.0)


def benchmark_record_status_kind(record: Dict[str, object]) -> str:
    status = str(record.get('status', '') or '').lower()
    if record.get('break_point'):
        return 'error'
    if status in ('ok', 'tests passed'):
        return 'success'
    if status in ('probe ok', 'time budget exhausted', 'context too small', 'tests failed', 'not hermes-ready', 'skipped'):
        return 'warning'
    if not status or status == '-':
        return 'normal'
    return 'error'


def agent_record_matches_winner(record: Dict[str, object], winner: Dict[str, object]) -> bool:
    if not record or not winner:
        return False
    comparable = ('ctx', 'ctx_per_slot', 'parallel', 'preset', 'tier', 'status')
    for key in comparable:
        if key in winner and key in record and str(record.get(key)) != str(winner.get(key)):
            return False
    record_score = float(record.get('score', 0.0) or 0.0)
    winner_score = float(winner.get('score', 0.0) or 0.0)
    return winner_score <= 0 or abs(record_score - winner_score) < 0.05


def benchmark_record_roles(record: Dict[str, object], winners: Dict[str, object], run_kind: str) -> List[str]:
    labels: List[str] = []
    raw_labels = str(record.get('spectrum_label', '') or '')
    for label in raw_labels.split(','):
        clean = label.strip()
        if clean and clean not in labels:
            labels.append(clean)
    if run_kind in ('opencode', 'hermes') or 'score' in record:
        winner = {}
        if isinstance(winners, dict):
            winner = winners.get(run_kind) or winners.get('agent') or {}
        if isinstance(winner, dict) and agent_record_matches_winner(record, winner):
            labels.append('Winner')
        if str(record.get('status', '') or '') in ('ok', 'tests passed'):
            labels.append('Passed')
    elif isinstance(winners, dict):
        for key, role_labels in SERVER_WINNER_LABELS.items():
            winner = winners.get(key) or {}
            if isinstance(winner, dict) and record_matches_profile(record, winner):
                labels.extend(role_labels)
    status_kind = benchmark_record_status_kind(record)
    if status_kind == 'warning' and str(record.get('status', '') or '').lower() in ('skipped', 'not hermes-ready', 'context too small'):
        labels.append('Skipped')
    if status_kind == 'error':
        labels.append('Break Point' if record.get('break_point') else 'Failed')
    if not labels:
        measurement = str(record.get('measurement_type', '') or '')
        labels.append('Probe' if measurement == 'probe' or str(record.get('status', '') or '') == 'probe ok' else 'Measured')
    deduped: List[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped


def benchmark_role_priority(labels: List[str]) -> int:
    return min((RANK_ROLE_PRIORITY.get(label, 12) for label in labels), default=12)


def ranked_benchmark_records(run: Dict[str, object]) -> List[Tuple[Dict[str, object], List[str]]]:
    winners = run.get('winners') or {}
    records = list(run.get('records', []) or [])
    kind = str(run.get('kind', '') or '')
    ranked: List[Tuple[Dict[str, object], List[str]]] = [
        (record, benchmark_record_roles(record, winners if isinstance(winners, dict) else {}, kind))
        for record in records
        if isinstance(record, dict)
    ]
    agent_run = kind in ('opencode', 'hermes') or any('score' in row for row in records if isinstance(row, dict))

    def sort_key(item: Tuple[Dict[str, object], List[str]]) -> Tuple[object, ...]:
        record, labels = item
        status_kind = benchmark_record_status_kind(record)
        status_group = 0 if status_kind == 'success' else 1 if status_kind == 'warning' else 2 if status_kind == 'error' else 1
        measurement = str(record.get('measurement_type', 'full') or 'full')
        if not agent_run and status_kind == 'success' and measurement == 'probe':
            status_group = 1
        score_label, score = benchmark_record_score(record)
        ctx_slot = int(record.get('ctx_per_slot', 0) or 0)
        seconds = float(record.get('seconds', 0.0) or 0.0)
        if agent_run:
            return (status_group, -score, -ctx_slot, seconds)
        return (status_group, benchmark_role_priority(labels), -score, -ctx_slot, seconds, score_label)

    return sorted(ranked, key=sort_key)


def benchmark_run_is_agent(run: Dict[str, object]) -> bool:
    kind = str((run or {}).get('kind', '') or '')
    records = list((run or {}).get('records', []) or [])
    return kind in ('opencode', 'hermes') or any(isinstance(row, dict) and 'score' in row for row in records)


def _table_row(values: List[object], widths: List[int]) -> str:
    cells = []
    for value, width in zip(values, widths):
        text = ellipsize(str(value), max(1, int(width or 1)))
        cells.append(f'{text:{max(1, int(width or 1))}}')
    return ' '.join(cells).rstrip()


def _table_rule(widths: List[int]) -> str:
    return ' '.join('-' * max(1, int(width or 1)) for width in widths).rstrip()


def _table_widths(fixed: List[int], total_width: int) -> List[int]:
    width = max(1, int(total_width or 1))
    widths = [max(1, int(item or 1)) for item in fixed]
    spaces = len(widths)
    detail_w = max(1, width - sum(widths) - spaces)
    widths.append(detail_w)
    while sum(widths) + len(widths) - 1 > width:
        shrinkable = [idx for idx, value in enumerate(widths[:-1]) if value > 1]
        if not shrinkable:
            break
        target = max(shrinkable, key=lambda idx: widths[idx])
        widths[target] -= 1
    return widths


def _status_attr_for_record(
    record: Dict[str, object],
    success_attr: int,
    warning_attr: int,
    error_attr: int,
    normal_attr: int,
) -> int:
    status_kind = benchmark_record_status_kind(record)
    if status_kind == 'success':
        return success_attr
    if status_kind == 'warning':
        return warning_attr
    if status_kind == 'error':
        return error_attr
    return normal_attr


def benchmark_rank_line(rank: int, record: Dict[str, object], labels: List[str]) -> str:
    score_label, score = benchmark_record_score(record)
    seconds = float(record.get('seconds', 0.0) or 0.0)
    ctx = int(record.get('ctx', 0) or 0)
    parallel = int(record.get('parallel', 0) or 0)
    slot = int(record.get('ctx_per_slot', 0) or 0) or (ctx // max(1, parallel or 1))
    status = str(record.get('status', '-') or '-')
    role_text = ', '.join(labels or ['Measured'])
    detail = compact_message(str(record.get('detail', '') or ''))
    left = (
        f'#{rank:02d} [{role_text}] {status} '
        f'{score:.2f} {score_label} {seconds:.1f}s '
        f'ctx={ctx} slot={slot} par={parallel}'
    )
    return f'{left}  {detail}' if detail else left


def benchmark_rank_table_items(
    run: Dict[str, object],
    width: int = 120,
    success_attr: int = 0,
    warning_attr: int = 0,
    error_attr: int = 0,
    heading_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    if not run:
        return [('No benchmark run selected.', warning_attr)]
    ranked = ranked_benchmark_records(run)
    if not ranked:
        return [('No benchmark rows yet.', warning_attr)]

    width = max(24, int(width or 120))
    agent_run = benchmark_run_is_agent(run)
    items: List[Tuple[str, int]] = []
    if agent_run:
        if width >= 100:
            widths = _table_widths([4, 18, 22, 8, 6, 7, 7, 7], width)
            headers = ['Rank', 'Role', 'Status', 'Score', 'Sec', 'Pass', 'Ctx', 'Slot', 'Detail']
            columns = ('rank', 'role', 'status', 'score', 'seconds', 'pass', 'ctx', 'slot', 'detail')
        elif width >= 72:
            widths = _table_widths([4, 14, 12, 7, 5, 5, 6], width)
            headers = ['Rank', 'Role', 'Status', 'Score', 'Sec', 'Pass', 'Ctx', 'Detail']
            columns = ('rank', 'role', 'status', 'score', 'seconds', 'pass', 'ctx', 'detail')
        elif width >= 40:
            widths = _table_widths([4, 10, 10, 6, 5], width)
            headers = ['Rank', 'Role', 'Status', 'Score', 'Pass', 'Detail']
            columns = ('rank', 'role', 'status', 'score', 'pass', 'detail')
        else:
            widths = _table_widths([4, 8, 6], width)
            headers = ['Rank', 'Role', 'Score', 'Detail']
            columns = ('rank', 'role', 'score', 'detail')
        items.append((_table_row(headers, widths), heading_attr))
        items.append((_table_rule(widths), heading_attr))
        for index, (record, labels) in enumerate(ranked, 1):
            score_label, score = benchmark_record_score(record)
            seconds = float(record.get('seconds', 0.0) or 0.0)
            ctx = int(record.get('ctx', 0) or 0)
            parallel = max(1, int(record.get('parallel', 1) or 1))
            slot = int(record.get('ctx_per_slot', 0) or 0) or (ctx // parallel)
            passed = record.get('passed')
            tasks = record.get('tasks')
            pass_text = f'{int(passed or 0)}/{int(tasks or 0)}' if passed is not None or tasks is not None else '-'
            detail = compact_message(str(record.get('detail', '') or ''))
            record_values = {
                'rank': f'{index:02d}',
                'role': ', '.join(labels or ['Measured']),
                'status': str(record.get('status', '-') or '-'),
                'score': f'{score:.2f}' if score_label == 'score' else '-',
                'seconds': f'{seconds:.1f}' if seconds > 0 else '-',
                'pass': pass_text,
                'ctx': ctx or '-',
                'slot': slot or '-',
                'detail': detail,
            }
            values = [record_values[column] for column in columns]
            items.append((_table_row(values, widths), _status_attr_for_record(record, success_attr, warning_attr, error_attr, normal_attr)))
        return items

    if width >= 112:
        widths = _table_widths([4, 20, 14, 8, 6, 7, 7, 3, 8, 14], width)
        headers = ['Rank', 'Role', 'Status', 'Tok/s', 'Sec', 'Ctx', 'Slot', 'Par', 'Variant', 'Reason', 'Detail']
        columns = ('rank', 'role', 'status', 'score', 'seconds', 'ctx', 'slot', 'parallel', 'variant', 'reason', 'detail')
    elif width >= 72:
        widths = _table_widths([4, 14, 12, 7, 5, 6, 6], width)
        headers = ['Rank', 'Role', 'Status', 'Tok/s', 'Sec', 'Ctx', 'Slot', 'Detail']
        columns = ('rank', 'role', 'status', 'score', 'seconds', 'ctx', 'slot', 'detail')
    elif width >= 40:
        widths = _table_widths([4, 10, 9, 6, 6], width)
        headers = ['Rank', 'Role', 'Status', 'Tok/s', 'Ctx', 'Detail']
        columns = ('rank', 'role', 'status', 'score', 'ctx', 'detail')
    else:
        widths = _table_widths([4, 8, 5], width)
        headers = ['Rank', 'Role', 'Tok/s', 'Detail']
        columns = ('rank', 'role', 'score', 'detail')
    items.append((_table_row(headers, widths), heading_attr))
    items.append((_table_rule(widths), heading_attr))
    for index, (record, labels) in enumerate(ranked, 1):
        _score_label, score = benchmark_record_score(record)
        seconds = float(record.get('seconds', 0.0) or 0.0)
        ctx = int(record.get('ctx', 0) or 0)
        parallel = int(record.get('parallel', 0) or 0)
        slot = int(record.get('ctx_per_slot', 0) or 0) or (ctx // max(1, parallel or 1))
        reason = str(record.get('selection_reason') or record.get('planner_reason') or record.get('scan_level') or record.get('measurement_type') or '')
        detail = compact_message(str(record.get('detail', '') or ''))
        detail_text = detail or (reason if width < 112 else '')
        record_values = {
            'rank': f'{index:02d}',
            'role': ', '.join(labels or ['Measured']),
            'status': str(record.get('status', '-') or '-'),
            'score': f'{score:.2f}' if score > 0 else '-',
            'seconds': f'{seconds:.1f}' if seconds > 0 else '-',
            'ctx': ctx or '-',
            'slot': slot or '-',
            'parallel': parallel or '-',
            'variant': str(record.get('variant', '') or 'default'),
            'reason': reason or '-',
            'detail': detail_text,
        }
        values = [record_values[column] for column in columns]
        items.append((_table_row(values, widths), _status_attr_for_record(record, success_attr, warning_attr, error_attr, normal_attr)))
    return items


def benchmark_ranking_items(
    run: Dict[str, object],
    width: int = 120,
    success_attr: int = 0,
    warning_attr: int = 0,
    error_attr: int = 0,
    heading_attr: int = 0,
    normal_attr: int = 0,
) -> List[Tuple[str, int]]:
    return benchmark_rank_table_items(
        run,
        width=width,
        success_attr=success_attr,
        warning_attr=warning_attr,
        error_attr=error_attr,
        heading_attr=heading_attr,
        normal_attr=normal_attr,
    )


def benchmark_ranking_rows(run: Dict[str, object]) -> List[str]:
    return [line for line, _attr in benchmark_ranking_items(run)]


def benchmark_plan_summary_lines(
    engine: str,
    binary: str,
    capability_summary: Sequence[str],
    candidate_names: Sequence[str],
    skipped: Sequence[Tuple[str, str]] = (),
    strategy_id: str = '',
) -> List[Tuple[str, str]]:
    """Build the benchmark plan preview as ``(text, kind)`` lines.

    Pure/string-only so it is unit testable without an app or curses.
    """
    lines: List[Tuple[str, str]] = [('Benchmark Plan', 'heading')]
    lines.append((f'Engine: {engine or "-"}', 'normal'))
    lines.append((f'Binary: {binary or "-"}', 'muted'))
    if strategy_id:
        lines.append((f'Strategy: {strategy_id}', 'normal'))
    lines.append(('', 'normal'))
    lines.append(('Detected capabilities', 'heading'))
    caps = list(capability_summary or [])
    for cap in (caps or ['none detected']):
        lines.append((f'- {cap}', 'normal' if caps else 'warning'))
    lines.append(('', 'normal'))
    names = [str(item) for item in (candidate_names or []) if str(item or '').strip()]
    lines.append((f'Generated candidates: {len(names)}', 'success' if names else 'warning'))
    important = [n for n in names if any(tag in n.lower() for tag in ('mtp', 'baseline', 'fit'))]
    shown = (important or names)[:8]
    for name in shown:
        lines.append((f'  - {name}', 'normal'))
    if len(names) > len(shown):
        lines.append((f'  ... +{len(names) - len(shown)} more', 'muted'))
    skipped_list = list(skipped or [])
    if skipped_list:
        lines.append(('', 'normal'))
        lines.append(('Skipped', 'heading'))
        for name, reason in skipped_list:
            lines.append((f'  - {name}: {compact_message(str(reason))}', 'warning'))
    return lines


def benchmark_plan_lines(app, model, depth: str = 'fast') -> List[Tuple[str, str]]:
    """Resolve engine/binary/capabilities/candidates and format a plan preview."""
    from .benchmark import active_engine_runtime_profiles, benchmark_strategy_for_app
    from .engines import resolve_runtime_engine_context

    try:
        context = resolve_runtime_engine_context(app, model=model)
    except Exception:
        context = None
    engine = str(getattr(context, 'engine_id', '') or '')
    binary = str(getattr(context, 'command', '') or '')
    caps = getattr(context, 'capabilities', None)
    cap_summary: List[str] = []
    if caps is not None:
        if getattr(caps, 'supports_spec_type', False):
            cap_summary.append(f'--spec-type ({getattr(context, "selected_mtp_spec_type", "") or "none"})')
        if getattr(caps, 'supports_spec_draft_n_max', False):
            cap_summary.append('--spec-draft-n-max')
        if getattr(caps, 'supports_fit', False):
            cap_summary.append('fit')
        if getattr(caps, 'supports_no_mmap', False):
            cap_summary.append('--no-mmap')
        if getattr(caps, 'supports_ctk_ctv', False):
            cap_summary.append('-ctk/-ctv')
        cap_summary.append('MTP capable: ' + ('yes' if getattr(context, 'supports_mtp', False) else 'no'))
    strategy_id = ''
    skipped: List[Tuple[str, str]] = []
    try:
        strategy = benchmark_strategy_for_app(app, model, depth=depth, objective='quick_sanity')
        strategy_id = str(getattr(strategy, 'id', '') or '')
        if getattr(strategy, 'blocked_reason', ''):
            skipped.append((strategy_id or 'strategy', str(strategy.blocked_reason)))
    except Exception:
        pass
    candidate_names: List[str] = []
    try:
        hardware = app.hardware_profile()
        profiles = active_engine_runtime_profiles(app, model, hardware, depth=depth)
        candidate_names = [str(getattr(p, 'name', '') or '') for p in profiles]
    except Exception:
        candidate_names = []
    return benchmark_plan_summary_lines(engine, binary, cap_summary, candidate_names, skipped, strategy_id)
