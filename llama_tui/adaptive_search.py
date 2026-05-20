"""Adaptive context-size search and the context-ladder helpers.

Second extraction of audit finding #6 (decompose ``benchmark.py``).
This module is pure: it takes a probe callable and budgets and returns
sorted (success, failure) context lists or refinement ladders. Nothing
in here touches subprocess, hardware, AppConfig, or runtime profiles —
which is what made it a low-risk next pick after
``failure_classification``.

The deadline-aware search at the top is the same algorithm shipped in
audit finding #4 (adaptive-context-search-deadline-aware), moved
verbatim. Tests in ``tests/test_opencode_stack.py`` cover the search
and the surrounding ladder helpers; the re-export at
``benchmark.from .adaptive_search import ...`` keeps every existing
caller working without churn.
"""

from typing import Callable, Dict, List, Optional, Tuple


ADAPTIVE_CONTEXT_ROUNDING = 256
ADAPTIVE_BINARY_STEPS = 4
ADAPTIVE_MAX_CONTEXT_PROBES = 12
EXHAUSTIVE_CONTEXT_STEP = 2048
COARSE_CONTEXT_LOW_LIMIT = 16_384
COARSE_CONTEXT_MID_LIMIT = 65_536
COARSE_CONTEXT_LOW_STEP = 2_048
COARSE_CONTEXT_MID_STEP = 4_096
COARSE_CONTEXT_HIGH_STEP = 8_192
CONTEXT_REFINE_STEP = 2_048
CONTEXT_KNEE_ROUNDING = 1_024
SMART_MAX_FULL_CONTEXTS_PER_VARIANT = 5


def round_context(value: int, step: int = ADAPTIVE_CONTEXT_ROUNDING) -> int:
    value = max(1, int(value or 1))
    step = max(1, int(step or 1))
    return max(step, int(round(value / step) * step))


def round_context_down(value: int, step: int = ADAPTIVE_CONTEXT_ROUNDING) -> int:
    value = max(1, int(value or 1))
    step = max(1, int(step or 1))
    return max(step, int(value // step) * step)


def round_context_up(value: int, step: int = ADAPTIVE_CONTEXT_ROUNDING) -> int:
    value = max(1, int(value or 1))
    step = max(1, int(step or 1))
    return max(step, int(((value + step - 1) // step) * step))


def adaptive_context_search(
    ctx_min: int,
    ctx_upper: int,
    probe: Callable[[int], bool],
    max_probes: int = ADAPTIVE_MAX_CONTEXT_PROBES,
    deadline_expired: Optional[Callable[[], bool]] = None,
) -> Tuple[List[int], List[int]]:
    """Search the largest working context size via exponential growth then
    binary refinement and gap-filling.

    ``deadline_expired``, when provided, is checked before every probe and
    between each refinement step. If it returns True, the search exits early
    with whatever successes/failures were collected — this prevents a single
    long-context candidate from consuming the entire global benchmark budget.
    """
    ctx_min = round_context(max(256, ctx_min))
    ctx_upper = max(ctx_min, round_context_down(ctx_upper))
    successes: List[int] = []
    failures: List[int] = []
    seen = set()

    def _out_of_time() -> bool:
        return bool(deadline_expired and deadline_expired())

    def run_probe(value: int) -> bool:
        value = max(ctx_min, min(ctx_upper, round_context(value)))
        if value in seen or len(seen) >= max_probes:
            return value in successes
        seen.add(value)
        ok = bool(probe(value))
        (successes if ok else failures).append(value)
        return ok

    current = ctx_min
    last_success = 0
    first_failure = 0
    while len(seen) < max_probes:
        if _out_of_time():
            return sorted(set(successes)), sorted(set(failures))
        ok = run_probe(current)
        if ok:
            last_success = current
            if current >= ctx_upper:
                break
            current = min(ctx_upper, max(current + ADAPTIVE_CONTEXT_ROUNDING, current * 2))
            continue
        first_failure = current
        break

    if last_success and not first_failure and last_success < ctx_upper and len(seen) < max_probes:
        if _out_of_time():
            return sorted(set(successes)), sorted(set(failures))
        if run_probe(ctx_upper):
            last_success = ctx_upper
        else:
            first_failure = ctx_upper

    if last_success and first_failure:
        low = min(last_success, first_failure)
        high = max(last_success, first_failure)
        for _ in range(ADAPTIVE_BINARY_STEPS):
            if len(seen) >= max_probes or _out_of_time():
                break
            midpoint = round_context((low + high) // 2)
            if midpoint <= low or midpoint >= high:
                break
            if run_probe(midpoint):
                low = midpoint
            else:
                high = midpoint

    while len(seen) < max_probes and len(successes) >= 2:
        if _out_of_time():
            break
        ordered = sorted(set(successes))
        gaps = [(ordered[idx + 1] - ordered[idx], ordered[idx], ordered[idx + 1]) for idx in range(len(ordered) - 1)]
        gaps = [gap for gap in sorted(gaps, reverse=True) if gap[0] > ADAPTIVE_CONTEXT_ROUNDING * 2]
        if not gaps:
            break
        _gap, low, high = gaps[0]
        midpoint = round_context((low + high) // 2)
        if midpoint in seen:
            break
        run_probe(midpoint)

    return sorted(set(successes)), sorted(set(failures))


def coarse_context_step(ctx: int) -> int:
    ctx = max(1, int(ctx or 1))
    if ctx < COARSE_CONTEXT_LOW_LIMIT:
        return COARSE_CONTEXT_LOW_STEP
    if ctx < COARSE_CONTEXT_MID_LIMIT:
        return COARSE_CONTEXT_MID_STEP
    return COARSE_CONTEXT_HIGH_STEP


def exhaustive_context_ladder(ctx_min: int, ctx_max: int, step: int = EXHAUSTIVE_CONTEXT_STEP) -> List[int]:
    ctx_min = max(1, int(ctx_min or 1))
    ctx_max = max(ctx_min, int(ctx_max or ctx_min))
    values = [ctx_min]
    current = ctx_min
    while current < ctx_max:
        current = min(ctx_max, current + coarse_context_step(current))
        if current != values[-1]:
            values.append(current)
    if values[-1] != ctx_max:
        values.append(ctx_max)
    return values


def break_refinement_contexts(last_success_ctx: int, break_ctx: int, tested: set) -> List[int]:
    last_success_ctx = int(last_success_ctx or 0)
    break_ctx = int(break_ctx or 0)
    if last_success_ctx <= 0 or break_ctx <= last_success_ctx + CONTEXT_REFINE_STEP:
        return []
    values = []
    current = last_success_ctx + CONTEXT_REFINE_STEP
    while current < break_ctx:
        if current not in tested:
            values.append(current)
        current += CONTEXT_REFINE_STEP
    return values


def context_knee_refinement_contexts(
    records: List[Dict[str, object]],
    tested: set,
    ctx_max: int,
) -> List[int]:
    successful = sorted(
        [record for record in records if record.get('status') == 'ok'],
        key=lambda record: int(record.get('ctx', 0) or 0),
    )
    if len(successful) < 2:
        return []
    ctx_max = max(1, int(ctx_max or 1))
    max_tps = max(float(record.get('tokens_per_sec', 0.0) or 0.0) for record in successful) or 1.0
    max_ctx = max(int(record.get('ctx_per_slot', 0) or record.get('ctx', 0) or 0) for record in successful) or 1
    candidates = set()
    scored = []
    for idx in range(len(successful) - 1):
        left = successful[idx]
        right = successful[idx + 1]
        left_ctx = int(left.get('ctx', 0) or 0)
        right_ctx = int(right.get('ctx', 0) or 0)
        gap = right_ctx - left_ctx
        if gap <= CONTEXT_REFINE_STEP:
            continue
        left_tps = float(left.get('tokens_per_sec', 0.0) or 0.0)
        right_tps = float(right.get('tokens_per_sec', 0.0) or 0.0)
        drop = max(0.0, left_tps - right_tps) / max(left_tps, 1.0)
        ctx_gain = gap / max(ctx_max, 1)
        midpoint = round_context((left_ctx + right_ctx) // 2, CONTEXT_KNEE_ROUNDING)
        if left_ctx < midpoint < right_ctx and midpoint not in tested:
            if drop >= 0.18 or (drop >= 0.05 and ctx_gain >= 0.20):
                candidates.add(midpoint)
        left_score = 0.55 * (left_tps / max_tps) + 0.45 * (int(left.get('ctx_per_slot', left_ctx) or left_ctx) / max_ctx)
        scored.append((left_score, idx))
    last = successful[-1]
    last_score = 0.55 * (float(last.get('tokens_per_sec', 0.0) or 0.0) / max_tps) + 0.45 * (
        int(last.get('ctx_per_slot', last.get('ctx', 0)) or 0) / max_ctx
    )
    scored.append((last_score, len(successful) - 1))
    if scored:
        _score, best_idx = max(scored)
        for neighbor_idx in (best_idx - 1, best_idx):
            if 0 <= neighbor_idx < len(successful) - 1:
                left_ctx = int(successful[neighbor_idx].get('ctx', 0) or 0)
                right_ctx = int(successful[neighbor_idx + 1].get('ctx', 0) or 0)
                if right_ctx - left_ctx > CONTEXT_REFINE_STEP:
                    midpoint = round_context((left_ctx + right_ctx) // 2, CONTEXT_KNEE_ROUNDING)
                    if left_ctx < midpoint < right_ctx and midpoint not in tested:
                        candidates.add(midpoint)
    return sorted(candidates)


def smart_break_refinement_contexts(last_success_ctx: int, break_ctx: int, tested: set) -> List[int]:
    last_success_ctx = int(last_success_ctx or 0)
    break_ctx = int(break_ctx or 0)
    if last_success_ctx <= 0 or break_ctx <= last_success_ctx:
        return []
    candidates = {
        last_success_ctx + CONTEXT_REFINE_STEP,
        round_context((last_success_ctx + break_ctx) // 2, CONTEXT_KNEE_ROUNDING),
        break_ctx - CONTEXT_REFINE_STEP,
    }
    return sorted(
        value for value in candidates
        if last_success_ctx < value < break_ctx and value not in tested
    )


def _nearest_context_at_or_above(contexts: List[int], target: int) -> int:
    if not contexts:
        return 0
    target = int(target or 0)
    above = [ctx for ctx in contexts if ctx >= target]
    if above:
        return min(above, key=lambda ctx: (ctx - target, ctx))
    return max(contexts)


def smart_measurement_contexts(
    successes: List[int],
    failures: List[int],
    ctx_min: int,
    ctx_max: int,
    chat_floor: int,
    opencode_floor: int = 0,
) -> List[int]:
    ordered = sorted(set(int(ctx) for ctx in successes if int(ctx) > 0))
    if not ordered:
        return []
    selected = {ordered[0], ordered[-1]}
    if len(ordered) >= 2:
        selected.add(ordered[-2])
    span = max(0, ordered[-1] - ordered[0])
    if span > CONTEXT_REFINE_STEP:
        midpoint = round_context((ordered[0] + ordered[-1]) // 2, CONTEXT_KNEE_ROUNDING)
        selected.add(min(ordered, key=lambda ctx: (abs(ctx - midpoint), ctx)))
    for target in (chat_floor, opencode_floor, ctx_min, ctx_max):
        if int(target or 0) > 0:
            selected.add(_nearest_context_at_or_above(ordered, int(target)))
    positive_failures = [int(ctx) for ctx in failures if int(ctx) > 0]
    if positive_failures:
        first_break = min(positive_failures)
        below_break = [ctx for ctx in ordered if ctx < first_break]
        if below_break:
            selected.add(max(below_break))
    return sorted(ctx for ctx in selected if ctx in ordered)[:SMART_MAX_FULL_CONTEXTS_PER_VARIANT]


def smart_fast_contexts(successes: List[int], chat_floor: int) -> List[int]:
    ordered = sorted(set(int(ctx) for ctx in successes if int(ctx) > 0))
    if not ordered:
        return []
    primary = _nearest_context_at_or_above(ordered, chat_floor)
    selected = {primary}
    above = [ctx for ctx in ordered if ctx > primary]
    if above and primary <= max(chat_floor, 1) * 2:
        selected.add(above[0])
    return sorted(selected)
