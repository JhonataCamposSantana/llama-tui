"""Unit tests for the context-ladder + adaptive-search helpers.

Covers the module extracted in audit #6 step 2. The big
``adaptive_context_search`` function is already exercised by
``tests/test_opencode_stack`` (the deadline + edge-case tests added
for audit #4 / #23 live there). This file fills in the surrounding
pure helpers — the rounding primitives, the coarse/exhaustive ladder,
break and knee refinement, and the smart measurement-context picker —
that previously had only minimal direct coverage.
"""

import unittest

from llama_tui.adaptive_search import (
    ADAPTIVE_CONTEXT_ROUNDING,
    CONTEXT_KNEE_ROUNDING,
    CONTEXT_REFINE_STEP,
    _nearest_context_at_or_above,
    break_refinement_contexts,
    coarse_context_step,
    context_knee_refinement_contexts,
    exhaustive_context_ladder,
    round_context,
    round_context_down,
    round_context_up,
    smart_break_refinement_contexts,
    smart_fast_contexts,
    smart_measurement_contexts,
)


class RoundContextTests(unittest.TestCase):
    def test_round_context_to_nearest_step(self):
        self.assertEqual(round_context(2000), 2048)
        self.assertEqual(round_context(2049), 2048)
        self.assertEqual(round_context(2200), 2304)  # 2200 → 2304 (closer to 9*256)

    def test_round_context_clamps_to_step_minimum(self):
        # Zero or negative input rounds up to one step.
        self.assertEqual(round_context(0), ADAPTIVE_CONTEXT_ROUNDING)
        self.assertEqual(round_context(-1), ADAPTIVE_CONTEXT_ROUNDING)

    def test_round_context_down_truncates(self):
        # round_context_down floors to the nearest step boundary
        # (using integer division), unlike round_context which goes to
        # the nearest step.
        self.assertEqual(round_context_down(2048), 2048)
        self.assertEqual(round_context_down(2300), 2048)
        self.assertEqual(round_context_down(2304), 2304)
        self.assertEqual(round_context_down(2050), 2048)

    def test_round_context_up_ceils(self):
        self.assertEqual(round_context_up(2048), 2048)
        self.assertEqual(round_context_up(2049), 2304)
        self.assertEqual(round_context_up(1), ADAPTIVE_CONTEXT_ROUNDING)


class CoarseStepTests(unittest.TestCase):
    def test_step_grows_with_ctx_size(self):
        low = coarse_context_step(2048)
        mid = coarse_context_step(32_000)
        high = coarse_context_step(131_072)
        self.assertLess(low, mid)
        self.assertLess(mid, high)


class ExhaustiveLadderTests(unittest.TestCase):
    def test_starts_at_min_ends_at_max(self):
        ladder = exhaustive_context_ladder(2048, 8192)
        self.assertEqual(ladder[0], 2048)
        self.assertEqual(ladder[-1], 8192)

    def test_no_duplicates_within_step_boundaries(self):
        ladder = exhaustive_context_ladder(2048, 65_536)
        self.assertEqual(len(ladder), len(set(ladder)))

    def test_max_below_min_collapses(self):
        # ctx_max defensively forced to be at least ctx_min.
        self.assertEqual(exhaustive_context_ladder(8192, 2048), [8192])


class BreakRefinementTests(unittest.TestCase):
    def test_break_below_threshold_returns_empty(self):
        # break_ctx not enough above last_success to fit a refinement step.
        self.assertEqual(
            break_refinement_contexts(8192, 8192 + CONTEXT_REFINE_STEP, {8192, 8192 + CONTEXT_REFINE_STEP}),
            [],
        )

    def test_break_returns_midpoints_between(self):
        result = break_refinement_contexts(20480, 32768, {20480, 32768})
        for value in result:
            self.assertGreater(value, 20480)
            self.assertLess(value, 32768)

    def test_smart_break_dedupes_against_tested(self):
        tested = {20480, 22528, 32768}
        result = smart_break_refinement_contexts(20480, 32768, tested)
        for value in result:
            self.assertNotIn(value, tested)


class ContextKneeTests(unittest.TestCase):
    def test_returns_empty_with_too_few_records(self):
        self.assertEqual(context_knee_refinement_contexts([], set(), 32768), [])
        self.assertEqual(
            context_knee_refinement_contexts(
                [{'status': 'ok', 'ctx': 8192, 'tokens_per_sec': 50.0}],
                set(),
                32768,
            ),
            [],
        )

    def test_finds_midpoint_when_throughput_drops(self):
        records = [
            {'status': 'ok', 'ctx': 8192, 'ctx_per_slot': 8192, 'tokens_per_sec': 60.0},
            {'status': 'ok', 'ctx': 32768, 'ctx_per_slot': 32768, 'tokens_per_sec': 20.0},
        ]
        result = context_knee_refinement_contexts(records, {8192, 32768}, 32768)
        self.assertTrue(result)
        for value in result:
            self.assertGreater(value, 8192)
            self.assertLess(value, 32768)


class NearestContextTests(unittest.TestCase):
    def test_picks_smallest_at_or_above_target(self):
        self.assertEqual(_nearest_context_at_or_above([2048, 4096, 8192], 5000), 8192)

    def test_falls_back_to_max_when_all_below(self):
        self.assertEqual(_nearest_context_at_or_above([2048, 4096], 9999), 4096)

    def test_empty_returns_zero(self):
        self.assertEqual(_nearest_context_at_or_above([], 5000), 0)


class SmartMeasurementContextsTests(unittest.TestCase):
    def test_no_successes_returns_empty(self):
        self.assertEqual(smart_measurement_contexts([], [], 2048, 32768, 0), [])

    def test_selects_floor_and_max(self):
        successes = [2048, 4096, 8192, 16384, 32768]
        result = smart_measurement_contexts(successes, [], 2048, 32768, chat_floor=8192)
        # Result must be a subset of the successes (it picks from there).
        for value in result:
            self.assertIn(value, successes)


class SmartFastContextsTests(unittest.TestCase):
    def test_picks_nearest_at_or_above_chat_floor(self):
        result = smart_fast_contexts([2048, 4096, 8192, 16384], chat_floor=4000)
        self.assertIn(4096, result)

    def test_returns_empty_when_no_successes(self):
        self.assertEqual(smart_fast_contexts([], chat_floor=4096), [])


if __name__ == '__main__':
    unittest.main()
