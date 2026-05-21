import unittest

from llama_tui.hardware import HardwareProfile
from llama_tui.memory_guardrail import (
    GIB,
    MemoryGuardrailPolicy,
    MemoryGuardrailState,
    memory_guardrail_decision,
    memory_guardrail_record_fields,
)


def _profile(ram_avail, vram_free=6 * GIB):
    return HardwareProfile(
        memory_total=32 * GIB,
        memory_available=ram_avail,
        gpu_memory_total=8 * GIB,
        gpu_memory_free=vram_free,
    )


class PolicyFloorTests(unittest.TestCase):
    def test_dynamic_floor_scales_with_total(self):
        policy = MemoryGuardrailPolicy.balanced()
        # 7% of 32 GiB exceeds the 1 GiB absolute floor.
        self.assertEqual(policy.ram_critical_floor(32 * GIB), int(32 * GIB * 0.07))
        # Absolute floor wins for a tiny machine.
        self.assertEqual(policy.ram_critical_floor(4 * GIB), 1 * GIB)


class DecisionTests(unittest.TestCase):
    def test_healthy_profile_allows(self):
        decision = memory_guardrail_decision(_profile(20 * GIB))
        self.assertEqual(decision.action, 'allow')
        self.assertFalse(decision.critical)

    def test_critical_ram_stops_during_runtime(self):
        decision = memory_guardrail_decision(_profile(1 * GIB), phase='runtime')
        self.assertEqual(decision.action, 'stop')
        self.assertTrue(decision.critical)
        self.assertEqual(decision.status, 'memory_guardrail_stopped')

    def test_critical_ram_skips_during_admission(self):
        decision = memory_guardrail_decision(_profile(1 * GIB), phase='admission')
        self.assertEqual(decision.action, 'skip')
        self.assertEqual(decision.status, 'memory_guardrail_skipped')

    def test_warning_band_warns(self):
        # 3 GiB sits between the critical (~2.24 GiB) and warning (~3.84 GiB) floors.
        decision = memory_guardrail_decision(_profile(3 * GIB))
        self.assertEqual(decision.action, 'warn')
        self.assertTrue(decision.warning)
        self.assertFalse(decision.critical)


class RecordFieldsAndStateTests(unittest.TestCase):
    def test_record_fields_empty_for_allow(self):
        decision = memory_guardrail_decision(_profile(20 * GIB))
        self.assertEqual(memory_guardrail_record_fields(decision), {})

    def test_record_fields_present_for_stop(self):
        decision = memory_guardrail_decision(_profile(1 * GIB))
        fields = memory_guardrail_record_fields(decision)
        self.assertEqual(fields['memory_guardrail_action'], 'stop')
        self.assertIn('memory_guardrail_reason', fields)

    def test_state_tracks_minimums_and_stop(self):
        state = MemoryGuardrailState()
        state.observe(_profile(20 * GIB))
        state.observe(_profile(1 * GIB))
        self.assertIsNotNone(state.stop_decision)
        self.assertEqual(state.min_ram_available, 1 * GIB)
        self.assertEqual(state.record_fields()['memory_guardrail_min_ram_available'], 1 * GIB)


class AdmissionSkipTests(unittest.TestCase):
    """Pressure-based admission skip fires only above the 0.70 threshold."""

    _ADMISSION_KWARGS = dict(
        phase='admission',
        candidate_ctx=2048,
        safe_ctx=529,
    )

    def test_old_trigger_pressure_no_longer_skips(self):
        # pressure=0.46 used to fire at the old 0.45 threshold — must not skip now.
        decision = memory_guardrail_decision(
            _profile(20 * GIB), pressure_score=0.46, **self._ADMISSION_KWARGS
        )
        self.assertNotEqual(decision.action, 'skip')

    def test_mid_band_pressure_does_not_skip(self):
        decision = memory_guardrail_decision(
            _profile(20 * GIB), pressure_score=0.65, **self._ADMISSION_KWARGS
        )
        self.assertNotEqual(decision.action, 'skip')

    def test_high_pressure_skips(self):
        decision = memory_guardrail_decision(
            _profile(20 * GIB), pressure_score=0.75, **self._ADMISSION_KWARGS
        )
        self.assertEqual(decision.action, 'skip')
        self.assertEqual(decision.status, 'memory_guardrail_skipped')

    def test_required_for_floor_bypasses_skip(self):
        decision = memory_guardrail_decision(
            _profile(20 * GIB),
            pressure_score=0.75,
            required_for_floor=True,
            **self._ADMISSION_KWARGS,
        )
        self.assertNotEqual(decision.action, 'skip')

    def test_memory_warning_alone_does_not_skip(self):
        # VRAM near warning floor with low pressure should warn, not skip.
        decision = memory_guardrail_decision(
            _profile(20 * GIB, vram_free=1 * GIB),
            pressure_score=0.10,
            **self._ADMISSION_KWARGS,
        )
        self.assertNotEqual(decision.action, 'skip')
        self.assertEqual(decision.warning, True)


if __name__ == '__main__':
    unittest.main()
