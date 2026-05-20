"""Unit tests for ``load_model_from_payload``.

Covers the deserialiser extracted in audit #9 step 3. The function is
the safety net between a raw, possibly-old, possibly-hand-edited
``models.json`` entry and a fully-typed ``ModelConfig`` instance, so
the tests focus on edge cases: missing required fields, wrong types
in optional fields, legacy values, and the dataclass_payload filter
that drops unknown keys.
"""

import unittest

from llama_tui.model_loader import (
    VERIFICATION_STATUSES,
    load_model_from_payload,
)
from llama_tui.models import ModelConfig


_MIN_PAYLOAD = {
    'id': 'm', 'name': 'M', 'path': '/m.gguf', 'alias': 'm', 'port': 18080,
}


class LoadModelFromPayloadTests(unittest.TestCase):
    def test_minimum_payload_yields_modelconfig(self):
        model = load_model_from_payload(dict(_MIN_PAYLOAD), 0)
        self.assertIsInstance(model, ModelConfig)
        self.assertEqual(model.id, 'm')
        self.assertEqual(model.port, 18080)

    def test_non_dict_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_model_from_payload('not a dict', 0)
        with self.assertRaises(ValueError):
            load_model_from_payload(None, 0)
        with self.assertRaises(ValueError):
            load_model_from_payload([], 0)

    def test_missing_required_fields_raises_value_error(self):
        bad = {'id': 'm', 'name': 'M', 'path': '/m.gguf'}  # missing alias, port
        with self.assertRaises(ValueError) as ctx:
            load_model_from_payload(bad, 0)
        self.assertIn('alias', str(ctx.exception))
        self.assertIn('port', str(ctx.exception))

    def test_default_values_applied_to_optional_fields(self):
        model = load_model_from_payload(dict(_MIN_PAYLOAD), 0)
        self.assertEqual(model.ctx, 8192)
        self.assertEqual(model.ctx_min, 2048)
        self.assertEqual(model.ctx_max, 131072)
        self.assertEqual(model.threads, 6)
        self.assertEqual(model.ngl, 999)
        self.assertEqual(model.parallel, 1)
        self.assertEqual(model.temp, 0.7)
        self.assertEqual(model.top_p, 0.95)
        self.assertEqual(model.top_k, 40)
        self.assertEqual(model.no_context_shift, False)
        self.assertEqual(model.preserve_thinking, 'auto')

    def test_invalid_preserve_thinking_falls_back_to_auto(self):
        payload = dict(_MIN_PAYLOAD)
        payload['preserve_thinking'] = 'nonsense'
        self.assertEqual(load_model_from_payload(payload, 0).preserve_thinking, 'auto')

    def test_invalid_architecture_type_falls_back_to_unknown(self):
        payload = dict(_MIN_PAYLOAD)
        payload['architecture_type'] = 'transformer'
        self.assertEqual(load_model_from_payload(payload, 0).architecture_type, 'unknown')

    def test_known_architecture_types_pass_through(self):
        for valid in ('dense', 'moe', 'unknown'):
            payload = dict(_MIN_PAYLOAD)
            payload['architecture_type'] = valid
            self.assertEqual(load_model_from_payload(payload, 0).architecture_type, valid)

    def test_invalid_verification_status_falls_back_to_unknown(self):
        payload = dict(_MIN_PAYLOAD)
        payload['verification_status'] = 'made_up'
        self.assertEqual(load_model_from_payload(payload, 0).verification_status, 'unknown')

    def test_known_verification_statuses_pass_through(self):
        for valid in VERIFICATION_STATUSES:
            payload = dict(_MIN_PAYLOAD)
            payload['verification_status'] = valid
            self.assertEqual(load_model_from_payload(payload, 0).verification_status, valid)

    def test_wrong_typed_tensor_overrides_coerces_to_empty(self):
        payload = dict(_MIN_PAYLOAD)
        payload['tensor_overrides'] = 'not a list'
        self.assertEqual(load_model_from_payload(payload, 0).tensor_overrides, [])

    def test_wrong_typed_extra_args_coerces_to_empty(self):
        payload = dict(_MIN_PAYLOAD)
        payload['extra_args'] = {'not': 'a list'}
        self.assertEqual(load_model_from_payload(payload, 0).extra_args, [])

    def test_wrong_typed_launch_overrides_coerces_to_empty(self):
        payload = dict(_MIN_PAYLOAD)
        payload['launch_overrides'] = ['not a dict']
        self.assertEqual(load_model_from_payload(payload, 0).launch_overrides, {})

    def test_unknown_keys_are_dropped_via_dataclass_payload(self):
        # A field added in a newer build must not crash a deserialisation
        # done by an older one — and an extra typo'd key must not slip
        # through unfiltered.
        payload = dict(_MIN_PAYLOAD)
        payload['this_is_not_a_real_field'] = 'should be ignored'
        payload['typoed_field_name'] = 42
        model = load_model_from_payload(payload, 0)
        # The loader returned a clean ModelConfig; the extra keys did
        # not propagate as instance attributes (slots prevents it
        # anyway, but the test also confirms dataclass_payload filtered).
        self.assertFalse(hasattr(model, 'this_is_not_a_real_field'))
        self.assertFalse(hasattr(model, 'typoed_field_name'))

    def test_index_seeds_sort_rank_when_missing(self):
        # When the config has no explicit sort_rank, the loader stamps
        # the entry's index so reload-order is stable.
        model_zero = load_model_from_payload(dict(_MIN_PAYLOAD), 0)
        model_five = load_model_from_payload(dict(_MIN_PAYLOAD), 5)
        self.assertEqual(model_zero.sort_rank, 1)
        self.assertEqual(model_five.sort_rank, 6)

    def test_explicit_sort_rank_takes_precedence(self):
        payload = dict(_MIN_PAYLOAD)
        payload['sort_rank'] = 42
        self.assertEqual(load_model_from_payload(payload, 0).sort_rank, 42)

    def test_n_cpu_moe_clamps_to_zero_minimum(self):
        payload = dict(_MIN_PAYLOAD)
        payload['n_cpu_moe'] = -5
        self.assertEqual(load_model_from_payload(payload, 0).n_cpu_moe, 0)


if __name__ == '__main__':
    unittest.main()
