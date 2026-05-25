import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.engines import (
    ENGINE_LLAMA_CPP,
    ENGINE_LLAMA_CPP_MTP,
    ENGINE_TURBOQUANT,
)
from llama_tui.model_compat import detect_model_runtime_features, engine_shows_model, engine_supports_model
from llama_tui.models import ModelConfig
from llama_tui.provenance import parse_hf_cache_provenance
from llama_tui.runtime_profiles import EngineCapabilities, make_runtime_profile
from llama_tui.ui import browser_models


def model(model_id: str, path: str, **kwargs) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        name=kwargs.pop('name', model_id),
        path=path,
        alias=model_id,
        **kwargs,
    )


class ModelCompatibilityTests(unittest.TestCase):
    def test_tq3_native_is_unsupported_on_every_engine(self):
        # TQ3 engine removed (2026-05): a TQ3-native GGUF is detected by
        # name/tensors but no remaining engine can run its weights.
        tq3 = model('tq3', '/models/generic-native-TQ3_4S.gguf')

        self.assertIn('tq3_native', detect_model_runtime_features(tq3))
        for engine in (ENGINE_LLAMA_CPP, ENGINE_TURBOQUANT, ENGINE_LLAMA_CPP_MTP):
            self.assertEqual(engine_supports_model(engine, tq3, EngineCapabilities(supports_mtp=True)).status, 'unsupported')

    def test_mtp_native_prefers_capable_binary_on_llama_cpp_family(self):
        # Audit #7: MTP is a binary capability now. Any llama.cpp-family
        # engine becomes 'preferred' for an MTP-native model when the
        # binary advertises the speculative MTP flags; without those
        # flags it's 'compatible_with_warning' (it will still load,
        # just without MTP acceleration). The legacy MTP engine alias
        # normalises to llama.cpp so its result is identical.
        mtp = model('mtp', '/models/generic-GGUF-MTP.gguf', supports_mtp='yes')

        self.assertIn('mtp_native', detect_model_runtime_features(mtp))
        capable_caps = EngineCapabilities(
            supports_spec_type=True,
            supports_mtp=True,
            mtp_spec_type='mtp',
            supports_spec_draft_n_max=True,
        )
        for engine in (ENGINE_LLAMA_CPP, ENGINE_LLAMA_CPP_MTP, ENGINE_TURBOQUANT):
            result = engine_supports_model(engine, mtp, capable_caps)
            self.assertEqual(result.status, 'preferred', f'{engine!r} should be preferred with MTP caps')
            self.assertTrue(result.compatible)
        # Without MTP caps the model still loads but with a warning.
        for engine in (ENGINE_LLAMA_CPP, ENGINE_TURBOQUANT):
            self.assertEqual(engine_supports_model(engine, mtp).status, 'compatible_with_warning')

    def test_mtp_native_with_uncapable_binary_loads_with_warning(self):
        # Audit #7: previously an MTP-native model on the MTP engine
        # without MTP flags was 'unsupported'/'compatible_with_warning'.
        # Now with the engine collapsed, an MTP-native model on plain
        # llama.cpp without MTP flags is consistently
        # 'compatible_with_warning' — it still loads, just without MTP
        # acceleration, and the warning tells the user how to upgrade.
        mtp = model('mtp', '/models/generic-native-mtp.gguf', supports_mtp='yes')
        not_ready_caps = EngineCapabilities(
            supports_spec_type=True,
            supports_mtp=False,
            supports_spec_draft_n_max=True,
        )

        launch = engine_supports_model(ENGINE_LLAMA_CPP, mtp, not_ready_caps)
        visibility = engine_shows_model(ENGINE_LLAMA_CPP, mtp, not_ready_caps)

        self.assertTrue(launch.compatible)
        self.assertEqual(launch.status, 'compatible_with_warning')
        self.assertIn('draft-mtp', launch.reason)
        self.assertTrue(visibility.compatible)
        self.assertEqual(visibility.status, 'compatible_with_warning')

        ready_caps = EngineCapabilities(
            supports_spec_type=True,
            supports_mtp=True,
            mtp_spec_type='mtp',
            supports_spec_draft_n_max=True,
        )
        self.assertEqual(engine_supports_model(ENGINE_LLAMA_CPP, mtp, ready_caps).status, 'preferred')
        self.assertTrue(engine_shows_model(ENGINE_LLAMA_CPP, mtp, ready_caps).compatible)

    def test_mtp_hint_detection_is_not_family_specific(self):
        explicit = model('explicit', '/models/custom-name.gguf', supports_mtp='yes')
        hinted = model('hinted', '/models/custom-native-mtp.gguf')
        unrelated = model('unrelated', '/models/future-family.gguf')

        self.assertIn('mtp_native', detect_model_runtime_features(explicit))
        self.assertIn('mtp_native', detect_model_runtime_features(hinted))
        self.assertNotIn('mtp_native', detect_model_runtime_features(unrelated))

    def test_parse_hf_cache_provenance_is_generic(self):
        provenance = parse_hf_cache_provenance(Path('/cache/hub/models--some-owner--some-model-MTP-GGUF/snapshots/abc123/Generic.gguf'))

        self.assertEqual(provenance['repo_folder'], 'models--some-owner--some-model-MTP-GGUF')
        self.assertEqual(provenance['repo_id'], 'some-owner/some-model-MTP-GGUF')
        self.assertEqual(provenance['snapshot'], 'abc123')

    def test_detects_mtp_from_hf_cache_repo_folder_even_if_file_name_lacks_mtp(self):
        candidate = model(
            'generic',
            '/cache/hub/models--some-owner--some-model-MTP-GGUF/snapshots/abc123/Generic-UD-Q3_K_XL.gguf',
            name='Generic UD Q3 K XL',
        )

        self.assertIn('mtp_native', detect_model_runtime_features(candidate))

    def test_detects_tq3_from_hf_cache_repo_folder_even_if_display_name_is_generic(self):
        candidate = model(
            'generic-tq3',
            '/cache/hub/models--some-owner--some-model-TQ3_4S-GGUF/snapshots/abc123/Generic.gguf',
            name='Generic Model',
        )

        self.assertIn('tq3_native', detect_model_runtime_features(candidate))

    def test_normal_gguf_is_uncertain_for_specialized_engines(self):
        # Audit #7: ENGINE_LLAMA_CPP_MTP collapses to ENGINE_LLAMA_CPP,
        # so a normal GGUF is 'compatible' across the llama.cpp family.
        normal = model('normal', '/models/llama-q4_k_m.gguf')

        self.assertIn('normal_gguf', detect_model_runtime_features(normal))
        self.assertIn('dense', detect_model_runtime_features(normal))
        self.assertTrue(engine_supports_model(ENGINE_LLAMA_CPP, normal).compatible)
        self.assertTrue(engine_supports_model(ENGINE_TURBOQUANT, normal).compatible)
        # Legacy alias resolves identically to llama.cpp now.
        self.assertTrue(engine_supports_model(ENGINE_LLAMA_CPP_MTP, normal).compatible)

    def test_unknown_gguf_is_warning_on_standard_engine(self):
        unknown = model('unknown', '/models/mystery.gguf')

        result = engine_supports_model(ENGINE_LLAMA_CPP, unknown)

        self.assertIn('unknown_quant', detect_model_runtime_features(unknown))
        self.assertEqual(result.status, 'compatible_with_warning')
        self.assertTrue(result.compatible)

    def test_browser_defaults_can_filter_by_active_engine_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            app.models = [
                model('normal', '/models/normal-q4.gguf', port=18080),
                model('tq3', '/models/native-TQ3_4S.gguf', port=18081),
                model('mtp', '/models/generic-native-mtp.gguf', port=18082, supports_mtp='yes'),
            ]
            statuses = {item.id: ('STOPPED', '') for item in app.models}

            app.runtime_profile = make_runtime_profile('llama.cpp', 'llama-server')
            self.assertEqual(
                [item.id for item in browser_models(app, statuses, compatibility_filter='active')],
                ['normal', 'mtp'],
            )

            # Audit #7: --engine llama.cpp-mtp now collapses to llama.cpp.
            # An MTP-capable binary makes the MTP-native model 'preferred',
            # and the normal model is also compatible, so both surface
            # under the active filter.
            app.runtime_profile = make_runtime_profile('llama.cpp-mtp', 'llama-server')
            app.engine_capabilities = lambda: EngineCapabilities(
                supports_spec_type=True,
                supports_mtp=True,
                mtp_spec_type='mtp',
                supports_spec_draft_n_max=True,
            )
            self.assertEqual(
                [item.id for item in browser_models(app, statuses, compatibility_filter='active')],
                ['normal', 'mtp'],
            )

            self.assertEqual(
                [item.id for item in browser_models(app, statuses, compatibility_filter='all')],
                ['normal', 'tq3', 'mtp'],
            )

    def test_browser_active_filter_shows_mtp_model_when_binary_is_not_launch_ready(self):
        # Audit #7: the legacy MTP engine collapses to llama.cpp. With a
        # binary that lacks MTP flags, the MTP-native model is still
        # browsable (warning), and the normal GGUF remains compatible.
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            app.models = [
                model('normal', '/models/normal-q4.gguf', port=18080),
                model('tq3', '/models/native-TQ3_4S.gguf', port=18081),
                model('mtp', '/models/generic-native-mtp.gguf', port=18082, supports_mtp='yes'),
            ]
            statuses = {item.id: ('STOPPED', '') for item in app.models}

            app.runtime_profile = make_runtime_profile('llama.cpp-mtp', 'llama-server')
            app.engine_capabilities = lambda: EngineCapabilities(
                supports_spec_type=True,
                supports_mtp=False,
                supports_spec_draft_n_max=True,
            )

            self.assertEqual(
                [item.id for item in browser_models(app, statuses, compatibility_filter='active')],
                ['normal', 'mtp'],
            )
            visible, visibility_reason = app.active_engine_model_visibility(app.models[2])
            compatible, compatibility_reason = app.active_engine_model_compatibility(app.models[2])
            self.assertTrue(visible)
            self.assertTrue(compatible)
            self.assertIn('draft-mtp', compatibility_reason)

    def test_hf_and_llama_cache_discovery_dedupes_by_resolved_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / 'home'
            hf_root = root / 'hf'
            gguf = hf_root / 'models--owner--repo' / 'snapshots' / 'abc' / 'generic-native-mtp.gguf'
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(b'not a real gguf but enough for discovery')

            with patch.dict(
                os.environ,
                {
                    'HF_HUB_CACHE': str(hf_root),
                    'HUGGINGFACE_HUB_CACHE': '',
                    'HF_HOME': '',
                    'LLAMA_CACHE': '',
                },
                clear=False,
            ), \
                 patch('llama_tui.app.Path.home', return_value=home):
                app = AppConfig(root / 'models.json')
                app.hf_cache_root = str(hf_root)

                discovered, _notes = app.discover_source_files()
                self.assertEqual(len(discovered), 1)
                _path, source, provenance = next(iter(discovered.values()))
                self.assertEqual(source, 'huggingface,hf_cache')
                self.assertEqual(provenance['source_labels'], ['huggingface', 'hf_cache'])
                self.assertEqual(provenance['source_repo_id'], 'owner/repo')
                self.assertEqual(provenance['source_snapshot'], 'abc')

                added, _messages = app.detect_models()
                self.assertEqual(added, 1)
                self.assertEqual(len(app.models), 1)
                self.assertEqual(app.models[0].supports_mtp, 'yes')
                self.assertEqual(app.models[0].source, 'huggingface,hf_cache')
                self.assertEqual(app.models[0].source_labels, ['huggingface', 'hf_cache'])
                self.assertEqual(app.models[0].source_repo_id, 'owner/repo')
                self.assertEqual(app.models[0].source_snapshot, 'abc')
                self.assertEqual(app.models[0].source_path, str(gguf))
                self.assertEqual(app.models[0].source_root, str(hf_root))

                added_again, _messages = app.detect_models()
                self.assertEqual(added_again, 0)
                self.assertEqual(len(app.models), 1)


if __name__ == '__main__':
    unittest.main()
