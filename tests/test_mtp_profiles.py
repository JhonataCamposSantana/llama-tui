import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_tui.app import AppConfig
from llama_tui.benchmark_strategies import select_benchmark_strategy
from llama_tui.engines import (
    ENGINE_LLAMA_CPP,
    ENGINE_LLAMA_CPP_MTP,
    resolve_runtime_engine_context,
)
from llama_tui.models import ModelConfig
from llama_tui.benchmark_mtp import (
    annotate_mtp_optimizer_records,
    mtp_optimizer_profile_recommendations,
)
from llama_tui.runtime_profiles import (
    RuntimeProfile,
    command_spec_type_value,
    default_engine_capabilities,
    make_runtime_profile,
    validate_mtp_command,
)


def mtp_caps(spec_values=('none', 'draft-mtp'), *, fit=False):
    from llama_tui.runtime_profiles import select_mtp_spec_type_value

    spec_type = select_mtp_spec_type_value(spec_values)
    base = replace(
        default_engine_capabilities(ENGINE_LLAMA_CPP),
        help_text='--spec-type VALUE\n--spec-draft-n-max N\n--no-warmup -ctk TYPE -ctv TYPE',
        supports_spec_type=True,
        supports_mtp=bool(spec_type),
        spec_type_values=tuple(spec_values),
        mtp_spec_type=spec_type,
        mtp_spec_type_value=spec_type,
        supports_spec_draft_n_max=True,
        supports_no_warmup=True,
        supports_ctk_ctv=True,
        supports_spec_draft_type_kv=True,
    )
    if fit:
        base = replace(
            base,
            supports_fit=True,
            supports_fit_ctx=True,
            supports_fit_target=True,
            supports_cache_ram=True,
            supports_no_mmap=True,
        )
    return base


def plain_caps():
    return replace(
        default_engine_capabilities(ENGINE_LLAMA_CPP),
        help_text='--ctx-size N --parallel N',
        supports_spec_type=False,
        supports_mtp=False,
        spec_type_values=(),
    )


def make_model(**overrides):
    fields = dict(
        id='mtp',
        name='Generic native-mtp',
        path='/models/generic-native-mtp.gguf',
        alias='mtp',
        port=18080,
        supports_mtp='yes',
        mtp_enabled=True,
        mtp_draft_n_max=1,
    )
    fields.update(overrides)
    return ModelConfig(**fields)


class MtpCommandValidatorTests(unittest.TestCase):
    def test_validator_rejects_unadvertised_spec_type(self):
        ok, msg = validate_mtp_command(
            ['llama-server', '-m', 'x.gguf', '--spec-type', 'mtp'],
            mtp_caps(('none', 'draft-mtp')),
        )
        self.assertFalse(ok)
        self.assertIn('INVALID_MTP_SPEC_TYPE', msg)
        self.assertIn('requested mtp', msg)
        self.assertIn('draft-mtp', msg)

    def test_validator_accepts_advertised_spec_type(self):
        ok, msg = validate_mtp_command(
            ['llama-server', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '1'],
            mtp_caps(('none', 'draft-mtp')),
        )
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_validator_allows_command_without_spec_type(self):
        ok, msg = validate_mtp_command(['llama-server', '-m', 'x.gguf'], mtp_caps())
        self.assertTrue(ok)
        self.assertEqual(msg, '')

    def test_validator_allows_when_capabilities_unknown(self):
        ok, _msg = validate_mtp_command(['llama-server', '--spec-type', 'mtp'], plain_caps())
        self.assertTrue(ok)


class MtpSpecTypeSelectionTests(unittest.TestCase):
    def test_draft_mtp_only_selects_draft_mtp(self):
        self.assertEqual(mtp_caps(('none', 'draft-mtp')).mtp_spec_type_value, 'draft-mtp')

    def test_mtp_only_selects_mtp_legacy(self):
        self.assertEqual(mtp_caps(('none', 'mtp')).mtp_spec_type_value, 'mtp')

    def test_both_advertised_prefers_draft_mtp(self):
        self.assertEqual(mtp_caps(('none', 'mtp', 'draft-mtp')).mtp_spec_type_value, 'draft-mtp')

    def test_neither_advertised_is_unsupported(self):
        caps = mtp_caps(('none', 'ngram-simple'))
        self.assertEqual(caps.mtp_spec_type_value, '')
        self.assertFalse(caps.supports_mtp)


class MtpBuildCommandTests(unittest.TestCase):
    def _app(self, engine_id):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return AppConfig(
            Path(tmp.name) / 'models.json',
            runtime_profile=make_runtime_profile(engine_id, 'llama-server'),
        )

    def test_build_command_never_emits_legacy_mtp_when_binary_has_draft_mtp(self):
        app = self._app(ENGINE_LLAMA_CPP_MTP)
        model = make_model()
        # A stale profile still carrying the old `mtp` dialect must not leak
        # through: the binary only advertises draft-mtp.
        profile = RuntimeProfile(
            engine_id=ENGINE_LLAMA_CPP_MTP,
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            mtp_enabled=True,
            mtp_draft_n_max=1,
            mtp_spec_type='mtp',
        )
        with patch.object(app, 'engine_capabilities', return_value=mtp_caps(('none', 'draft-mtp'))):
            cmd = app.build_command(model, runtime_profile=profile)
        self.assertEqual(command_spec_type_value(cmd), 'draft-mtp')

    def test_build_command_uses_runtime_profile_specific_capabilities(self):
        # Global engine is plain llama.cpp with a non-MTP binary; the benchmark
        # passes an MTP runtime profile and build_command must resolve MTP
        # capabilities from that profile's engine, not the global state.
        app = self._app(ENGINE_LLAMA_CPP)
        model = make_model()
        mtp_profile = RuntimeProfile(
            engine_id=ENGINE_LLAMA_CPP_MTP,
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            mtp_enabled=True,
            mtp_draft_n_max=2,
        )

        def fake_caps(engine_profile=None, *, server_bin=None, engine_id=None):
            resolved = engine_id or getattr(engine_profile, 'engine_id', None) or app.runtime_profile.engine_id
            if resolved == ENGINE_LLAMA_CPP_MTP:
                return mtp_caps(('none', 'draft-mtp'))
            return plain_caps()

        with patch.object(app, 'engine_capabilities', side_effect=fake_caps):
            cmd = app.build_command(model, runtime_profile=mtp_profile)
        self.assertEqual(command_spec_type_value(cmd), 'draft-mtp')
        self.assertEqual(cmd[cmd.index('--spec-draft-n-max') + 1], '2')

    def test_fit_q8_nommap_profile_emits_full_flag_set(self):
        app = self._app(ENGINE_LLAMA_CPP_MTP)
        model = make_model(name='Qwen3.6-35B-A3B-MTP-GGUF')
        profile = RuntimeProfile(
            engine_id=ENGINE_LLAMA_CPP_MTP,
            name='mtp_fit_q8_draftq8_nommap_draft1_128k',
            ctx_size=131072,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/q8_0',
            flash_attn='on',
            batch_size=128,
            ubatch_size=64,
            fit=True,
            fit_context=4096,
            fit_target=1024,
            cache_ram=0,
            no_mmap=True,
            no_warmup=True,
            mtp_enabled=True,
            mtp_draft_n_max=1,
            mtp_draft_kv_preset='q8_0/q8_0',
            mtp_spec_type='draft-mtp',
        )
        with patch.object(app, 'engine_capabilities', return_value=mtp_caps(('none', 'draft-mtp'), fit=True)):
            cmd = app.build_command(model, runtime_profile=profile)

        def value_after(flag):
            return cmd[cmd.index(flag) + 1]

        self.assertEqual(value_after('-fit'), 'on')
        self.assertEqual(value_after('-fitc'), '4096')
        self.assertEqual(value_after('-fitt'), '1024')
        self.assertEqual(value_after('-ctk'), 'q8_0')
        self.assertEqual(value_after('-ctv'), 'q8_0')
        self.assertEqual(value_after('--spec-draft-type-k'), 'q8_0')
        self.assertEqual(value_after('--spec-draft-type-v'), 'q8_0')
        self.assertEqual(value_after('--cache-ram'), '0')
        self.assertIn('--no-mmap', cmd)
        self.assertIn('--no-warmup', cmd)
        self.assertEqual(value_after('--spec-type'), 'draft-mtp')
        self.assertEqual(value_after('--spec-draft-n-max'), '1')
        self.assertNotIn('--n-gpu-layers', cmd)


class MtpEngineContextResolverTests(unittest.TestCase):
    def test_resolver_uses_runtime_profile_engine_over_global(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = AppConfig(
            Path(tmp.name) / 'models.json',
            runtime_profile=make_runtime_profile(ENGINE_LLAMA_CPP, 'llama-server'),
        )
        profile = RuntimeProfile(
            engine_id=ENGINE_LLAMA_CPP_MTP,
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            mtp_enabled=True,
        )

        def fake_caps(engine_profile=None, *, server_bin=None, engine_id=None):
            resolved = engine_id or getattr(engine_profile, 'engine_id', None) or app.runtime_profile.engine_id
            if resolved == ENGINE_LLAMA_CPP_MTP:
                return mtp_caps(('none', 'draft-mtp'))
            return plain_caps()

        with patch.object(app, 'engine_capabilities', side_effect=fake_caps):
            context = resolve_runtime_engine_context(app, runtime_profile=profile)
        self.assertEqual(context.engine_id, ENGINE_LLAMA_CPP_MTP)
        self.assertTrue(context.supports_mtp)
        self.assertEqual(context.selected_mtp_spec_type, 'draft-mtp')


class MtpScoringTests(unittest.TestCase):
    def _records(self, draft3_accept=0.53, draft1_accept=0.77):
        baseline = {
            'status': 'ok',
            'mtp_enabled': False,
            'benchmark_phase': 'baseline_no_mtp',
            'tokens_per_sec': 20.0,
            'seconds': 10.0,
            'ctx': 131072,
        }
        draft1 = {
            'status': 'ok',
            'mtp_enabled': True,
            'benchmark_phase': 'fit_q8_draftq8_draft_n1',
            'mtp_draft_n_max': 1,
            'tokens_per_sec': 31.25,
            'seconds': 6.0,
            'ctx': 131072,
            'kv_preset': 'q8_0/q8_0',
            'mtp_draft_kv_preset': 'q8_0/q8_0',
            'no_mmap': True,
            'accept_rate': draft1_accept,
        }
        draft3 = {
            'status': 'ok',
            'mtp_enabled': True,
            'benchmark_phase': 'fit_q8_draftq8_draft_n3',
            'mtp_draft_n_max': 3,
            'tokens_per_sec': 22.0,
            'seconds': 9.0,
            'ctx': 131072,
            'kv_preset': 'q8_0/q8_0',
            'mtp_draft_kv_preset': 'q8_0/q8_0',
            'no_mmap': True,
            'accept_rate': draft3_accept,
        }
        return [baseline, draft1, draft3]

    def test_scoring_rejects_draft_n3_low_acceptance(self):
        annotated = annotate_mtp_optimizer_records(self._records(), spec_type='draft-mtp')
        draft3 = next(item for item in annotated if item.get('mtp_draft_n_max') == 3)
        self.assertEqual(draft3['mtp_risk_level'], 'failed')

    def test_scoring_picks_draft_n1_for_safe_and_long_context(self):
        annotated = annotate_mtp_optimizer_records(self._records(), spec_type='draft-mtp')
        recommendations = mtp_optimizer_profile_recommendations(annotated)
        self.assertIn('mtp_safe', recommendations)
        self.assertIn('mtp_long_context', recommendations)
        self.assertEqual(recommendations['mtp_safe']['mtp_draft_n_max'], 1)
        self.assertEqual(recommendations['mtp_long_context']['mtp_draft_n_max'], 1)


class MtpStrategySelectionTests(unittest.TestCase):
    def test_upstream_llama_cpp_with_draft_mtp_selects_mtp_strategy(self):
        # A plain llama.cpp engine whose binary advertises draft-mtp must route
        # an MTP-native model to the acceptance matrix strategy.
        model = make_model(name='Generic native-mtp')
        strategy = select_benchmark_strategy(
            ENGINE_LLAMA_CPP,
            model,
            capabilities=mtp_caps(('none', 'draft-mtp')),
            objective='quick_sanity',
            depth='fast',
        )
        self.assertEqual(strategy.id, 'mtp_acceptance_matrix')
        self.assertEqual(strategy.engine_id, ENGINE_LLAMA_CPP)
        self.assertEqual(strategy.blocked_reason, '')
        self.assertIn('spec_type=draft-mtp', strategy.reason)

    def test_llama_cpp_without_mtp_binary_does_not_select_mtp_strategy(self):
        model = make_model(name='Generic native-mtp')
        strategy = select_benchmark_strategy(
            ENGINE_LLAMA_CPP,
            model,
            capabilities=plain_caps(),
            objective='quick_sanity',
            depth='fast',
        )
        self.assertNotEqual(strategy.id, 'mtp_acceptance_matrix')


class MtpStartLaunchTests(unittest.TestCase):
    def _setup(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        model_path = root / 'generic-native-mtp.gguf'
        model_path.write_bytes(b'GGUF' + b'\0' * 64)
        global_bin = root / 'llama-server'
        global_bin.write_text('#!/bin/sh\n')
        mtp_bin = root / 'mtp' / 'llama-server'
        mtp_bin.parent.mkdir(parents=True, exist_ok=True)
        mtp_bin.write_text('#!/bin/sh\n')
        app = AppConfig(
            root / 'models.json',
            runtime_profile=make_runtime_profile(ENGINE_LLAMA_CPP, str(global_bin)),
        )
        model = ModelConfig(
            id='mtp',
            name='Generic native-mtp',
            path=str(model_path),
            alias='mtp',
            port=18080,
            supports_mtp='yes',
            mtp_enabled=True,
            mtp_draft_n_max=1,
        )
        mtp_profile = RuntimeProfile(
            engine_id=ENGINE_LLAMA_CPP_MTP,
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            mtp_enabled=True,
            mtp_draft_n_max=1,
        )
        return app, model, mtp_profile, str(global_bin), str(mtp_bin)

    def test_start_preflights_runtime_profile_binary_not_global(self):
        # Global engine binary differs from the runtime profile's engine
        # binary; start() must launch the runtime-profile binary and emit the
        # advertised draft-mtp spec type.
        app, model, mtp_profile, global_bin, mtp_bin = self._setup()
        caps = mtp_caps(('none', 'draft-mtp'), fit=True)
        launched = {}

        class FakeProc:
            pid = 4242

        def fake_popen(cmd, *args, **kwargs):
            launched['cmd'] = list(cmd)
            return FakeProc()

        with patch.dict('os.environ', {'LLAMA_CPP_MTP_PATH': mtp_bin}), \
             patch.object(app, 'engine_capabilities', return_value=caps), \
             patch.object(app, 'runtime_command_ready', return_value=(True, '')), \
             patch.object(app, 'get_pid', return_value=None), \
             patch('llama_tui.app.subprocess.Popen', side_effect=fake_popen):
            ok, msg = app.start(model, runtime_profile=mtp_profile)

        self.assertTrue(ok, msg)
        self.assertEqual(launched.get('cmd', [None])[0], mtp_bin)
        self.assertNotEqual(launched['cmd'][0], global_bin)
        self.assertEqual(command_spec_type_value(launched['cmd']), 'draft-mtp')

    def test_start_blocks_invalid_spec_type_before_subprocess(self):
        app, model, mtp_profile, _global_bin, mtp_bin = self._setup()
        caps = mtp_caps(('none', 'draft-mtp'), fit=True)
        bad_command = [mtp_bin, '-m', model.path, '--spec-type', 'mtp', '--spec-draft-n-max', '1']
        popen_calls = []

        with patch.dict('os.environ', {'LLAMA_CPP_MTP_PATH': mtp_bin}), \
             patch.object(app, 'engine_capabilities', return_value=caps), \
             patch.object(app, 'runtime_command_ready', return_value=(True, '')), \
             patch.object(app, 'get_pid', return_value=None), \
             patch.object(app, 'build_command', return_value=bad_command), \
             patch('llama_tui.app.subprocess.Popen', side_effect=lambda *a, **k: popen_calls.append(a) or MagicMock()):
            ok, msg = app.start(model, runtime_profile=mtp_profile)

        self.assertFalse(ok)
        self.assertIn('INVALID_MTP_SPEC_TYPE', msg)
        self.assertEqual(popen_calls, [])


if __name__ == '__main__':
    unittest.main()
