import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.engines import ENGINE_LLAMA_CPP_MTP, EngineInstall
from llama_tui.models import ModelConfig
from llama_tui.mtp_doctor import build_mtp_doctor_report, mtp_status_for_model
from llama_tui.runtime_profiles import default_engine_capabilities, make_runtime_profile
from llama_tui.ui import mtp_doctor_items


def mtp_capabilities(spec_type: str = 'draft-mtp'):
    return replace(
        default_engine_capabilities(ENGINE_LLAMA_CPP_MTP),
        help_text=f'--spec-type none,{spec_type}\n--spec-draft-n-max N\n--no-warmup\n--parallel N\n-ctk TYPE -ctv TYPE --no-mmap',
        supports_spec_type=True,
        supports_mtp=True,
        spec_type_values=('none', spec_type),
        mtp_spec_type=spec_type,
        mtp_spec_type_value=spec_type,
        supports_spec_draft_n_max=True,
        supports_no_warmup=True,
        supports_parallel=True,
        supports_ctk_ctv=True,
        supports_no_mmap=True,
    )


class MtpDoctorTests(unittest.TestCase):
    def make_app(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch.dict('os.environ', {'LLAMA_CPP_MTP_PATH': '/opt/mtp/bin'}):
            app = AppConfig(
                Path(tmp.name) / 'models.json',
                runtime_profile=make_runtime_profile(ENGINE_LLAMA_CPP_MTP, 'llama-server'),
            )
        return app

    def test_doctor_reports_ready_launch_with_draft_mtp_command_preview(self):
        app = self.make_app()
        model = ModelConfig(
            id='mtp',
            name='MTP',
            path='/models/native-mtp.gguf',
            alias='mtp',
            port=18080,
            supports_mtp='yes',
            mtp_enabled=True,
            mtp_draft_n_max=2,
        )
        caps = mtp_capabilities('draft-mtp')
        install = EngineInstall(
            id=ENGINE_LLAMA_CPP_MTP,
            resolved_command='/opt/mtp/bin/llama-server',
            source='env:LLAMA_CPP_MTP_PATH',
            exists=True,
            executable=True,
            resolved_path='/opt/mtp/bin/llama-server',
            checked_paths=['/opt/mtp/bin/llama-server'],
        )

        with patch('llama_tui.mtp_doctor.resolve_engine_install', return_value=install), \
             patch('llama_tui.mtp_doctor.detect_engine_capabilities', return_value=caps), \
             patch.object(app, 'engine_capabilities', return_value=caps):
            report = build_mtp_doctor_report(app, model)
            summary = mtp_status_for_model(app, model)

        self.assertEqual(report.launch_status, 'ready')
        self.assertEqual(summary.status, 'ready')
        self.assertEqual(report.selected_spec_type, 'draft-mtp')
        self.assertTrue(report.supports_mtp)
        self.assertTrue(report.model_allowed)
        self.assertTrue(report.launch.includes_spec_type)
        self.assertTrue(report.launch.includes_spec_draft_n_max)
        self.assertEqual(report.launch.selected_spec_type, 'draft-mtp')
        self.assertEqual(report.launch.draft_n_max, 2)
        self.assertEqual(report.launch.added_flags, ('--spec-type', 'draft-mtp', '--spec-draft-n-max', '2'))
        self.assertIn('--spec-type draft-mtp', report.launch.command_preview)
        self.assertIn('--spec-draft-n-max 2', report.launch.command_preview)

    def test_doctor_blocks_missing_mtp_spec_value_before_launch(self):
        app = self.make_app()
        model = ModelConfig(
            id='mtp',
            name='MTP',
            path='/models/native-mtp.gguf',
            alias='mtp',
            port=18080,
            supports_mtp='yes',
            mtp_enabled=True,
            mtp_draft_n_max=2,
        )
        caps = replace(
            default_engine_capabilities(ENGINE_LLAMA_CPP_MTP),
            help_text='--spec-type none,ngram-simple\n--spec-draft-n-max N',
            supports_spec_type=True,
            supports_mtp=False,
            spec_type_values=('none', 'ngram-simple'),
            supports_spec_draft_n_max=True,
        )
        install = EngineInstall(
            id=ENGINE_LLAMA_CPP_MTP,
            resolved_command='/opt/mtp/bin/llama-server',
            source='env:LLAMA_CPP_MTP_PATH',
            exists=True,
            executable=True,
        )

        with patch('llama_tui.mtp_doctor.resolve_engine_install', return_value=install), \
             patch('llama_tui.mtp_doctor.detect_engine_capabilities', return_value=caps), \
             patch.object(app, 'engine_capabilities', return_value=caps):
            report = build_mtp_doctor_report(app, model)

        self.assertEqual(report.launch_status, 'blocked')
        self.assertEqual(report.reason, 'missing mtp/draft-mtp value')
        self.assertEqual(report.launch.blocked_reason, 'missing mtp/draft-mtp value')
        self.assertIn('--spec-type', report.launch.skipped_flags)
        self.assertFalse(report.launch.includes_spec_type)
        self.assertIn('Build/select a llama.cpp MTP binary', report.next_action)

    def test_doctor_marks_generic_auto_model_as_unknown(self):
        app = self.make_app()
        model = ModelConfig(
            id='generic',
            name='Generic',
            path='/models/generic.gguf',
            alias='generic',
            port=18080,
            supports_mtp='auto',
            mtp_enabled=False,
        )
        caps = mtp_capabilities('mtp')
        install = EngineInstall(
            id=ENGINE_LLAMA_CPP_MTP,
            resolved_command='/opt/mtp/bin/llama-server',
            source='env:LLAMA_CPP_MTP_PATH',
            exists=True,
            executable=True,
        )

        with patch('llama_tui.mtp_doctor.resolve_engine_install', return_value=install), \
             patch('llama_tui.mtp_doctor.detect_engine_capabilities', return_value=caps), \
             patch.object(app, 'engine_capabilities', return_value=caps):
            report = build_mtp_doctor_report(app, model)

        self.assertEqual(report.launch_status, 'unknown')
        self.assertFalse(report.model_allowed)
        self.assertIn('unknown', report.reason)

    def test_auto_without_hints_blocks_when_mtp_enabled_consistently(self):
        app = self.make_app()
        model = ModelConfig(
            id='generic',
            name='Generic',
            path='/models/generic.gguf',
            alias='generic',
            port=18080,
            supports_mtp='auto',
            mtp_enabled=True,
        )
        caps = mtp_capabilities('mtp')
        install = EngineInstall(
            id=ENGINE_LLAMA_CPP_MTP,
            resolved_command='/opt/mtp/bin/llama-server',
            source='env:LLAMA_CPP_MTP_PATH',
            exists=True,
            executable=True,
        )

        with patch('llama_tui.mtp_doctor.resolve_engine_install', return_value=install), \
             patch('llama_tui.mtp_doctor.detect_engine_capabilities', return_value=caps), \
             patch.object(app, 'engine_capabilities', return_value=caps):
            report = build_mtp_doctor_report(app, model)
            launch_ok, launch_msg = app.validate_mtp_launch(model)

        self.assertFalse(launch_ok)
        self.assertEqual(report.launch_status, 'blocked')
        self.assertFalse(report.model_allowed)
        self.assertEqual(report.risk_level, 'block')
        self.assertIn('MTP capability is unknown', launch_msg)
        self.assertIn('MTP capability is unknown', report.reason)

    def test_ui_items_include_status_capabilities_model_and_command(self):
        app = self.make_app()
        model = ModelConfig(
            id='mtp',
            name='MTP',
            path='/models/native-mtp.gguf',
            alias='mtp',
            port=18080,
            supports_mtp='yes',
            mtp_enabled=True,
            mtp_draft_n_max=3,
        )
        caps = mtp_capabilities('draft-mtp')
        install = EngineInstall(
            id=ENGINE_LLAMA_CPP_MTP,
            resolved_command='/opt/mtp/bin/llama-server',
            source='env:LLAMA_CPP_MTP_PATH',
            exists=True,
            executable=True,
        )

        with patch('llama_tui.mtp_doctor.resolve_engine_install', return_value=install), \
             patch('llama_tui.mtp_doctor.detect_engine_capabilities', return_value=caps), \
             patch.object(app, 'engine_capabilities', return_value=caps):
            text = '\n'.join(line for line, _kind in mtp_doctor_items(app, model))

        self.assertIn('MTP Doctor', text)
        self.assertIn('selected spec value: draft-mtp', text)
        self.assertIn('model allowed for MTP: yes', text)
        self.assertIn('--spec-type included: yes', text)
        self.assertIn('--spec-draft-n-max 3', text)

    def test_doctor_warns_about_old_bad_mtp_profile_shapes_and_good_fit_profile(self):
        app = self.make_app()
        model = ModelConfig(
            id='mtp',
            name='MTP',
            path='/models/native-mtp.gguf',
            alias='mtp',
            port=18080,
            supports_mtp='yes',
            mtp_enabled=True,
            mtp_draft_n_max=3,
            last_benchmark_results=[
                {
                    'status': 'start failed',
                    'failure_category': 'FIXED_GPU_LAYERS_BLOCKED_FIT',
                    'benchmark_phase': 'draft_n3',
                    'mtp_enabled': True,
                },
                {
                    'status': 'ok',
                    'runtime_profile': 'mtp_fit_q8_draftq8_nommap_draft1_128k',
                    'ctx': 131072,
                    'kv_preset': 'q8_0/q8_0',
                    'mtp_draft_kv_preset': 'q8_0/q8_0',
                    'no_mmap': True,
                    'mtp_enabled': True,
                    'mtp_draft_n_max': 1,
                    'tokens_per_sec': 31.25,
                    'accept_rate': 0.7671,
                },
                {
                    'status': 'ok',
                    'mtp_enabled': True,
                    'mtp_draft_n_max': 2,
                    'accept_rate': 0.6453,
                    'warning': 'MMAP_CPU_OVERRIDE_SLOWPATH',
                },
            ],
        )
        caps = mtp_capabilities('draft-mtp')
        install = EngineInstall(
            id=ENGINE_LLAMA_CPP_MTP,
            resolved_command='/opt/mtp/bin/llama-server',
            source='env:LLAMA_CPP_MTP_PATH',
            exists=True,
            executable=True,
        )

        with patch('llama_tui.mtp_doctor.resolve_engine_install', return_value=install), \
             patch('llama_tui.mtp_doctor.detect_engine_capabilities', return_value=caps), \
             patch.object(app, 'engine_capabilities', return_value=caps):
            text = '\n'.join(line for line, _kind in mtp_doctor_items(app, model))

        self.assertIn('MTP fit was blocked because --n-gpu-layers was fixed', text)
        self.assertIn('--no-mmap may improve performance', text)
        self.assertIn('Draft acceptance below 70%', text)
        self.assertIn('MTP fit-assisted q8/no-mmap profile is usable: 31.25 tok/s, 77% acceptance.', text)


if __name__ == '__main__':
    unittest.main()
