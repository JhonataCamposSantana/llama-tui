import json
import re
import unittest
from pathlib import Path

from llama_tui.main import build_cli_parser


class DocumentationConsistencyTests(unittest.TestCase):
    def test_readme_engine_examples_match_cli_choices(self):
        text = Path('README.md').read_text(encoding='utf-8')
        parser = build_cli_parser()
        engine_action = next(action for action in parser._actions if '--engine' in action.option_strings)
        supported = set(engine_action.choices)

        documented = set(re.findall(r'--engine\s+([A-Za-z0-9._+-]+)', text))

        self.assertTrue(documented)
        self.assertLessEqual(documented, supported)

    def test_removed_engine_commands_are_not_documented(self):
        text = Path('README.md').read_text(encoding='utf-8')

        self.assertNotRegex(text, r'--engine\s+(?:tq3|buun|vllm)\b')
        self.assertNotIn('vllm_command', text)
        self.assertNotIn('vllm serve', text)

    def test_sample_config_uses_current_schema_fields(self):
        sample = json.loads(Path('examples/models.sample.json').read_text(encoding='utf-8'))

        self.assertNotIn('vllm_command', sample)
        self.assertEqual(sample['ui']['browser_view'], 'compact')
        self.assertEqual(sample['models'][0]['runtime'], 'llama.cpp')


if __name__ == '__main__':
    unittest.main()
