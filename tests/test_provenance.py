import unittest
from pathlib import Path

from llama_tui.provenance import (
    model_source_provenance,
    normalize_source_labels,
    parse_hf_cache_provenance,
    source_labels_text,
)


class ParseHfCacheProvenanceTests(unittest.TestCase):
    def test_parses_repo_and_snapshot(self):
        path = Path('/cache/hub/models--unsloth--Qwen3-GGUF/snapshots/abc123/model.gguf')
        self.assertEqual(
            parse_hf_cache_provenance(path),
            {
                'repo_folder': 'models--unsloth--Qwen3-GGUF',
                'repo_id': 'unsloth/Qwen3-GGUF',
                'snapshot': 'abc123',
            },
        )

    def test_non_hf_path_returns_empty(self):
        self.assertEqual(parse_hf_cache_provenance(Path('/models/local/model.gguf')), {})

    def test_malformed_repo_folder_keeps_raw(self):
        result = parse_hf_cache_provenance(Path('/hub/models--singletoken/x.gguf'))
        self.assertEqual(result['repo_id'], 'models--singletoken')


class SourceLabelTests(unittest.TestCase):
    def test_normalize_dedupes_and_splits(self):
        self.assertEqual(
            normalize_source_labels('a,b', ['c', 'a'], 'd'),
            ['a', 'b', 'c', 'd'],
        )

    def test_source_labels_text_defaults_to_manual(self):
        self.assertEqual(source_labels_text(), 'manual')
        self.assertEqual(source_labels_text('lmstudio', 'manual'), 'lmstudio,manual')


class ModelSourceProvenanceTests(unittest.TestCase):
    def test_includes_repo_and_labels(self):
        path = Path('/hub/models--owner--repo/snapshots/deadbeef/m.gguf')
        result = model_source_provenance(path, source='lmstudio')
        self.assertEqual(result['source_repo_id'], 'owner/repo')
        self.assertEqual(result['source_snapshot'], 'deadbeef')
        self.assertEqual(result['source_labels'], ['lmstudio'])


if __name__ == '__main__':
    unittest.main()
