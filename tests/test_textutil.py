import tempfile
import unittest
from pathlib import Path

from llama_tui.textutil import (
    _collapse_repeated_lines,
    compact_message,
    ellipsize,
    is_error_message,
    tail_text,
    wrap_display_lines,
)


class EllipsizeTests(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(ellipsize('hi', 10), 'hi')

    def test_long_text_gets_ellipsis(self):
        self.assertEqual(ellipsize('hello world', 8), 'hello...')

    def test_tiny_width_truncates_without_ellipsis(self):
        self.assertEqual(ellipsize('hello', 3), 'hel')

    def test_zero_width(self):
        self.assertEqual(ellipsize('hello', 0), '')


class CompactMessageTests(unittest.TestCase):
    def test_joins_nonblank_lines(self):
        self.assertEqual(compact_message('a\n\n  b \nc'), 'a | b | c')

    def test_empty(self):
        self.assertEqual(compact_message(''), '')


class IsErrorMessageTests(unittest.TestCase):
    def test_detects_failure_words(self):
        self.assertTrue(is_error_message('benchmark failed'))
        self.assertTrue(is_error_message('error: boom'))

    def test_negations_are_not_errors(self):
        self.assertFalse(is_error_message('completed without errors'))
        self.assertFalse(is_error_message('0 errors'))
        self.assertFalse(is_error_message('no errors captured'))

    def test_empty_is_not_error(self):
        self.assertFalse(is_error_message(''))


class WrapAndCollapseTests(unittest.TestCase):
    def test_wrap_respects_width(self):
        lines = wrap_display_lines('one two three four', 8)
        self.assertTrue(all(len(line) <= 8 for line in lines))
        self.assertEqual(' '.join(lines).split(), ['one', 'two', 'three', 'four'])

    def test_wrap_zero_width(self):
        self.assertEqual(wrap_display_lines('anything', 0), [])

    def test_collapse_repeated_lines(self):
        self.assertEqual(
            _collapse_repeated_lines(['x', 'x', 'x', 'y']),
            ['x (repeated 3x)', 'y'],
        )


class TailTextTests(unittest.TestCase):
    def test_missing_file(self):
        self.assertEqual(tail_text(Path('/no/such/file.log')), ['<no log file yet>'])

    def test_returns_last_lines(self):
        with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as handle:
            handle.write('\n'.join(f'line{i}' for i in range(50)))
            path = Path(handle.name)
        try:
            self.assertEqual(tail_text(path, max_lines=3), ['line47', 'line48', 'line49'])
        finally:
            path.unlink()


if __name__ == '__main__':
    unittest.main()
