import unittest

from subtitle_tool.utils import sanitize_int


class TestSanitizeInt(unittest.TestCase):
    def test_with_none(self):
        result = sanitize_int(None)
        self.assertEqual(result, 0)

    def test_with_number(self):
        result = sanitize_int(10)
        self.assertEqual(result, 10)

    def test_with_string(self):
        with self.assertRaises(ValueError):
            sanitize_int("blah")  # type: ignore
            self.fail("Should never get here")
