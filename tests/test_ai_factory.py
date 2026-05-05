import unittest

from subtitle_tool.ai import AISubtitler
from subtitle_tool.ai.gemini import GeminiSubtitler


class TestAISubtitlerFactory(unittest.TestCase):
    def test_defaults_to_gemini_subtitler(self):
        subtitler = AISubtitler(api_key="test-api-key", model_name="test-model")

        self.assertIsInstance(subtitler, GeminiSubtitler)

    def test_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "Unsupported AI subtitle provider"):
            AISubtitler(
                provider="unknown", api_key="test-api-key", model_name="test-model"
            )


if __name__ == "__main__":
    unittest.main()
