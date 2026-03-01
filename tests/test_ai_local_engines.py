import unittest
from unittest.mock import Mock

from subtitle_tool.ai.engine import voxtral, whisper_mlx
from subtitle_tool.subtitles import SubtitleEvent


class TestVoxtralJsonPayload(unittest.TestCase):
    def test_parse_json_array_with_millisecond_timestamps(self):
        payload = """
        [
            {"start": 0, "end": 1200, "text": "Hello"},
            {"start": 1200, "end": 2600, "text": "world"}
        ]
        """

        events = voxtral._json_to_subtitles(payload, 5000)
        self.assertEqual(
            events,
            [
                SubtitleEvent(start=0, end=1200, text="Hello"),
                SubtitleEvent(start=1200, end=2600, text="world"),
            ],
        )

    def test_parse_wrapped_payload_with_timecode_strings(self):
        payload = """
        {
            "segments": [
                {"start": "00:00:00.500", "end": "00:00:01.900", "text": "Hi there"}
            ]
        }
        """

        events = voxtral._json_to_subtitles(payload, 5000)
        self.assertEqual(
            events,
            [SubtitleEvent(start=500, end=1900, text="Hi there")],
        )

    def test_overlapping_events_are_normalized(self):
        payload = """
        [
            {"start": 0, "end": 1200, "text": "First"},
            {"start": 1000, "end": 2200, "text": "Second"}
        ]
        """

        events = voxtral._json_to_subtitles(payload, 5000)
        self.assertEqual(
            events,
            [
                SubtitleEvent(start=0, end=1200, text="First"),
                SubtitleEvent(start=1200, end=2200, text="Second"),
            ],
        )

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            voxtral._json_to_subtitles("not-json", 5000)

    def test_empty_json_array_is_valid(self):
        events = voxtral._json_to_subtitles("[]", 5000)
        self.assertEqual(events, [])

    def test_build_prompt_stays_transcript_shaped(self):
        subtitler = Mock()
        subtitler.media_lang = "English"
        subtitler.subtitle_lang = ""

        prompt = voxtral._build_prompt(subtitler, 5000)

        self.assertEqual(
            prompt,
            "English transcript. Clean punctuation and capitalization. Faithful subtitle wording. Keep the spoken language.",
        )
        self.assertNotIn("[INST]", prompt)
        self.assertNotIn("[TRANSCRIBE]", prompt)

    def test_build_prompt_can_request_translation_when_target_language_is_set(self):
        subtitler = Mock()
        subtitler.media_lang = "Spanish"
        subtitler.subtitle_lang = "English"

        prompt = voxtral._build_prompt(subtitler, 5000)

        self.assertIn(
            "Translate to English when needed.",
            prompt,
        )
        self.assertIn("Write subtitle text only in English.", prompt)

    def test_plain_text_output_is_split_into_subtitles(self):
        raw = (
            "Up next, he promises to avenge his sister's murder. "
            "For years, he tracks her killer without success. "
            "Every day was another blow to the stomach."
        )

        events = voxtral._plain_text_to_subtitles(raw, 12000)

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0].start, 0)
        self.assertEqual(events[-1].end, 12000)
        self.assertTrue(all(event.text for event in events))
        self.assertTrue(all(event.start < event.end for event in events))

    def test_plain_text_cleanup_removes_wrapper_prefixes(self):
        raw = 'Transcript: "Hello there." "General Kenobi."'

        events = voxtral._plain_text_to_subtitles(raw, 4000)

        self.assertEqual(
            events,
            [
                SubtitleEvent(start=0, end=2000, text="Hello there."),
                SubtitleEvent(start=2000, end=4000, text="General Kenobi."),
            ],
        )


class TestWhisperMlxEngine(unittest.TestCase):
    def _subtitler(self) -> Mock:
        subtitler = Mock()
        subtitler.media_lang = "English"
        subtitler.subtitle_lang = ""
        subtitler.model_name = "mlx-community/whisper-large-v3-turbo"
        subtitler.whisper = Mock()
        subtitler._normalize_language.return_value = "en"
        subtitler._segment_value.side_effect = (
            lambda payload, field, default=None: payload.get(field, default)
            if isinstance(payload, dict)
            else default
        )
        subtitler._generation_error_class.return_value = RuntimeError
        return subtitler

    def test_requires_timestamped_segments(self):
        subtitler = self._subtitler()
        subtitler.whisper.transcribe.return_value = {"text": "hello"}

        with self.assertRaises(RuntimeError):
            whisper_mlx.generate_subtitles(
                subtitler,
                duration_ms=3000,
                audio_path="/tmp/chunk.wav",
            )

    def test_uses_segments_payload_for_normalization(self):
        subtitler = self._subtitler()
        segments = [{"start": 0.0, "end": 1.5, "text": "hello"}]
        subtitler.whisper.transcribe.return_value = {"segments": segments}
        subtitler._normalize_segments.return_value = [
            SubtitleEvent(start=0, end=1500, text="hello")
        ]

        events = whisper_mlx.generate_subtitles(
            subtitler,
            duration_ms=3000,
            audio_path="/tmp/chunk.wav",
        )

        self.assertEqual(events, [SubtitleEvent(start=0, end=1500, text="hello")])
        subtitler._normalize_segments.assert_called_once_with(
            {"segments": segments}, 3000
        )
        _, kwargs = subtitler.whisper.transcribe.call_args
        self.assertEqual(kwargs["task"], "transcribe")
        self.assertEqual(
            kwargs["initial_prompt"],
            "English transcript. Clean punctuation and capitalization. Accurate subtitle wording.",
        )

    def test_prompt_stays_short_and_does_not_try_to_translate(self):
        subtitler = self._subtitler()
        subtitler.media_lang = "Spanish"
        subtitler.subtitle_lang = "English"

        prompt = whisper_mlx._build_prompt(subtitler)

        self.assertEqual(
            prompt,
            "Spanish transcript. Clean punctuation and capitalization. Accurate subtitle wording.",
        )

    def test_prompt_can_bias_spelling_when_languages_match(self):
        subtitler = self._subtitler()
        subtitler.media_lang = "Portuguese"
        subtitler.subtitle_lang = "Portuguese"

        prompt = whisper_mlx._build_prompt(subtitler)

        self.assertEqual(
            prompt,
            "Portuguese transcript. Clean punctuation and capitalization. Accurate subtitle wording. Portuguese spelling.",
        )
