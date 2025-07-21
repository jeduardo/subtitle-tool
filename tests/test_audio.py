import unittest

from pydub import AudioSegment
from pydub.generators import WhiteNoise

from subtitle_tool.audio import AudioSplitter


class TestAudioSplitter(unittest.TestCase):
    def test_split_audio_exact_segment(self):
        # durations in milliseconds
        noise_duration_ms = 3 * 1000  # 3 seconds of noise
        silence_duration_ms = 2 * 1000  # 2 seconds of silence

        # generate the first noise segment
        noise1 = WhiteNoise().to_audio_segment(duration=noise_duration_ms)
        noise1.apply_gain(120)  # make it ultra loud
        silence = AudioSegment.silent(duration=silence_duration_ms)
        noise2 = WhiteNoise().to_audio_segment(duration=noise_duration_ms)
        noise2.apply_gain(120)
        result = noise1 + silence + noise2

        splitter = AudioSplitter(
            silence_threshold=-16,
            min_silence_length=silence_duration_ms,
        )
        segments = splitter.split_audio(result, segment_length=3, keep_silence=False)

        self.assertIsInstance(segments, list)
        self.assertEqual(len(result), 8000)
        self.assertIsInstance(result[0], AudioSegment)
        self.assertEqual(round(segments[0].duration_seconds), 3)

    def test_split_audio_default_options(self):
        # durations in milliseconds
        noise_duration_ms = 3 * 1000  # 3 seconds of noise
        silence_duration_ms = 2 * 1000  # 2 seconds of silence

        # generate the first noise segment
        noise1 = WhiteNoise().to_audio_segment(duration=noise_duration_ms)
        silence = AudioSegment.silent(duration=silence_duration_ms)
        noise2 = WhiteNoise().to_audio_segment(duration=noise_duration_ms)
        result = noise1 + silence + noise2

        splitter = AudioSplitter()
        segments = splitter.split_audio(result)
        total_time = sum(segment.duration_seconds for segment in segments)

        self.assertIsInstance(segments, list)
        self.assertEqual(len(result), 8000)
        self.assertAlmostEqual(total_time, result.duration_seconds, places=1)
        self.assertIsInstance(result[0], AudioSegment)


if __name__ == "__main__":
    unittest.main()
