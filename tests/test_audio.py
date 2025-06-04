import unittest
from pydub import AudioSegment
from pydub.generators import WhiteNoise

from subtitle_tool.audio import split_audio


class TestAudio(unittest.TestCase):

    def test_split_audio(self):
        # durations in milliseconds
        noise_duration_ms = 3 * 1000  # 3 seconds of noise
        silence_duration_ms = 2 * 1000  # 2 seconds of silence

        # generate the first noise segment
        noise1 = WhiteNoise().to_audio_segment(duration=noise_duration_ms)
        silence = AudioSegment.silent(duration=silence_duration_ms)
        noise2 = WhiteNoise().to_audio_segment(duration=noise_duration_ms)
        result = noise1 + silence + noise2

        segments = split_audio(result)
        total_time = sum(segment.duration_seconds for segment in segments)

        self.assertIsInstance(segments, list)
        self.assertEqual(len(result), 8000)
        self.assertEqual(round(total_time), round(result.duration_seconds))
        self.assertIsInstance(result[0], AudioSegment)


if __name__ == "__main__":
    unittest.main()
