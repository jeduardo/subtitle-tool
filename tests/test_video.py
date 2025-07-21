import tempfile
import unittest
from unittest.mock import MagicMock, patch

import ffmpeg
import pytest
from pydub import AudioSegment
from pydub.generators import WhiteNoise

from subtitle_tool.video import VideoProcessingError, extract_audio


class TestVideo(unittest.TestCase):
    def _create_test_audio(self):
        """Helper method to create test audio segment"""
        noise1 = WhiteNoise().to_audio_segment(duration=3_000)  # 3 s of noise
        silence = AudioSegment.silent(duration=2_000)  # 2 s of silence
        noise2 = WhiteNoise().to_audio_segment(duration=3_000)  # 3 s of noise
        return noise1 + silence + noise2

    def _test_extract_audio_with_codec(self, acodec, container_format="mp4"):
        """Helper method to test audio extraction with different codecs"""
        audio_segment = self._create_test_audio()

        # Create temporary audio and video files with auto-delete
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio,
            tempfile.NamedTemporaryFile(suffix=f".{container_format}") as tmp_video,
        ):
            # Export audio to temp file
            audio_segment.export(tmp_audio.name, format="wav")

            # Inputs
            video_input = ffmpeg.input(
                "color=black:s=1280x720:d=8:rate=30",  # 8 seconds black
                f="lavfi",
            )
            audio_input = ffmpeg.input(tmp_audio.name)

            # Merging inputs into video
            out = ffmpeg.output(
                video_input,
                audio_input,
                tmp_video.name,
                vcodec="libx264",
                acodec=acodec,
                pix_fmt="yuv420p",
                shortest=None,
                movflags="+faststart",
            )

            # Run ffmpeg command
            ffmpeg.run(out, overwrite_output=True, quiet=True)

            # Extract the audio
            result = extract_audio(tmp_video.name)

            # Assert the result
            self.assertIsInstance(result, AudioSegment)
            self.assertAlmostEqual(
                result.duration_seconds,
                audio_segment.duration_seconds,
                places=1,  # Allow small differences due to encoding
            )

    def test_extract_audio_wav(self):
        """Test audio extraction from video with WAV audio codec"""
        self._test_extract_audio_with_codec("pcm_s16le", "avi")

    def test_extract_audio_aac(self):
        """Test audio extraction from video with AAC audio codec"""
        self._test_extract_audio_with_codec("aac")

    def test_extract_audio_mp3(self):
        """Test audio extraction from video with MP3 audio codec"""
        self._test_extract_audio_with_codec("libmp3lame")

    def test_extract_audio_ac3(self):
        """Test audio extraction from video with AC3 audio codec"""
        self._test_extract_audio_with_codec("ac3")

    def test_extract_audio_opus(self):
        """Test audio extraction from video with Opus audio codec"""
        audio_segment = self._create_test_audio()

        # Create temporary audio and video files with auto-delete
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio,
            tempfile.NamedTemporaryFile(suffix=".mkv") as tmp_video,
        ):
            # Export audio to temp file
            audio_segment.export(tmp_audio.name, format="wav")

            # Inputs
            video_input = ffmpeg.input(
                "color=black:s=1280x720:d=8:rate=30",  # 8 seconds black
                f="lavfi",
            )
            audio_input = ffmpeg.input(tmp_audio.name)

            # MKV container with H.264 video and Opus audio
            out = ffmpeg.output(
                video_input,
                audio_input,
                tmp_video.name,
                vcodec="libx264",
                acodec="libopus",
                pix_fmt="yuv420p",
                shortest=None,
                # MKV doesn't use movflags
            )

            # Run ffmpeg command
            ffmpeg.run(out, overwrite_output=True, quiet=True)

            # Extract the audio
            result = extract_audio(tmp_video.name)

            # Assert the result
            self.assertIsInstance(result, AudioSegment)
            self.assertAlmostEqual(
                result.duration_seconds,
                audio_segment.duration_seconds,
                places=1,  # Allow small differences due to encoding
            )

    def test_extract_audio_empty_path(self):
        with pytest.raises(VideoProcessingError) as exc_info:
            extract_audio("")
        self.assertEqual(exc_info.value.args[0], "Path to video file is mandatory")

    def test_extract_audio_invalid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mkv") as tmp_video:
            with pytest.raises(VideoProcessingError) as exc_info:
                extract_audio(tmp_video.name)
            self.assertEqual(exc_info.value.args[0], "Error probing for file metadata")

    def test_extract_audio_no_audio_streams(self):
        with tempfile.NamedTemporaryFile(suffix=".mkv") as tmp_video:
            # Inputs
            video_input = ffmpeg.input(
                "color=black:s=1280x720:d=8:rate=30",  # 8 seconds black
                f="lavfi",
            )

            # MKV container with H.264 video and Opus audio
            out = ffmpeg.output(
                video_input,
                tmp_video.name,
                vcodec="libx264",
                acodec="libopus",
                pix_fmt="yuv420p",
                shortest=None,
                # MKV doesn't use movflags
            )

            # Run ffmpeg command
            ffmpeg.run(out, overwrite_output=True, quiet=True)

            with pytest.raises(VideoProcessingError) as exc_info:
                extract_audio(tmp_video.name)
            self.assertEqual(
                exc_info.value.args[0],
                f"No audio streams found in file {tmp_video.name}",
            )

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run_async")
    def test_extract_audio_ffmpeg_failure(self, mock_run_async, mock_probe):
        # Mock ffmpeg.probe to return a valid stream
        mock_probe.return_value = {
            "streams": [{"codec_type": "audio", "codec_name": "aac"}]
        }

        # Mock the process returned by ffmpeg.run_async
        mock_process = MagicMock()
        mock_process.returncode = 1  # Simulate a non-zero return code
        mock_error_output = b"ffmpeg error"  # Simplified error message
        mock_process.communicate.return_value = (b"", mock_error_output)
        mock_run_async.return_value = mock_process

        with pytest.raises(VideoProcessingError) as exc_info:
            extract_audio("dummy_video.mp4")

        self.assertTrue(exc_info.value.args[0].startswith("Extraction error:"))


if __name__ == "__main__":
    unittest.main()
