#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
import ffmpeg
import logging
import tempfile
import shutil
import unittest
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from click.testing import CliRunner
from pydub import AudioSegment
from pydub.generators import WhiteNoise

from subtitle_tool.ai import AISubtitler
from subtitle_tool.audio import AudioSplitter
from subtitle_tool.cli import main, setup_logging, API_KEY_NAME, AI_DEFAULT_MODEL
from subtitle_tool.subtitles import SubtitleEvent
from subtitle_tool.video import VideoProcessingError


class TestSetupLogging(unittest.TestCase):
    """Test the logging setup functionality"""

    def test_setup_logging_normal(self):
        """Test normal logging setup (ERROR level)"""
        setup_logging(verbose=False, debug=False)
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.ERROR)

    def test_setup_logging_verbose(self):
        """Test verbose logging setup (DEBUG for subtitle_tool only)"""
        setup_logging(verbose=True, debug=False)
        root_logger = logging.getLogger()
        subtitle_logger = logging.getLogger("subtitle_tool")

        self.assertEqual(root_logger.level, logging.ERROR)
        self.assertEqual(subtitle_logger.level, logging.DEBUG)

    def test_setup_logging_debug(self):
        """Test debug logging setup (DEBUG for everything)"""
        setup_logging(verbose=False, debug=True)
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)


class TestMainCommand(unittest.TestCase):
    """Test the main CLI command functionality"""

    def setUp(self):
        """Setup for each test method"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = f"{self.temp_dir}/test_video.mp4"
        self.test_audio_path = f"{self.temp_dir}/test_audio.wav"

        # Create audio track
        noise1 = WhiteNoise().to_audio_segment(duration=3_000)  # 3 s of noise
        silence = AudioSegment.silent(duration=2_000)  # 2 s of silence
        noise2 = WhiteNoise().to_audio_segment(duration=3_000)  # 3 s of noise
        audio_segment = noise1 + silence + noise2
        audio_segment.export(self.test_audio_path, format="wav")

        # Create video track
        video_input = ffmpeg.input(
            "color=black:s=1280x720:d=8:rate=30",  # 8 seconds black
            f="lavfi",
        )
        audio_input = ffmpeg.input(self.test_audio_path)

        # Create dummy video with audio track
        out = ffmpeg.output(
            video_input,
            audio_input,
            self.test_video_path,
            vcodec="libx264",
            acodec="pcm_s16le",
            pix_fmt="yuv420p",
            shortest=None,
            movflags="+faststart",
        )

        # Run ffmpeg command
        ffmpeg.run(out, overwrite_output=True, quiet=True)

    def tearDown(self):
        """Cleanup after each test method"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_api_key(self):
        """Test that missing API key raises proper error"""
        os.environ.pop(API_KEY_NAME, None)
        result = self.runner.invoke(main, ["--video", str(self.test_video_path)])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("API key not informed", result.output)

    def test_missing_media_file_arguments(self):
        """Test that missing both video and audio arguments raises error"""
        result = self.runner.invoke(main, ["--api-key", "test_key"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Either --video or --audio need to be specified", result.output)

    def test_both_video_and_audio_specified(self):
        """Test that specifying both video and audio raises error"""
        result = self.runner.invoke(
            main,
            [
                "--api-key",
                "test_key",
                "--video",
                str(self.test_video_path),
                "--audio",
                str(self.test_audio_path),
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Either --video or --audio need to be specified", result.output)

    def test_nonexistent_file(self):
        """Test that nonexistent file raises proper error"""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.mp4"
        result = self.runner.invoke(
            main, ["--api-key", "test_key", "--video", str(nonexistent_path)]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(f"File '{nonexistent_path}' does not exist", result.output)

    def test_directory_instead_of_file(self):
        """Test that directory path raises proper error"""
        result = self.runner.invoke(
            main, ["--api-key", "test_key", "--video", str(self.temp_dir)]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(f"File '{self.temp_dir}' is a directory", result.output)

    @patch("subtitle_tool.video.extract_audio")
    @patch.object(AudioSplitter, "split_audio")
    @patch.object(AISubtitler, "transcribe_audio")
    @patch.object(ThreadPoolExecutor, "map")
    def test_successful_video_processing(
        self,
        mock_map,
        mock_transcribe_audio,
        mock_split_audio,
        mock_extract_audio,
    ):
        """Test successful video processing flow"""
        # Setup mocks
        mock_audio_segment = Mock()
        mock_audio_segment.duration_seconds = 10.0
        mock_extract_audio.return_value = mock_audio_segment

        mock_segments = [Mock(duration_seconds=5.0), Mock(duration_seconds=5.0)]
        mock_split_audio.return_value = mock_segments

        mock_map.return_value = [
            [
                SubtitleEvent(start=1000, end=2000, text="First"),
                SubtitleEvent(start=3000, end=4000, text="Second"),
            ],
            [
                SubtitleEvent(start=1000, end=2000, text="Third"),
                SubtitleEvent(start=3000, end=4000, text="Fourth"),
            ],
        ]

        # Run command
        # Patching time.sleep to speed up the retry mechanism
        with patch("time.sleep", lambda _: None):
            with patch("builtins.open", mock_open()) as mock_file:
                result = self.runner.invoke(
                    main,
                    [
                        "--api-key",
                        "test_key",
                        "--video",
                        str(self.test_video_path),
                    ],
                )

        # Assertions
        self.assertEqual(result.exit_code, 0)
        mock_split_audio.assert_called_once()
        self.assertIn("Subtitle saved at", result.output)

    @patch("subtitle_tool.video.extract_audio")
    @patch.object(AudioSplitter, "split_audio")
    @patch.object(AISubtitler, "transcribe_audio")
    @patch.object(ThreadPoolExecutor, "map")
    def test_successful_audio_processing(
        self,
        mock_map,
        mock_transcribe_audio,
        mock_split_audio,
        mock_extract_audio,
    ):
        """Test successful video processing flow"""
        # Setup mocks
        mock_audio_segment = Mock()
        mock_audio_segment.duration_seconds = 10.0
        mock_extract_audio.return_value = mock_audio_segment

        mock_segments = [Mock(duration_seconds=5.0), Mock(duration_seconds=5.0)]
        mock_split_audio.return_value = mock_segments

        mock_map.return_value = [
            [
                SubtitleEvent(start=1000, end=2000, text="First"),
                SubtitleEvent(start=3000, end=4000, text="Second"),
            ],
            [
                SubtitleEvent(start=1000, end=2000, text="Third"),
                SubtitleEvent(start=3000, end=4000, text="Fourth"),
            ],
        ]

        # Run command
        # Patching time.sleep to speed up the retry mechanism
        with patch("time.sleep", lambda _: None):
            with patch("builtins.open", mock_open()) as mock_file:
                result = self.runner.invoke(
                    main,
                    [
                        "--api-key",
                        "test_key",
                        "--audio",
                        str(self.test_audio_path),
                    ],
                )

        # Assertions
        self.assertEqual(result.exit_code, 0)
        mock_split_audio.assert_called_once()
        self.assertIn("Subtitle saved at", result.output)

    @patch("subtitle_tool.cli.extract_audio")
    def test_video_audio_extraction_error(self, mock_extract_audio):
        """Test error handling when audio extraction fails"""
        mock_extract_audio.side_effect = VideoProcessingError("Audio extraction failed")

        result = self.runner.invoke(
            main, ["--api-key", "test_key", "--video", str(self.test_video_path)]
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error loading audio stream", result.output)

    @patch("subtitle_tool.cli.extract_audio")
    @patch.object(AudioSplitter, "split_audio")
    @patch("subtitle_tool.cli.AISubtitler")
    @patch.object(ThreadPoolExecutor, "map")
    def test_keyboard_interrupt_handling(
        self, mock_map, mock_ai_subtitler, mock_split_audio, mock_extract_audio
    ):
        """Test graceful handling of KeyboardInterrupt"""
        # Setup mocks
        mock_audio_segment = Mock()
        mock_audio_segment.duration_seconds = 60.0
        mock_extract_audio.return_value = mock_audio_segment

        mock_segments = [Mock(duration_seconds=30.0)]
        mock_split_audio.return_value = mock_segments

        mock_subtitler_instance = Mock()
        mock_ai_subtitler.return_value = mock_subtitler_instance

        mock_map.side_effect = KeyboardInterrupt()

        result = self.runner.invoke(
            main, ["--api-key", "test_key", "--video", str(self.test_video_path)]
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Control-C pressed", result.output)

    @unittest.skip("work in progress")
    @patch("subtitle_tool.video.extract_audio")
    @patch.object(AudioSplitter, "split_audio")
    @patch.object(AISubtitler, "transcribe_audio")
    @patch.object(ThreadPoolExecutor, "map")
    @patch("subtitle_tool.subtitles.merge_subtitle_events")
    @patch("subtitle_tool.subtitles.events_to_subtitles")
    @patch("shutil.move")
    def test_existing_subtitle_backup(
        self,
        mock_move,
        mock_events_to_subtitles,
        mock_merge_events,
        mock_executor,
        mock_ai_subtitler,
        mock_split_audio,
        mock_extract_audio,
    ):
        """Test that existing subtitle files are backed up"""
        # Setup mocks
        mock_audio_segment = Mock()
        mock_audio_segment.duration_seconds = 60.0
        mock_extract_audio.return_value = mock_audio_segment

        mock_segments = [Mock(duration_seconds=30.0)]
        mock_split_audio.return_value = mock_segments

        mock_subtitler_instance = Mock()
        mock_ai_subtitler.return_value = mock_subtitler_instance

        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.map.return_value = [["subtitle1"]]

        mock_subtitle_events = [Mock()]
        mock_merge_events.return_value = mock_subtitle_events

        mock_subtitles = Mock()
        mock_events_to_subtitles.return_value = mock_subtitles

        # Create existing subtitle file
        subtitle_path = Path(self.temp_dir) / "test_video.srt"
        subtitle_path.touch()

        # Run command
        with patch("builtins.open", mock_open()) as mock_file:
            result = self.runner.invoke(
                main, ["--api-key", "test_key", "--video", str(self.test_video_path)]
            )

        # Assertions
        self.assertEqual(result.exit_code, 0)
        mock_move.assert_called_once()
        self.assertIn("backed up to", result.output)

        self.assertNotIn("API key not informed", result.output)

    @unittest.skip("work in progress")
    def test_custom_ai_model(self):
        """Test that custom AI model parameter is accepted"""
        custom_model = "custom-model-name"

        with patch("subtitle_tool.video.extract_audio") as mock_extract:
            mock_extract.side_effect = Exception("Stopping early for test")

            result = self.runner.invoke(
                main,
                [
                    "--api-key",
                    "test_key",
                    "--ai-model",
                    custom_model,
                    "--video",
                    str(self.test_video_path),
                ],
            )

            # The test should fail at audio extraction, not at model validation
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Error loading audio stream", result.output)

    @unittest.skip("work in progress")
    def test_keep_temp_files_flag(self):
        """Test that keep-temp-files flag is properly passed"""
        with (
            patch("subtitle_tool.video.extract_audio") as mock_extract,
            patch("subtitle_tool.ai.AISubtitler") as mock_ai_subtitler,
        ):

            mock_extract.side_effect = Exception("Stopping early for test")

            result = self.runner.invoke(
                main,
                [
                    "--api-key",
                    "test_key",
                    "--keep-temp-files",
                    "--video",
                    str(self.test_video_path),
                ],
            )

            # Check that the flag doesn't cause parsing errors
            self.assertNotEqual(result.exit_code, 0)  # Should fail on audio extraction
            self.assertIn("Error loading audio stream", result.output)

    def test_verbose_and_debug_flags(self):
        """Test that verbose and debug flags work"""
        # Test verbose flag
        result = self.runner.invoke(
            main, ["--api-key", "test_key", "--verbose", "--video", "/nonexistent/path"]
        )
        # Should fail on file not existing
        self.assertIn("does not exist", result.output)

        # Test debug flag
        result = self.runner.invoke(
            main, ["--api-key", "test_key", "--debug", "--video", "/nonexistent/path"]
        )
        # Should fail on file not existing
        self.assertIn("does not exist", result.output)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""

    def setUp(self):
        """Setup for each test method"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = Path(self.temp_dir) / "test_video.mp4"
        self.test_video_path.touch()

    def tearDown(self):
        """Cleanup after each test method"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("subtitle_tool.video.extract_audio")
    @patch.object(AudioSplitter, "split_audio")
    @patch("subtitle_tool.ai.AISubtitler")
    def test_internal_error_handling(
        self, mock_ai_subtitler, mock_split_audio, mock_extract_audio
    ):
        """Test that internal errors are properly caught and reported"""
        # Setup mocks to raise an unexpected exception
        mock_extract_audio.side_effect = RuntimeError("Unexpected internal error")

        result = self.runner.invoke(
            main, ["--api-key", "test_key", "--video", str(self.test_video_path)]
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error: ", result.output)


if __name__ == "__main__":
    unittest.main()
