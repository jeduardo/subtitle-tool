import json
import logging
import unittest
from unittest.mock import MagicMock, Mock, patch

from google.genai.errors import ClientError, ServerError
from pydub import AudioSegment
from tenacity import RetryCallState

from subtitle_tool.ai import (
    DEFAULT_WAIT_TIME,
    AIGenerationError,
    AISubtitler,
)
from subtitle_tool.subtitles import SubtitleEvent

# Running tests in DEBUG will help to troubleshoot errors on changes
logging.getLogger("subtitle_tool").setLevel(logging.DEBUG)


CLIENT_ERROR_429_RATE_LIMIT_MINUTE = """
{
    "error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
            {
                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                "quotaDimensions": {
                "location": "global",
                "model": "gemini-2.5-flash"
                },
                "quotaValue": "10"
            }
            ]
        },
        {
            "@type": "type.googleapis.com/google.rpc.Help",
            "links": [
            {
                "description": "Learn more about Gemini API quotas",
                "url": "https://ai.google.dev/gemini-api/docs/rate-limits"
            }
            ]
        },
        {
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "33s"
        }
        ]
    }
}
"""  # noqa: E501

CLIENT_ERROR_429_RATE_LIMIT_DAY = """
{
    "error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
            {
                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                "quotaDimensions": {
                "location": "global",
                "model": "gemini-2.5-flash"
                },
                "quotaValue": "10"
            }
            ]
        },
        {
            "@type": "type.googleapis.com/google.rpc.Help",
            "links": [
            {
                "description": "Learn more about Gemini API quotas",
                "url": "https://ai.google.dev/gemini-api/docs/rate-limits"
            }
            ]
        },
        {
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "33s"
        }
        ]
    }
}
"""  # noqa: E501

CLIENT_ERROR_403_AUTH = """
{
    "error": {
        "code": 403,
        "message": "Auth exceptiom",
        "status": "AUTH ERROR",
        "details": [
        ]
    }
}
"""

SERVER_ERROR_500_INTERNAL = """
{
    "error": {
        "code": 500,
        "message": "An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting",
        "status": "INTERNAL"
    }
}
"""

SERVER_ERROR_503_UNAVAILABLE = """
{
    "message": "",
    "status": "Service Unavailable"
}
"""


class TestIsRecoverable(unittest.TestCase):
    def setUp(self) -> None:
        self.subtitler = AISubtitler(api_key="test-api-key", model_name="test-model")

    def test_client_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )
        self.assertTrue(self.subtitler._is_recoverable_exception(error))

    def test_client_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_DAY)
        )
        self.assertFalse(self.subtitler._is_recoverable_exception(error))

    def test_client_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH))
        self.assertTrue(self.subtitler._is_recoverable_exception(error))

    def test_server_internal_error(self):
        error = ServerError(
            code=500, response_json=json.loads(SERVER_ERROR_500_INTERNAL)
        )
        self.assertTrue(self.subtitler._is_recoverable_exception(error))  # type: ignore

    def test_server_unavailable_error(self):
        error = ServerError(
            code=503, response_json=json.loads(SERVER_ERROR_503_UNAVAILABLE)
        )
        self.assertTrue(self.subtitler._is_recoverable_exception(error))  # type: ignore

    def test_generic_exception(self):
        error = Exception("Generic Exception")
        self.assertTrue(self.subtitler._is_recoverable_exception(error))  # type: ignore


class TestExtractRetryDelay(unittest.TestCase):
    def setUp(self) -> None:
        self.subtitler = AISubtitler(api_key="test-api-key", model_name="test-model")

    def test_client_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )
        delay = self.subtitler._extract_retry_delay(error)
        self.assertEqual(delay, 33.0)

    def test_client_rate_limit_per_minute_zero_delay(self):
        error = ClientError(
            code=429,
            response_json=json.loads(
                CLIENT_ERROR_429_RATE_LIMIT_MINUTE.replace("33s", "0s")
            ),
        )
        delay = self.subtitler._extract_retry_delay(error)
        self.assertEqual(delay, DEFAULT_WAIT_TIME)

    def test_client_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_DAY)
        )
        delay = self.subtitler._extract_retry_delay(error)
        self.assertEqual(delay, 33.0)

    def test_client_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH))
        delay = self.subtitler._extract_retry_delay(error)
        self.assertEqual(delay, DEFAULT_WAIT_TIME)

    def test_server_internal_error(self):
        error = ServerError(
            code=500, response_json=json.loads(SERVER_ERROR_500_INTERNAL)
        )
        delay = self.subtitler._extract_retry_delay(error)  # type: ignore
        self.assertEqual(delay, DEFAULT_WAIT_TIME)

    def test_server_unavailable_error(self):
        error = ServerError(
            code=503, response_json=json.loads(SERVER_ERROR_503_UNAVAILABLE)
        )
        delay = self.subtitler._extract_retry_delay(error)  # type: ignore
        self.assertEqual(delay, DEFAULT_WAIT_TIME)

    def test_generic_exception(self):
        error = Exception("Generic exception")
        delay = self.subtitler._extract_retry_delay(error)  # type: ignore
        self.assertEqual(delay, DEFAULT_WAIT_TIME)


class TestWaitApiLimit(unittest.TestCase):
    def setUp(self) -> None:
        self.subtitler = AISubtitler(api_key="test-api-key", model_name="test-model")

    def test_client_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = self.subtitler._wait_api_limit(retry_state)
        self.assertEqual(result, 33.0)

    def test_client_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_DAY)
        )

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = self.subtitler._wait_api_limit(retry_state)
        self.assertEqual(result, 33.0)

    def test_client_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH))

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = self.subtitler._wait_api_limit(retry_state)
        self.assertEqual(result, DEFAULT_WAIT_TIME)

    def test_server_internal_error(self):
        error = ServerError(
            code=500, response_json=json.loads(SERVER_ERROR_500_INTERNAL)
        )

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = self.subtitler._wait_api_limit(retry_state)
        self.assertEqual(result, DEFAULT_WAIT_TIME)

    def test_server_unavailable_error(self):
        error = ServerError(
            code=503, response_json=json.loads(SERVER_ERROR_503_UNAVAILABLE)
        )

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = self.subtitler._wait_api_limit(retry_state)
        self.assertEqual(result, DEFAULT_WAIT_TIME)

    def test_generic_exception(self):
        error = Exception("Generic Exception")

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = self.subtitler._wait_api_limit(retry_state)
        self.assertEqual(result, DEFAULT_WAIT_TIME)


class TestRetryHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.subtitler = AISubtitler(api_key="test-api-key", model_name="test-model")

    def test_client_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )
        result = self.subtitler._ai_retry_handler(error)
        self.assertTrue(result)

    def test_client_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_DAY)
        )
        result = self.subtitler._ai_retry_handler(error)
        self.assertFalse(result)

    def test_client_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH))
        result = self.subtitler._ai_retry_handler(error)
        self.assertTrue(result)

    def test_server_internal_error(self):
        error = ServerError(
            code=500, response_json=json.loads(SERVER_ERROR_500_INTERNAL)
        )
        result = self.subtitler._ai_retry_handler(error)
        self.assertTrue(result)

    def test_server_unavailable_error(self):
        error = ServerError(
            code=503, response_json=json.loads(SERVER_ERROR_503_UNAVAILABLE)
        )
        result = self.subtitler._ai_retry_handler(error)
        self.assertTrue(result)


class TestAISubtitler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.mock_audio_segment = Mock(spec=AudioSegment)
        self.mock_audio_segment.duration_seconds = 10.0
        self.api_key = "test_api_key"

        # Instantiate the actual class
        self.subtitler = AISubtitler(
            api_key=self.api_key, model_name="test_model", delete_temp_files=True
        )

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_success(self, mock_client_class, mock_temp_file):
        """Test successful audio upload and cleanup"""
        # Setup mocks needed for the method to operate
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        # Run the method within the context manager
        with self.subtitler.upload_audio(self.mock_audio_segment) as result:  # type: ignore
            # Verify the result is the upload reference
            self.assertEqual(result, mock_ref)

        # Verify cleanup happened after context manager exit
        mock_client.files.delete.assert_called_once_with(name="files/test_upload_id")

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_with_delete_temp_files_false(
        self, mock_client_class, mock_temp_file
    ):
        """Test that delete_temp_files parameter is passed to NamedTemporaryFile"""
        self.subtitler.delete_temp_files = False

        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        with self.subtitler.upload_audio(self.mock_audio_segment):  # type: ignore
            pass

        # Verify NamedTemporaryFile was called with delete=False
        mock_temp_file.assert_called_once_with(suffix=".wav", delete=False)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_cleanup_on_exception(self, mock_client_class, mock_temp_file):
        """Test that cleanup happens even if an exception occurs in the context"""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        # Test that cleanup happens even when exception is raised in context
        with self.assertRaises(ValueError):
            with self.subtitler.upload_audio(self.mock_audio_segment):
                raise ValueError("Test exception")

        # Verify cleanup still happened
        mock_client.files.delete.assert_called_once_with(name="files/test_upload_id")

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_export_failure(self, mock_client_class, mock_temp_file):
        """Test handling of audio export failure"""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        # Make export raise an exception
        self.mock_audio_segment.export.side_effect = Exception("Export failed")

        with self.assertRaises(Exception) as context:
            with self.subtitler.upload_audio(self.mock_audio_segment):
                pass

        self.assertEqual(str(context.exception), "Export failed")

        # Verify that genai.Client was not called since export failed
        mock_client_class.assert_not_called()

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_upload_failure(self, mock_client_class, mock_temp_file):
        """Test handling of file upload failure"""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Make upload raise an exception
        mock_client.files.upload.side_effect = Exception("Upload failed")

        # Patching time.sleep to speed up the retry mechanism
        with patch("time.sleep", lambda _: None):
            # The exception should be raised when trying to enter the context manager
            with self.assertRaises(Exception) as context:
                with self.subtitler.upload_audio(self.mock_audio_segment):
                    # This block should never be reached
                    self.fail("Context manager should not have been entered")

        self.assertEqual(str(context.exception), "Upload failed")

        # Verify export was called and client was created
        self.mock_audio_segment.export.assert_called_once()
        mock_client.files.upload.assert_called_with(file="/tmp/test_audio.wav")
        self.assertEqual(mock_client.files.upload.call_count, 5)

        # Delete should not be called since upload failed before yield
        mock_client.files.delete.assert_not_called()

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_upload_status_not_finalized_error(
        self, mock_client_class, mock_temp_file
    ):
        """Test handling of specific genai upload status error"""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Make upload raise the specific ValueError you encountered
        mock_client.files.upload.side_effect = ValueError(
            "Failed to upload file: Upload status is not finalized."
        )

        # Patching time.sleep to speed up the retry mechanism
        with patch("time.sleep", lambda _: None):
            # The exception should be raised when trying to enter the context manager
            with self.assertRaises(ValueError) as context:
                with self.subtitler.upload_audio(self.mock_audio_segment):
                    # This block should never be reached
                    self.fail("Context manager should not have been entered")

        self.assertEqual(
            str(context.exception),
            "Failed to upload file: Upload status is not finalized.",
        )

        # Verify export was called and client was created
        self.mock_audio_segment.export.assert_called_once()
        mock_client.files.upload.assert_called_with(file="/tmp/test_audio.wav")
        self.assertEqual(mock_client.files.upload.call_count, 5)

        # Delete should not be called since upload failed before yield
        mock_client.files.delete.assert_not_called()

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_various_upload_errors(
        self, mock_client_class, mock_temp_file
    ):
        """Test handling of various upload-related errors"""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Test different types of upload errors
        upload_errors = [
            ValueError("Failed to upload file: Upload status is not finalized."),
            ConnectionError("Connection failed"),
            TimeoutError("Upload timeout"),
            Exception("Server error 500"),
        ]

        for er in upload_errors:
            with self.subTest(error=er):
                # Reset mocks for each test
                mock_client.reset_mock()
                mock_client_class.reset_mock()
                self.mock_audio_segment.reset_mock()

                # Make upload raise the specific error
                mock_client.files.upload.side_effect = er

                # Patching time.sleep to speed up the retry mechanism
                with patch("time.sleep", lambda _: None):
                    # The exception should be raised when trying
                    # to enter the context manager
                    with self.assertRaises(type(er)) as context:
                        with self.subtitler.upload_audio(self.mock_audio_segment):
                            # This block should never be reached
                            self.fail(
                                f"Context manager should not have been entered for {er}"
                            )

                self.assertEqual(str(context.exception), str(er))

                # Verify the sequence of calls before the failure
                self.mock_audio_segment.export.assert_called_once()
                mock_client.files.upload.assert_called_with(file="/tmp/test_audio.wav")
                self.assertEqual(mock_client.files.upload.call_count, 5)

        # Delete should not be called since upload failed before yield
        mock_client.files.delete.assert_not_called()

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_delete_failure_logged(
        self, mock_client_class, mock_temp_file
    ):
        """Test that an exception during file deletion is logged and not re-raised"""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        # Make client.files.delete raise an exception
        mock_client.files.delete.side_effect = Exception("Deletion failed unexpectedly")

        # Patch the logger to capture warnings
        with patch("subtitle_tool.ai.logger.warning") as mock_logger_warning:
            # The context manager should not raise an exception
            with self.subtitler.upload_audio(self.mock_audio_segment):
                pass  # Simulate normal operation within the context

            # Verify that delete was called and the warning was logged
            mock_client.files.delete.assert_called_once_with(
                name="files/test_upload_id"
            )
            mock_logger_warning.assert_called_once()
            # The warning message is a single f-string, so it's
            # at index 0 of the args tuple
            full_warning_message = mock_logger_warning.call_args[0][0]
            self.assertIn(
                "Error while removing uploaded file files/test_upload_id",
                full_warning_message,
            )
            self.assertIn("Deletion failed unexpectedly", full_warning_message)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_transcribe_audio_success(self, mock_client_class, mock_temp_file):
        """Test successful audio transcription"""
        # Setup mocks needed for the method to operate
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        mock_response_usage_metadata = Mock()
        mock_response_usage_metadata.cache_tokens_details = "Internal object"
        mock_response_usage_metadata.cached_content_token_count = 0
        mock_response_usage_metadata.prompt_token_count = 0
        mock_response_usage_metadata.thoughts_token_count = 0
        mock_response_usage_metadata.candidates_token_count = 0

        mock_response = Mock()
        mock_response.usage_metadata = mock_response_usage_metadata
        mock_response.parsed = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=3000, end=4000, text="Second"),
        ]

        mock_client.models.generate_content.return_value = mock_response

        result = self.subtitler.transcribe_audio(self.mock_audio_segment)
        self.assertEqual(len(result), 2)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_wrong_type(self, mock_client_class, mock_temp_file):
        """Test handling of wrong type for audio segment in upload_audio"""
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Pass a non-AudioSegment object
        wrong_type_audio = "not an audio segment"

        with self.assertRaises(
            AttributeError
        ):  # pydub.AudioSegment.export will raise AttributeError
            with self.subtitler.upload_audio(wrong_type_audio):  # type: ignore
                pass

        mock_temp_file.assert_called_once()
        # Client should not be called if export fails
        mock_client_class.assert_not_called()

    @patch("google.genai.Client")
    def test_generate_subtitles_parsed_not_list(self, mock_client_class):
        """Test _generate_subtitles when response.parsed is not a list"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.thoughts_token_count = 5
        mock_response.parsed = "not a list"  # Simulate wrong type

        mock_client.models.generate_content.return_value = mock_response

        mock_file_ref = Mock()
        mock_file_ref.name = "files/test_upload_id"

        expected_retries = 10
        with patch("time.sleep", lambda _: None):
            with self.assertRaisesRegex(
                AIGenerationError, "Parsed response is not a list"
            ):
                self.subtitler._generate_subtitles(mock_file_ref)

        self.assertEqual(
            self.subtitler.metrics.input_token_count, 10 * expected_retries
        )
        self.assertEqual(
            self.subtitler.metrics.output_token_count, 25 * expected_retries
        )
        self.assertEqual(self.subtitler.metrics.retries, expected_retries)

    @patch("google.genai.Client")
    def test_generate_subtitles_empty_response(self, mock_client_class):
        """Test _generate_subtitles when response is None"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.models.generate_content.return_value = (
            None  # Simulate empty response
        )

        mock_file_ref = Mock()
        mock_file_ref.name = "files/test_upload_id"

        expected_retries = 10
        with patch("time.sleep", lambda _: None):
            with self.assertRaisesRegex(AIGenerationError, "Response is empty"):
                self.subtitler._generate_subtitles(mock_file_ref)

        self.assertEqual(self.subtitler.metrics.input_token_count, 0)
        self.assertEqual(self.subtitler.metrics.output_token_count, 0)
        self.assertEqual(self.subtitler.metrics.retries, expected_retries)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.mock_audio_segment = Mock(spec=AudioSegment)
        self.mock_audio_segment.duration_seconds = 10.0
        self.api_key = "test_api_key"

        self.mock_response_usage_metadata = Mock()
        self.mock_response_usage_metadata.cache_tokens_details = "Internal object"
        self.mock_response_usage_metadata.cached_content_token_count = 0
        self.mock_response_usage_metadata.prompt_token_count = 1000
        self.mock_response_usage_metadata.thoughts_token_count = 2000
        self.mock_response_usage_metadata.candidates_token_count = 2000

        # Instantiate the actual class
        self.subtitler = AISubtitler(
            api_key=self.api_key, model_name="test_model", delete_temp_files=True
        )

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_input_output(self, mock_client_class, mock_temp_file):
        # Setup mocks needed for the method to operate
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        mock_response = Mock()
        mock_response.usage_metadata = self.mock_response_usage_metadata
        mock_response.parsed = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=3000, end=4000, text="Second"),
        ]

        mock_client.models.generate_content.return_value = mock_response

        self.subtitler.transcribe_audio(self.mock_audio_segment)
        metrics = self.subtitler.metrics
        self.assertEqual(metrics.input_token_count, 1000)
        self.assertEqual(metrics.output_token_count, 4000)
        self.assertEqual(metrics.client_errors, 0)
        self.assertEqual(metrics.server_errors, 0)
        self.assertEqual(metrics.invalid_subtitles, 0)
        self.assertEqual(metrics.throttles, 0)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_client_errors(self, mock_client_class, mock_temp_file):
        # Setup mocks needed for the method to operate
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.models.generate_content.side_effect = ClientError(
            code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH)
        )

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        # Error control for Gemini will retry 10 times
        expected_client_errors = 10
        with patch("time.sleep", lambda _: None):
            with self.assertRaises(Exception):  # noqa: B017
                self.subtitler.transcribe_audio(self.mock_audio_segment)
                self.fail("Should never get here")

        metrics = self.subtitler.metrics
        self.assertEqual(metrics.input_token_count, 0)
        self.assertEqual(metrics.output_token_count, 0)
        self.assertEqual(metrics.client_errors, expected_client_errors)
        self.assertEqual(metrics.server_errors, 0)
        self.assertEqual(metrics.invalid_subtitles, 0)
        self.assertEqual(metrics.throttles, 0)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_server_errors(self, mock_client_class, mock_temp_file):
        # Setup mocks needed for the method to operate
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.models.generate_content.side_effect = ServerError(
            code=503, response_json=json.loads(SERVER_ERROR_503_UNAVAILABLE)
        )

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        # Error control for Gemini will retry 10 times
        expected_server_errors = 10
        with patch("time.sleep", lambda _: None):
            with self.assertRaises(Exception):  # noqa: B017
                self.subtitler.transcribe_audio(self.mock_audio_segment)
                self.fail("Should never get here")

        metrics = self.subtitler.metrics
        self.assertEqual(metrics.input_token_count, 0)
        self.assertEqual(metrics.output_token_count, 0)
        self.assertEqual(metrics.client_errors, 0)
        self.assertEqual(metrics.server_errors, expected_server_errors)
        self.assertEqual(metrics.invalid_subtitles, 0)
        self.assertEqual(metrics.throttles, 0)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_throttles(self, mock_client_class, mock_temp_file):
        # Setup mocks needed for the method to operate
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.models.generate_content.side_effect = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        # Error control for Gemini will retry 10 times
        expected_throttles = 10
        with patch("time.sleep", lambda _: None):
            with self.assertRaises(Exception):  # noqa: B017
                self.subtitler.transcribe_audio(self.mock_audio_segment)
                self.fail("Should never get here")

        metrics = self.subtitler.metrics
        self.assertEqual(metrics.input_token_count, 0)
        self.assertEqual(metrics.output_token_count, 0)
        self.assertEqual(metrics.client_errors, 0)
        self.assertEqual(metrics.server_errors, 0)
        self.assertEqual(metrics.invalid_subtitles, 0)
        self.assertEqual(metrics.throttles, expected_throttles)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_invalid_subtitles(self, mock_client_class, mock_temp_file):
        # Setup mocks needed for the method to operate
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        mock_response = Mock()
        mock_response.usage_metadata = self.mock_response_usage_metadata
        mock_response.parsed = [
            SubtitleEvent(start=1000, end=2000, text="First"),
            SubtitleEvent(start=1000, end=2000, text="Second"),
        ]

        mock_client.models.generate_content.return_value = mock_response

        # The complete subtitle generation process will try for
        # 20 times before giving up.
        expected_invalid_subtitles = 20
        with patch("time.sleep", lambda _: None):
            with self.assertRaises(Exception):  # noqa: B017
                self.subtitler.transcribe_audio(self.mock_audio_segment)
                self.fail("Should never get here")

        metrics = self.subtitler.metrics
        # 20 times with 1000 tokens
        self.assertEqual(metrics.input_token_count, 20000)
        # 20 times with 2000 tokens + 2000 thinking tokens
        self.assertEqual(metrics.output_token_count, 80000)
        self.assertEqual(metrics.client_errors, 0)
        self.assertEqual(metrics.server_errors, 0)
        self.assertEqual(metrics.invalid_subtitles, expected_invalid_subtitles)
        self.assertEqual(metrics.throttles, 0)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_generation_errors(self, mock_client_class, mock_temp_file):
        # Setup mocks needed for the method to operate
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        mock_client.models.generate_content.side_effect = AIGenerationError(
            "Test generation error"
        )

        # Error control for Gemini will retry 10 times
        expected_generation_errors = 10
        with patch("time.sleep", lambda _: None):
            with self.assertRaises(AIGenerationError):
                self.subtitler.transcribe_audio(self.mock_audio_segment)

        metrics = self.subtitler.metrics
        self.assertEqual(metrics.input_token_count, 0)
        self.assertEqual(metrics.output_token_count, 0)
        self.assertEqual(metrics.client_errors, 0)
        self.assertEqual(metrics.server_errors, 0)
        self.assertEqual(metrics.invalid_subtitles, 0)
        self.assertEqual(metrics.throttles, 0)
        self.assertEqual(metrics.generation_errors, expected_generation_errors)


if __name__ == "__main__":
    unittest.main()
