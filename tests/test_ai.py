import tempfile
import unittest

import json

from unittest.mock import MagicMock, Mock, patch
from google.genai.errors import ClientError, ServerError
from tenacity import RetryCallState
from pydub import AudioSegment

from subtitle_tool.ai import (
    AISubtitler,
    is_recoverable_exception,
    extract_retry_delay,
    wait_api_limit,
    retry_handler,
    DEFAULT_WAIT_TIME,
)

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
"""

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
"""

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

    def test_client_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )
        self.assertTrue(is_recoverable_exception(error))

    def test_client_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_DAY)
        )
        self.assertFalse(is_recoverable_exception(error))

    def test_client_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH))
        self.assertTrue(is_recoverable_exception(error))

    def test_server_internal_error(self):
        error = ServerError(
            code=500, response_json=json.loads(SERVER_ERROR_500_INTERNAL)
        )
        self.assertTrue(is_recoverable_exception(error))  # type: ignore

    def test_server_unavailable_error(self):
        error = ServerError(
            code=503, response_json=json.loads(SERVER_ERROR_503_UNAVAILABLE)
        )
        self.assertTrue(is_recoverable_exception(error))  # type: ignore

    def test_generic_exception(self):
        error = Exception("Generic Exception")
        self.assertTrue(is_recoverable_exception(error))  # type: ignore


class TestExtractRetryDelay(unittest.TestCase):

    def test_client_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )
        delay = extract_retry_delay(error)
        self.assertEqual(delay, 33.0)

    def test_client_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_DAY)
        )
        delay = extract_retry_delay(error)
        self.assertEqual(delay, 33.0)

    def test_client_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH))
        delay = extract_retry_delay(error)
        self.assertEqual(delay, DEFAULT_WAIT_TIME)

    def test_server_internal_error(self):
        error = ServerError(
            code=500, response_json=json.loads(SERVER_ERROR_500_INTERNAL)
        )
        delay = extract_retry_delay(error)  # type: ignore
        self.assertEqual(delay, DEFAULT_WAIT_TIME)

    def test_server_unavailable_error(self):
        error = ServerError(
            code=503, response_json=json.loads(SERVER_ERROR_503_UNAVAILABLE)
        )
        delay = extract_retry_delay(error)  # type: ignore
        self.assertEqual(delay, DEFAULT_WAIT_TIME)

    def test_generic_exception(self):
        error = Exception("Generic exception")
        delay = extract_retry_delay(error)  # type: ignore
        self.assertEqual(delay, DEFAULT_WAIT_TIME)


class TestWaitApiLimit(unittest.TestCase):

    def test_client_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = wait_api_limit(retry_state)
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

        result = wait_api_limit(retry_state)
        self.assertEqual(result, 33.0)

    def test_client_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH))

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = wait_api_limit(retry_state)
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

        result = wait_api_limit(retry_state)
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

        result = wait_api_limit(retry_state)
        self.assertEqual(result, DEFAULT_WAIT_TIME)

    def test_generic_exception(self):
        error = Exception("Generic Exception")

        mock_outcome = Mock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = error

        retry_state = Mock(spec=RetryCallState)
        retry_state.outcome = mock_outcome

        result = wait_api_limit(retry_state)
        self.assertEqual(result, DEFAULT_WAIT_TIME)


class TestRetryHandler(unittest.TestCase):

    def test_client_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_MINUTE)
        )
        result = retry_handler(error)
        self.assertTrue(result)

    def test_client_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(CLIENT_ERROR_429_RATE_LIMIT_DAY)
        )
        result = retry_handler(error)
        self.assertFalse(result)

    def test_client_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(CLIENT_ERROR_403_AUTH))
        result = retry_handler(error)
        self.assertTrue(result)

    def test_server_internal_error(self):
        error = ServerError(
            code=500, response_json=json.loads(SERVER_ERROR_500_INTERNAL)
        )
        result = retry_handler(error)
        self.assertTrue(result)

    def test_server_unavailable_error(self):
        error = ServerError(
            code=503, response_json=json.loads(SERVER_ERROR_503_UNAVAILABLE)
        )
        result = retry_handler(error)
        self.assertTrue(result)


class TestAISubtitler(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.mock_audio_segment = Mock(spec=AudioSegment)
        self.api_key = "test_api_key"

        # Instantiate the actual class
        self.uploader = AISubtitler(
            api_key=self.api_key, model_name="test_model", delete_temp_files=True
        )

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    @patch("logging.getLogger")
    def test_upload_audio_success(self, mock_logger, mock_client_class, mock_temp_file):
        """Test successful audio upload and cleanup"""
        # Setup mocks
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        # Execute the context manager
        with self.uploader.upload_audio(self.mock_audio_segment) as result:
            # Verify the result is the upload reference
            self.assertEqual(result, mock_ref)

            # Verify audio segment was exported
            self.mock_audio_segment.export.assert_called_once_with(
                "/tmp/test_audio.wav", format="wav"
            )

            # Verify client was created with correct API key
            mock_client_class.assert_called_once_with(api_key=self.api_key)

            # Verify file was uploaded
            mock_client.files.upload.assert_called_once_with(file="/tmp/test_audio.wav")

        # Verify cleanup happened after context manager exit
        mock_client.files.delete.assert_called_once_with(name="files/test_upload_id")

        # Verify NamedTemporaryFile was called with correct parameters
        mock_temp_file.assert_called_once_with(suffix=".wav", delete=True)

    @patch("tempfile.NamedTemporaryFile")
    @patch("google.genai.Client")
    def test_upload_audio_with_delete_temp_files_false(
        self, mock_client_class, mock_temp_file
    ):
        """Test that delete_temp_files parameter is passed to NamedTemporaryFile"""
        self.uploader.delete_temp_files = False

        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/test_audio.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance
        mock_temp_file.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_ref = Mock()
        mock_ref.name = "files/test_upload_id"
        mock_client.files.upload.return_value = mock_ref

        with self.uploader.upload_audio(self.mock_audio_segment):
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
            with self.uploader.upload_audio(self.mock_audio_segment):
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
            with self.uploader.upload_audio(self.mock_audio_segment):
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

        with self.assertRaises(Exception) as context:
            with self.uploader.upload_audio(self.mock_audio_segment):
                pass

        self.assertEqual(str(context.exception), "Upload failed")

        # Verify export was called but delete was not (since upload failed)
        self.mock_audio_segment.export.assert_called_once()
        mock_client.files.delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
