import unittest

import json

from unittest.mock import Mock
from google.genai.errors import ClientError, ServerError
from tenacity import RetryCallState
from subtitle_tool.ai import (
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


if __name__ == "__main__":
    unittest.main()
