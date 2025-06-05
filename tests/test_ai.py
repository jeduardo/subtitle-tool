import unittest

import json

from google.genai.errors import ClientError
from subtitle_tool.ai import (
    is_recoverable_exception,
    extract_retry_delay,
    DEFAULT_WAIT_TIME,
)


ERROR_429_RATE_LIMIT_MINUTE = """
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

ERROR_429_RATE_LIMIT_DAY = """
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

ERROR_403_AUTH = """
{
    "error": {
        "code": 403,
        "message": "Auth exceptiom",
        "status": "AUTH ERROR",
        "details": [
        {
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": "33s"
        }
        ]
    }
}
"""

ERROR_500_INTERNAL = """
{
    "error": {
        "code": 500,
        "message": "An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting",
        "status": "INTERNAL"
    }
}
"""


class TestResilienceMethods(unittest.TestCase):

    def test_is_recoverable_exception_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(ERROR_429_RATE_LIMIT_MINUTE)
        )
        self.assertTrue(is_recoverable_exception(error))

    def test_is_recoverable_exception_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(ERROR_429_RATE_LIMIT_DAY)
        )
        self.assertFalse(is_recoverable_exception(error))

    def test_is_recoverable_exception_auth_error(self):
        error = ClientError(code=403, response_json=json.loads(ERROR_403_AUTH))
        self.assertTrue(is_recoverable_exception(error))

    def test_is_recoverable_exception_internal_error(self):
        error = ClientError(code=500, response_json=json.loads(ERROR_500_INTERNAL))
        self.assertTrue(is_recoverable_exception(error))

    def test_extract_retry_delay_rate_limit_per_minute(self):
        error = ClientError(
            code=429, response_json=json.loads(ERROR_429_RATE_LIMIT_MINUTE)
        )
        delay = extract_retry_delay(error)
        self.assertEqual(delay, 33.0)

    def test_extract_retry_delay_rate_limit_per_day(self):
        error = ClientError(
            code=429, response_json=json.loads(ERROR_429_RATE_LIMIT_DAY)
        )
        delay = extract_retry_delay(error)
        self.assertEqual(delay, 33.0)

    def test_extract_retry_delay_auth(self):
        error = ClientError(code=403, response_json=json.loads(ERROR_403_AUTH))
        delay = extract_retry_delay(error)
        self.assertEqual(delay, 33.0)

    def test_extract_retry_delay_internal(self):
        error = ClientError(code=500, response_json=json.loads(ERROR_500_INTERNAL))
        delay = extract_retry_delay(error)
        self.assertEqual(delay, DEFAULT_WAIT_TIME)


if __name__ == "__main__":
    unittest.main()
