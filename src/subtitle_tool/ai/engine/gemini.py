import logging
import tempfile
from contextlib import contextmanager
from typing import Any

from pydub import AudioSegment
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from subtitle_tool.subtitles import SubtitleEvent, SubtitleValidationError
from subtitle_tool.utils import sanitize_int

logger = logging.getLogger("subtitle_tool.ai")

try:
    from google import genai
    from google.genai.errors import ClientError, ServerError
    from google.genai.types import (
        BlockedReason,
        File,
        FinishReason,
        GenerateContentConfig,
        GenerateContentResponse,
        HarmBlockThreshold,
        HarmCategory,
        HttpOptions,
        SafetySetting,
        ThinkingConfig,
    )
except ImportError:  # pragma: no cover - optional at runtime
    genai = None

    class ClientError(Exception):
        code: int = 0
        details: dict[str, Any] = {}

    class ServerError(Exception):
        code: int = 0

    class File:  # type: ignore[no-redef]
        name: str = ""

    class GenerateContentResponse:  # type: ignore[no-redef]
        usage_metadata: Any = None
        parsed: Any = None
        candidates: Any = None
        prompt_feedback: Any = None

    class BlockedReason:  # type: ignore[no-redef]
        PROHIBITED_CONTENT = "PROHIBITED_CONTENT"

    class FinishReason:  # type: ignore[no-redef]
        PROHIBITED_CONTENT = "PROHIBITED_CONTENT"

    class HarmCategory:  # type: ignore[no-redef]
        HARM_CATEGORY_DANGEROUS_CONTENT = ""
        HARM_CATEGORY_HARASSMENT = ""
        HARM_CATEGORY_HATE_SPEECH = ""
        HARM_CATEGORY_SEXUALLY_EXPLICIT = ""

    class HarmBlockThreshold:  # type: ignore[no-redef]
        BLOCK_NONE = ""

    def GenerateContentConfig(**kwargs):  # type: ignore[no-redef]
        return kwargs

    def HttpOptions(**kwargs):  # type: ignore[no-redef]
        return kwargs

    def SafetySetting(**kwargs):  # type: ignore[no-redef]
        return kwargs

    def ThinkingConfig(**kwargs):  # type: ignore[no-redef]
        return kwargs


def is_recoverable_exception(exception) -> bool:
    if isinstance(exception, ClientError):
        if exception.code == 429:
            details = (
                getattr(exception, "details", {})
                .get("error", {})
                .get("details", [])
            )
            for detail in details:
                if detail.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure":
                    for violation in detail.get("violations", []):
                        if "PerDay" in violation.get("quotaId", ""):
                            return False

    return True


def wait_api_limit(retry_state, default: float = 1.0) -> float | None:
    if retry_state.outcome and retry_state.outcome.failed:
        ex = retry_state.outcome.exception()
        if not ex or not hasattr(ex, "details"):
            return None

        for detail in ex.details.get("error", {}).get("details", []) or []:  # type: ignore
            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                rd = detail.get("retryDelay", "")
                if rd.endswith("s"):
                    try:
                        secs = float(rd[:-1])
                        logger.debug(
                            f"Rate limit hit, sleeping for {secs} seconds "
                            + "as suggested by API"
                        )
                    except ValueError:
                        return default
                    return secs or default
    return None


def setup(subtitler) -> None:
    if genai is None:
        raise ImportError(
            "google-genai is not installed. Install dependencies with `uv sync` "
            + "or `pip install google-genai`."
        )

    if not subtitler.api_key:
        raise ValueError("Gemini engine requires api_key")

    subtitler.client = genai.Client(api_key=subtitler.api_key)
    subtitle_lang_prompt = ""
    subtitle_lang_directive = ""
    if subtitler.subtitle_lang:
        subtitle_lang_prompt = f"to {subtitler.subtitle_lang}"
        subtitle_lang_directive = f"in {subtitler.subtitle_lang}"
    subtitler.system_prompt = subtitler.system_prompt % (
        subtitler.media_lang,
        subtitle_lang_prompt,
        subtitle_lang_directive,
    )


def ai_retry_handler(subtitler, exception: BaseException) -> bool:
    should_ret = False

    if isinstance(exception, ServerError):
        logger.debug(f"Server error caught: {exception}")
        subtitler.metrics.add_metrics(server_errors=1)
        should_ret = True
    elif isinstance(exception, ClientError):
        logger.debug(f"Client error caught: {exception}")
        if exception.code == 429:
            subtitler.metrics.add_metrics(throttles=1)
        else:
            subtitler.metrics.add_metrics(client_errors=1)
        should_ret = is_recoverable_exception(exception)
    elif isinstance(exception, subtitler._generation_error_class()):
        logger.debug(f"AI Generation error caught: {exception}")
        subtitler.metrics.add_metrics(generation_errors=1)
        should_ret = True

    if should_ret:
        subtitler.metrics.add_metrics(retries=1)

    return should_ret


def upload_file(subtitler, file_name: str) -> File:
    ret = File()

    for attempt in Retrying(
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    ):
        with attempt:
            ret = subtitler.client.files.upload(file=file_name)

    return ret


def remove_file(subtitler, ref_name: str):
    try:
        subtitler.client.files.delete(name=ref_name)
    except Exception as e:
        logger.warning(f"Error while removing uploaded file {ref_name}: {e!r}")


@contextmanager
def upload_audio(subtitler, segment: AudioSegment):
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=subtitler.delete_temp_files
    ) as temp_file:
        logger.debug(
            f"Temporary file created at {temp_file.name}. "
            + f"It will {'be' if subtitler.delete_temp_files else 'not be'} removed."
        )

        segment.export(temp_file.name, format="wav")
        logger.debug(f"Audio segment exported to {temp_file.name}")

        ref = upload_file(subtitler, temp_file.name)
        logger.debug(f"Temporary file {temp_file.name} uploaded as {ref.name}")
        try:
            yield ref
        finally:
            remove_file(subtitler, f"{ref.name}")
            logger.debug(f"Removed temporary file {temp_file.name} upload {ref.name}")


def generate_subtitles(
    subtitler, duration: int, file_ref: File, temp_adj: float = 0.0
) -> list[SubtitleEvent]:
    ret = []

    for attempt in Retrying(
        retry=retry_if_exception(subtitler._ai_retry_handler),
        wait=subtitler._wait_strategy(),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    ):
        with attempt:
            temp = subtitler.temperature + temp_adj
            response = GenerateContentResponse()
            try:
                logger.debug(f"Asking Gemini to generate subtitles (temp: {temp})...")
                safety_settings = [
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=HarmBlockThreshold.BLOCK_NONE,
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=HarmBlockThreshold.BLOCK_NONE,
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=HarmBlockThreshold.BLOCK_NONE,
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=HarmBlockThreshold.BLOCK_NONE,
                    ),
                ]

                user_prompt = (
                    "Create subtitles for this audio file that has a duration of "
                    + f"{duration} milliseconds"
                )
                logger.debug(f"User prompt: {user_prompt}")
                logger.debug(f"System prompt: {subtitler.system_prompt}")

                response = subtitler.client.models.generate_content(
                    model=subtitler.model_name,
                    contents=[user_prompt, file_ref],
                    config=GenerateContentConfig(
                        safety_settings=safety_settings,
                        system_instruction=subtitler.system_prompt,
                        temperature=temp,
                        top_k=50,
                        http_options=HttpOptions(timeout=5 * 60 * 1000),
                        response_mime_type="application/json",
                        response_schema=list[SubtitleEvent],
                        thinking_config=ThinkingConfig(thinking_budget=24576),
                    ),
                )
            finally:
                if response and response.usage_metadata:
                    metadata = response.usage_metadata

                    cached_token_count = sanitize_int(
                        metadata.cached_content_token_count
                    )
                    thoughts_token_count = sanitize_int(metadata.thoughts_token_count)
                    input_token_count = sanitize_int(metadata.prompt_token_count)
                    output_token_count = sanitize_int(metadata.candidates_token_count)
                    subtitler.metrics.add_metrics(
                        input_token_count=input_token_count - cached_token_count,
                        output_token_count=output_token_count + thoughts_token_count,
                    )

            if response:
                if isinstance(response.parsed, list):
                    ret = response.parsed
                else:
                    for candidate in response.candidates or []:
                        if candidate.finish_reason == FinishReason.PROHIBITED_CONTENT:
                            raise SubtitleValidationError(
                                "Output flagged as prohibited"
                            )

                    if (
                        response.prompt_feedback
                        and response.prompt_feedback.block_reason
                        == BlockedReason.PROHIBITED_CONTENT
                    ):
                        raise SubtitleValidationError("Input flagged as prohibited")

                    raise subtitler._generation_error_class()(
                        "Parsed response is not a list"
                    )
            else:
                raise subtitler._generation_error_class()("Response is empty")

    return ret
