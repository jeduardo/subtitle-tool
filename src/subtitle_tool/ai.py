import logging
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    File,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    HttpOptions,
    SafetySetting,
    ThinkingConfig,
)
from pydub import AudioSegment
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from subtitle_tool.subtitles import (
    SubtitleEvent,
    SubtitleValidationException,
    validate_subtitles,
)
from subtitle_tool.utils import sanitize_int

logger = logging.getLogger("subtitle_tool.ai")

DEFAULT_WAIT_TIME = 15.0


class AIGenerationError(BaseException):
    pass


@dataclass
class OperationMetrics:
    """
    Usage tracker for interesting metrics.

    Args:
        input_token_count (int): number of input tokens, derived from the model prompt.
        output_token_count (int): number of output tokens, comprising both output tokens
            and thought tokens.
        client_errors (int): how many errors the client has seen
        server_errors (int): how many errors the client has seen
        throttles (int): how many errors the client has seen
        retries (int): how many retries the client has seen
        invalid_subtitles (int): how many invalid subtitles were generated
        generation_errors (int): how many malformed responses the AI returned
    """

    input_token_count: int = 0
    output_token_count: int = 0
    client_errors: int = 0
    server_errors: int = 0
    throttles: int = 0
    retries: int = 0
    invalid_subtitles: int = 0
    generation_errors: int = 0

    def __post_init__(self):
        self.lock = Lock()

    def add_metrics(
        self,
        input_token_count: int = 0,
        output_token_count: int = 0,
        client_errors: int = 0,
        server_errors: int = 0,
        throttles: int = 0,
        retries: int = 0,
        invalid_subtitles: int = 0,
        generation_errors: int = 0,
    ) -> None:
        """
        Add usage counters from the current client.
        To allow the client to be used by multple threads, updates are
        performed under a lock.

        Args:
            input_token_count (int): number of input tokens to add to the
                current counter
            output_token_count (int): number of output tokens to add to the
                current counter
            client_errors (int): number of client errors to add to the
                current counter
            server_errors (int): number of server errors to add to the
                current counter
            throttles (int): number of throttles to add to the current counter
            retries (int): number of retries to add to the current counter
            invalid_subtitles (int): number of invalid subtitles to add
            generation_errors (int): number of generation errors to add
        """
        with self.lock:
            self.input_token_count += input_token_count
            self.output_token_count += output_token_count
            self.client_errors += client_errors
            self.server_errors += server_errors
            self.throttles += throttles
            self.retries += retries
            self.invalid_subtitles += invalid_subtitles
            self.generation_errors += generation_errors


@dataclass
class AISubtitler:
    """
    AI Subtitler implementation using Gemini.

    Args:
        model_name (str): Gemini model to be used (mandatory)
        api_key (str): Gemini API key (mandatory)
        delete_temp_files (bool): whether any temporary files created
            should be deleted (default: True)
        system_prompt (str): system prompt driving the model. There is
            a default prompt already provided, override only if necessary.
    """

    model_name: str
    api_key: str
    delete_temp_files: bool = True
    system_prompt: str = """
        You are a professional transcriber of audio clips into subtitles.
        You recognize which language is being spoken in the title and write the subtitle accordingly.
        You take an audio file and you output a high-quality, perfect transcription.
        Your output is only the subtitle content in the JSON format specified.

        You follow these MANDATORY GUIDELINES:
        1. The output is done in the JSON format specified.
        2. Each segment should be of 1-2 lines and a maximum of 5 seconds. Check the example for more reference.
        3. Use proper punctuation and capitalization.
        4. Keep original meaning but clean up filler words like "um", "uh", "like", "you know", etc.
        5. Clean up stutters like "I I I" or "uh uh uh".
        6. After you generate the subtitles, you will MAKE ABSOLUTELY SURE that the last subtitle does not end after the audio file.

        Example JSON subtitle for an audio file of 34000 milliseconds. Notice how the end of the last subtitle ends before the end of the audio file:
        [
            {
                "start": 0,
                "end": 5000,
                "text": "Up next, he promises to avenge his sister's"
            },
            {
                "start": 5000,
                "end": 7100,
                "text": "murder. I prayed to God that I would be led to be"
            },
            {
                "start": 7100,
                "end": 8500,
                "text": "in the right place at the right time."
            },
            {
                "start": 8500,
                "end": 12000,
                "text": "For years, he tracks her killer without success."
            },
            {
                "start": 12000,
                "end": 14500,
                "text": "Every day was another blow to the stomach."
            },
            {
                "start": 14900,
                "end": 18842,
                "text": "Somewhere deep in the Houston crime files are the secrets to solve"
            },
            {
                "start": 18842,
                "end": 19842,
                "text": "the case."
            },
            {
                "start": 19900,
                "end": 21500,
                "text": "He just had to find them."
            },
            {
                "start": 22000,
                "end": 26800,
                "text": "Houston had 500,000 prints. Everybody has 10 fingers. That's 5 million prints."
            },
            {
                "start": 27300,
                "end": 30800,
                "text": "34 years later, investigators find the answer."
            },
            {
                "start": 31500,
                "end": 33500,
                "text": "I want to know who killed Diane."
            }
        ]

        """  # noqa: E501

    def __post_init__(self):
        self.metrics = OperationMetrics()

    def _is_recoverable_exception(self, exception: ClientError) -> bool:
        """
        This is an overly optimistic function that deems that all exceptions
        are recoverable except ones that fail because of exceeded daily
        quotas.

        Args:
            exception (ClientError): a Gemini ClientError

        Returns:
            bool: able to recover or not
        """
        if isinstance(exception, ClientError):
            if exception.code == 429:
                details = exception.details["error"]["details"]
                for detail in details:
                    if (
                        detail.get("@type")
                        == "type.googleapis.com/google.rpc.QuotaFailure"
                    ):
                        for violation in detail.get("violations"):
                            # min: GenerateRequestsPerMinutePerProjectPerModel-FreeTier
                            # day: GenerateRequestsPerDayPerProjectPerModel-FreeTier
                            if "PerDay" in violation["quotaId"]:
                                return False

        return True

    def _extract_retry_delay(self, exception: ClientError) -> float:
        """
        Extract retry delay from rate-limit message.
        It will return 60 seconds on parsing error.

        Args:
            exception: The exception object

        Returns:
            float: Retry delay in seconds, defaults to 15 if not found
        """
        try:
            for detail in exception.details["error"]["details"]:
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    retry_delay = detail.get("retryDelay", "")
                    if retry_delay.endswith("s"):
                        delay = float(retry_delay[:-1])
                        if delay == 0:
                            logger.warning(
                                f"Delay returned is zero, waiting the default {DEFAULT_WAIT_TIME} instead"  # noqa: E501
                            )
                            return DEFAULT_WAIT_TIME
                        else:
                            return delay
        except Exception as e:
            logger.warning(f"Could not parse retry delay from exception: {e}")

        # Default fallback delay
        return DEFAULT_WAIT_TIME

    def _wait_api_limit(self, retry_state: RetryCallState) -> float:
        """

        Extracts the retry delay from rate limit messages.
        From internal exceptions, it retries after 15 seconds.

        Args:
            retry_state: The retry state object from tenacity

        Returns:
            float: The sleep duration
        """
        if retry_state.outcome and retry_state.outcome.failed:
            exception = retry_state.outcome.exception()

            if isinstance(exception, ClientError):
                if exception.code == 429:
                    delay = self._extract_retry_delay(exception)
                    logger.debug(
                        f"Rate limit hit, sleeping for {delay} seconds as "
                        + "suggested by API"
                    )
                    return delay

        # Default delay for other cases
        return DEFAULT_WAIT_TIME

    def _ai_retry_handler(self, exception: BaseException) -> bool:
        """
        This handler defines the cases when tenacity should retry
        calling the Gemini API.

        We will retry the API when:
        - It's an error issued by the Gemini Client
        - It's a 500 INTERNAL error, which Gemini sometimes issues and they
            recommend to retry.
        - It's a 429 rate limit error for quotas that are replenished by the minute.
        - It's a Server error.
        - It's an AI Generation Error issued when validating the Gemini responses.
        For all other issues, we will not ask tenacity to retry.

        Args:
            exception (BaseException): The exception that occurred

        Returns:
            bool: True if we should retry
        """

        # We want to return False on all exceptions we don't know so we can avoid
        # unneeded retries for problems we don't know about.
        should_ret = False

        if isinstance(exception, ServerError):
            logger.debug(f"Server error caught: {exception}")
            self.metrics.add_metrics(server_errors=1)
            should_ret = True
        elif isinstance(exception, ClientError):
            logger.debug(f"Client error caught: {exception}")
            if exception.code == 429:
                self.metrics.add_metrics(throttles=1)
            else:
                self.metrics.add_metrics(client_errors=1)
            should_ret = self._is_recoverable_exception(exception)
        elif isinstance(exception, AIGenerationError):
            logger.debug(f"AI Generation error caught: {exception}")
            self.metrics.add_metrics(generation_errors=1)
            should_ret = True

        if should_ret:
            self.metrics.add_metrics(retries=1)

        return should_ret

    def _subtitles_retry_handler(self, exception: BaseException) -> bool:
        """
        This handler defines the cases when tenacity should try calling
        the entire generation process again.

        This happens when Gemini has generated a subtitle but this subtitle
        is invalid.

        Args:
            exception (BaseException): The exception that occurred

        Returns:
            bool: True if we should retry
        """

        # We want to return False on all exceptions we don't know so we can avoid
        # unneeded retries for problems we don't know about.
        should_ret = False

        if isinstance(exception, SubtitleValidationException):
            logger.debug(f"Invalid subtitles generated: {exception}")
            self.metrics.add_metrics(invalid_subtitles=1)
            should_ret = True

        if should_ret:
            self.metrics.add_metrics(retries=1)

        return should_ret

    def _upload_file(self, file_name: str) -> File:
        """
        Wrapper to retry file uploads to the Gemini file server.
        It will apply exponential backoff for retries and try it for 5 times.

        Args:
            file_name (str): Path to file to be uploaded

        Returns:
            File: file upload reference
        """

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=5),
            stop=stop_after_attempt(5),
            before_sleep=before_sleep_log(logger, logging.DEBUG),
            reraise=True,
        )
        def _inner_upload_file() -> File:
            client = genai.Client(api_key=self.api_key)
            return client.files.upload(file=file_name)

        return _inner_upload_file()

    def _remove_file(self, ref_name: str):
        """
        Wrapper to remove files from the Gemini file server.

        Args:
            ref_name (str): Upload reference
        """
        try:
            client = genai.Client(api_key=self.api_key)
            client.files.delete(name=ref_name)
        except Exception as e:
            # Google deletes the files in 48h, so cleanup is a courtesy.
            # This means we just issue a warning here.
            logger.warning(f"Error while removing uploaded file {ref_name}: {e!r}")

    @contextmanager
    def upload_audio(self, segment: AudioSegment):
        """
        Context to upload and remove a file from Gemini servers

        Arguments:
            segment (AudioSegment): segment representation

        Returns:
            File: upload identifier
        """
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=self.delete_temp_files
        ) as temp_file:
            # Export AudioSegment to temporary file
            # I found out that wav files can avoid some unexplained
            # 500 errors with Gemini.
            logger.debug(
                f"Temporary file created at {temp_file.name}. "
                + f"It will {'be' if self.delete_temp_files else 'not be'} removed."
            )

            segment.export(temp_file.name, format="wav")
            logger.debug(f"Audio segment exported to {temp_file.name}")

            # Upload the temporary file (API will infer mime type from content)
            ref = self._upload_file(temp_file.name)
            logger.debug(f"Temporary file {temp_file.name} uploaded as {ref.name}")
            try:
                yield ref
            finally:
                self._remove_file(f"{ref.name}")
                logger.debug(
                    f"Removed temporary file {temp_file.name} upload {ref.name}"
                )

    def _audio_to_subtitles(
        self, audio_segment: AudioSegment, file_ref: File
    ) -> list[SubtitleEvent]:
        """
        Generate subtitles for an audio segment.

        This function will call Gemini to generate subtitles and will
        validate the result before returning. If the subtitle is invalid,
        it will ask Gemini to recreate the subtitles up to 20 times.

        It will only retry the generation on subtitle validation errors.

        Args:
            audio_segment (AudioSegment): segment to be transcribed
            file_ref (types.File): reference to uploaded file

        Returns:
            list[SubtitleEvent]: subtitles extracted from audio track

        Throws:
            SubtitleValidationException in case the merged subtitles are invalid.

        """

        @retry(
            retry=retry_if_exception(self._subtitles_retry_handler),
            stop=stop_after_attempt(20),
            before_sleep=before_sleep_log(logger, logging.DEBUG),
        )
        def _inner_audio_to_subtitles() -> list[SubtitleEvent]:
            subtitle_events = self._generate_subtitles(file_ref)
            validate_subtitles(subtitle_events, audio_segment.duration_seconds)
            logger.debug("Valid subtitles generated for segment")
            return subtitle_events

        return _inner_audio_to_subtitles()

    def _generate_subtitles(self, file_ref) -> list[SubtitleEvent]:
        """
        Generate subtitles for the file uploaded onto Gemini servers.

        If there is a rate limit, it will retrieve the waiting time from the
        ClientError before retrying.

        Due to its probabilistic nature, Gemini sometimes generates empty
        response payloads, which will trigger a retry.

        Finally, sometimes Gemini issues ServerErrors, which will trigger
        a retry.

        This function will retry fetching the response from Gemini up to
        10 times.

        Args:
            file_id (str): identifier of uploaded file

        Returns:
            list[SubtitleEvent]: subtitles extracted from audio track
        """

        @retry(
            retry=retry_if_exception(self._ai_retry_handler),
            wait=self._wait_api_limit,
            stop=stop_after_attempt(10),
            before_sleep=before_sleep_log(logger, logging.DEBUG),
            reraise=True,
        )
        def _inner_generate_subtitles() -> list[SubtitleEvent]:
            response = None
            try:
                logger.debug("Asking Gemini to generate subtitles...")
                client = genai.Client(api_key=self.api_key)
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

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=["Create subtitles for this audio file", file_ref],
                    config=GenerateContentConfig(
                        # Don't want to censor any subtitles
                        safety_settings=safety_settings,
                        system_instruction=self.system_prompt,
                        temperature=0.1,
                        http_options=HttpOptions(timeout=2 * 60 * 1000),  # 2 minutes
                        response_mime_type="application/json",
                        response_schema=list[SubtitleEvent],
                        thinking_config=ThinkingConfig(thinking_budget=24576),
                    ),
                )
            finally:
                if response and response.usage_metadata:
                    metadata = response.usage_metadata
                    logger.debug(f"Cached token info: {metadata.cache_tokens_details}")
                    logger.debug(
                        f"Cached token count: {metadata.cached_content_token_count}"
                    )
                    logger.debug(
                        f"Thoughts token count: {metadata.thoughts_token_count}"
                    )
                    logger.debug(
                        f"Output token count: {metadata.candidates_token_count}"
                    )
                    self.metrics.add_metrics(
                        input_token_count=sanitize_int(metadata.prompt_token_count),
                        output_token_count=sanitize_int(metadata.candidates_token_count)
                        + sanitize_int(metadata.thoughts_token_count),
                    )
            if response:
                if isinstance(response.parsed, list):
                    return response.parsed
                else:
                    logger.debug(f"Parsed response is not a list: {response}")
                    raise AIGenerationError("Parsed response is not a list")
            else:
                logger.debug("Response is empty")
                raise AIGenerationError("Response is empty")

        return _inner_generate_subtitles()

    def transcribe_audio(self, audio_segment: AudioSegment) -> list[SubtitleEvent]:
        """
        Transcribe the audio of a given segment into subtitle.

        This function will upload the audio file to Gemini servers,
        removing it after processing.

        Args:
            audio_segment (AudioSegment): segment to be transcribed

        Return:
            list[SubtitleEvent: list of validated subtitles
        """
        with self.upload_audio(audio_segment) as file_ref:
            subtitle_events = self._audio_to_subtitles(audio_segment, file_ref)

        return subtitle_events
