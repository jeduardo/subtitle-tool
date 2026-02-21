import logging
import math
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import signature
from threading import Lock
from typing import Any

from humanize import precisedelta
from pydub import AudioSegment
from tenacity import (
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from subtitle_tool.subtitles import (
    SubtitleEvent,
    SubtitleValidationError,
    validate_subtitles,
)

try:
    from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor
except ImportError:  # pragma: no cover - handled at runtime
    VoxtralForConditionalGeneration = None  # type: ignore[assignment]
    VoxtralProcessor = None  # type: ignore[assignment]

logger = logging.getLogger("subtitle_tool.ai")


class AIGenerationError(BaseException):
    pass


def _is_recoverable_exception(exception) -> bool:
    """
    Determine whether a local generation exception should be retried.

    Args:
        exception: error raised during local transcription

    Returns:
        bool: True if retrying may help.
    """

    if isinstance(exception, FileNotFoundError):
        return False

    # Model loading and local runtime errors can be transient.
    return True


def _wait_api_limit(retry_state: RetryCallState, default: float = 1.0) -> float | None:
    """
    Keep compatibility with previous retry behavior API.

    Local inference has no remote rate-limit headers, so this returns None.
    """

    del retry_state, default
    return None


class WaitExponentialOrServerDelay:
    def __init__(self, multiplier=1, max=16, default_wait=1):
        self._exp = wait_exponential(multiplier=multiplier, max=max)
        self._default = default_wait

    def __call__(self, retry_state):
        api_delay = _wait_api_limit(retry_state, default=self._default)
        if api_delay is not None:
            return api_delay

        api_delay = self._exp(retry_state)
        logger.debug(
            "No wait time suggested by runtime, using exponential backoff wait "
            + f"time: {api_delay}"
        )
        return api_delay


@dataclass
class OperationMetrics:
    """
    Usage tracker for interesting metrics.

    Args:
        input_token_count (int): always zero for local voxtral.
        output_token_count (int): always zero for local voxtral.
        client_errors (int): how many client/runtime errors the client has seen.
        server_errors (int): retained for compatibility.
        throttles (int): retained for compatibility.
        retries (int): how many retries the client has seen.
        invalid_subtitles (int): how many invalid subtitles were generated.
        generation_errors (int): how many malformed responses were generated.
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
        with self.lock:
            self.input_token_count += input_token_count
            self.output_token_count += output_token_count
            self.client_errors += client_errors
            self.server_errors += server_errors
            self.throttles += throttles
            self.retries += retries
            self.invalid_subtitles += invalid_subtitles
            self.generation_errors += generation_errors


def _normalize_language(language: str | None) -> str | None:
    if not language:
        return None

    normalized = language.strip().lower()
    language_map = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "portuguese": "pt",
        "german": "de",
        "italian": "it",
        "dutch": "nl",
        "hindi": "hi",
    }
    return language_map.get(normalized, normalized)


@dataclass
class AISubtitler:
    """
    AI subtitler implementation using local mlx-voxtral on Apple Silicon.

    Args:
        model_name (str): Voxtral model repository/path.
        subtitle_lang (str): subtitle language hint (best-effort).
        media_lang (str): media language hint.
        delete_temp_files (bool): whether temporary files should be deleted.
        api_key (str): retained for compatibility; ignored.
        temperature (float): generation temperature.
        temperature_adj (float): retry temperature increment.
        system_prompt (str): retained for compatibility; ignored.
    """

    model_name: str
    subtitle_lang: str
    media_lang: str = "English"
    delete_temp_files: bool = True
    api_key: str | None = None
    temperature: float = 0.0
    temperature_adj: float = 0.01
    system_prompt: str = ""

    def __post_init__(self):
        if VoxtralForConditionalGeneration is None or VoxtralProcessor is None:
            raise ImportError(
                "mlx-voxtral is not installed. Install dependencies with `uv sync` "
                + "or `pip install mlx-voxtral` and ensure `transformers` is "
                + "available (latest recommended)."
            )

        self.processor = VoxtralProcessor.from_pretrained(self.model_name)
        self.model = VoxtralForConditionalGeneration.from_pretrained(self.model_name)
        self.inference_lock = Lock()
        self.metrics = OperationMetrics()

    def _ai_retry_handler(self, exception: BaseException) -> bool:
        should_ret = False

        if isinstance(exception, AIGenerationError):
            logger.debug(f"AI generation error caught: {exception}")
            self.metrics.add_metrics(generation_errors=1)
            should_ret = True
        elif _is_recoverable_exception(exception):
            logger.debug(f"Runtime error caught: {exception!r}")
            self.metrics.add_metrics(client_errors=1)
            should_ret = True
        else:
            logger.debug(f"Unrecoverable runtime error caught: {exception!r}")
            self.metrics.add_metrics(client_errors=1)
            should_ret = False

        if should_ret:
            self.metrics.add_metrics(retries=1)

        return should_ret

    def _subtitles_retry_handler(self, exception: BaseException) -> bool:
        if isinstance(exception, SubtitleValidationError):
            logger.debug(f"Invalid subtitles generated: {exception}")
            self.metrics.add_metrics(invalid_subtitles=1)
            self.metrics.add_metrics(retries=1)
            return True

        return False

    @contextmanager
    def upload_audio(self, segment: AudioSegment):
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=self.delete_temp_files
        ) as temp_file:
            logger.debug(
                f"Temporary file created at {temp_file.name}. "
                + f"It will {'be' if self.delete_temp_files else 'not be'} removed."
            )

            segment.export(temp_file.name, format="wav")
            logger.debug(f"Audio segment exported to {temp_file.name}")
            yield temp_file.name

    def _build_request_inputs(self, audio_path: str, language: str | None):
        # Old Voxtral processors used apply_transcrition_request (typo).
        request_builder = getattr(self.processor, "apply_transcription_request", None)
        if request_builder is None:
            request_builder = getattr(self.processor, "apply_transcrition_request", None)

        if request_builder is None:
            raise AIGenerationError(
                "Voxtral processor does not expose transcription request helper"
            )

        kwargs: dict[str, Any] = {"audio": audio_path}
        try:
            params = signature(request_builder).parameters
        except (TypeError, ValueError):
            params = {}

        if language and ("language" in params or not params):
            kwargs["language"] = language

        try:
            return request_builder(**kwargs)
        except TypeError as error:
            # Handle version mismatches gracefully.
            if "language" in kwargs and "language" in str(error):
                return request_builder(audio=audio_path)
            raise

    def _request_value(self, inputs: Any, field: str) -> Any:
        if isinstance(inputs, dict):
            return inputs.get(field)

        return getattr(inputs, field, None)

    def _extract_prompt_length(self, inputs: Any) -> int:
        input_ids = self._request_value(inputs, "input_ids")

        if input_ids is None:
            return 0

        shape = getattr(input_ids, "shape", None)
        if shape is None:
            return 0

        if len(shape) >= 2:
            return int(shape[1])

        return int(shape[0])

    def _decode_outputs(self, inputs: Any, outputs: Any) -> str:
        prompt_length = self._extract_prompt_length(inputs)

        output_tokens = outputs
        if isinstance(outputs, (list, tuple)) and outputs:
            output_tokens = outputs[0]

        ndim = getattr(output_tokens, "ndim", None)
        shape = getattr(output_tokens, "shape", None)
        if ndim == 2 or (shape is not None and len(shape) == 2):
            output_tokens = output_tokens[0]

        if prompt_length:
            try:
                output_tokens = output_tokens[prompt_length:]
            except Exception:
                logger.debug("Could not trim prompt tokens from output; decoding raw")

        text = self.processor.decode(output_tokens, skip_special_tokens=True)
        return (text or "").strip()

    def _chunk_text(self, text: str, target_chunks: int) -> list[str]:
        words = text.split()
        if not words:
            return []

        target_chunks = max(1, target_chunks)
        if len(words) <= target_chunks:
            return words

        chunk_size = math.ceil(len(words) / target_chunks)
        ret = []
        for idx in range(0, len(words), chunk_size):
            ret.append(" ".join(words[idx : idx + chunk_size]))

        return ret

    def _text_to_subtitles(self, text: str, duration_ms: int) -> list[SubtitleEvent]:
        text = (text or "").strip()
        if not text:
            return []

        # Keep subtitle events near a maximum of 5s each.
        target_chunks = max(1, math.ceil(duration_ms / 5000))
        chunks = self._chunk_text(text, target_chunks)
        if not chunks:
            return []

        base_duration = max(1, duration_ms // len(chunks))
        remainder = max(0, duration_ms - (base_duration * len(chunks)))

        ret: list[SubtitleEvent] = []
        current_start = 0
        for index, chunk in enumerate(chunks):
            extra = 1 if index < remainder else 0
            end = current_start + base_duration + extra
            if index == len(chunks) - 1:
                end = duration_ms

            if end <= current_start:
                end = min(duration_ms, current_start + 1)

            ret.append(
                SubtitleEvent(
                    start=current_start,
                    end=end,
                    text=chunk,
                )
            )
            current_start = end

        return ret

    def _generate_subtitles(
        self, duration_ms: int, audio_path: str, temp_adj: float = 0.0
    ) -> list[SubtitleEvent]:
        language = _normalize_language(self.media_lang)
        temperature = max(0.0, self.temperature + temp_adj)

        with self.inference_lock:
            inputs = self._build_request_inputs(audio_path, language)
            input_ids = self._request_value(inputs, "input_ids")
            input_features = self._request_value(inputs, "input_features")
            attention_mask = self._request_value(inputs, "attention_mask")

            generation_kwargs = {
                "input_ids": input_ids,
                "input_features": input_features,
            }
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask

            outputs = self.model.generate(
                **generation_kwargs,
                max_new_tokens=2048,
                temperature=temperature,
            )

        text = self._decode_outputs(inputs, outputs)
        ret = self._text_to_subtitles(text, duration_ms)
        if ret:
            return ret

        logger.debug(f"Unexpected transcription response payload: {text}")
        raise AIGenerationError("Transcription payload has no subtitle content")

    def _audio_to_subtitles(
        self, audio_segment: AudioSegment, audio_path: str
    ) -> list[SubtitleEvent]:
        subtitle_events: list[SubtitleEvent] = []

        temp_adj = 0.0
        for attempt in Retrying(
            retry=retry_if_exception(self._subtitles_retry_handler),
            stop=stop_after_attempt(10),
            before_sleep=before_sleep_log(logger, logging.DEBUG),
        ):
            with attempt:
                duration_ms = int(audio_segment.duration_seconds * 1000)

                for generation_attempt in Retrying(
                    retry=retry_if_exception(self._ai_retry_handler),
                    wait=WaitExponentialOrServerDelay(
                        multiplier=1, max=16, default_wait=1
                    ),
                    stop=stop_after_attempt(5),
                    before_sleep=before_sleep_log(logger, logging.DEBUG),
                    reraise=True,
                ):
                    with generation_attempt:
                        cur_temp_adj = temp_adj
                        temp_adj += self.temperature_adj
                        subtitle_events = self._generate_subtitles(
                            duration_ms, audio_path, cur_temp_adj
                        )

                validate_subtitles(subtitle_events, audio_segment.duration_seconds)
                logger.debug("Valid subtitles generated for segment")

        return subtitle_events

    def transcribe_audio(self, audio_segment: AudioSegment) -> list[SubtitleEvent]:
        with self.upload_audio(audio_segment) as audio_path:
            segment_dur = precisedelta(int(audio_segment.duration_seconds))
            logger.debug(f"Transcribing audio segment of {segment_dur}")
            subtitle_events = self._audio_to_subtitles(audio_segment, audio_path)

        return subtitle_events
