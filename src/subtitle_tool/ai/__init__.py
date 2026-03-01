import logging
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
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

from subtitle_tool.ai.engine import gemini as gemini_engine
from subtitle_tool.ai.engine import voxtral as voxtral_engine
from subtitle_tool.ai.engine import whisper_mlx as whisper_mlx_engine
from subtitle_tool.subtitles import (
    SubtitleEvent,
    SubtitleValidationError,
    validate_subtitles,
)

logger = logging.getLogger("subtitle_tool.ai")

SUPPORTED_ENGINES = ("gemini", "voxtral", "whisper-mlx")


class AIGenerationError(BaseException):
    pass


def _is_recoverable_exception(exception) -> bool:
    """Compatibility wrapper for Gemini recoverable-exception behavior."""

    return gemini_engine.is_recoverable_exception(exception)


def _wait_api_limit(retry_state: RetryCallState, default: float = 1.0) -> float | None:
    """Compatibility wrapper for Gemini retry-delay extraction."""

    return gemini_engine.wait_api_limit(retry_state, default=default)


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
            f"No wait time suggested by runtime/API, using exponential backoff logic wait time: {api_delay}"  # noqa: E501
        )
        return api_delay


@dataclass
class OperationMetrics:
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
    Unified subtitler with pluggable engines.

    `engine` options:
    - `gemini`: Google Gemini cloud API
    - `voxtral`: local MLX Voxtral
    - `whisper-mlx`: local MLX Whisper
    """

    model_name: str
    api_key: str | None = None
    subtitle_lang: str = ""
    media_lang: str = "English"
    delete_temp_files: bool = True
    temperature: float = 0.1
    temperature_adj: float = 0.01
    engine: str = "gemini"
    system_prompt: str = """
        # YOUR ROLE
        - You work as a transcriber of audio clips in %s %s, delivering perfect transcriptions %s.
        - You know many languages, and you can recognize the language spoken in the audio and write the subtitle accordingly.
        - Your work is to take an audio file and output a high-quality, perfect transcription synchronized with spoken dialogue,
        - You strictly follow the JSON format specified, and your output is only the subtitle content in this JSON format.
        - You *DO NOT* subtitle music or music moods.
        - You *NEVER* generate a subtitle that ends after the audio clip.
        - You *ALWAYS* check your work before delivering it.
        - You *NEVER DEVIATE* from the mandatory guidelines below.

        # MANDATORY GUIDELINES
        1. The output is done in the JSON format specified.
        2. Each segment should be of 1-2 lines and a maximum of 5 seconds. Check the example for more reference.
        3. Use proper punctuation and capitalization.
        4. Keep original meaning but clean up filler words like "um", "uh", "like", "you know", etc.
        5. Clean up stutters like "I I I" or "uh uh uh".
        6. For every subtitle entry, you ensure that both the start and end times do not have a higher value in milliseconds than the end time of the audio segment in milliseconds.

        # EXAMPLE
        Here is an example of a JSON subtitle for an audio file of 34000 milliseconds. Notice how the last entry in the subtitle DOES NOT go beyond 34000 milliseconds.
        <EXAMPLE>
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
        </EXAMPLE>
        """  # noqa: E501

    def __post_init__(self):
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError(
                "Unsupported engine "
                + f"'{self.engine}'. Valid engines: {SUPPORTED_ENGINES}"
            )

        self.metrics = OperationMetrics()
        self.inference_lock = Lock()

        if self.engine == "gemini":
            gemini_engine.setup(self)
        elif self.engine == "voxtral":
            voxtral_engine.setup(self)
        else:
            whisper_mlx_engine.setup(self)

    def _generation_error_class(self):
        return AIGenerationError

    def _normalize_language(self, language: str | None) -> str | None:
        return _normalize_language(language)

    def _wait_strategy(self):
        return WaitExponentialOrServerDelay(multiplier=1, max=16, default_wait=1)

    def _ai_retry_handler(self, exception: BaseException) -> bool:
        should_ret = False

        if self.engine == "gemini":
            should_ret = gemini_engine.ai_retry_handler(self, exception)
        else:
            if isinstance(exception, self._generation_error_class()):
                logger.debug(f"AI Generation error caught: {exception}")
                self.metrics.add_metrics(generation_errors=1)
                should_ret = True
            elif isinstance(exception, FileNotFoundError):
                logger.debug(f"Runtime error not recoverable: {exception}")
                self.metrics.add_metrics(client_errors=1)
                should_ret = False
            else:
                logger.debug(f"Runtime error caught: {exception}")
                self.metrics.add_metrics(client_errors=1)
                should_ret = True

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
        if self.engine == "gemini":
            with gemini_engine.upload_audio(self, segment) as file_ref:
                yield file_ref
            return

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

    def _segment_value(self, segment: Any, field: str, default: Any = None) -> Any:
        if isinstance(segment, dict):
            return segment.get(field, default)

        return getattr(segment, field, default)

    def _to_milliseconds(self, value: Any, duration_ms: int) -> int:
        if value is None:
            return 0

        as_float = float(value)
        if abs(as_float) <= (duration_ms / 1000.0 + 1.0):
            as_float *= 1000.0

        return int(round(as_float))

    def _normalize_segments(
        self, response: Any, duration_ms: int
    ) -> list[SubtitleEvent]:
        segments = self._segment_value(response, "segments")
        ret: list[SubtitleEvent] = []
        max_end = max(1, duration_ms)

        if not segments:
            text = self._segment_value(response, "text", "")
            if isinstance(text, str) and text.strip():
                ret.append(SubtitleEvent(start=0, end=max_end, text=text.strip()))
            return ret

        for segment in segments:
            text = (self._segment_value(segment, "text", "") or "").strip()
            if not text:
                continue

            start = self._to_milliseconds(
                self._segment_value(segment, "start", 0), duration_ms
            )
            end = self._to_milliseconds(
                self._segment_value(segment, "end", start), duration_ms
            )

            start = max(0, min(start, max_end - 1))
            end = max(start + 1, min(end, max_end))
            ret.append(SubtitleEvent(start=start, end=end, text=text))

        ret.sort(key=lambda event: event.start)

        normalized: list[SubtitleEvent] = []
        prev_end = 0
        for event in ret:
            if prev_end >= max_end:
                break

            if event.start < prev_end:
                event.start = prev_end
            if event.start >= max_end:
                break

            if event.end <= event.start:
                event.end = min(max_end, event.start + 1)
            if event.end <= event.start:
                continue

            normalized.append(event)
            prev_end = event.end

        return normalized

    def _generate_subtitles(self, duration: int, file_ref: Any, temp_adj: float = 0.0):
        return gemini_engine.generate_subtitles(self, duration, file_ref, temp_adj)

    def _generate_local_subtitles(
        self, duration_ms: int, audio_path: str, temp_adj: float = 0.0
    ) -> list[SubtitleEvent]:
        if self.engine == "whisper-mlx":
            return whisper_mlx_engine.generate_subtitles(
                self, duration_ms, audio_path, temp_adj
            )

        return voxtral_engine.generate_subtitles(
            self, duration_ms, audio_path, temp_adj
        )

    def _audio_to_subtitles(
        self, audio_segment: AudioSegment, file_ref: Any
    ) -> list[SubtitleEvent]:
        subtitle_events = []

        if self.engine == "gemini":
            temp_adj = 0.0
            for attempt in Retrying(
                retry=retry_if_exception(self._subtitles_retry_handler),
                stop=stop_after_attempt(30),
                before_sleep=before_sleep_log(logger, logging.DEBUG),
            ):
                with attempt:
                    duration = int(audio_segment.duration_seconds)
                    cur_temp_adj = temp_adj
                    temp_adj += self.temperature_adj

                    subtitle_events = self._generate_subtitles(
                        duration, file_ref, cur_temp_adj
                    )
                    validate_subtitles(subtitle_events, duration)
                    logger.debug("Valid subtitles generated for segment")

            return subtitle_events

        temp_adj = 0.0
        duration_seconds = audio_segment.duration_seconds
        duration_ms = int(duration_seconds * 1000)

        for attempt in Retrying(
            retry=retry_if_exception(self._subtitles_retry_handler),
            stop=stop_after_attempt(30),
            before_sleep=before_sleep_log(logger, logging.DEBUG),
        ):
            with attempt:
                for generation_attempt in Retrying(
                    retry=retry_if_exception(self._ai_retry_handler),
                    wait=self._wait_strategy(),
                    stop=stop_after_attempt(5),
                    before_sleep=before_sleep_log(logger, logging.DEBUG),
                    reraise=True,
                ):
                    with generation_attempt:
                        cur_temp_adj = temp_adj
                        temp_adj += self.temperature_adj
                        subtitle_events = self._generate_local_subtitles(
                            duration_ms, file_ref, cur_temp_adj
                        )

                validate_subtitles(subtitle_events, duration_seconds)
                logger.debug("Valid subtitles generated for segment")

        return subtitle_events

    def transcribe_audio(self, audio_segment: AudioSegment) -> list[SubtitleEvent]:
        with self.upload_audio(audio_segment) as file_ref:
            segment_dur = precisedelta(int(audio_segment.duration_seconds))
            logger.debug(f"Transcribing audio segment of {segment_dur}")
            subtitle_events = self._audio_to_subtitles(audio_segment, file_ref)

        return subtitle_events


__all__ = [
    "AISubtitler",
    "AIGenerationError",
    "SUPPORTED_ENGINES",
    "OperationMetrics",
    "WaitExponentialOrServerDelay",
    "_is_recoverable_exception",
    "_wait_api_limit",
]
