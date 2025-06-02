import io
import logging
import tempfile

from contextlib import contextmanager
from dataclasses import dataclass
from google import genai
from google.genai import types
from pydub import AudioSegment
from pysubs2 import SSAFile
from subtitle_tool.subtitles import SubtitleEvent, subtitles_to_dict, validate_subtitles
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential
from warnings import deprecated

logger = logging.getLogger("subtitle_tool.ai")


@dataclass
class Gemini(object):
    model_name: str
    api_key: str
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
                "text": "For years, he tracks her killer\\Nwithout success."
            },
            {
                "start": 12000,
                "end": 14500,
                "text": "Every day was another blow to the stomach."
            },
            {
                "start": 14900,
                "end": 18842,
                "text": "Somewhere deep in the Houston crime files\\Nare the secrets to solve"
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
                "text": "Houston had 500,000 prints. Everybody has 10 fingers.\\NThat's 5 million prints."
            },
            {
                "start": 27300,
                "end": 30800,
                "text": "34 years later, investigators\\Nfind the answer."
            },
            {
                "start": 31500,
                "end": 33500,
                "text": "I want to know who killed Diane."
            }
        ]

        """

    def __post_init__(self):
        self.client = genai.Client(api_key=self.api_key)

    @contextmanager
    def upload_audio(self, segment: AudioSegment):
        """
        Context to upload and remove a file from Gemini servers

        Arguments:
            segment (AudioSegment): segment representation

        Returns:
            File: upload identifier
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            # Export AudioSegment to temporary file
            # AudioSegment will be loaded as RAW audio, so we can export it to
            # whatever format we want. We will choose MP3.
            logger.debug(f"Temporary file created at {temp_file.name}")

            segment.export(temp_file.name, format="mp3")
            logger.debug(f"Audio segment exported to {temp_file.name}")

            # Upload the temporary file (API will infer mime type from extension)
            ref = self.client.files.upload(file=temp_file.name)  # type: ignore
            logger.debug(f"Temporary file uploaded as {ref.name}")
            try:
                yield ref
            finally:
                self.client.files.delete(name=f"{ref.name}")
                logger.debug(f"Removed temporary file upload {ref.name}")

    @retry(
        stop=stop_after_attempt(50),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _audio_to_subtitles(
        self, audio_segment: AudioSegment, file_ref
    ) -> list[SubtitleEvent]:
        """
        Generate subtitles for an audio segment.

        This function will call Gemini to generate subtitles and will
        validate the result before returning. If the subtitle is invalid,
        it will ask Gemini to recreate the subtitles up to 50 times.

        Args:
            audio_segment (AudioSegment): segment to be transcribed
            file_ref (types.File): reference to uploaded file

        Returns:
            list[SubtitleEvent]: subtitles extracted from audio track

        """
        subtitle_events = self._generate_subtitles(file_ref)
        validate_subtitles(subtitle_events, audio_segment.duration_seconds)
        logger.debug("Valid subtitles generated for segment")
        return subtitle_events

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=300),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_subtitles(self, file_ref) -> list[SubtitleEvent]:
        """
        Generate subtitles for the file uploaded onto Gemini servers.
        Given that sometimes Gemini will not generate a valid output, or
        that some limits might be surpassed, this function will retry
        for up to 5 minutes with exponential backoff.

        Args:
            file_id (str): identifier of uploaded file

        Returns:
            list[SubtitleEvent]: subtitles extracted from audio track
        """

        logger.debug("Asking Gemini to generate subtitles...")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=["Create subtitles for this audio file", file_ref],
            config=types.GenerateContentConfig(
                # Don't want to censor any subtitles
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
                system_instruction=self.system_prompt,
                temperature=0.1,
                http_options=types.HttpOptions(timeout=2 * 60 * 1000),  # 2 minutes
                response_mime_type="application/json",
                response_schema=list[SubtitleEvent],
            ),
        )

        logger.debug(f"Cached token info: {response.usage_metadata.cache_tokens_details}")  # type: ignore
        logger.debug(f"Cached token count: {response.usage_metadata.cached_content_token_count}")  # type: ignore
        logger.debug(f"Thoughts token count: {response.usage_metadata.thoughts_token_count}")  # type: ignore
        logger.debug(f"Output token count: {response.usage_metadata.candidates_token_count}")  # type: ignore

        return response.parsed  # type: ignore

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
        with self.upload_audio(audio_segment) as file_ref:  # type: ignore
            subtitle_events = self._audio_to_subtitles(audio_segment, file_ref)

        return subtitle_events
