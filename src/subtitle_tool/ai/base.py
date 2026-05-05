from typing import Protocol

from pydub import AudioSegment

from subtitle_tool.ai.metrics import OperationMetrics
from subtitle_tool.subtitles import SubtitleEvent


class Subtitler(Protocol):
    metrics: OperationMetrics

    def transcribe_audio(self, audio_segment: AudioSegment) -> list[SubtitleEvent]:
        """Transcribe an audio segment into subtitle events."""
        ...  # pragma: no cover
