import json
import logging
import re

from functools import reduce
from humanize.time import precisedelta
from pydantic import BaseModel, Field, ConfigDict
from pysubs2 import SSAFile, SSAEvent

logger = logging.getLogger("subtitle_tool.subtitles")


class SubtitleEvent(BaseModel):
    start: int = Field(
        description="Individual subtitle start in millseconds from the video start"
    )
    end: int = Field(
        description="Individual subtitle end in millseconds from the video start"
    )
    text: str = Field(description="Lines of text spoken during the individual subtitle")
    model_config = ConfigDict(title="Individual subtitle text")


def subtitles_to_events(subtitle: SSAFile) -> list[SubtitleEvent]:
    """
    Return a list of SubtitleEvent from a subtitle file.

    Args:
        subtitle: SSAFile: parsed subtitle file

    Returns:
        List[SubtitleEvent]: list of formatted events
    """
    return [
        SubtitleEvent(start=obj.start, end=obj.end, text=obj.text)
        for obj in subtitle.events
    ]


def subtitles_to_dict(subtitle: SSAFile) -> list[dict]:
    """
    Return a dict from a subtitle file.

    Args:
        subtitle: SSAFile: parsed subtitle file

    Returns:
        dict]: list of formatted events
    """
    return [
        {"start": obj.start, "end": obj.end, "text": obj.text}
        for obj in subtitle.events
    ]


def events_to_subtitles(events: list[SubtitleEvent]) -> SSAFile:
    """
    Return a SSAFile from a list of SubtitleEvent

    Args:
        events: List[SubtitleEvent]: list of subtitle events

    Returns:
        SSAFile: subtitle representation
    """
    subtitle = SSAFile()
    subtitle.events = [
        SSAEvent(start=obj.start, end=obj.end, text=obj.text) for obj in events
    ]
    return subtitle


def validate_subtitles(subtitles: list[SubtitleEvent], duration: float):
    """
    Check whether a group of subtitles is valid.
    The subtitles will be valid when all segments do not overlap and
    when the last subtitle does not exceed the duration of the segment.

    Args:
        subtitles (list[SubtitleEvent]): subtitle group
        duration (float): duration of the segment in seconds

    Returns:
        Exception: This method will return an exception if the subtitle is invalid.
    """

    if subtitles[-1].end > (duration * 1000):
        raise Exception(
            f"Subtitle ends at {subtitles[-1].end} ({precisedelta(int(subtitles[-1].end / 1000))}) while audio segment ends at {duration * 1000} ({precisedelta(int(duration))})"
        )

    prev_end = 0
    for index, event in enumerate(subtitles):
        if event.start > event.end:
            raise Exception(
                f"Subtitle {index} starts at {event.start} ({precisedelta(int(event.start / 1000))}) but ends at {event.end} ({precisedelta(int(event.end / 1000))})"
            )

        if index == 0:
            prev_end = event.end
            continue

        if event.start < prev_end:
            raise Exception(
                f"Subtitle {index} starts at {event.start} (({precisedelta(int(event.start / 1000))})) but the previous subtitle finishes at {prev_end} (({precisedelta(int(prev_end / 1000))}))"
            )

        prev_end = event.end


def save_to_json(subtitles: list[SubtitleEvent], name):
    with open(name, "w") as f:
        f.write(json.dumps(subtitles_to_dict(events_to_subtitles(subtitles))))


def merge_subtitle_events(
    subtitle_groups: list[list[SubtitleEvent]], segment_durations: list[float]
) -> list[SubtitleEvent]:
    """
    Join several groups of subtitle events into a single stream of events,
    adjusting the timestamps.

    Args:
        subtitle_groups (list[list[SubtitleEvent]]): groups of subtitles
        segment_durations (list[float]): how long each segment lasts (in seconds)

    Returns:
        list[SubtitleEvent]: merge subtitle stream
    """
    time_shift = 0
    all_events = []
    total_duration = reduce(lambda x, y: x + y, segment_durations)
    for index, events in enumerate(subtitle_groups):
        duration = segment_durations[index]
        # Adjust for timeshift, loop once
        for event in events:
            event.start += time_shift
            event.end += time_shift
            all_events.append(event)
        # Accumulating time played for current segment for next start time
        time_shift += int(duration)

    validate_subtitles(all_events, total_duration)

    return all_events


class SubtitleBalancer:
    def __init__(self, max_words_per_screen=12, min_duration_ms=1000):
        """
        Initialize the subtitle balancer.

        Args:
            max_words_per_screen (int): Maximum words to display per subtitle
            min_duration_ms (int): Minimum duration for each subtitle in milliseconds
        """
        self.max_words_per_screen = max_words_per_screen
        self.min_duration_ms = min_duration_ms

    def count_words(self, text: str) -> int:
        """Count words in subtitle text, ignoring HTML tags."""
        # Remove HTML tags and formatting
        clean_text = re.sub(r"<[^>]+>", "", text)
        # Split by whitespace and filter empty strings
        words = [word for word in clean_text.split() if word.strip()]
        return len(words)

    def split_text_smartly(self, text: str, max_words: int) -> list[str]:
        """Split text into chunks with smart line breaking."""
        lines = text.split("\\N")  # pysubs2 uses \\N for line breaks
        chunks = []
        current_chunk = []
        current_word_count = 0

        for line in lines:
            line_words = len([w for w in line.split() if w.strip()])

            # If adding this line would exceed the limit, start a new chunk
            if current_word_count + line_words > max_words and current_chunk:
                chunks.append("\\N".join(current_chunk))
                current_chunk = [line]
                current_word_count = line_words
            else:
                current_chunk.append(line)
                current_word_count += line_words

        # Add the last chunk
        if current_chunk:
            chunks.append("\\N".join(current_chunk))

        # If chunks are still too long, split by words
        final_chunks = []
        for chunk in chunks:
            if self.count_words(chunk) <= max_words:
                final_chunks.append(chunk)
            else:
                # Split by words
                words = chunk.replace("\\N", " ").split()
                temp_chunk = []

                for word in words:
                    if len(temp_chunk) + 1 > max_words and temp_chunk:
                        final_chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                    else:
                        temp_chunk.append(word)

                if temp_chunk:
                    final_chunks.append(" ".join(temp_chunk))

        return final_chunks

    def balance_subtitles(self, subs: SSAFile) -> SSAFile:
        """Balance subtitles by splitting long entries."""
        new_subs = SSAFile()

        for line in subs:
            word_count = self.count_words(line.text)

            if word_count <= self.max_words_per_screen:
                # Keep as is
                new_subs.append(line)
            else:
                # Split the subtitle
                text_chunks = self.split_text_smartly(
                    line.text, self.max_words_per_screen
                )

                if len(text_chunks) == 1:
                    # Couldn't split effectively, keep as is
                    new_subs.append(line)
                else:
                    # Calculate timing for each chunk
                    total_duration = line.end - line.start
                    chunk_word_counts = [
                        self.count_words(chunk) for chunk in text_chunks
                    ]
                    total_words = sum(chunk_word_counts)

                    current_start = line.start

                    for i, (chunk_text, chunk_words) in enumerate(
                        zip(text_chunks, chunk_word_counts)
                    ):
                        # Calculate duration proportional to word count
                        if i == len(text_chunks) - 1:
                            # Last chunk gets remaining time
                            chunk_end = line.end
                        else:
                            duration_ratio = chunk_words / total_words
                            chunk_duration = max(
                                int(total_duration * duration_ratio),
                                self.min_duration_ms,
                            )
                            chunk_end = current_start + chunk_duration

                        # Ensure minimum duration
                        if chunk_end - current_start < self.min_duration_ms:
                            chunk_end = current_start + self.min_duration_ms

                        # Create new subtitle line
                        new_line = SSAEvent(
                            start=current_start, end=chunk_end, text=chunk_text
                        )
                        new_subs.append(new_line)

                        current_start = chunk_end

        return new_subs
