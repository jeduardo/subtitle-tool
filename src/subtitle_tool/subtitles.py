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
