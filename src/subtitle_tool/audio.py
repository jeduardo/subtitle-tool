import logging

from humanize.time import precisedelta
from pydub import AudioSegment, silence

logger = logging.getLogger("subtitle_tool.audio")


def split_audio(
    audio_clip: AudioSegment, segment_length: int = 30, keep_silence: bool = True
) -> list[AudioSegment]:
    """
    Splits an audio file into segments based on silence.

    Args:
        audio_clip (AudioSegment): Audio clip to be split
        segment_length (int): Audio segment length in seconds (default: 30)
        keep_silence (bool): Whether silence should be kept in the seguments (default: True)

    Returns:
        list[AudioSegment]: List of segments from the audio file.
    """

    chunks: list[AudioSegment] = silence.split_on_silence(
        audio_clip,
        min_silence_len=200,  # 200ms of silence is considered silence
        silence_thresh=-40,  # -40dB of silence
        keep_silence=keep_silence,  # keep silence in the chunks
    )  # type: ignore
    logging.debug(f"Extracted a total of {len(chunks)} chunks")

    # Creating a new segment group with the top segment empty
    cur_segment = AudioSegment.silent(duration=0)
    segments = []
    for chunk in chunks:
        # Current segment is the last in the list
        segment_dur = cur_segment.duration_seconds
        chunk_dur = chunk.duration_seconds

        if segment_dur + chunk_dur < segment_length:
            logger.debug(f"Adding chunk ({chunk_dur}) to segment ({segment_dur})")
            cur_segment += chunk
        else:
            logging.debug(
                f"Adding chunk ({chunk_dur}) overflows the segment ({segment_dur})"
            )
            segments.append(cur_segment)
            cur_segment = chunk
            logger.debug(f"Most recent segment is new chunk ({chunk_dur})")
    # Add the cur_segment to the list to complete the pass
    segments.append(cur_segment)

    segments_length = sum(group.duration_seconds for group in segments)
    logger.debug(f"Grouped segments {len(segments)}")
    logger.debug(
        f"Segments playtime: {segments_length} ({precisedelta(segments_length)})"
    )

    return segments
