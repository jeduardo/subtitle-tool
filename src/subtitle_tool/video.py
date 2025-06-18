import io
import logging

import ffmpeg
from pydub import AudioSegment

logger = logging.getLogger("subtitle_tool.video")


class VideoProcessingError(Exception):
    pass


def extract_audio(video_path: str) -> AudioSegment:
    """
    Extract an audio stream from the video file.
    This method will always extract the main audio stream.
    The audio stream will be extracted to wav as it's the closest from
    the PCM format that pydubs uses internally.

    Args:
        video_path (str): path to the video file

    Returns:
        AudioSegment: in-memory representation of the extracted audio stream.
    """
    if not video_path:
        raise VideoProcessingError("Path to video file is mandatory")

    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        logger.error(f"Error probing for file metadata: {e}")
        raise VideoProcessingError("Error probing for file metadata") from e

    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    if not audio_streams:
        raise VideoProcessingError(f"No audio streams found in file {video_path}")

    audio_stream = audio_streams[0]
    audio_codec = audio_stream.get("codec_name", "")
    logger.info(f"Audio stream detected: {audio_codec}")

    # Extract audio
    logger.debug(f"Converting {audio_codec} to wav")
    audio_format = "wav"
    process = (
        ffmpeg.input(video_path)
        .output("pipe:", format=audio_format, acodec="pcm_s16le")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    out, err = process.communicate()

    # Check the return code instead of stderr content
    if process.returncode != 0:
        logger.error(f"Extraction error with ffmpeg: {err.decode()}")
        raise VideoProcessingError(f"Extraction error: {err.decode()}")

    audio_buffer = io.BytesIO(out)

    # Convert audio to AudioSegment
    audio_buffer.seek(0)
    return AudioSegment.from_file(audio_buffer, format=audio_format)
