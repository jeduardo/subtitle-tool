import ffmpeg
import io
import logging

from pydub import AudioSegment

# Mapping only the formats supported natively by Gemini
CODEC_TO_FORMAT = {
    "pcm_s16le": "wav",
    "pcm_s24le": "wav",
    "pcm_s32le": "wav",
    "pcm_f32le": "wav",
    "mp3": "mp3",
    "pcm_s16be": "aiff",
    "aac": "aac",
    "vorbis": "ogg",
    "flac": "flac",
}

logger = logging.getLogger("subtitle_tool.video")


def extract_audio(video_path: str) -> AudioSegment:
    """
    Extract an audio stream from the video file.
    This method will always extract the main audio stream.

    Args:
        video_path (str): path to the video file

    Returns:
        AudioSegment: in-memory representation of the extracted audio stream.
    """
    if not video_path:
        raise Exception("Path to video file is mandatory")

    probe = ffmpeg.probe(video_path)
    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]

    if not audio_streams:
        raise Exception("No audio streams found")

    audio_stream = audio_streams[0]
    audio_codec = audio_stream.get("codec_name", "")

    # Extract audio
    if audio_codec in CODEC_TO_FORMAT:
        audio_format = CODEC_TO_FORMAT[audio_codec]
        logger.debug(f"Copying {audio_codec} stream directly to {audio_format}")
        process = (
            ffmpeg.input(video_path)
            .output("pipe:", format=audio_format, acodec="copy")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
    else:
        logger.debug(f"Converting {audio_codec} to mp3")
        audio_format = "mp3"
        process = (
            ffmpeg.input(video_path)
            .output("pipe:", format=audio_format, acodec="mp3")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

    out, err = process.communicate()

    # Check the return code instead of stderr content
    if process.returncode != 0:
        raise Exception(f"Extraction error: {err.decode()}")

    audio_buffer = io.BytesIO(out)

    # Convert audio to AudioSegment
    audio_buffer.seek(0)
    return AudioSegment.from_file(audio_buffer, format=audio_format)
