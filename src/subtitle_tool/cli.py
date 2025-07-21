import logging
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import click
from humanize.time import precisedelta
from pydub import AudioSegment

from subtitle_tool.ai import AISubtitler
from subtitle_tool.audio import AudioSplitter
from subtitle_tool.subtitles import (
    equalize_subtitles,
    events_to_subtitles,
    merge_subtitle_events,
)
from subtitle_tool.video import VideoProcessingError, extract_audio

API_KEY_NAME = "GEMINI_API_KEY"
AI_DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"


def setup_logging(verbose=False, debug=False):
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(threadName)s] "
        + "%(filename)s:%(lineno)d:%(funcName)s(): %(message)s",
        datefmt="%H:%M:%S",
    )

    # Setup handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    if debug:
        # Debug flag: enable DEBUG for everything (root level)
        root_logger.setLevel(logging.DEBUG)
    elif verbose:
        # Verbose flag: enable DEBUG only for subtitle_tool loggers
        root_logger.setLevel(logging.ERROR)

        # Set DEBUG level for all subtitle_tool loggers
        subtitle_logger = logging.getLogger("subtitle_tool")
        subtitle_logger.setLevel(logging.DEBUG)
    else:
        # Normal operation
        root_logger.setLevel(logging.ERROR)


@click.command()
@click.option(
    "--api-key",
    envvar=API_KEY_NAME,
    type=click.STRING,
    help="Google Gemini API key",
)
@click.option(
    "-m",
    "--ai-model",
    type=click.STRING,
    default=AI_DEFAULT_MODEL,
    help="Gemini model to use",
)
@click.option(
    "-vf",
    "--video",
    help="Path to video file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-af",
    "--audio",
    help="Path to audio file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-s",
    "--subtitle-path",
    help="Path to save subtitles",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable debug logging for subtitle_tool modules",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    help="Enable debug logging for all modules",
)
@click.option(
    "-k",
    "--keep-temp-files",
    is_flag=True,
    help="Do not erase temporary files",
)
@click.option(
    "-l",
    "--audio-segment-length",
    type=click.INT,
    help="Length of audio segments to be subtitled in seconds (default: 30)",
    default=30,
)
@click.option(
    "-p",
    "--parallel-segments",
    help="Number of segments subtitled in parallel",
    type=click.INT,
    default=5,
)
def main(
    api_key: str,
    ai_model: str,
    video: Path,
    audio: Path,
    subtitle_path: str,
    verbose: bool,
    debug: bool,
    keep_temp_files: bool,
    audio_segment_length: int,
    parallel_segments: int,
) -> None:
    setup_logging(debug=debug, verbose=verbose)

    start = time.time()

    if not api_key:
        raise click.MissingParameter(
            "API key not informed with --api-key or not present "
            + "in the environment variable {API_KEY_NAME}"
        )

    if not audio and not video or audio and video:
        raise click.MissingParameter("Either --video or --audio need to be specified")

    click.echo(f"Generating subtitles for {video if video else audio}")

    executor = None

    try:
        # 1. Load audio stream from either video or audio file
        media_path = video if video else audio
        try:
            audio_stream = (
                extract_audio(video) if video else AudioSegment.from_file(audio)
            )
        except VideoProcessingError as e:
            raise click.ClickException(f"Error loading audio stream: {e}") from e
        click.echo(f"Audio loaded ({precisedelta(int(audio_stream.duration_seconds))})")

        # 2. Split the audio stream into 30-second segments
        click.echo(
            f"Segmenting audio stream in {audio_segment_length} "
            + f"{'second' if audio_segment_length <= 1 else 'seconds'} chunks..."
        )
        segments = AudioSplitter().split_audio(
            audio_stream, segment_length=audio_segment_length
        )
        click.echo(f"Audio split into {len(segments)} segments")

        # 3. Ask Gemini to create subtitles
        click.echo(f"Generating subtitles with {ai_model}...")

        subtitler = AISubtitler(
            api_key=api_key, model_name=ai_model, delete_temp_files=not keep_temp_files
        )

        executor = ThreadPoolExecutor(max_workers=parallel_segments)
        try:
            subtitle_groups = list(executor.map(subtitler.transcribe_audio, segments))
        except (KeyboardInterrupt, click.Abort) as e:
            click.echo("Control-C pressed, shutting down processing")
            executor.shutdown(wait=False, cancel_futures=True)
            raise click.Abort() from e

        # 4. Join all subtitles into a single one
        segment_durations = [segment.duration_seconds * 1000 for segment in segments]
        subtitle_events = merge_subtitle_events(subtitle_groups, segment_durations)

        # 5. Convert subtitle events into subtitle file
        ai_subtitles = events_to_subtitles(subtitle_events)
        subtitles = equalize_subtitles(ai_subtitles)
        click.echo("New subtitle adjusted for viewing")

        # 6. Backup existing subtitle (if exists)
        if not subtitle_path:
            subtitle_path = f"{media_path.parent}/{media_path.stem}.srt"

        if Path(subtitle_path).exists():
            dst = f"{subtitle_path}.bak"
            shutil.move(subtitle_path, dst)
            click.echo(f"Existing subtitle backed up to {dst}")

        # 7. Write AI response
        with open(subtitle_path, "w") as f:
            subtitles.to_file(f, "srt")

        # 8. Output processing info
        end = time.time()
        duration = timedelta(seconds=round(end - start, 2))
        metrics = subtitler.metrics

        click.echo(f"Subtitle generation retries: {metrics.invalid_subtitles}")
        click.echo(
            f"AI tokens used: {metrics.input_token_count} "
            + f"input / {metrics.output_token_count} output"
        )
        click.echo(
            f"AI errors: {metrics.client_errors} client / {metrics.server_errors} "
            + f"server / {metrics.generation_errors} generation"
        )
        click.echo(
            f"AI calls: {metrics.throttles} throttled / {metrics.retries} retried"
        )
        click.echo(f"Processing time: {precisedelta(duration)}")
        click.echo(f"Subtitles saved at {subtitle_path}")

    except click.ClickException:
        # Re-raise them for click to handle
        raise
    except Exception as e:
        click.echo(f"Internal error: {e!r}", err=True)
        click.echo(traceback.format_exc())
        if executor:
            click.echo("Force-stopping all transcription tasks...", nl=False)
            executor.shutdown(wait=True, cancel_futures=True)
            click.echo("done.")

        sys.exit(1)
