#!/usr/bin/env python3

import click
import logging
import shutil
import time
import os
import sys

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from humanize.time import naturaldelta, precisedelta
from pathlib import Path
from subtitle_tool.ai import AISubtitler
from subtitle_tool.audio import split_audio
from subtitle_tool.subtitles import events_to_subtitles, merge_subtitle_events
from subtitle_tool.video import extract_audio
from pydub import AudioSegment

API_KEY_NAME = "GEMINI_API_KEY"
AI_DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"


def setup_logging(verbose=False, debug=False):
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(threadName)s] %(filename)s:%(lineno)d:%(funcName)s(): %(message)s",
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
    help="Google Gemini API key",
)
@click.option(
    "--ai-model",
    default=AI_DEFAULT_MODEL,
    help=f"Gemini model to use (default {AI_DEFAULT_MODEL}",
)
@click.option("--video", help="Path to video file")
@click.option("--audio", help="Path to audio file")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging for subtitle_tool modules",
)
@click.option("--debug", is_flag=True, help="Enable debug logging for all modules")
@click.option("--keep-temp-files", is_flag=True, help="Do not erase temporary files")
@click.pass_context
def main(
    ctx: click.Context,
    api_key: str,
    ai_model: str,
    video: str,
    audio: str,
    verbose: bool,
    debug: bool,
    keep_temp_files: bool = False,
) -> None:
    setup_logging(debug=debug, verbose=verbose)

    start = time.time()
    executor = None
    completed = False

    def cleanup():
        if not completed:
            click.echo("\nForce killing all tasks...")
            if executor:
                # Don't wait for stuck tasks - just shutdown immediately
                executor.shutdown(wait=False, cancel_futures=True)

            os._exit(-1)

    # Register cleanup function
    ctx.call_on_close(cleanup)

    if not api_key:
        raise click.ClickException(
            f"API key not informed or not present in the environment variable {API_KEY_NAME}"
        )

    if not audio and not video or audio and video:
        raise click.ClickException(f"Either --video or --audio need to be specified")

    click.echo(f"Generating subtitle for {video if video else audio}")

    # 1. Load audio stream from either video or audio file
    media_path = Path(video) if video else Path(audio)
    if not media_path.exists():
        raise click.ClickException(f"{media_path} does not exist")
    if not media_path.is_file()
        raise click.ClickException(f"{media_path} is not a file")

    try:
        audio_stream = extract_audio(video) if video else AudioSegment.from_file(audio)
    except Exception as e:
        raise click.ClickException(f"Error loading audio stream: {e}")
    click.echo(f"Audio loaded ({precisedelta(int(audio_stream.duration_seconds))})")

    # 2. Split the audio stream into 30-second segments
    click.echo("Segmenting audio stream...")
    segments = split_audio(audio_stream, segment_length=30)
    click.echo(f"Audio split into {len(segments)} segments")

    # 3. Ask Gemini to create subtitles
    click.echo(f"Generating subtitles with {ai_model}...")

    gemini = AISubtitler(
        api_key=api_key, model_name=ai_model, delete_temp_files=not keep_temp_files
    )
    executor = ThreadPoolExecutor(max_workers=5)
    subtitle_groups = list(executor.map(gemini.transcribe_audio, segments))

    # 4. Join all subtitles into a single one
    segment_durations = [segment.duration_seconds * 1000 for segment in segments]
    subtitle_events = merge_subtitle_events(subtitle_groups, segment_durations)

    # 5. Convert subtitle events into subtitle file
    subtitles = events_to_subtitles(subtitle_events)

    # 6. Backup existing subtitle (if exists)
    subtitle_path = Path(f"{media_path.parent}/{media_path.stem}.srt")

    if subtitle_path.exists():
        dst = f"{subtitle_path}.bak"
        shutil.move(subtitle_path, dst)
        click.echo(f"Existing subtitle backed up to {dst}")

    # 7. Write AI response
    with open(subtitle_path, "w") as f:
        subtitles.to_file(f, "srt")

    # 8. Output processing info
    end = time.time()
    duration = timedelta(seconds=round(end - start, 2))
    click.echo(
        f"Subtitle saved at {subtitle_path} (Processed for {naturaldelta(duration)})"
    )
    completed = True


if __name__ == "__main__":
    main()
