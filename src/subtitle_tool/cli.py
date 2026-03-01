import logging
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import click
from humanize.time import precisedelta

from subtitle_tool.ai import SUPPORTED_ENGINES, AISubtitler
from subtitle_tool.audio import AudioExtractionError, AudioSplitter, extract_audio
from subtitle_tool.subtitles import (
    equalize_subtitles,
    events_to_subtitles,
    merge_subtitle_events,
)

GEMINI_API_KEY_NAME = "GEMINI_API_KEY"
API_KEY_NAME = GEMINI_API_KEY_NAME
ENGINE_DEFAULT_MODEL = {
    "gemini": "gemini-2.5-flash",
    "voxtral": "mistralai/Voxtral-Mini-3B-2507",
    "whisper-mlx": "mlx-community/whisper-large-v3-turbo",
}


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
        root_logger.setLevel(logging.DEBUG)
    elif verbose:
        root_logger.setLevel(logging.ERROR)
        subtitle_logger = logging.getLogger("subtitle_tool")
        subtitle_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.ERROR)


@click.command()
@click.argument(
    "mediafile",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "--api-key",
    envvar=GEMINI_API_KEY_NAME,
    type=click.STRING,
    help="Google Gemini API key (required only for engine=gemini)",
)
@click.option(
    "--engine",
    type=click.Choice(SUPPORTED_ENGINES),
    default="gemini",
    show_default=True,
    help="Transcription engine to use",
)
@click.option(
    "--gemini",
    is_flag=True,
    default=False,
    help="Shortcut for --engine gemini",
)
@click.option(
    "-m",
    "--ai-model",
    type=click.STRING,
    default=None,
    help="Model to use (defaults depend on --engine)",
)
@click.option(
    "-s",
    "--subtitle-path",
    help="Subtitle file name [default: MEDIAFILE.srt]",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging for subtitle_tool modules",
    show_default=True,
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging for all modules",
    show_default=True,
)
@click.option(
    "-k",
    "--keep-temp-files",
    is_flag=True,
    default=False,
    help="Do not erase temporary files",
    show_default=True,
)
@click.option(
    "-l",
    "--audio-segment-length",
    type=click.INT,
    help="Length of audio segments to be subtitled in seconds",
    default=30,
    show_default=True,
)
@click.option(
    "-p",
    "--parallel-segments",
    help="Number of segments subtitled in parallel",
    type=click.INT,
    default=5,
    show_default=True,
)
@click.option(
    "--media-lang", help="Media file language", type=click.STRING, default="English"
)
@click.option("--subtitle-lang", help="Language for subtitle output", type=click.STRING)
def main(
    mediafile: Path,
    api_key: str,
    engine: str,
    gemini: bool,
    ai_model: str | None,
    subtitle_path: str,
    verbose: bool,
    debug: bool,
    keep_temp_files: bool,
    audio_segment_length: int,
    parallel_segments: int,
    media_lang: str,
    subtitle_lang: str,
) -> None:
    """Generate subtitles for a media file"""

    setup_logging(debug=debug, verbose=verbose)

    start = time.time()

    selected_engine = "gemini" if gemini else engine
    selected_model = ai_model or ENGINE_DEFAULT_MODEL[selected_engine]

    if selected_engine == "gemini" and not api_key:
        raise click.MissingParameter(
            "API key not informed with --api-key or not present "
            + f"in the environment variable {GEMINI_API_KEY_NAME}"
        )

    click.echo(f"Generating subtitles for {mediafile}")

    executor = None

    try:
        try:
            audio_stream = extract_audio(f"{mediafile}")
        except AudioExtractionError as e:
            raise click.ClickException(f"Error loading audio stream: {e}") from e
        click.echo(f"Audio loaded ({precisedelta(int(audio_stream.duration_seconds))})")

        click.echo(
            f"Segmenting audio stream in {audio_segment_length} "
            + f"{'second' if audio_segment_length <= 1 else 'seconds'} chunks..."
        )
        segments = AudioSplitter().split_audio(
            audio_stream, segment_length=audio_segment_length
        )
        click.echo(f"Audio split into {len(segments)} segments")

        click.echo(
            f"Generating subtitles with {selected_model} ({selected_engine})..."
        )

        subtitler = AISubtitler(
            api_key=api_key,
            model_name=selected_model,
            delete_temp_files=not keep_temp_files,
            media_lang=media_lang,
            subtitle_lang=subtitle_lang,
            engine=selected_engine,
        )

        executor = ThreadPoolExecutor(max_workers=parallel_segments)
        try:
            subtitle_groups = list(
                executor.map(
                    lambda segment, idx: setattr(
                        threading.current_thread(), "name", f"segment-{idx}"
                    )
                    or subtitler.transcribe_audio(segment),
                    segments,
                    range(len(segments)),
                )
            )

        except (KeyboardInterrupt, click.Abort) as e:
            click.echo("Control-C pressed, shutting down processing")
            executor.shutdown(wait=False, cancel_futures=True)
            raise click.Abort() from e

        segment_durations = [segment.duration_seconds * 1000 for segment in segments]
        subtitle_events = merge_subtitle_events(subtitle_groups, segment_durations)

        ai_subtitles = events_to_subtitles(subtitle_events)
        subtitles = equalize_subtitles(ai_subtitles)
        click.echo("New subtitle adjusted for viewing")

        if not subtitle_path:
            subtitle_path = f"{mediafile.parent}/{mediafile.stem}.srt"

        if Path(subtitle_path).exists():
            dst = f"{subtitle_path}.bak"
            shutil.move(subtitle_path, dst)
            click.echo(f"Existing subtitle backed up to {dst}")

        with open(subtitle_path, "w") as f:
            subtitles.to_file(f, "srt")

        end = time.time()
        duration = timedelta(seconds=round(end - start, 2))
        metrics = subtitler.metrics

        click.echo(f"Audio segments processed: {len(segments)}")
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
        raise
    except Exception as e:
        click.echo(f"Internal error: {e!r}", err=True)
        click.echo(traceback.format_exc())
        if executor:
            click.echo("Force-stopping all transcription tasks...", nl=False)
            executor.shutdown(wait=True, cancel_futures=True)
            click.echo("done.")

        sys.exit(1)
