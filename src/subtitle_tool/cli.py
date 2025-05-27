#!/usr/bin/env python3

import click
import datetime
import ffmpeg
import humanize.time
import json
import pysubs2
import re
import shutil
import time

from google import genai
from google.genai import types
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import List
from pysubs2 import SSAFile, SSAEvent
from tenacity import retry, stop_after_attempt


API_KEY_NAME = "GEMINI_API_KEY"
AI_DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"

# https://subtle-subtitlers.org.uk/wp-content/uploads/2023/01/SUBTLE-Recommended-Quality-Criteria-for-Subtitling.pdf
GUIDELINES = """
# Code of Good Practice in AVT: Recommended Quality Criteria For Subtitling (January 2023) © SUBTLE

## Recommended Quality Criteria for Subtitling

The following quality criteria have been compiled with the aim of establishing
a single basic set of European standards for interlingual subtitling. They are aimed
at stakeholders in the film, broadcast, and VOD industry, as well as any video creator
wishing to subtitle their content. The quality criteria contain guidance to help them
reach their audiences by giving them the best possible viewing experience.
These criteria are based on widely recognised academic references as well as the
consolidated professional expertise of Subtle’s membership. They shall serve as
a general common code that neither undermines nor annuls but complements the
already established conventions within the various countries (please see their list
at the end of this document).

## 1. THE PURPOSE OF SUBTITLING
Subtitles must convey the content and intent of an original work that would
otherwise be inaccessible or incomprehensible to a given audience.
The general practice of the production [...] of [TV] subtitles
should be guided by the aim to provide maximum appreciation
and comprehension of the target film as a whole by maximising
the legibility and readability of the inserted subtitled text.
Fotios Karamitroglou,
A Proposed Set of Subtitling Standards in Europe, 1998.

## 2. THE ELEMENTS OF SUBTITLING
Subtitles are defined as lines of written text reproducing language content
in films and other types of audiovisual media. Due to the multimodality
of these media, the message is expressed by means of verbal and nonverbal
signs and via two different channels of communication — audio and visual.
Therefore, the process of subtitling involves interpreting the combination
of these different modes while also requiring full awareness of the following
three key elements:

· Time
· Space
· Content

The timing, brevity, and content of the subtitles are factors that need to be
accounted for to convey the intent of the original work. If any of these aspects
are compromised, the viewer's enjoyment of the production will be
jeopardised. Successful subtitling enhances the overall experience for the
target audience. To achieve that, it must strike the right balance between wellsynced timing and precise dialogue editing, while taking care that subtitles are
displayed for long enough for viewers to read them and that their presence on
the screen maintains an even rhythm.

## 3. CRITERIA DETAILS

A lot of creative input, work, and money goes into all aspects of producing
a film, series, TV programme or other types of audiovisual product. Subtitles
are an added element and, consequently, certain rules must be observed
in order to ensure that they blend in with the original work as much as
possible.

### 3.1 Time

The viewer must be given enough time to take in the image, sound, and
subtitle as part of the flow of viewing.

#### Time-cueing guidelines

Subtitles should be timed to be in sync with the dialogue, avoid crossing shot
changes whenever possible, and adhere to the overall rhythm of the original.
- In-time: on or within 2-3 frames of the onset of speech. If the speech starts
within a few frames before/after a shot change, it’s preferable to cue in the
subtitle on the shot change.

- Out-time: with the end of the speech or a few frames (up to 1 sec if
necessary) thereafter, but not before. If speech ends within a few frames
before/after a shot change, the subtitle should preferably be cued out
before the shot change. 

- In order to stay in sync with the speech and/or to enhance readability,
subtitles may be displayed over a shot change if this is within the same
scene but must be cued out before a scene change. In exceptional cases,
e.g., when a sound bridge is present, subtitles can also cross scene
changes. Respecting the filmmaker’s choices should be prioritised
whenever possible.

- Apart from time-cueing subtitles with regard to editing, other
cinematography aspects must also be considered. Subtitles should be
appropriately timed to focus pulls, camera tilts and pans, etc.
Minimum gap between subtitles
Consecutive subtitles must be cued using a fixed interval to create a clear
distinction between them and to promote an even rhythm and pace. This
interval can be set to a minimum of 2 and a maximum of 6 frames depending
on the frame rate (recommended: 3-4 frames) and must be consistent
throughout the entire subtitle file. It is recommended to use the fixed interval
if the pause between two subtitles is less than 0.5 seconds unless any cuts
occur during this time.

#### Subtitle duration/reading speed

Context, target language, audio, and visual editing all have a bearing on the
amount of text one subtitle may or should contain. These factors should guide
the decision of how concise the subtitle should be and for how long it should
be on-screen for a comfortable reading speed.

Subtitles with a duration of less than 1 second should be avoided wherever
possible. The maximum duration of a subtitle should not exceed 6 seconds.
Exceptions may apply, e.g. to subtitled song lyrics and on-screen text.
Subtitle display rates are typically measured in characters per second (cps)
or words per minute (wpm). Depending on the target audience, language, type
of programme and medium, it is recommended to keep the average reading
speed to a maximum of 12 to 15 cps or 150-180 wpm, with maximum speeds
not exceeding 16-17 cps/190-200 wpm. Extra time should be given to subtitles
with very short duration; complex content, syntax, and vocabulary; sudden
change of position or format; and low contrast against the image.

### 3.2 Space

#### Layout & number of lines

Subtitles consist of 1 or 2 lines of text. In rare cases (e.g. tri-lingual subtitles)

3 lines can be used, but it’s generally not recommended to exceed 2 lines.
Characters per line

The length of each line varies a lot between countries and the medium used.
It typically ranges from 34 to 50 characters (incl. spaces).
Positioning

For languages based on the Latin alphabet, subtitles are generally positioned
at the bottom of the screen. The subtitles must be re-positioned or raised
to avoid overlapping with on-screen text, signs, the mouth of the speakers,
or other relevant imagery. However, rapidly alternating positioning should
be avoided.


### 3.3 Content

#### Dialogue editing

If the amount of text in the subtitle is too high in relation to its exposure time
on the screen, the audience may not have time to read it. Attaining an amount
of text that is consistent with the viewers' assumed reading speed (see above)
usually requires compressing the dialogue. When editing, special attention
must be paid to maintaining coherence and consistency – both at the
sentence level and throughout the whole subtitle file.

#### Language/register

The translation must recreate the message conveyed in the original work
as closely as possible, and the style and register of the target language should
be equivalent to those of the source language. The translation of culturespecific terms and expressions requires careful consideration of the source
and target cultures.
Translation, style, or grammatical errors, as well as lengthy, convoluted
sentences, risk distracting the viewer. Since subtitling is also seen as a
language learning tool, accuracy and readability are of utmost importance.
Idiomatic phrases and expressions should usually be used unless
a character’s unidiomatic language plays an important role. 

#### Numbers/measurements

Converted measurements and numbers should be rounded to the nearest
whole number, unless the exact figure is important in the context.
Segmentation/line breaks
Each subtitle must be semantically and grammatically self-contained,
especially if a sentence goes over two or more subtitles. Likewise, line breaks
should preferably occur after a semantic unit. Long chains of short subtitles
and sentences spanning three or more subtitles should be avoided whenever
possible.
When deciding where to divide the line within one subtitle, closely related
logical, syntactic, and semantic units should be kept together. This rule
supersedes the aesthetic consideration that favours bottom-heavy subtitles.
When multiple equally readable line break options are possible, it is usually
recommended to shape a subtitle like a pyramid or a rectangle while avoiding
short lines of one or two words.
Songs
Notwithstanding copyright issues, song lyrics should be subtitled if pertinent
to the plot. When possible, the rhyme and accent distribution should be
preserved.
#### Style guidelines

National subtitling standards for punctuation, formatting (e.g. italicization),
text alignment, continuity, dialogue indicators, text/border design, etc. must
always be followed.

#### Synchrony with source
The subtitle content should not contradict the dialogue, soundtrack, or image
(e.g. body language) or pre-empt any pieces of information before they are
uttered or are visible on screen.
"""


class SubtitleEvent(BaseModel):
    start: int = Field(
        description="Individual subtitle start in millseconds from the video start"
    )
    end: int = Field(
        description="Individual subtitle end in millseconds from the video start"
    )
    text: str = Field(description="Lines of text spoken during the individual subtitle")
    model_config = ConfigDict(title="Individual subtitle text")


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

    def split_text_smartly(self, text: str, max_words: int) -> List[str]:
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
        new_subs = pysubs2.SSAFile()

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
                        new_line = pysubs2.SSAEvent(
                            start=current_start, end=chunk_end, text=chunk_text
                        )
                        new_subs.append(new_line)

                        current_start = chunk_end

        return new_subs


def subtitles_to_events(subtitle: SSAFile) -> List[SubtitleEvent]:
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


def subtitles_to_dict(subtitle: SSAFile) -> List[dict]:
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


def events_to_subtitles(events: List[SubtitleEvent]) -> SSAFile:
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


def get_codec_file_ending(video: str) -> str:
    """
    Return the file ending for the codec.

    Args:
        video (str): path to the video file

    Returns:
        str: path to the extracted audio file
    """

    if not video:
        raise click.ClickException("Path to video file is mandatory")

    try:
        metadata = ffmpeg.probe(video)
        audio_streams = [
            stream for stream in metadata["streams"] if stream["codec_type"] == "audio"
        ]
        if not audio_streams:
            raise click.ClickException("No audio stream found")

        return audio_streams[0]["codec_name"]
    except ffmpeg.Error as e:
        raise click.ClickException(f"ffmpeg error: {e}")


def extract_audio(video: str) -> str:
    """
    Extract an audio file from the video file.
    Equivalent to the command: ffmpeg -i input-video.avi -vn -acodec copy output-audio.aac

    Args:
        video (str): path to the video file

    Returns:
        str: path to the extracted audio file
    """
    if not video:
        raise click.ClickException("Path to video file is mandatory")

    codec = get_codec_file_ending(video)
    output = Path(f"{Path(video).stem}.{codec}")
    if output.exists():
        output.unlink()
    try:
        ffmpeg.input(video).output(
            str(output), vn=None, acodec="copy", loglevel="quiet"
        ).run()
        return str(output)
    except ffmpeg.Error as e:
        raise click.ClickException(f"ffmpeg error: {e}")


def subtitle_audio(
    api_key: str, model_name: str, audio_path: str, subtitle_path: str
) -> SSAFile:
    """
    Subtitles an audio file using Gemini in SRT format.

    Args:
        api_key (str): Google Gemini API key
        model_name (str): Google Gemini model to use
        audio_path (str): path to the audio file
        subtitle_path (str): path to existing subtitle file (optional)

    Returns:
        SSAFile: transcribed subtitles

    """
    if not audio_path:
        raise click.ClickException("Path to audio file is mandatory")

    client = genai.Client(api_key=api_key)
    upload = client.files.upload(file=audio_path)

    prompt = [
        "You are a subtitle generator that receives audio and generates subtitles in the .srt format.",
        "Your output is only the subtitle content.",
    ]

    prompt = prompt + [
        "When creating the subtitles, you MUST FOLLOW these guidelines to ensure maximum quality:",
        "<GUIDELINES>",
        GUIDELINES,
        "</GUIDELINES>",
    ]

    if subtitle_path:
        subs = subtitles_to_dict(pysubs2.load(subtitle_path))

        prompt = prompt + [
            "There is an existing subtitle for the video that is out of sync or with minor incorrections.",
            "If the subtitle is all caps, do not make the final subtitles to be in all caps, but have it in a more natural format instead.",
            "THE GUIDELINES SUPERSEDE THE FORMAT OF THE EXAMPLE SUBTITLE",
            "No single subtitle should have more than 105 characters",
            "You should use these subtitles in JSON format as a reference:",
            "<SUBTITLE>",
            json.dumps(subs),
            "</SUBTITLE>",
        ]

    prompt.append(
        "Based on all the information provided, create a subtitle in JSON format for this audio clip:"
    )

    model_info = client.models.get(model=model_name)
    click.echo(f"{model_info.input_token_limit=}")
    click.echo(f"{model_info.output_token_limit=}")

    contents = ["\n".join(prompt), upload]
    tokens = client.models.count_tokens(
        model=model_name, contents=contents
    ).total_tokens
    click.echo(f"Input token count: {tokens}")

    try:
        response = generate_subtitle(model_name, client, contents)

        metadata = response.usage_metadata
        if metadata:
            click.echo(f"Thoughts token count: {metadata.thoughts_token_count}")
            click.echo(f"Output token count: {metadata.total_token_count - tokens}")  # type: ignore

        result_subs = events_to_subtitles(response.parsed)  # type: ignore

        return result_subs
    except Exception as e:
        raise click.ClickException(str(e))


@retry(reraise=True, stop=stop_after_attempt(7))
def generate_subtitle(model_name: str, client: genai.Client, contents: List[str]):
    """
    Call Google Gemini to generate a subtitle.

    This method will be automatically retried if the response is empty.

    Args:
        model_name (str): model to be used
        client (genai.Client): existing client to Gemini
        contents (List[str]): list of prompts to be sent to Gemiin
    """
    response = client.models.generate_content(
        model=model_name,
        contents=(contents,),
        config=types.GenerateContentConfig(
            temperature=0.0,  # Don't be creative
            http_options=types.HttpOptions(timeout=5 * 60 * 1000),  # 5 minutes
            thinking_config=types.ThinkingConfig(
                thinking_budget=1024  # Small thinking budget
            ),
            response_mime_type="application/json",
            response_schema=list[SubtitleEvent],
        ),
    )

    if not response.parsed:
        click.echo("Got empty response")
        if response.candidates:
            reason = response.candidates[0].finish_reason
        else:
            reason = "unknown"

        raise Exception(f"No subtitles generated by Gemini due to: {reason}")
    return response


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
@click.option("--video", help="Path to video file", required=True)
@click.option("--subtitle", help="Path to subtitle file")
@click.option(
    "--max-words-per-screen",
    default=12,
    help=f"Maximum words to display per screen (default: 12)",
)
@click.option(
    "--min-duration",
    default=1000,
    help=f"Minimum time a subtitle should stay on screen (default: 1000ms)",
)
def main(
    api_key: str,
    ai_model: str,
    video: str,
    subtitle: str,
    max_words_per_screen: int,
    min_duration: int,
) -> None:
    start = time.time()

    if not api_key:
        raise click.ClickException(
            f"API key not informed or not present in the environment variable {API_KEY_NAME}"
        )
    # 1. Extract audio from video
    audio_path = extract_audio(video)
    click.echo(f"Audio saved to {audio_path}")

    # 2. Send request to Gemini
    click.echo("Asking Gemini to generate the subtitle...")
    new_subtitles = subtitle_audio(
        api_key=api_key,
        model_name=ai_model,
        audio_path=audio_path,
        subtitle_path=subtitle,
    )
    click.echo("Subtitle generated by Gemini")

    # 3. Backup existing subtitle (if exists)
    root = Path(video).stem
    subtitle_path = Path(f"{root}.srt")
    if subtitle_path.exists():
        dst = f"{subtitle_path}.bak"
        shutil.move(subtitle_path, dst)
        click.echo(f"Existing subtitle backed up to {dst}")

    # 4. Post-process subtitle to rebalance the amount of words per screen
    balancer = SubtitleBalancer(
        max_words_per_screen=max_words_per_screen, min_duration_ms=min_duration
    )
    balanced_subs = balancer.balance_subtitles(new_subtitles)

    # 5. Write AI response
    with open(subtitle_path, "w") as f:
        balanced_subs.to_file(f, "srt")

    # 6. Output processing info
    end = time.time()
    duration = datetime.timedelta(seconds=round(end - start, 2))
    click.echo(
        f"Subtitle saved at {subtitle_path} (Processed for {humanize.time.naturaldelta(duration)})"
    )


if __name__ == "__main__":
    main()
