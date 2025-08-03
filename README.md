# Subtitle tool

[![codecov](https://codecov.io/gh/jeduardo/subtitle-tool/graph/badge.svg?token=TPA3UXF5OC)](https://codecov.io/gh/jeduardo/subtitle-tool)

This utility uses Google Gemini to generate subtitles to audio and video files.

## Dependencies

`ffmpeg` needs to be installed for audio extraction.

## Process

1. Extract the audio from the video
2. Send the audio to Gemini for transcription
3. Backup the existing subtitle
4. Save the new subtitle

## Dependencies

- Export the API key for Gemini to the environment variable `GEMINI_API_KEY`
  **or** specify it in the command line with the flag `--api-key`.

- `ffmpeg` needs to be installed (`brew install ffmpeg`, `apt-get install ffmpeg` or `dnf install ffmpeg`)

- Ensure `uv` installs its dev dependencies with `uv sync --extra dev`.

## Installation

```shell
pip install subtitle-tool
```

## Developing

For local development it is useful to install the binary from the development
location into the user's `PATH`. For this, run the following commands:

```shell
uv tool install -e .
uv tool update-shell
```

## Usage

- Subtitle files with the defaults:

```shell
subtitle-tool video.avi
subtitle-tool audio.mp3
```

- Usage options:

```text
Usage: subtitle-tool [OPTIONS] MEDIAFILE

  Generate subtitles for a media file

Options:
  --api-key TEXT                  Google Gemini API key
  -m, --ai-model TEXT             Gemini model to use  [default:
                                  gemini-2.5-flash]
  -s, --subtitle-path TEXT        Subtitle file name [default: MEDIAFILE.srt]
  -v, --verbose                   Enable debug logging for subtitle_tool
                                  modules
  -d, --debug                     Enable debug logging for all modules
  -k, --keep-temp-files           Do not erase temporary files
  -l, --audio-segment-length INTEGER
                                  Length of audio segments to be subtitled in
                                  seconds  [default: 30]
  -p, --parallel-segments INTEGER
                                  Number of segments subtitled in parallel
                                  [default: 5]
  --help                          Show this message and exit.
```
