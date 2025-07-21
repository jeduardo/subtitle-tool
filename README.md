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

Export the API key for Gemini to the environment variable `GEMINI_API_KEY`
**or** specify it in the command line with the flag `--api-key`.

`ffmpeg` needs to be installed (`brew install ffmpeg`, `apt-get install ffmpeg` or `dnf install ffmpeg`)

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

```shell
subtitle-tool --video myvideo.avi
```
