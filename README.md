# Subtitle tool

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
# Leaving it possible to change the local code
uv tool install -e .
```

## Usage

```shell
subtitle-tool --video myvideo.avi
```
