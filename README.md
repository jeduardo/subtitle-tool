# Subtitle tool

This utility uses Google Gemini to sync subtitles to videos.

## Dependencies

`ffmpeg` needs to be installed for audio extraction.

## Process

1. Extract the audio from the video
2. Send the audio to Gemini for transcription, with the existing subtitle
   also sent over if available.
3. Backup the existing subtitle
4. Rebalance the generated subtitle to limit the words per screen,
5. Save the adjusted subtitle.

## Dependencies

Export the API key for Gemini to the environment variable `GEMINI_API_KEY`
**or** specify it in the command line with the flag `--api-key`.

`ffmpeg` needs to be installed (`brew install ffmpeg`, `apt-get install ffmpeg` or `dnf install ffmpeg`)

## Installation

```shell
# Leaving it possible to change the local code
uv install tool -e .
```

## Usage

```shell
subtitle-tool --video myvideo.avi --subtitle out-of-sync.srt
```

## References

- [Subtle Recommended Quality Criteria for Subtitling](https://subtle-subtitlers.org.uk/wp-content/uploads/2023/01/SUBTLE-Recommended-Quality-Criteria-for-Subtitling.pdf), © SUBTLE 2023: used to guide the AI in terms of subtitle quality.
