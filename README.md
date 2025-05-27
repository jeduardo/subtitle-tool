# Subtitle sync

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
