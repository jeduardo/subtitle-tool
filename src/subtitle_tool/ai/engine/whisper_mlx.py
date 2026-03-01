
def _build_prompt(subtitler) -> str:
    parts: list[str] = []

    language = (subtitler.media_lang or "").strip()
    if language:
        parts.append(f"{language} transcript.")

    # Whisper's initial_prompt works best as a short transcript primer rather
    # than a block of instructions. Keep it brief and transcript-shaped.
    parts.extend(
        [
            "Clean punctuation and capitalization.",
            "Accurate subtitle wording.",
        ]
    )

    subtitle_lang = (subtitler.subtitle_lang or "").strip()
    if subtitle_lang and subtitle_lang.casefold() == language.casefold():
        parts.append(f"{subtitle_lang} spelling.")

    return " ".join(parts)


def setup(subtitler) -> None:
    try:
        import mlx_whisper
    except ImportError as ex:  # pragma: no cover - optional runtime dep
        raise ImportError(
            "mlx-whisper is not installed. Install dependencies with `uv sync` "
            + "or `pip install mlx-whisper`."
        ) from ex

    subtitler.whisper = mlx_whisper


def generate_subtitles(
    subtitler, duration_ms: int, audio_path: str, temp_adj: float = 0.0
):
    del temp_adj
    language = subtitler._normalize_language(subtitler.media_lang)
    prompt = _build_prompt(subtitler)
    response = subtitler.whisper.transcribe(
        audio_path,
        path_or_hf_repo=subtitler.model_name,
        language=language,
        task="transcribe",
        initial_prompt=prompt,
        word_timestamps=False,
    )
    segments = subtitler._segment_value(response, "segments")
    if segments is None:
        raise subtitler._generation_error_class()(
            "Whisper payload has no timestamped segments"
        )

    return subtitler._normalize_segments({"segments": segments}, duration_ms)
