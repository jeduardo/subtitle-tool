import json
import logging
import math
import re
from inspect import signature
from threading import Lock
from typing import Any

from subtitle_tool.subtitles import SubtitleEvent

logger = logging.getLogger("subtitle_tool.ai")


def _translation_directive(subtitler) -> str:
    target_language = (subtitler.subtitle_lang or "").strip()
    source_language = (subtitler.media_lang or "").strip()
    if target_language:
        if source_language and target_language.casefold() == source_language.casefold():
            return f"{target_language} spelling."
        return (
            f"Translate to {target_language} when needed. "
            f"Write subtitle text only in {target_language}."
        )

    return "Keep the spoken language."


def _build_prompt(subtitler, duration_ms: int) -> str:
    del duration_ms
    parts: list[str] = []

    language = (subtitler.media_lang or "").strip()
    if language:
        parts.append(f"{language} transcript.")

    parts.extend(
        [
            "Clean punctuation and capitalization.",
            "Faithful subtitle wording.",
            _translation_directive(subtitler),
        ]
    )
    return " ".join(parts)


def setup(subtitler) -> None:
    try:
        from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor
    except ImportError as ex:  # pragma: no cover - optional runtime dep
        raise ImportError(
            "mlx-voxtral is not installed. Install dependencies with `uv sync` "
            + "or `pip install mlx-voxtral transformers`."
        ) from ex

    subtitler.processor = VoxtralProcessor.from_pretrained(subtitler.model_name)
    subtitler.model = VoxtralForConditionalGeneration.from_pretrained(
        subtitler.model_name
    )
    subtitler.inference_lock = Lock()


def _inject_prompt_tokens(subtitler, inputs: Any, prompt: str | None) -> Any:
    if not prompt:
        return inputs

    input_ids = request_value(inputs, "input_ids")
    if input_ids is None:
        return inputs

    token_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    if not isinstance(token_ids, list) or not token_ids:
        return inputs

    prompt_tokens = subtitler.processor.tokenizer.encode(
        " " + prompt.strip(), add_special_tokens=False
    )
    if hasattr(prompt_tokens, "tolist"):
        prompt_tokens = prompt_tokens.tolist()
    if not isinstance(prompt_tokens, list) or not prompt_tokens:
        return inputs

    special_ids = getattr(subtitler.processor, "_special_token_ids", {}) or {}
    inst_end_id = special_ids.get("inst_end")
    insert_idx = len(token_ids)
    if inst_end_id is not None:
        try:
            insert_idx = token_ids.index(inst_end_id)
        except ValueError:
            insert_idx = len(token_ids)

    merged_tokens = token_ids[:insert_idx] + prompt_tokens + token_ids[insert_idx:]

    import mlx.core as mx

    dtype = getattr(input_ids, "dtype", None)
    if dtype:
        new_input_ids = mx.array([merged_tokens], dtype=dtype)
    else:
        new_input_ids = mx.array([merged_tokens])

    if isinstance(inputs, dict):
        inputs["input_ids"] = new_input_ids
    else:
        inputs.input_ids = new_input_ids

    return inputs


def build_request_inputs(
    subtitler, audio_path: str, language: str | None, prompt: str | None = None
):
    request_builder = getattr(subtitler.processor, "apply_transcription_request", None)
    if request_builder is None:
        request_builder = getattr(
            subtitler.processor, "apply_transcrition_request", None
        )

    if request_builder is None:
        raise subtitler._generation_error_class()(  # noqa: B904
            "Voxtral processor does not expose transcription request helper"
        )

    kwargs: dict[str, Any] = {"audio": audio_path}
    try:
        params = signature(request_builder).parameters
    except (TypeError, ValueError):
        params = {}

    if language and ("language" in params or not params):
        kwargs["language"] = language

    try:
        inputs = request_builder(**kwargs)
    except TypeError as error:
        if "language" in kwargs and "language" in str(error):
            inputs = request_builder(audio=audio_path)
        else:
            raise

    return _inject_prompt_tokens(subtitler, inputs, prompt)


def request_value(inputs: Any, field: str) -> Any:
    if isinstance(inputs, dict):
        return inputs.get(field)

    return getattr(inputs, field, None)


def extract_prompt_length(inputs: Any) -> int:
    input_ids = request_value(inputs, "input_ids")

    if input_ids is None:
        return 0

    shape = getattr(input_ids, "shape", None)
    if shape is None:
        return 0

    if len(shape) >= 2:
        return int(shape[1])

    return int(shape[0])


def decode_outputs(subtitler, inputs: Any, outputs: Any) -> str:
    prompt_length = extract_prompt_length(inputs)

    output_tokens = outputs
    if isinstance(outputs, list | tuple) and outputs:
        output_tokens = outputs[0]

    ndim = getattr(output_tokens, "ndim", None)
    shape = getattr(output_tokens, "shape", None)
    if ndim == 2 or (shape is not None and len(shape) == 2):
        output_tokens = output_tokens[0]

    if prompt_length:
        try:
            output_tokens = output_tokens[prompt_length:]
        except Exception:
            logger.debug("Could not trim prompt tokens from output; decoding raw")

    text = subtitler.processor.decode(output_tokens, skip_special_tokens=True)
    return (text or "").strip()


def _extract_json_payload(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model output")

    candidates: list[str] = [text]
    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates.extend(match.strip() for match in fenced_matches if match.strip())

    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        candidates.append(text[bracket_start : bracket_end + 1].strip())

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1].strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Model output is not valid JSON")


def _parse_timecode_ms(value: str) -> int:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Unsupported timecode format: {value!r}")

    hours = 0
    if len(parts) == 3:
        hours = int(parts[0].strip())
        minutes = int(parts[1].strip())
        sec_part = parts[2].strip()
    else:
        minutes = int(parts[0].strip())
        sec_part = parts[1].strip()

    if "." in sec_part:
        seconds_text, frac_text = sec_part.split(".", 1)
    elif "," in sec_part:
        seconds_text, frac_text = sec_part.split(",", 1)
    else:
        seconds_text, frac_text = sec_part, ""

    seconds = int(seconds_text.strip())
    fraction = "".join(ch for ch in frac_text.strip() if ch.isdigit())
    millis = int((fraction + "000")[:3]) if fraction else 0

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds * 1000 + millis


def _timestamp_to_ms(value: Any, duration_ms: int) -> int:
    if value is None:
        raise ValueError("Missing timestamp value")

    if isinstance(value, int | float):
        as_float = float(value)
        if abs(as_float) <= (duration_ms / 1000.0 + 1.0):
            as_float *= 1000.0
        return int(round(as_float))

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError("Empty timestamp value")

        lowered = raw.lower()
        if lowered.endswith("ms"):
            return int(round(float(lowered[:-2].strip())))
        if lowered.endswith("s"):
            return int(round(float(lowered[:-1].strip()) * 1000.0))

        try:
            number = float(raw)
            if abs(number) <= (duration_ms / 1000.0 + 1.0):
                number *= 1000.0
            return int(round(number))
        except ValueError:
            return _parse_timecode_ms(raw)

    raise ValueError(f"Unsupported timestamp type: {type(value)!r}")


def _select_value(entry: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
    return None


def _extract_segments(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("segments", "subtitles", "events", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if {"start", "end", "text"}.issubset(payload):
            return [payload]

    raise ValueError("JSON payload must be a subtitle array or object with segments")


def _normalize_events(
    events: list[SubtitleEvent], duration_ms: int
) -> list[SubtitleEvent]:
    max_end = max(1, duration_ms)
    ret: list[SubtitleEvent] = []

    prev_end = 0
    for event in sorted(events, key=lambda item: item.start):
        if prev_end >= max_end:
            break

        start = max(0, min(event.start, max_end - 1))
        end = max(start + 1, min(event.end, max_end))

        if start < prev_end:
            start = prev_end
        if start >= max_end:
            break

        if end <= start:
            end = min(max_end, start + 1)
        if end <= start:
            continue

        text = (event.text or "").strip()
        if not text:
            continue

        ret.append(SubtitleEvent(start=start, end=end, text=text))
        prev_end = end

    return ret


def _json_to_subtitles(raw_output: str, duration_ms: int) -> list[SubtitleEvent]:
    payload = _extract_json_payload(raw_output)
    segments = _extract_segments(payload)

    parsed: list[SubtitleEvent] = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Subtitle segment entries must be JSON objects")

        text = str(_select_value(segment, "text", "subtitle", "value", "content") or "")
        text = text.strip()
        if not text:
            continue

        start_value = _select_value(
            segment,
            "start",
            "start_ms",
            "startMs",
            "from",
            "begin",
        )
        end_value = _select_value(
            segment,
            "end",
            "end_ms",
            "endMs",
            "to",
            "finish",
        )

        start = _timestamp_to_ms(start_value, duration_ms)
        end = _timestamp_to_ms(end_value, duration_ms)
        parsed.append(SubtitleEvent(start=start, end=end, text=text))

    return _normalize_events(parsed, duration_ms)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _clean_transcript_text(raw_output: str) -> str:
    text = (raw_output or "").strip()
    if not text:
        return ""

    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        r"^(?:here is the )?(?:transcript|translation|subtitle|subtitles)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'(?<=[.!?])"', "", text)
    text = re.sub(r'^["\']+|["\']+$', "", text)
    return text.strip()


def _clean_chunk_text(text: str) -> str:
    return re.sub(r'^["\']+|["\']+$', "", text.strip())


def _split_text_units(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])["\']?\s+', text)
    ret: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        clauses = re.split(r"(?<=[,;:])\s+", sentence)
        ret.extend(_clean_chunk_text(clause) for clause in clauses if clause.strip())

    return ret or ([text] if text else [])


def _split_long_unit(unit: str, max_words: int) -> list[str]:
    words = unit.split()
    if len(words) <= max_words:
        return [unit]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        for idx in range(end - 1, start, -1):
            if re.search(r'[,:;.!?]["\']?$', words[idx]):
                end = idx + 1
                break

        chunks.append(_clean_chunk_text(" ".join(words[start:end])))
        start = end

    return [chunk for chunk in chunks if chunk]


def _chunk_plain_text(text: str, duration_ms: int) -> list[str]:
    total_words = max(1, _word_count(text))
    words_per_window = max(
        5,
        min(12, int(math.ceil(total_words * 5000 / max(duration_ms, 1)))),
    )

    units = _split_text_units(text)
    chunks: list[str] = []
    current = ""
    current_words = 0

    for unit in units:
        for fragment in _split_long_unit(unit, words_per_window):
            fragment_words = _word_count(fragment)
            if (
                current
                and current_words + fragment_words <= words_per_window
                and len(current) + 1 + len(fragment) <= 84
                and not re.search(r'[.!?]["\']?$', current)
            ):
                current = f"{current} {fragment}"
                current_words += fragment_words
                continue

            if current:
                chunks.append(_clean_chunk_text(current))
            current = _clean_chunk_text(fragment)
            current_words = fragment_words

    if current:
        chunks.append(_clean_chunk_text(current))

    return chunks


def _plain_text_to_subtitles(raw_output: str, duration_ms: int) -> list[SubtitleEvent]:
    text = _clean_transcript_text(raw_output)
    if not text:
        return []

    chunks = _chunk_plain_text(text, duration_ms)
    if not chunks:
        return []

    weights = [max(1, _word_count(chunk)) for chunk in chunks]
    total_weight = sum(weights)
    events: list[SubtitleEvent] = []
    consumed = 0
    start = 0

    for index, (chunk, weight) in enumerate(zip(chunks, weights, strict=True)):
        consumed += weight
        if index == len(chunks) - 1:
            end = duration_ms
        else:
            end = int(round(duration_ms * consumed / total_weight))
        end = max(start + 1, min(end, duration_ms))
        events.append(SubtitleEvent(start=start, end=end, text=chunk))
        start = end

    return _normalize_events(events, duration_ms)


def generate_subtitles(
    subtitler, duration_ms: int, audio_path: str, temp_adj: float = 0.0
) -> list[SubtitleEvent]:
    language = subtitler._normalize_language(subtitler.media_lang)
    prompt = _build_prompt(subtitler, duration_ms)
    temperature = max(0.0, subtitler.temperature + temp_adj)

    with subtitler.inference_lock:
        inputs = build_request_inputs(subtitler, audio_path, language, prompt=prompt)
        input_ids = request_value(inputs, "input_ids")
        input_features = request_value(inputs, "input_features")
        attention_mask = request_value(inputs, "attention_mask")

        generation_kwargs = {
            "input_ids": input_ids,
            "input_features": input_features,
        }
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask

        outputs = subtitler.model.generate(
            **generation_kwargs,
            max_new_tokens=2048,
            temperature=temperature,
        )

    text = decode_outputs(subtitler, inputs, outputs)
    try:
        return _json_to_subtitles(text, duration_ms)
    except ValueError as error:
        plain_events = _plain_text_to_subtitles(text, duration_ms)
        if plain_events:
            logger.debug(
                "Voxtral returned plain transcript output; using heuristic subtitle segmentation"
            )
            return plain_events
        raise subtitler._generation_error_class()(  # noqa: B904
            "Invalid Voxtral subtitle payload: "
            + str(error)
        )
