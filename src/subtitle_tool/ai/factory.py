from subtitle_tool.ai.base import Subtitler
from subtitle_tool.ai.gemini import GeminiSubtitler


def AISubtitler(provider: str = "gemini", **kwargs) -> Subtitler:
    if provider == "gemini":
        return GeminiSubtitler(**kwargs)

    raise ValueError(f"Unsupported AI subtitle provider: {provider}")
