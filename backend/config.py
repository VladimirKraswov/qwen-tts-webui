import os
from pathlib import Path
from typing import Final


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


APP_NAME: Final[str] = os.getenv("APP_NAME", "Qwen TTS Web UI")
APP_VERSION: Final[str] = os.getenv("APP_VERSION", "2.0.0")

MODEL_NAME: Final[str] = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEVICE: Final[str] = os.getenv("TTS_DEVICE", "cuda:0")
DTYPE: Final[str] = os.getenv("TTS_DTYPE", "bfloat16")
SAMPLE_RATE: Final[int] = _env_int("TTS_SAMPLE_RATE", 24000)
STREAM_EMIT_EVERY_FRAMES: Final[int] = _env_int("TTS_STREAM_EMIT_FRAMES", 6)
STREAM_DECODE_WINDOW_FRAMES: Final[int] = _env_int("TTS_STREAM_DECODE_WINDOW", 72)

DEFAULT_VOICE: Final[str] = os.getenv("TTS_DEFAULT_VOICE", "Vivian")
DEFAULT_LANGUAGE: Final[str] = os.getenv("TTS_DEFAULT_LANGUAGE", "Russian")
VOICES: Final[list[str]] = _env_list(
    "TTS_VOICES",
    ["Vivian", "Ryan", "Serena", "Aiden", "Eric", "Dylan"],
)

ALLOWED_AUDIO_FORMATS: Final[set[str]] = {"mp3", "wav", "pcm"}
SUPPORTED_LANGUAGES: Final[list[str]] = ["Russian", "English", "Chinese", "Auto"]

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR: Final[Path] = BASE_DIR / "frontend"
BOOK_UPLOAD_DIR: Final[Path] = Path(os.getenv("BOOK_UPLOAD_DIR", "/tmp/book_uploads"))
BOOK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_TTS_TEXT_LENGTH: Final[int] = _env_int("TTS_MAX_TEXT_LENGTH", 8000)
BOOK_PREVIEW_SEGMENTS: Final[int] = _env_int("BOOK_PREVIEW_SEGMENTS", 12)
BOOK_MAX_PARAGRAPHS: Final[int] = _env_int("BOOK_MAX_PARAGRAPHS", 5000)
BOOK_MAX_FILE_SIZE_MB: Final[int] = _env_int("BOOK_MAX_FILE_SIZE_MB", 20)
