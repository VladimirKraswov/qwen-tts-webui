import os

MODEL_NAME = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEVICE = os.getenv("TTS_DEVICE", "cuda:0")
DTYPE = os.getenv("TTS_DTYPE", "bfloat16")
ATTN_IMPLEMENTATION = os.getenv("TTS_ATTN", "sdpa")
SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
STREAM_CHUNK_MS = int(os.getenv("TTS_STREAM_CHUNK_MS", "200"))

DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "Vivian")
DEFAULT_LANGUAGE = os.getenv("TTS_DEFAULT_LANGUAGE", "Russian")

STATIC_VOICES = [
    "Vivian",
    "Serena",
    "Ryan",
    "Aiden",
    "Eric",
    "Dylan",
    "Uncle_Fu",
    "Ono_Anna",
    "Sohee",
]
