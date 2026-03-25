import os

MODEL_NAME = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEVICE = os.getenv("TTS_DEVICE", "cuda:0")
DTYPE = os.getenv("TTS_DTYPE", "bfloat16")
STREAM_EMIT_EVERY_FRAMES = int(os.getenv("TTS_STREAM_EMIT_FRAMES", "6"))
STREAM_DECODE_WINDOW_FRAMES = int(os.getenv("TTS_STREAM_DECODE_WINDOW", "72"))
SAMPLE_RATE = 24000  # Hz
VOICES = ["Vivian", "Ryan", "Serena", "Aiden", "Eric", "Dylan"]