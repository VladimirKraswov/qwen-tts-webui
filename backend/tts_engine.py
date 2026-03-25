import torch
import numpy as np
import io
from pydub import AudioSegment
import asyncio
import logging

from config import (
    MODEL_NAME, DEVICE, DTYPE,
    STREAM_EMIT_EVERY_FRAMES, STREAM_DECODE_WINDOW_FRAMES,
    SAMPLE_RATE, VOICES
)

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        try:
            from qwen_tts import Qwen3TTSModel
            self.model = Qwen3TTSModel.from_pretrained(
                MODEL_NAME,
                device_map=DEVICE,
                dtype=torch.bfloat16 if DTYPE == "bfloat16" else torch.float16,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )
            logger.info("Модель TTS загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def generate_audio(self, text: str, voice: str, language: str = "Russian"):
        if self.model is None:
            self.load_model()
        try:
            pcm = self.model.generate_pcm(
                text=text,
                speaker=voice,
                language=language,
                return_numpy=True
            )
            audio = AudioSegment(
                data=pcm.tobytes(),
                sample_width=2,
                frame_rate=SAMPLE_RATE,
                channels=1
            )
            mp3_io = io.BytesIO()
            audio.export(mp3_io, format="mp3", bitrate="192k")
            return mp3_io.getvalue()
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            raise

    async def stream_generate(self, text: str, voice: str, language: str = "Russian"):
        if self.model is None:
            self.load_model()

        try:
            if hasattr(self.model, 'stream_generate_pcm'):
                gen = self.model.stream_generate_pcm(
                    text=text,
                    speaker=voice,
                    language=language,
                    emit_every_frames=STREAM_EMIT_EVERY_FRAMES,
                    decode_window_frames=STREAM_DECODE_WINDOW_FRAMES
                )
                for chunk in gen:
                    yield chunk.tobytes()
            else:
                pcm = self.model.generate_pcm(
                    text=text,
                    speaker=voice,
                    language=language,
                    return_numpy=True
                )
                chunk_size = SAMPLE_RATE // 4
                for i in range(0, len(pcm), chunk_size):
                    yield pcm[i:i+chunk_size].tobytes()
        except Exception as e:
            logger.error(f"Ошибка стриминга: {e}")
            raise

engine = TTSEngine()