import asyncio
import io
import logging
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment

from config import (
    MODEL_NAME,
    DEVICE,
    DTYPE,
    ATTN_IMPLEMENTATION,
    STREAM_CHUNK_MS,
    DEFAULT_LANGUAGE,
    STATIC_VOICES,
)

logger = logging.getLogger(__name__)


def _torch_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), torch.bfloat16)


class TTSEngine:
    def __init__(self):
        self.model = None
        self.voices = list(STATIC_VOICES)

    def load_model(self) -> None:
        """Загружает модель один раз."""
        if self.model is not None:
            return

        try:
            from qwen_tts import Qwen3TTSModel

            self.model = Qwen3TTSModel.from_pretrained(
                MODEL_NAME,
                device_map=DEVICE,
                dtype=_torch_dtype(DTYPE),
                attn_implementation=ATTN_IMPLEMENTATION,
            )

            if hasattr(self.model, "get_supported_speakers"):
                speakers = self.model.get_supported_speakers()
                if speakers:
                    self.voices = list(speakers)

            logger.info("Qwen TTS model loaded: %s", MODEL_NAME)
        except Exception:
            logger.exception("Ошибка загрузки модели")
            raise

    def get_voices(self) -> list[str]:
        return list(self.voices)

    def synthesize_wav(
        self,
        text: str,
        voice: str,
        language: str = DEFAULT_LANGUAGE,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """Возвращает float32 wav и sample rate."""
        if self.model is None:
            self.load_model()

        if not text or not text.strip():
            raise ValueError("Text is empty")

        try:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
                instruct=instruct or "",
            )
        except Exception:
            logger.exception("Ошибка синтеза")
            raise

        if not wavs:
            raise RuntimeError("Модель не вернула аудио")

        wav = np.asarray(wavs[0], dtype=np.float32).squeeze()
        if wav.ndim != 1:
            wav = wav.reshape(-1)

        wav = np.clip(wav, -1.0, 1.0)
        return wav, int(sr)

    @staticmethod
    def _float32_to_pcm16(wav: np.ndarray) -> np.ndarray:
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767.0).astype(np.int16)

    def generate_audio(
        self,
        text: str,
        voice: str,
        language: str = DEFAULT_LANGUAGE,
        response_format: str = "mp3",
        instruct: Optional[str] = None,
    ) -> tuple[bytes, str, int, str]:
        """
        Возвращает:
        - bytes
        - media_type
        - sample_rate
        - extension
        """
        wav, sr = self.synthesize_wav(text, voice, language, instruct)
        fmt = response_format.lower()

        if fmt == "mp3":
            pcm16 = self._float32_to_pcm16(wav)
            audio = AudioSegment(
                data=pcm16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            out = io.BytesIO()
            audio.export(out, format="mp3", bitrate="192k")
            return out.getvalue(), "audio/mpeg", sr, "mp3"

        if fmt == "wav":
            out = io.BytesIO()
            sf.write(out, wav, sr, format="WAV", subtype="PCM_16")
            return out.getvalue(), "audio/wav", sr, "wav"

        if fmt == "pcm":
            pcm16 = self._float32_to_pcm16(wav)
            return pcm16.tobytes(), "audio/L16", sr, "pcm"

        raise ValueError(f"Unsupported response_format: {response_format}")

    async def stream_generate(
        self,
        text: str,
        voice: str,
        language: str = DEFAULT_LANGUAGE,
        instruct: Optional[str] = None,
    ):
        """
        Совместимый fallback-стриминг:
        сначала синтезируем аудио, затем отдаём PCM чанками.
        """
        wav, sr = await asyncio.to_thread(self.synthesize_wav, text, voice, language, instruct)
        pcm = self._float32_to_pcm16(wav).tobytes()

        chunk_size = max(2, int(sr * 2 * STREAM_CHUNK_MS / 1000))
        if chunk_size % 2 != 0:
            chunk_size += 1

        for i in range(0, len(pcm), chunk_size):
            yield pcm[i:i + chunk_size]
            await asyncio.sleep(0)


engine = TTSEngine()
