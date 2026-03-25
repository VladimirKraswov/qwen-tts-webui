# Qwen TTS Web UI

Обновлённый веб-интерфейс для синтеза речи на базе Qwen3-TTS.

## Что изменено в сборке

Тяжёлые AI-зависимости вынесены в отдельный базовый образ:

- `backend/Dockerfile.ai-base` — CUDA, Python, ffmpeg, PyTorch, torchaudio и прочие AI-библиотеки
- `backend/Dockerfile` — лёгкий runtime-образ приложения поверх готовой AI-базы
- `docker-compose.yml` — отдельный build-сервис `tts-base` и основной сервис `tts`

Это нужно, чтобы при обычных правках backend/frontend не пересобирать и не скачивать заново тяжёлые слои с PyTorch и системными пакетами.

## Первый запуск

Собери AI-базу один раз:

```bash
docker compose build tts-base