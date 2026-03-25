# Qwen TTS Web UI

Обновлённый веб-интерфейс для синтеза речи на базе Qwen3-TTS.

## Что изменено в сборке

Тяжёлые AI-зависимости вынесены в отдельный базовый образ:

- `backend/Dockerfile.ai-base` — CUDA, Python, ffmpeg, PyTorch, transformers и прочие AI-библиотеки;
- `backend/Dockerfile` — лёгкий runtime-образ приложения поверх готовой AI-базы;
- `docker-compose.yml` — отдельный build-сервис `tts-base` и основной сервис `tts`.

Это нужно, чтобы при обычных правках backend/frontend не пересобирать и не скачивать заново тяжёлые слои с PyTorch и системными пакетами.

## Первый запуск

Собери AI-базу один раз:

```bash
docker compose build tts-base
```

Потом подними приложение:

```bash
docker compose up --build tts
```

или через `make`:

```bash
make base
make up
```

## Обычный цикл разработки

Когда меняешь только код:

```bash
docker compose up --build tts
```

или просто:

```bash
docker compose up tts
```

если контейнер уже был собран.

### Важно

Для правок кода **не используй** каждый раз:

```bash
docker compose down --volumes --rmi all
docker compose build --no-cache
```

Именно эти флаги убивают весь эффект кеша и заставляют Docker заново скачивать и пересобирать тяжёлые зависимости.

## Когда нужно пересобрать AI-базу

Пересобирай `tts-base`, если изменилось одно из этого:

- `backend/Dockerfile.ai-base`
- `backend/requirements.ai.txt`
- версия CUDA / PyTorch
- системные пакеты для модели

Команда:

```bash
docker compose build --no-cache tts-base
docker compose build --no-cache tts
```

или:

```bash
make rebuild-base
```

## Структура зависимостей

### AI-слой

`backend/requirements.ai.txt`

- `torch`
- `transformers`
- `qwen-tts`
- `ctranslate2`
- `faster-whisper`
- `pydub`
- `soundfile`
- `numpy`

### App-слой

`backend/requirements.app.txt`

- `fastapi`
- `uvicorn`
- `python-multipart`
- `pyyaml`
- `openai`
- `requests`
- `aiofiles`

## API

- `GET /api/health`
- `GET /api/voices`
- `POST /api/audio/speech`
- `POST /api/audio/stream`
- `POST /api/books/upload`
- `POST /api/books/{job_id}/start`
- `GET /api/books/{job_id}`
- `GET /api/books/{job_id}/download`

## GPU

Для Docker Compose используется reservation через `deploy.resources.reservations.devices`. Проверь, что на хосте установлен NVIDIA Container Toolkit.
