FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg sox git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r requirements.txt

# Установка форка с поддержкой стриминга
RUN git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git /tmp/qwen-tts-streaming && \
    cd /tmp/qwen-tts-streaming && \
    python3 -m pip install -e .

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]