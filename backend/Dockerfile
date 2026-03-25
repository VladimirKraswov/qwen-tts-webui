FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg sox git curl python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .

# Удаляем суффиксы типа +cu124 из всех строк (особенно torch)
RUN sed -i 's/+cu[0-9]*//g' requirements.txt

# Устанавливаем torch из официального репозитория PyTorch для CUDA 12.4
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

RUN git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git /tmp/qwen-tts-streaming && \
    cd /tmp/qwen-tts-streaming && \
    pip install -e .

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]