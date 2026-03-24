function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

async function loadVoices() {
    try {
        const res = await fetch("/voices");
        const data = await res.json();
        const voiceSelect = document.getElementById("tts-voice");
        const voices = Array.isArray(data.voices) ? data.voices : [];
        voiceSelect.innerHTML = voices
            .map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`)
            .join("");
    } catch (e) {
        document.getElementById("tts-status").textContent = "Не удалось загрузить список голосов";
    }
}

function getTtsPayload(format = "mp3", stream = false) {
    return {
        text: document.getElementById("tts-text").value,
        voice: document.getElementById("tts-voice").value,
        language: document.getElementById("tts-language").value,
        instruct: document.getElementById("tts-instruct").value || null,
        response_format: format,
        stream,
    };
}

async function requestAudio(format) {
    const status = document.getElementById("tts-status");
    const audio = document.getElementById("audio-element");
    const payload = getTtsPayload(format, false);

    if (!payload.text.trim()) {
        status.textContent = "Введите текст";
        return;
    }

    status.textContent = "Синтез...";
    try {
        const response = await fetch("/v1/audio/speech", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || "Ошибка генерации");
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        audio.src = url;
        await audio.play().catch(() => {});
        status.textContent = `Готово: ${format.toUpperCase()}`;
    } catch (e) {
        status.textContent = `Ошибка: ${e.message}`;
    }
}

document.getElementById("generate-btn").addEventListener("click", () => requestAudio("mp3"));
document.getElementById("wav-btn").addEventListener("click", () => requestAudio("wav"));

document.getElementById("stream-btn").addEventListener("click", async () => {
    const status = document.getElementById("tts-status");
    const payload = getTtsPayload("pcm", true);

    if (!payload.text.trim()) {
        status.textContent = "Введите текст";
        return;
    }

    status.textContent = "Открываю PCM поток...";
    try {
        const response = await fetch("/v1/audio/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok || !response.body) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || "Ошибка стриминга");
        }

        const sampleRate = parseInt(response.headers.get("X-Sample-Rate") || "24000", 10);
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
        await audioContext.resume();

        const reader = response.body.getReader();
        let nextTime = audioContext.currentTime + 0.05;
        let pendingByte = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (!value || value.length === 0) continue;

            let chunk = value;

            if (pendingByte !== null) {
                const merged = new Uint8Array(chunk.length + 1);
                merged[0] = pendingByte;
                merged.set(chunk, 1);
                chunk = merged;
                pendingByte = null;
            }

            if (chunk.length % 2 !== 0) {
                pendingByte = chunk[chunk.length - 1];
                chunk = chunk.slice(0, -1);
            }

            if (chunk.length === 0) continue;

            const int16 = new Int16Array(chunk.buffer, chunk.byteOffset, chunk.byteLength / 2);
            const float32 = new Float32Array(int16.length);
            for (let i = 0; i < int16.length; i++) {
                float32[i] = int16[i] / 32768.0;
            }

            const buffer = audioContext.createBuffer(1, float32.length, sampleRate);
            buffer.copyToChannel(float32, 0);

            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);

            const startAt = Math.max(nextTime, audioContext.currentTime + 0.03);
            source.start(startAt);
            nextTime = startAt + buffer.duration;
        }

        status.textContent = "PCM поток завершён";
    } catch (e) {
        status.textContent = `Ошибка: ${e.message}`;
    }
});

let currentJobId = null;

document.getElementById("upload-book-btn").addEventListener("click", async () => {
    const fileInput = document.getElementById("book-file");
    const file = fileInput.files[0];
    const progress = document.getElementById("book-progress");

    if (!file) {
        progress.textContent = "Выберите файл";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    progress.textContent = "Загрузка книги...";
    try {
        const res = await fetch("/upload_text", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || "Ошибка загрузки");
        }

        currentJobId = data.job_id;
        const previewDiv = document.getElementById("book-preview");
        previewDiv.innerHTML =
            `<h3>Предпросмотр (${data.total} абзацев)</h3>` +
            data.segments
                .map(s => `<p><strong>${s.id + 1}</strong>: ${escapeHtml(s.text)}</p>`)
                .join("");

        document.getElementById("generate-book-btn").style.display = "inline-block";
        progress.textContent = "Книга загружена";
    } catch (e) {
        progress.textContent = `Ошибка: ${e.message}`;
    }
});

document.getElementById("generate-book-btn").addEventListener("click", async () => {
    const progress = document.getElementById("book-progress");
    if (!currentJobId) {
        progress.textContent = "Сначала загрузите книгу";
        return;
    }

    progress.textContent = "Запуск озвучивания...";
    try {
        const res = await fetch(`/generate_book/${currentJobId}`, { method: "POST" });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || "Ошибка запуска");
        }

        progress.textContent = `Озвучивание начато. Job ID: ${data.job_id}`;
    } catch (e) {
        progress.textContent = `Ошибка: ${e.message}`;
    }
});

document.getElementById("train-btn").addEventListener("click", async () => {
    const voiceName = document.getElementById("voice-name").value.trim();
    const audioFiles = document.getElementById("train-audio").files;
    const transcriptsText = document.getElementById("train-transcripts").value;
    const status = document.getElementById("train-status");

    if (!voiceName || audioFiles.length === 0 || !transcriptsText.trim()) {
        status.textContent = "Заполните все поля";
        return;
    }

    const transcripts = transcriptsText.split("\n").map(s => s.trim()).filter(Boolean);
    if (audioFiles.length !== transcripts.length) {
        status.textContent = "Количество аудиофайлов должно совпадать с количеством строк транскрипций";
        return;
    }

    const formData = new FormData();
    formData.append("voice_name", voiceName);
    for (let i = 0; i < audioFiles.length; i++) {
        formData.append("audio_files", audioFiles[i]);
        formData.append("transcriptions", transcripts[i]);
    }

    try {
        const res = await fetch("/train_voice", { method: "POST", body: formData });
        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
            throw new Error(data.detail || "Не реализовано");
        }

        status.textContent = "Обучение запущено";
    } catch (e) {
        status.textContent = `Ошибка: ${e.message}`;
    }
});

document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));

        btn.classList.add("active");
        const tabId = btn.getAttribute("data-tab");
        document.getElementById(`${tabId}-tab`).classList.add("active");
    });
});

loadVoices();
