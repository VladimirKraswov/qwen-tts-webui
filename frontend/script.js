const state = {
    currentAudioUrl: null,
    currentBookJobId: null,
    bookPollTimer: null,
    audioContext: null,
};

function $(selector) {
    return document.querySelector(selector);
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

async function apiFetch(url, options = {}) {
    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : await response.blob();

    if (!response.ok) {
        const message = payload?.detail || `HTTP ${response.status}`;
        throw new Error(message);
    }

    return payload;
}

function setStatus(element, message, type = "default") {
    element.textContent = message;
    element.dataset.state = type;
}

function setButtonBusy(button, busy, busyLabel = "Загрузка…") {
    if (!button.dataset.originalLabel) {
        button.dataset.originalLabel = button.textContent;
    }
    button.disabled = busy;
    button.textContent = busy ? busyLabel : button.dataset.originalLabel;
}

function revokeAudioUrl() {
    if (state.currentAudioUrl) {
        URL.revokeObjectURL(state.currentAudioUrl);
        state.currentAudioUrl = null;
    }
}

function updateTtsCounter() {
    const length = $("#tts-text").value.length;
    $("#tts-counter").textContent = `${length} символов`;
}

function populateSelect(select, values, selectedValue) {
    select.innerHTML = values.map((value) => {
        const selected = value === selectedValue ? " selected" : "";
        return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(value)}</option>`;
    }).join("");
}

function getTtsPayload(format = "mp3", stream = false) {
    return {
        text: $("#tts-text").value.trim(),
        voice: $("#tts-voice").value,
        language: $("#tts-language").value,
        instruct: $("#tts-instruct").value.trim() || null,
        response_format: format,
        stream,
    };
}

async function loadMeta() {
    const [voicesData, healthData] = await Promise.all([
        apiFetch("/api/voices"),
        apiFetch("/api/health"),
    ]);

    populateSelect($("#tts-voice"), voicesData.voices, voicesData.default_voice);
    populateSelect($("#book-voice"), voicesData.voices, voicesData.default_voice);
    populateSelect($("#tts-language"), voicesData.languages, voicesData.default_language);
    populateSelect($("#book-language"), voicesData.languages, voicesData.default_language);

    $("#meta-voices").textContent = voicesData.voices.length;
    $("#meta-languages").textContent = voicesData.languages.join(" · ");

    const chip = $("#engine-chip");
    const engine = healthData.engine || {};
    if (engine.loaded) {
        chip.textContent = "Модель загружена";
        chip.dataset.state = "success";
    } else if (engine.last_error) {
        chip.textContent = "Ошибка загрузки модели";
        chip.dataset.state = "danger";
    } else {
        chip.textContent = "Модель прогревается";
        chip.dataset.state = "warning";
    }
}

async function requestAudio(format) {
    const statusEl = $("#tts-status");
    const audioEl = $("#audio-element");
    const downloadLink = $("#download-link");
    const payload = getTtsPayload(format, false);

    if (!payload.text) {
        setStatus(statusEl, "Введите текст для синтеза.", "danger");
        return;
    }

    setButtonBusy($(format === "mp3" ? "#generate-btn" : "#wav-btn"), true, "Генерирую…");
    setStatus(statusEl, `Синтезирую ${format.toUpperCase()}…`, "warning");

    try {
        const response = await fetch("/api/audio/speech", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || "Не удалось сгенерировать аудио");
        }

        const blob = await response.blob();
        revokeAudioUrl();
        state.currentAudioUrl = URL.createObjectURL(blob);
        audioEl.src = state.currentAudioUrl;
        downloadLink.href = state.currentAudioUrl;
        downloadLink.download = `speech.${format}`;
        downloadLink.classList.remove("hidden");
        await audioEl.play().catch(() => {});
        setStatus(statusEl, `Готово: ${format.toUpperCase()} создан.`, "success");
    } catch (error) {
        setStatus(statusEl, `Ошибка: ${error.message}`, "danger");
    } finally {
        setButtonBusy($(format === "mp3" ? "#generate-btn" : "#wav-btn"), false);
    }
}

async function streamAudio() {
    const statusEl = $("#tts-status");
    const payload = getTtsPayload("pcm", true);
    if (!payload.text) {
        setStatus(statusEl, "Введите текст для стриминга.", "danger");
        return;
    }

    setButtonBusy($("#stream-btn"), true, "Стримлю…");
    setStatus(statusEl, "Подключаю live PCM-поток…", "warning");

    try {
        const response = await fetch("/api/audio/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok || !response.body) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || "Ошибка стриминга");
        }

        const sampleRate = Number(response.headers.get("X-Sample-Rate") || 24000);
        state.audioContext ||= new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
        await state.audioContext.resume();

        const reader = response.body.getReader();
        let nextStart = state.audioContext.currentTime + 0.05;
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
            for (let i = 0; i < int16.length; i += 1) {
                float32[i] = int16[i] / 32768;
            }

            const audioBuffer = state.audioContext.createBuffer(1, float32.length, sampleRate);
            audioBuffer.copyToChannel(float32, 0);
            const source = state.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(state.audioContext.destination);
            const startAt = Math.max(nextStart, state.audioContext.currentTime + 0.03);
            source.start(startAt);
            nextStart = startAt + audioBuffer.duration;
        }

        setStatus(statusEl, "PCM-поток завершён.", "success");
    } catch (error) {
        setStatus(statusEl, `Ошибка: ${error.message}`, "danger");
    } finally {
        setButtonBusy($("#stream-btn"), false);
    }
}

function renderBookPreview(data) {
    const previewEl = $("#book-preview");
    const total = data.total || 0;
    const segments = Array.isArray(data.segments) ? data.segments : [];
    $("#book-summary").textContent = `${total} абзацев`;

    if (!segments.length) {
        previewEl.innerHTML = '<div class="preview-empty">Нет данных для предпросмотра.</div>';
        return;
    }

    previewEl.innerHTML = segments
        .map((segment) => `
            <article class="preview-item">
                <div class="preview-index">${segment.id + 1}</div>
                <p>${escapeHtml(segment.text)}</p>
            </article>
        `)
        .join("");
}

async function uploadBook() {
    const fileInput = $("#book-file");
    const statusEl = $("#book-progress");
    const file = fileInput.files[0];

    if (!file) {
        setStatus(statusEl, "Выберите .txt файл.", "danger");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setButtonBusy($("#upload-book-btn"), true, "Загружаю…");
    setStatus(statusEl, "Загружаю и разбираю книгу…", "warning");

    try {
        const data = await apiFetch("/api/books/upload", { method: "POST", body: formData });
        state.currentBookJobId = data.job_id;
        renderBookPreview(data);
        $("#generate-book-btn").classList.remove("hidden");
        $("#book-download-link").classList.add("hidden");
        $("#book-progress-bar-wrap").classList.add("hidden");
        setStatus(statusEl, `Файл загружен. Абзацев: ${data.total}.`, "success");
    } catch (error) {
        setStatus(statusEl, `Ошибка: ${error.message}`, "danger");
    } finally {
        setButtonBusy($("#upload-book-btn"), false);
    }
}

function updateBookProgress(data) {
    const total = Number(data.total || 0);
    const processed = Number(data.processed || 0);
    const percent = total > 0 ? Math.round((processed / total) * 100) : 0;
    $("#book-progress-bar-wrap").classList.remove("hidden");
    $("#book-progress-bar").style.width = `${percent}%`;

    const statusEl = $("#book-progress");
    if (data.status === "queued") {
        setStatus(statusEl, "Задача поставлена в очередь…", "warning");
    } else if (data.status === "running") {
        setStatus(statusEl, `Озвучка: ${processed}/${total} (${percent}%).`, "warning");
    } else if (data.status === "completed") {
        setStatus(statusEl, `Готово. Создано файлов: ${data.files?.length || 0}.`, "success");
        if (data.download_url) {
            const link = $("#book-download-link");
            link.href = data.download_url;
            link.classList.remove("hidden");
        }
    } else if (data.status === "failed") {
        setStatus(statusEl, `Ошибка озвучки: ${data.error || "неизвестно"}`, "danger");
    } else {
        setStatus(statusEl, `Статус: ${data.status}`, "default");
    }
}

function stopBookPolling() {
    if (state.bookPollTimer) {
        clearInterval(state.bookPollTimer);
        state.bookPollTimer = null;
    }
}

async function pollBookJob() {
    if (!state.currentBookJobId) return;
    try {
        const data = await apiFetch(`/api/books/${state.currentBookJobId}`);
        updateBookProgress(data);
        if (["completed", "failed"].includes(data.status)) {
            stopBookPolling();
        }
    } catch (error) {
        stopBookPolling();
        setStatus($("#book-progress"), `Ошибка статуса: ${error.message}`, "danger");
    }
}

async function startBookGeneration() {
    if (!state.currentBookJobId) {
        setStatus($("#book-progress"), "Сначала загрузите книгу.", "danger");
        return;
    }

    setButtonBusy($("#generate-book-btn"), true, "Запускаю…");
    try {
        const payload = {
            voice: $("#book-voice").value,
            language: $("#book-language").value,
            response_format: $("#book-format").value,
            instruct: $("#book-instruct").value.trim() || null,
        };
        const data = await apiFetch(`/api/books/${state.currentBookJobId}/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        updateBookProgress(data);
        stopBookPolling();
        state.bookPollTimer = setInterval(pollBookJob, 2000);
        await pollBookJob();
    } catch (error) {
        setStatus($("#book-progress"), `Ошибка: ${error.message}`, "danger");
    } finally {
        setButtonBusy($("#generate-book-btn"), false);
    }
}

function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach((item) => {
                item.classList.remove("active");
                item.setAttribute("aria-selected", "false");
            });
            document.querySelectorAll(".tab-content").forEach((item) => item.classList.remove("active"));
            button.classList.add("active");
            button.setAttribute("aria-selected", "true");
            $(`#${button.dataset.tab}-tab`).classList.add("active");
        });
    });
}

function initPromptChips() {
    document.querySelectorAll("#prompt-chips .chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            $("#tts-instruct").value = chip.dataset.prompt;
        });
    });
}

async function init() {
    initTabs();
    initPromptChips();
    $("#tts-text").addEventListener("input", updateTtsCounter);
    $("#generate-btn").addEventListener("click", () => requestAudio("mp3"));
    $("#wav-btn").addEventListener("click", () => requestAudio("wav"));
    $("#stream-btn").addEventListener("click", streamAudio);
    $("#upload-book-btn").addEventListener("click", uploadBook);
    $("#generate-book-btn").addEventListener("click", startBookGeneration);

    updateTtsCounter();
    try {
        await loadMeta();
    } catch (error) {
        setStatus($("#tts-status"), `Не удалось загрузить конфигурацию: ${error.message}`, "danger");
        $("#engine-chip").textContent = "Backend недоступен";
        $("#engine-chip").dataset.state = "danger";
    }
}

window.addEventListener("beforeunload", () => {
    stopBookPolling();
    revokeAudioUrl();
});

init();
