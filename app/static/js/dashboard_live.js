import { SensorBuffer } from "./sensor_buffer.js";
import { StressSocketClient } from "./socket_client.js";
import { WebSerialManager } from "./web_serial_manager.js";

const dashboard = document.querySelector(".dashboard-shell");
const faceIntervalMs = Math.max(Number(dashboard?.dataset.faceInterval || 1.4) * 1000, 1000);

const elements = {
    connect: document.getElementById("connectDevice"),
    disconnect: document.getElementById("disconnectDevice"),
    serialSupport: document.getElementById("serialSupport"),
    serialFallback: document.getElementById("serialFallback"),
    deviceStatus: document.getElementById("deviceStatus"),
    socketStatus: document.getElementById("socketStatus"),
    ackStatus: document.getElementById("ackStatus"),
    streamId: document.getElementById("streamId"),
    droppedCount: document.getElementById("droppedCount"),
    bufferStatus: document.getElementById("bufferStatus"),
    heartRate: document.getElementById("heartRate"),
    gsr: document.getElementById("gsr"),
    stressScore: document.getElementById("stressScore"),
    stressBanner: document.getElementById("stressBanner"),
    stressMeterFill: document.getElementById("stressMeterFill"),
    stressMeterValue: document.getElementById("stressMeterValue"),
    stressReasons: document.getElementById("stressReasons"),
    lastUpdated: document.getElementById("lastUpdated"),
    latestSeq: document.getElementById("latestSeq"),
    serialLog: document.getElementById("serialLog"),
    chartStatus: document.getElementById("chartStatus"),
    tipTitle: document.getElementById("tipTitle"),
    tipList: document.getElementById("tipList"),
    componentHr: document.getElementById("componentHr"),
    componentGsr: document.getElementById("componentGsr"),
    componentFace: document.getElementById("componentFace"),
    faceStart: document.getElementById("faceStart"),
    faceStop: document.getElementById("faceStop"),
    faceStatus: document.getElementById("faceStatus"),
    faceVideo: document.getElementById("faceVideo"),
    faceCanvas: document.getElementById("faceCanvas"),
    facePlaceholder: document.getElementById("facePlaceholder"),
    faceScore: document.getElementById("faceScore"),
    faceEmotion: document.getElementById("faceEmotion"),
    faceConfidence: document.getElementById("faceConfidence"),
    faceLatency: document.getElementById("faceLatency"),
    faceDistribution: document.getElementById("faceDistribution"),
    faceModel: document.getElementById("faceModel")
};

const TIPS = {
    LOW: ["Maintain the current rhythm.", "Hydrate before readings drift.", "Keep shoulders relaxed."],
    MEDIUM: ["Slow breathing for one minute.", "Look away from the screen.", "Release jaw and neck tension."],
    HIGH: ["Pause the current task.", "Use box breathing for two minutes.", "Notify someone nearby if symptoms persist."]
};

const sensorBuffer = new SensorBuffer({ maxBuffered: 1000 });
elements.streamId.textContent = sensorBuffer.streamId;

let pendingLocalReading = null;
let chart;
let chartQueue = [];
let lastChartDraw = 0;

const faceState = {
    running: false,
    stream: null,
    timer: null,
    inFlight: false,
    abortController: null,
    requestId: 0,
    errorCount: 0
};

const socketClient = new StressSocketClient({
    streamId: sensorBuffer.streamId,
    onStatus: (status) => {
        elements.socketStatus.textContent = status;
    },
    onAck: (payload) => {
        sensorBuffer.ackThrough(payload.last_accepted_seq);
        elements.ackStatus.textContent = `ACK through seq ${payload.last_accepted_seq}`;
        elements.droppedCount.textContent = payload.dropped_count ?? "0";
        updateBufferStats();
    },
    onLiveUpdate: (payload) => {
        queueLiveUpdate(payload);
    },
    onError: (message) => {
        appendLog(`socket error: ${message}`);
    }
});

const serialManager = new WebSerialManager({
    onStatus: (status) => {
        elements.deviceStatus.textContent = status;
    },
    onReading: (reading) => {
        const buffered = sensorBuffer.addReading(reading);
        pendingLocalReading = buffered;
        requestAnimationFrame(renderLocalReading);
        updateBufferStats();
    },
    onLine: (line) => {
        appendLog(line);
    },
    onError: (message) => {
        appendLog(`serial error: ${message}`);
    }
});

if (serialManager.isSupported()) {
    elements.serialSupport.textContent = "supported";
    elements.serialFallback.textContent = "Chromium serial access is available.";
} else {
    elements.serialSupport.textContent = "unsupported";
    elements.serialFallback.textContent = "Use Chromium on localhost or HTTPS.";
    elements.connect.disabled = true;
}

elements.connect.addEventListener("click", () => {
    serialManager.connect();
});

elements.disconnect.addEventListener("click", async () => {
    await serialManager.disconnect();
    socketClient.stop();
});

elements.faceStart.addEventListener("click", startFaceCamera);
elements.faceStop.addEventListener("click", stopFaceCamera);
elements.faceStop.disabled = true;

window.setInterval(flushSensorBatch, 60);
window.setInterval(drawChartIfNeeded, 240);
loadHistory();

function flushSensorBatch() {
    if (!socketClient.isConnected()) {
        updateBufferStats();
        return;
    }

    const batch = sensorBuffer.getSendableBatch({ maxCount: 12, retryAfterMs: 850 });
    if (!batch.length) {
        updateBufferStats();
        return;
    }

    const dropped = sensorBuffer.consumeDroppedCount();
    const sent = socketClient.sendSensorBatch(batch, dropped);
    if (sent) {
        sensorBuffer.markSent(batch);
    }
    updateBufferStats();
}

function renderLocalReading() {
    if (!pendingLocalReading) {
        return;
    }
    elements.heartRate.textContent = pendingLocalReading.heart_rate.toFixed(0);
    elements.gsr.textContent = pendingLocalReading.gsr.toFixed(2);
    elements.lastUpdated.textContent = new Date(pendingLocalReading.captured_at).toLocaleTimeString();
    elements.latestSeq.textContent = `local seq ${pendingLocalReading.seq}`;
}

function queueLiveUpdate(payload) {
    elements.heartRate.textContent = payload.heart_rate.toFixed(0);
    elements.gsr.textContent = payload.gsr.toFixed(2);
    elements.stressScore.textContent = payload.stress_score;
    elements.lastUpdated.textContent = new Date(payload.received_at).toLocaleTimeString();
    elements.latestSeq.textContent = `server seq ${payload.seq}`;
    elements.stressReasons.textContent = (payload.reasons || []).map(formatReason).join(", ");
    renderStressLevel(payload.stress_level, payload.stress_score);
    renderTips(payload.stress_level);
    renderComponents(payload.components || {});

    chartQueue.push(payload);
    if (chartQueue.length > 300) {
        chartQueue = chartQueue.slice(-300);
    }
}

function renderStressLevel(level, score) {
    const normalized = (level || "LOW").toLowerCase();
    elements.stressBanner.className = `stress-banner level-${normalized}`;
    elements.stressBanner.querySelector(".stress-state strong").textContent = level || "LOW";
    renderStressMeter(score);
}

function renderStressMeter(score) {
    const numericScore = Number(score);
    const boundedScore = Number.isFinite(numericScore) ? Math.max(0, Math.min(100, numericScore)) : 0;
    elements.stressMeterFill.style.width = `${boundedScore}%`;
    elements.stressMeterValue.textContent = Number.isFinite(numericScore) ? Math.round(boundedScore) : "--";
    elements.stressBanner.querySelector(".stress-meter").setAttribute("aria-valuenow", Math.round(boundedScore));
}

function renderTips(level) {
    const normalized = level || "LOW";
    const tips = TIPS[normalized] || TIPS.LOW;
    elements.tipTitle.textContent = normalized === "HIGH" ? "Take a reset" : normalized === "MEDIUM" ? "Ease the load" : "Stable";
    elements.tipList.replaceChildren(...tips.map((tip) => {
        const item = document.createElement("li");
        item.textContent = tip;
        return item;
    }));
}

function renderComponents(components) {
    elements.componentHr.textContent = formatComponent(components.heart_rate);
    elements.componentGsr.textContent = formatComponent(components.gsr);
    elements.componentFace.textContent = components.face === null || components.face === undefined
        ? "--"
        : formatComponent(components.face);
}

function formatComponent(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return "--";
    }
    return `${Math.round(value * 100)}%`;
}

function formatReason(reason) {
    return String(reason || "").replaceAll("_", " ");
}

function updateBufferStats() {
    const stats = sensorBuffer.stats();
    elements.bufferStatus.textContent = `${stats.pending} pending, ${stats.unacked} unacked`;
    if (stats.dropped) {
        elements.droppedCount.textContent = stats.dropped;
    }
}

function appendLog(line) {
    const current = elements.serialLog.textContent.split("\n").filter(Boolean);
    current.push(line);
    elements.serialLog.textContent = current.slice(-40).join("\n");
    elements.serialLog.scrollTop = elements.serialLog.scrollHeight;
}

function ensureChart() {
    if (chart || !window.Chart) {
        return chart;
    }

    const ctx = document.getElementById("readingsChart").getContext("2d");
    Chart.defaults.color = "#64736e";
    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Heart rate",
                    data: [],
                    borderColor: "#e4572e",
                    backgroundColor: "rgba(228, 87, 46, 0.08)",
                    fill: true,
                    tension: 0.25,
                    pointRadius: 0
                },
                {
                    label: "GSR",
                    data: [],
                    borderColor: "#0f766e",
                    backgroundColor: "rgba(15, 118, 110, 0.08)",
                    fill: true,
                    tension: 0.25,
                    pointRadius: 0
                },
                {
                    label: "Stress score",
                    data: [],
                    borderColor: "#f0b429",
                    backgroundColor: "rgba(240, 180, 41, 0.08)",
                    fill: false,
                    tension: 0.25,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            animation: false,
            plugins: {
                legend: {
                    position: "top"
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(100, 115, 110, 0.14)" }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: "rgba(100, 115, 110, 0.14)" }
                }
            }
        }
    });
    return chart;
}

function drawChartIfNeeded() {
    if (!chartQueue.length || Date.now() - lastChartDraw < 240) {
        return;
    }

    const chartInstance = ensureChart();
    if (!chartInstance) {
        elements.chartStatus.textContent = "Chart.js unavailable";
        return;
    }

    for (const reading of chartQueue.splice(0)) {
        chartInstance.data.labels.push(new Date(reading.received_at).toLocaleTimeString());
        chartInstance.data.datasets[0].data.push(reading.heart_rate);
        chartInstance.data.datasets[1].data.push(reading.gsr);
        chartInstance.data.datasets[2].data.push(reading.stress_score);
    }

    const maxPoints = 140;
    while (chartInstance.data.labels.length > maxPoints) {
        chartInstance.data.labels.shift();
        for (const dataset of chartInstance.data.datasets) {
            dataset.data.shift();
        }
    }

    chartInstance.update("none");
    lastChartDraw = Date.now();
    elements.chartStatus.textContent = `${chartInstance.data.labels.length} points`;
}

async function loadHistory() {
    try {
        const response = await fetch("/api/readings/history?limit=100");
        if (!response.ok) {
            return;
        }
        const history = await response.json();
        for (const reading of history) {
            chartQueue.push({
                ...reading,
                reasons: [],
                received_at: reading.received_at
            });
        }
        drawChartIfNeeded();
    } catch {
        elements.chartStatus.textContent = "History unavailable";
    }
}

async function startFaceCamera() {
    if (faceState.running) {
        return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
        setFaceStatus("camera unsupported", "error");
        return;
    }

    try {
        setFaceStatus("starting", "active");
        faceState.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });
        elements.faceVideo.srcObject = faceState.stream;
        await elements.faceVideo.play();
        faceState.running = true;
        elements.faceStart.disabled = true;
        elements.faceStop.disabled = false;
        elements.faceVideo.style.display = "block";
        elements.facePlaceholder.style.display = "none";
        faceState.errorCount = 0;
        setFaceStatus("analyzing", "active");
        scheduleFaceCapture(100);
    } catch (error) {
        stopFaceCamera({ updateStatus: false });
        setFaceStatus(error.message || "camera blocked", "error");
    }
}

function stopFaceCamera({ updateStatus = true } = {}) {
    faceState.running = false;
    window.clearTimeout(faceState.timer);
    faceState.timer = null;
    if (faceState.abortController) {
        faceState.abortController.abort();
        faceState.abortController = null;
    }
    if (faceState.stream) {
        for (const track of faceState.stream.getTracks()) {
            track.stop();
        }
    }
    faceState.stream = null;
    faceState.inFlight = false;
    faceState.errorCount = 0;
    elements.faceVideo.srcObject = null;
    elements.faceVideo.style.display = "none";
    elements.facePlaceholder.style.display = "grid";
    elements.faceStart.disabled = false;
    elements.faceStop.disabled = true;
    if (updateStatus) {
        setFaceStatus("idle");
    }
}

function scheduleFaceCapture(delay = faceIntervalMs) {
    window.clearTimeout(faceState.timer);
    faceState.timer = window.setTimeout(captureFaceFrame, delay);
}

async function captureFaceFrame() {
    if (!faceState.running) {
        return;
    }
    if (faceState.inFlight) {
        scheduleFaceCapture();
        return;
    }
    if (!drawVideoFrame(elements.faceVideo, elements.faceCanvas)) {
        scheduleFaceCapture(350);
        return;
    }

    const image = elements.faceCanvas.toDataURL("image/jpeg", 0.62);
    const requestId = ++faceState.requestId;
    faceState.inFlight = true;
    faceState.abortController = new AbortController();
    let nextDelay = faceIntervalMs;

    try {
        const response = await fetch("/api/face-inference", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image,
                captured_at: new Date().toISOString()
            }),
            signal: faceState.abortController.signal
        });
        const data = await response.json();
        if (requestId !== faceState.requestId || !faceState.running) {
            return;
        }
        if (!response.ok || data.status !== "ok") {
            const error = new Error(data.message || "face inference unavailable");
            error.retryAfterMs = Number(data.retry_after_seconds || 0) * 1000;
            throw error;
        }
        faceState.errorCount = 0;
        renderFaceResult(data);
        setFaceStatus("analyzing", "active");
    } catch (error) {
        if (error.name !== "AbortError") {
            faceState.errorCount += 1;
            nextDelay = inferenceRetryDelay(error);
            const seconds = Math.max(1, Math.round(nextDelay / 1000));
            setFaceStatus(`${error.message || "face model error"}; retrying in ${seconds}s`, "error");
        }
    } finally {
        faceState.inFlight = false;
        faceState.abortController = null;
        if (faceState.running) {
            scheduleFaceCapture(nextDelay);
        }
    }
}

function inferenceRetryDelay(error) {
    if (Number.isFinite(error.retryAfterMs) && error.retryAfterMs > 0) {
        return Math.min(Math.max(error.retryAfterMs, faceIntervalMs), 45000);
    }
    const multiplier = 2 ** Math.min(faceState.errorCount, 5);
    return Math.min(faceIntervalMs * multiplier, 45000);
}

function drawVideoFrame(video, canvas) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) {
        return false;
    }

    const size = Math.min(width, height);
    const sourceX = Math.floor((width - size) / 2);
    const sourceY = Math.floor((height - size) / 2);
    const context = canvas.getContext("2d", { alpha: false });
    context.save();
    context.scale(-1, 1);
    context.drawImage(video, sourceX, sourceY, size, size, -canvas.width, 0, canvas.width, canvas.height);
    context.restore();
    return true;
}

function renderFaceResult(data) {
    elements.faceScore.textContent = Math.round(data.stress_score * 100);
    elements.faceEmotion.textContent = titleCase(data.dominant_emotion || "unknown");
    elements.faceConfidence.textContent = `confidence ${Math.round((data.confidence || 0) * 100)}%`;
    elements.faceLatency.textContent = `latency ${data.latency_ms ?? "--"} ms`;
    elements.faceModel.textContent = `${data.provider || "hf"} / ${data.model || "model"}`;
    renderEmotionBars(data.distribution || []);
}

function renderEmotionBars(distribution) {
    const rows = distribution.slice(0, 5).map((row) => {
        const wrapper = document.createElement("div");
        wrapper.className = "emotion-bar";

        const label = document.createElement("span");
        label.textContent = titleCase(row.label);

        const track = document.createElement("div");
        track.className = "emotion-track";

        const fill = document.createElement("span");
        fill.className = "emotion-fill";
        fill.style.width = `${Math.round(row.score * 100)}%`;
        track.append(fill);

        const value = document.createElement("span");
        value.textContent = `${Math.round(row.score * 100)}%`;

        wrapper.append(label, track, value);
        return wrapper;
    });
    elements.faceDistribution.replaceChildren(...rows);
}

function setFaceStatus(text, state = "") {
    elements.faceStatus.textContent = text;
    elements.faceStatus.className = `status-pill ${state}`.trim();
}

function titleCase(value) {
    return String(value || "")
        .split(/[\s_-]+/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}
