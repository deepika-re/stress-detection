export class StressSocketClient {
    constructor({ streamId, onStatus, onAck, onLiveUpdate, onError }) {
        this.streamId = streamId;
        this.onStatus = onStatus;
        this.onAck = onAck;
        this.onLiveUpdate = onLiveUpdate;
        this.onError = onError;
        this.socket = io({ transports: ["websocket", "polling"], upgrade: true });
        this.bind();
    }

    bind() {
        this.socket.on("connect", () => {
            this.onStatus?.("connected");
            this.socket.emit("start_stream", { stream_id: this.streamId });
        });

        this.socket.on("disconnect", () => {
            this.onStatus?.("disconnected");
        });

        this.socket.on("connect_error", (error) => {
            this.onStatus?.("error");
            this.onError?.(error.message || "Socket connection failed");
        });

        this.socket.on("sensor_ack", (payload) => {
            this.onAck?.(payload);
        });

        this.socket.on("live_update", (payload) => {
            this.onLiveUpdate?.(payload);
        });

        this.socket.on("stream_error", (payload) => {
            this.onError?.(payload.message || "Stream error");
        });
    }

    isConnected() {
        return Boolean(this.socket?.connected);
    }

    sendSensorBatch(readings, droppedCount) {
        if (!this.isConnected() || !readings.length) {
            return false;
        }

        this.socket.emit("sensor_batch", {
            stream_id: this.streamId,
            readings,
            dropped_count: droppedCount
        });
        return true;
    }

    stop() {
        if (this.isConnected()) {
            this.socket.emit("stop_stream", { stream_id: this.streamId });
        }
    }
}
