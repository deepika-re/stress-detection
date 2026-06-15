import { parseSerialLine } from "./serial_line_parser.js";

export class WebSerialManager {
    constructor({ onStatus, onReading, onLine, onError }) {
        this.onStatus = onStatus;
        this.onReading = onReading;
        this.onLine = onLine;
        this.onError = onError;
        this.port = null;
        this.reader = null;
        this.keepReading = false;
        this.textBuffer = "";
    }

    isSupported() {
        return "serial" in navigator;
    }

    async connect() {
        if (!this.isSupported()) {
            this.onStatus?.("unsupported");
            this.onError?.("Web Serial is unsupported. Use Chromium on localhost/HTTPS, or a Local Bridge fallback.");
            return;
        }

        this.onStatus?.("connecting");
        try {
            this.port = await navigator.serial.requestPort();
            await this.port.open({ baudRate: 115200 });
            this.keepReading = true;
            this.onStatus?.("connected");
            this.readLoop();
        } catch (error) {
            this.onStatus?.("error");
            this.onError?.(error.message || "Unable to connect to serial device");
        }
    }

    async disconnect() {
        this.keepReading = false;
        try {
            if (this.reader) {
                await this.reader.cancel();
                this.reader.releaseLock();
            }
        } catch {
            // Reader cancellation can race with device removal.
        }
        try {
            if (this.port) {
                await this.port.close();
            }
        } catch {
            // Closing an already-detached device is harmless for the UI.
        }
        this.reader = null;
        this.port = null;
        this.onStatus?.("disconnected");
    }

    async readLoop() {
        this.onStatus?.("streaming");
        try {
            if (window.TextDecoderStream && this.port.readable?.pipeTo) {
                await this.readWithDecoderStream();
            } else {
                await this.readWithDecoder();
            }
        } catch (error) {
            if (this.keepReading) {
                this.onStatus?.("error");
                this.onError?.(error.message || "Serial stream failed");
            }
        }
    }

    async readWithDecoderStream() {
        while (this.port?.readable && this.keepReading) {
            const decoder = new TextDecoderStream();
            const closed = this.port.readable.pipeTo(decoder.writable).catch(() => {});
            this.reader = decoder.readable.getReader();
            try {
                while (this.keepReading) {
                    const { value, done } = await this.reader.read();
                    if (done) {
                        break;
                    }
                    this.handleChunk(value);
                }
            } finally {
                this.reader.releaseLock();
                await closed;
            }
        }
    }

    async readWithDecoder() {
        const decoder = new TextDecoder();
        while (this.port?.readable && this.keepReading) {
            this.reader = this.port.readable.getReader();
            try {
                while (this.keepReading) {
                    const { value, done } = await this.reader.read();
                    if (done) {
                        break;
                    }
                    this.handleChunk(decoder.decode(value, { stream: true }));
                }
            } finally {
                this.reader.releaseLock();
            }
        }
    }

    handleChunk(chunk) {
        this.textBuffer += chunk;
        const lines = this.textBuffer.split(/\r?\n/);
        this.textBuffer = lines.pop() || "";

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) {
                continue;
            }
            this.onLine?.(trimmed);
            const parsed = parseSerialLine(trimmed);
            if (parsed) {
                this.onReading?.(parsed);
            }
        }
    }
}
