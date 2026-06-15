export class SensorBuffer {
    constructor({ streamId, maxBuffered = 1000 } = {}) {
        this.streamId = streamId || this.createStreamId();
        this.maxBuffered = maxBuffered;
        this.nextSeq = 0;
        this.pendingSeqs = [];
        this.unacked = new Map();
        this.droppedCount = 0;
    }

    createStreamId() {
        if (crypto.randomUUID) {
            return crypto.randomUUID();
        }
        return `stream-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    addReading(reading) {
        const seq = this.nextSeq;
        this.nextSeq += 1;

        const normalized = {
            stream_id: this.streamId,
            seq,
            heart_rate: reading.heart_rate,
            gsr: reading.gsr,
            captured_at: new Date().toISOString(),
            source: "browser_serial",
            sent_at_ms: null
        };

        this.unacked.set(seq, normalized);
        this.pendingSeqs.push(seq);
        this.enforceLimit();
        return normalized;
    }

    enforceLimit() {
        while (this.unacked.size > this.maxBuffered) {
            const oldestSeq = this.unacked.keys().next().value;
            this.unacked.delete(oldestSeq);
            this.pendingSeqs = this.pendingSeqs.filter((seq) => seq !== oldestSeq);
            this.droppedCount += 1;
        }
    }

    getSendableBatch({ maxCount = 10, retryAfterMs = 1000 } = {}) {
        const now = Date.now();
        const batch = [];
        const uniqueSeqs = new Set(this.pendingSeqs);

        for (const [seq, reading] of this.unacked.entries()) {
            if (batch.length >= maxCount) {
                break;
            }
            const wasPending = uniqueSeqs.has(seq);
            const shouldRetry = reading.sent_at_ms && now - reading.sent_at_ms >= retryAfterMs;
            if (!reading.sent_at_ms || wasPending || shouldRetry) {
                batch.push(reading);
            }
        }

        return batch.slice(0, maxCount);
    }

    markSent(readings) {
        const now = Date.now();
        for (const reading of readings) {
            const tracked = this.unacked.get(reading.seq);
            if (tracked) {
                tracked.sent_at_ms = now;
            }
        }
        const sentSeqs = new Set(readings.map((reading) => reading.seq));
        this.pendingSeqs = this.pendingSeqs.filter((seq) => !sentSeqs.has(seq));
    }

    ackThrough(lastAcceptedSeq) {
        for (const seq of Array.from(this.unacked.keys())) {
            if (seq <= lastAcceptedSeq) {
                this.unacked.delete(seq);
            }
        }
        this.pendingSeqs = this.pendingSeqs.filter((seq) => seq > lastAcceptedSeq);
    }

    consumeDroppedCount() {
        const count = this.droppedCount;
        this.droppedCount = 0;
        return count;
    }

    stats() {
        return {
            streamId: this.streamId,
            pending: this.pendingSeqs.length,
            unacked: this.unacked.size,
            dropped: this.droppedCount
        };
    }
}
