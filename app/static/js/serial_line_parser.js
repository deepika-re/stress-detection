const SENSOR_LINE_PATTERN = /^HR\s*:\s*([+-]?\d+(?:\.\d+)?)\s*,\s*GSR\s*:\s*([+-]?\d+(?:\.\d+)?)$/i;

export function parseSerialLine(line) {
    if (typeof line !== "string") {
        return null;
    }

    const match = line.trim().match(SENSOR_LINE_PATTERN);
    if (!match) {
        return null;
    }

    const heartRate = Number(match[1]);
    const gsr = Number(match[2]);
    if (!Number.isFinite(heartRate) || !Number.isFinite(gsr)) {
        return null;
    }

    return {
        heart_rate: heartRate,
        gsr
    };
}
