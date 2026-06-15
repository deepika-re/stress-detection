import assert from "node:assert/strict";
import test from "node:test";

import { parseSerialLine } from "../app/static/js/serial_line_parser.js";

test("parses compact serial lines", () => {
  assert.deepEqual(parseSerialLine("HR:82,GSR:0.63"), {
    heart_rate: 82,
    gsr: 0.63
  });
});

test("parses serial lines with spaces", () => {
  assert.deepEqual(parseSerialLine("HR: 94, GSR: 1.25"), {
    heart_rate: 94,
    gsr: 1.25
  });
});

test("rejects malformed lines", () => {
  assert.equal(parseSerialLine("HR=94 GSR=1.25"), null);
  assert.equal(parseSerialLine("hello"), null);
});
