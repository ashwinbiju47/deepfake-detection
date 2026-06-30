import { describe, expect, it } from "vitest";
import fc from "fast-check";
import { ApiClient } from "../api/client";

/**
 * Tooling smoke test: verifies Vitest + fast-check are wired up and that the
 * API client stub builds the expected WebSocket stream URL. This is not one of
 * the spec's 22 correctness properties — those are added in their own tasks.
 */
describe("frontend test harness", () => {
  it("runs a fast-check property", () => {
    fc.assert(
      fc.property(fc.integer(), fc.integer(), (a, b) => {
        return a + b === b + a;
      }),
    );
  });

  it("builds a ws:// stream URL from an http origin", () => {
    const url = ApiClient.streamUrl("abc-123", "http://localhost:8000");
    expect(url).toBe("ws://localhost:8000/ws/analyses/abc-123/");
  });

  it("builds a wss:// stream URL from an https origin", () => {
    const url = ApiClient.streamUrl("s 1", "https://example.com");
    expect(url).toBe("wss://example.com/ws/analyses/s%201/");
  });
});
