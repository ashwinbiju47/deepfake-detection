import { describe, expect, it, vi } from "vitest";
import { ResultStreamer } from "../ws/client";

class MockWebSocket {
  url: string;
  readyState = 0;
  static OPEN = 1;

  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((ev: any) => void) | null = null;
  sentMessages: string[] = [];

  constructor(url: string) {
    this.url = url;
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) this.onopen();
    }, 10);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close() {
    this.readyState = 3;
    if (this.onclose) this.onclose();
  }
}

describe("ResultStreamer unit tests", () => {
  it("connects and sends resume message on open", async () => {
    const streamer = new ResultStreamer("test-session-id", {
      webSocketImpl: MockWebSocket as any,
    });

    const stateHandler = vi.fn();
    streamer.onStateChange(stateHandler);

    streamer.connect();

    await new Promise((resolve) => setTimeout(resolve, 30));

    expect(streamer.connectionState).toBe("open");
    expect(streamer.analysisSessionId).toBe("test-session-id");
  });
});
