/**
 * Typed WebSocket client stub for the Result_Streamer.
 *
 * Connects to /ws/analyses/{session_id}/ and relays sequence-numbered stream
 * events to subscribers. This stub establishes the typed surface and the
 * reconnect/resume scaffolding called out in the design:
 *
 *   - establish a connection within a 3s budget, retrying up to 3 times (5.1, 5.2)
 *   - track the last received `seq` and send a `resume {last_seq}` on reconnect (5.5)
 *
 * The concrete reconnect timing/backoff behaviour and Dashboard rendering are
 * completed in a later task (Task 17.2). Kept intentionally minimal here.
 */
import { ApiClient } from "../api/client";
import type { ResumeRequest, StreamEvent } from "./types";

export type StreamEventHandler = (event: StreamEvent) => void;
export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed"
  | "error";
export type ConnectionStateHandler = (state: ConnectionState) => void;

export interface ResultStreamerOptions {
  /** Max connection attempts before surfacing an error (default 3 per Req 5.2). */
  maxRetries?: number;
  /** Per-attempt connection timeout in ms (default 3000 per Req 5.1). */
  connectTimeoutMs?: number;
  /** Injectable WebSocket constructor (useful for tests). */
  webSocketImpl?: typeof WebSocket;
  /** Override the stream URL (defaults to ApiClient.streamUrl(sessionId)). */
  url?: string;
}

export class ResultStreamer {
  private readonly sessionId: string;
  private readonly url: string;
  private readonly maxRetries: number;
  private readonly connectTimeoutMs: number;
  private readonly WebSocketImpl: typeof WebSocket;

  private socket: WebSocket | null = null;
  private attempts = 0;
  private lastSeq = 0;
  private state: ConnectionState = "idle";

  private readonly eventHandlers = new Set<StreamEventHandler>();
  private readonly stateHandlers = new Set<ConnectionStateHandler>();

  constructor(sessionId: string, options: ResultStreamerOptions = {}) {
    this.sessionId = sessionId;
    this.url = options.url ?? ApiClient.streamUrl(sessionId);
    this.maxRetries = options.maxRetries ?? 3;
    this.connectTimeoutMs = options.connectTimeoutMs ?? 3000;
    this.WebSocketImpl = options.webSocketImpl ?? WebSocket;
  }

  /** The Analysis_Session this streamer is bound to. */
  get analysisSessionId(): string {
    return this.sessionId;
  }

  /** The highest stream sequence number observed so far. */
  get lastSequence(): number {
    return this.lastSeq;
  }

  /** Current connection state. */
  get connectionState(): ConnectionState {
    return this.state;
  }

  /** Subscribe to stream events. Returns an unsubscribe function. */
  onEvent(handler: StreamEventHandler): () => void {
    this.eventHandlers.add(handler);
    return () => this.eventHandlers.delete(handler);
  }

  /** Subscribe to connection-state transitions. Returns an unsubscribe function. */
  onStateChange(handler: ConnectionStateHandler): () => void {
    this.stateHandlers.add(handler);
    return () => this.stateHandlers.delete(handler);
  }

  /** Open the WebSocket connection (idempotent while open/connecting). */
  connect(): void {
    if (this.state === "open" || this.state === "connecting") {
      return;
    }
    this.attempts = 0;
    this.openSocket();
  }

  /** Close the connection and stop reconnect attempts. */
  close(): void {
    this.setState("closed");
    this.socket?.close();
    this.socket = null;
  }

  private openSocket(): void {
    this.setState(this.attempts === 0 ? "connecting" : "reconnecting");
    this.attempts += 1;

    const socket = new this.WebSocketImpl(this.url);
    this.socket = socket;

    const timeout = setTimeout(() => {
      if (socket.readyState !== socket.OPEN) {
        socket.close();
        this.handleConnectFailure();
      }
    }, this.connectTimeoutMs);

    socket.onopen = () => {
      clearTimeout(timeout);
      this.attempts = 0;
      this.setState("open");
      // Resume any events missed during a prior interruption (Req 5.5).
      this.sendResume();
    };

    socket.onmessage = (ev: MessageEvent) => {
      this.handleMessage(ev.data);
    };

    socket.onerror = () => {
      clearTimeout(timeout);
      this.handleConnectFailure();
    };

    socket.onclose = () => {
      clearTimeout(timeout);
      if (this.state !== "closed") {
        this.handleConnectFailure();
      }
    };
  }

  private handleConnectFailure(): void {
    if (this.attempts < this.maxRetries) {
      this.openSocket();
    } else {
      this.setState("error");
    }
  }

  private handleMessage(raw: unknown): void {
    let event: StreamEvent;
    try {
      event = typeof raw === "string" ? (JSON.parse(raw) as StreamEvent) : (raw as StreamEvent);
    } catch {
      return;
    }
    if (typeof event.seq === "number" && event.seq > this.lastSeq) {
      this.lastSeq = event.seq;
    }
    for (const handler of this.eventHandlers) {
      handler(event);
    }
  }

  private sendResume(): void {
    if (!this.socket || this.socket.readyState !== this.socket.OPEN) {
      return;
    }
    const request: ResumeRequest = { type: "resume", last_seq: this.lastSeq };
    this.socket.send(JSON.stringify(request));
  }

  private setState(state: ConnectionState): void {
    this.state = state;
    for (const handler of this.stateHandlers) {
      handler(state);
    }
  }
}
