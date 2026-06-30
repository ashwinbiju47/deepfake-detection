/**
 * Typed API client stub for the DRF backend.
 *
 * This module defines the surface the Dashboard uses to talk to the
 * Upload_Service and session/evaluation endpoints described in the design:
 *
 *   POST /api/analyses                -> submit a file (multipart) or URL (json)
 *   GET  /api/analyses/{id}           -> fetch session status/result metadata
 *   GET  /api/analyses/{id}/report    -> download PDF (WHERE REPORT_ENABLED)
 *   GET  /api/evaluations/{run_id}    -> retrieve persisted evaluation metrics
 *
 * The HTTP wiring is implemented here as a thin fetch wrapper; the full
 * Dashboard behaviour (forms, polling, rendering) is built in a later task.
 */
import type {
  AnalysisSession,
  EvaluationMetrics,
  UploadResult,
} from "./types";

export interface ApiClientOptions {
  /** Base URL for the REST API. Defaults to a same-origin "/api" prefix. */
  baseUrl?: string;
  /** Injectable fetch implementation (useful for tests). Defaults to global fetch. */
  fetchImpl?: typeof fetch;
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "/api").replace(/\/$/, "");
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  /** Build the WebSocket URL for a session's result stream. */
  static streamUrl(sessionId: string, origin: string = location.origin): string {
    const wsOrigin = origin.replace(/^http/, "ws");
    return `${wsOrigin}/ws/analyses/${encodeURIComponent(sessionId)}/`;
  }

  /**
   * Submit a video file for analysis (multipart). Returns 202 {session_id} on
   * acceptance or a typed error on rejection.
   */
  async submitFile(file: File): Promise<UploadResult> {
    const body = new FormData();
    body.append("file", file);
    const res = await this.fetchImpl(`${this.baseUrl}/analyses`, {
      method: "POST",
      body,
    });
    return this.toUploadResult(res);
  }

  /**
   * Submit an external video URL for analysis (json). Gated server-side by
   * EXTERNAL_URL_ENABLED.
   */
  async submitUrl(url: string): Promise<UploadResult> {
    const res = await this.fetchImpl(`${this.baseUrl}/analyses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    return this.toUploadResult(res);
  }

  /** Fetch status/result metadata for a session. */
  async getSession(sessionId: string): Promise<AnalysisSession> {
    const res = await this.fetchImpl(
      `${this.baseUrl}/analyses/${encodeURIComponent(sessionId)}`,
    );
    if (!res.ok) {
      throw new Error(`getSession failed: ${res.status}`);
    }
    return (await res.json()) as AnalysisSession;
  }

  /** Download the PDF report for a completed session (WHERE REPORT_ENABLED). */
  async getReport(sessionId: string): Promise<Blob> {
    const res = await this.fetchImpl(
      `${this.baseUrl}/analyses/${encodeURIComponent(sessionId)}/report`,
    );
    if (!res.ok) {
      throw new Error(`getReport failed: ${res.status}`);
    }
    return await res.blob();
  }

  /** Retrieve persisted model evaluation metrics for a run. */
  async getEvaluation(runId: string): Promise<EvaluationMetrics> {
    const res = await this.fetchImpl(
      `${this.baseUrl}/evaluations/${encodeURIComponent(runId)}`,
    );
    if (!res.ok) {
      throw new Error(`getEvaluation failed: ${res.status}`);
    }
    return (await res.json()) as EvaluationMetrics;
  }

  /** Normalize an upload response (202 accepted or 4xx/5xx rejection) into UploadResult. */
  private async toUploadResult(res: Response): Promise<UploadResult> {
    const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
    if (res.ok) {
      return {
        accepted: true,
        sessionId: (data.session_id as string) ?? null,
        errorCode: null,
        message: null,
      };
    }
    return {
      accepted: false,
      sessionId: null,
      errorCode: (data.error_code as UploadResult["errorCode"]) ?? null,
      message: (data.message as string) ?? null,
    };
  }
}
