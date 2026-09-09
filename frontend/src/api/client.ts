/**
 * Typed API client for the DRF backend.
 *
 * Endpoints (design: Components and Interfaces):
 *
 *   POST /api/analyses                -> submit a file (multipart) or URL (json)
 *   GET  /api/analyses/{id}           -> fetch session status/result metadata
 *   GET  /api/analyses/{id}/report    -> download PDF (WHERE REPORT_ENABLED)
 *   GET  /api/evaluations/{run_id}    -> retrieve persisted evaluation metrics
 *   GET  /api/evaluations/benchmark   -> Results-chapter benchmark tables
 *                                       (modality comparison + cross-dataset)
 */
import type {
  AnalysisSession,
  EvaluationBenchmark,
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
  static async submitFile(file: File, baseUrl = "/api"): Promise<UploadResult> {
    return new ApiClient({ baseUrl }).submitFileInstance(file);
  }

  /**
   * Submit an external video URL for analysis (json). Gated server-side by
   * EXTERNAL_URL_ENABLED.
   */
  static async submitUrl(url: string, baseUrl = "/api"): Promise<UploadResult> {
    return new ApiClient({ baseUrl }).submitUrlInstance(url);
  }

  /** Instance-based submission (used by the static helpers). */
  async submitFileInstance(file: File): Promise<UploadResult> {
    const body = new FormData();
    body.append("file", file);
    const res = await this.fetchImpl(`${this.baseUrl}/analyses`, {
      method: "POST",
      body,
    });
    return this.toUploadResult(res);
  }

  async submitUrlInstance(url: string): Promise<UploadResult> {
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
      `${this.baseUrl}/analyses/${encodeURIComponent(sessionId)}`
    );
    if (!res.ok) {
      throw new Error(`getSession failed: ${res.status}`);
    }
    return (await res.json()) as AnalysisSession;
  }

  /** Download the PDF report for a completed session (WHERE REPORT_ENABLED). */
  async getReport(sessionId: string): Promise<Blob> {
    const res = await this.fetchImpl(
      `${this.baseUrl}/analyses/${encodeURIComponent(sessionId)}/report`
    );
    if (!res.ok) {
      throw new Error(`getReport failed: ${res.status}`);
    }
    return await res.blob();
  }

  /** Retrieve persisted model evaluation metrics for a run. */
  async getEvaluation(runId: string): Promise<EvaluationMetrics> {
    const res = await this.fetchImpl(
      `${this.baseUrl}/evaluations/${encodeURIComponent(runId)}`
    );
    if (!res.ok) {
      throw new Error(`getEvaluation failed: ${res.status}`);
    }
    return (await res.json()) as EvaluationMetrics;
  }

  /**
   * Fetch the Results-chapter benchmark tables: modality comparison
   * (Multimodal > Visual-only > Audio-only) and cross-dataset generalization.
   */
  async getBenchmark(): Promise<EvaluationBenchmark> {
    const res = await this.fetchImpl(`${this.baseUrl}/evaluations/benchmark`);
    if (!res.ok) {
      throw new Error(`getBenchmark failed: ${res.status}`);
    }
    return (await res.json()) as EvaluationBenchmark;
  }

  /** Normalize an upload response (202 accepted or 4xx/5xx rejection) into UploadResult. */
  private async toUploadResult(res: Response): Promise<UploadResult> {
    const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
    if (res.ok) {
      return {
        accepted: true,
        session_id: (data.session_id as string) ?? null,
        error_code: null,
        message: null,
      };
    }
    return {
      accepted: false,
      session_id: null,
      error_code: (data.error_code as UploadResult["error_code"]) ?? null,
      message: (data.message as string) ?? null,
    };
  }
}