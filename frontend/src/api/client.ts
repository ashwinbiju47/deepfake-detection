/**
 * Typed API client for the DRF backend.
 *
 * Endpoints (design: Components and Interfaces):
 *
 *   POST /api/analyses                -> submit a media file (multipart)
 *   GET  /api/analyses/{id}           -> fetch last-analysis metrics/status
 *   GET  /api/analyses/{id}/report    -> download PDF (WHERE REPORT_ENABLED)
 *   GET  /api/evaluations/{run_id}    -> retrieve persisted evaluation metrics
 *   GET  /api/evaluations/benchmark   -> Results-chapter benchmark tables
 *                                       (modality comparison + cross-dataset)
 */
import type {
  AnalysisSessionDetail,
  EvaluationBenchmark,
  EvaluationMetrics,
  HealthStatus,
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
   * Submit a media file (video / image / audio) for analysis (multipart).
   * Returns 202 {session_id, media_kind} on acceptance or a typed error on
   * rejection. URL submission is not supported by the backend.
   */
  static async submitFile(file: File, baseUrl = "/api"): Promise<UploadResult> {
    return new ApiClient({ baseUrl }).submitFileInstance(file);
  }

  /** Instance-based submission (used by the static helper). */
  async submitFileInstance(file: File): Promise<UploadResult> {
    const body = new FormData();
    body.append("file", file);
    const res = await this.fetchImpl(`${this.baseUrl}/analyses`, {
      method: "POST",
      body,
    });
    return this.toUploadResult(res);
  }

  /**
   * Fetch the metrics of one analysis session (the last analysis when the
   * dashboard loads with its session id): per-modality evidence plus the
   * fused classification, straight from the database.
   */
  async getSession(sessionId: string): Promise<AnalysisSessionDetail> {
    const res = await this.fetchImpl(
      `${this.baseUrl}/analyses/${encodeURIComponent(sessionId)}`
    );
    if (!res.ok) {
      throw new Error(`getSession failed: ${res.status}`);
    }
    return (await res.json()) as AnalysisSessionDetail;
  }

  /**
   * Download the PDF report for a completed session (WHERE REPORT_ENABLED).
   *
   * On failure the backend's typed error (e.g. ``REPORT_DISABLED``,
   * ``SESSION_INCOMPLETE``) is surfaced as the thrown Error message so the UI
   * can explain *why* the download failed instead of showing a generic error.
   */
  async getReport(sessionId: string): Promise<Blob> {
    const res = await this.fetchImpl(
      `${this.baseUrl}/analyses/${encodeURIComponent(sessionId)}/report`
    );
    if (!res.ok) {
      let message = `Report unavailable (HTTP ${res.status}).`;
      try {
        const data = (await res.json()) as { message?: string; error_code?: string };
        if (data?.message) message = data.message;
        else if (data?.error_code) message = `Report unavailable: ${data.error_code}.`;
      } catch {
        // Non-JSON error body: keep the HTTP-status message.
      }
      throw new Error(message);
    }
    return await res.blob();
  }

  /**
   * Service liveness plus the active feature flags. Lets the UI hide or
   * explain features that are switched off server-side (e.g. the PDF report).
   */
  async getHealth(): Promise<HealthStatus> {
    const res = await this.fetchImpl(`${this.baseUrl}/health`);
    if (!res.ok) {
      throw new Error(`getHealth failed: ${res.status}`);
    }
    return (await res.json()) as HealthStatus;
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
        media_kind: (data.media_kind as UploadResult["media_kind"]) ?? null,
        error_code: null,
        message: null,
      };
    }
    return {
      accepted: false,
      session_id: null,
      media_kind: null,
      error_code: (data.error_code as UploadResult["error_code"]) ?? null,
      message: (data.message as string) ?? null,
    };
  }
}