/**
 * Shared types mirroring the backend (DRF) contracts described in the design's
 * Components and Interfaces section. These are kept in sync with the Django
 * serializers; they are the single source of truth on the client side.
 */

/** Supported video container formats accepted by the Upload_Service. */
export type SupportedFormat = "MP4" | "AVI";

/** Typed upload rejection reasons returned by the Upload_Service. */
export type UploadErrorCode =
  | "EMPTY_FILE"
  | "UNSUPPORTED_FORMAT"
  | "TOO_LARGE"
  | "UNDECODABLE"
  | "BAD_SCHEME"
  | "URL_UNREACHABLE";

/** Lifecycle status of an Analysis_Session. */
export type SessionStatus =
  | "QUEUED"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELED"
  | "INCONCLUSIVE";

/** Whether the transient media for a session is still present or purged. */
export type MediaState = "PRESENT" | "PURGED";

/** Classification label produced by the Fusion_Engine. */
export type ClassificationLabel = "authentic" | "deepfake";

/** Result of submitting a file or URL to the Upload_Service (POST /api/analyses). */
export interface UploadResult {
  accepted: boolean;
  /** Set iff `accepted` is true. */
  sessionId: string | null;
  /** Set iff `accepted` is false. */
  errorCode: UploadErrorCode | null;
  message: string | null;
}

/** Per-modality findings exposed via GET /api/analyses/{id}. */
export interface ModalityResult {
  modality: "visual" | "audio";
  likelihood: number | null;
  state: string;
}

/** Session status/result metadata returned by GET /api/analyses/{id}. */
export interface AnalysisSession {
  id: string;
  status: SessionStatus;
  mediaState: MediaState;
  /** Classification_Score in [0,1]; null when inconclusive or not yet produced. */
  score: number | null;
  label: ClassificationLabel | null;
  modalitiesUsed: Array<"visual" | "audio">;
  inconclusive: boolean;
  createdAt: string;
  completedAt: string | null;
}

/** Persisted evaluation metrics returned by GET /api/evaluations/{run_id}. */
export interface EvaluationMetrics {
  runId: string;
  dataset: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  meetsBaseline: boolean;
}
