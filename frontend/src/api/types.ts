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
  /** Set iff `accepted` is true. Wire format: snake_case session_id. */
  session_id: string | null;
  /** Set iff `accepted` is false. Wire format: snake_case error_code. */
  error_code: UploadErrorCode | null;
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
  run_id: string;
  dataset: string;
  train_dataset: string;
  variant: "multimodal" | "visual_only" | "audio_only";
  split: string;
  accuracy: number;
  meets_baseline: boolean;
  evaluated_at: string | null;
  metrics: {
    confusion_matrix: { tp: number; fp: number; tn: number; fn: number };
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc: number | null;
  } | null;
}

/** One row of the Results-chapter benchmark tables (snake_case, as served). */
export interface BenchmarkRow {
  variant: "multimodal" | "visual_only" | "audio_only";
  dataset: string;
  train_dataset: string;
  split: string;
  accuracy: number;
  accuracy_pct: number;
  precision: number;
  precision_pct: number;
  recall: number;
  recall_pct: number;
  f1_score: number;
  f1_pct: number;
  roc_auc: number;
  roc_auc_pct: number;
  confusion_matrix: { tp: number; fp: number; tn: number; fn: number };
  meets_baseline: boolean;
}

/** Modality comparison table + fusion improvement (GET /api/evaluations/benchmark). */
export interface ModalityComparison {
  table: BenchmarkRow[];
  ordering: Array<"multimodal" | "visual_only" | "audio_only">;
  ordering_satisfied: boolean;
  improvement_over_visual: number;
  improvement_over_audio: number;
  improvement_over_visual_pct: number;
  improvement_over_audio_pct: number;
}

/** One cross-dataset group (train on A, test on B). */
export interface CrossDatasetGroup {
  dataset: string;
  train_dataset: string;
  split: string;
  runs: BenchmarkRow[];
}

/** Response of GET /api/health — liveness plus the active feature flags. */
export interface HealthStatus {
  status: string;
  feature_flags: {
    external_url_enabled: boolean;
    heatmap_enabled: boolean;
    report_enabled: boolean;
  };
  worker_concurrency: number;
}

/** A single sampled point of a curve, as ``[x, y]``. */
export type CurvePoint = [number, number];

/** One model's ROC curve (all three are drawn on the same axes). */
export interface RocCurve {
  variant: "multimodal" | "visual_only" | "audio_only";
  label: string;
  /** Closed-form AUC of the plotted curve (equals the reported ROC-AUC). */
  auc: number;
  curve_auc_sampled: number;
  reported_auc: number;
  operating_point: { fpr: number; tpr: number };
  points: CurvePoint[];
}

/** One model's Precision-Recall curve. */
export interface PrCurve {
  variant: "multimodal" | "visual_only" | "audio_only";
  label: string;
  average_precision: number;
  points: CurvePoint[];
}

/** Confusion matrix for one modality configuration, with error readings. */
export interface ConfusionMatrixSummary {
  variant: "multimodal" | "visual_only" | "audio_only";
  label: string;
  dataset: string;
  train_dataset: string;
  split: string;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  false_positives: number;
  false_negatives: number;
  error_summary: string;
}

/** Curve data + confusion matrices behind the Results-chapter figures. */
export interface CurvesPayload {
  dataset: string;
  split: string;
  roc: RocCurve[];
  pr: PrCurve[];
  confusion_matrices: ConfusionMatrixSummary[];
  glossary: Record<string, string>;
}

/** One row of the fusion-weight ablation sweep. */
export interface AblationRow {
  alpha: number;
  audio_weight: number;
  visual_weight: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  accuracy: number;
  accuracy_pct: number;
  precision_pct: number;
  recall_pct: number;
  f1_pct: number;
  roc_auc_pct: number;
}

/** Fusion-weight ablation study (why 0.6/0.4). */
export interface WeightAblation {
  method: string;
  validation_size: number;
  validation_per_class: number;
  seed: number;
  threshold: number;
  configured_alpha: number;
  configured_audio_weight: number;
  best_alpha: number;
  best_accuracy: number;
  configured_matches_best: boolean;
  rows: AblationRow[];
  chosen_vs_audio_only_pp: number;
  chosen_vs_visual_only_pp: number;
}

/** A rendered figure delivered as a base64 PNG. */
export interface FigurePayload {
  png_base64: string;
}

/** Full benchmark payload returned by GET /api/evaluations/benchmark. */
export interface EvaluationBenchmark {
  modality_comparison: ModalityComparison;
  cross_dataset: { table: CrossDatasetGroup[] };
  curves: CurvesPayload;
  weight_ablation: WeightAblation;
  figures: Record<string, FigurePayload>;
}
