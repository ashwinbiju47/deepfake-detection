/**
 * WebSocket message protocol for the Result_Streamer, mirroring the design's
 * "Components and Interfaces > Result_Streamer" section. All server->client
 * messages carry a monotonically increasing `seq` to support lossless,
 * gap-free replay on reconnect (Requirement 5.5).
 *
 * The backend publishes *flat* payload objects: each event's payload dict is
 * augmented with `seq`, `type` and `session_id` before broadcast (see
 * StreamService.publish_event), so the client receives one flat object per
 * event. These types match that wire format exactly.
 */
import type { ClassificationLabel } from "../api/types";

/** Shared envelope fields added by the backend to every event payload. */
interface EventEnvelope {
  seq: number;
  type: string;
  session_id: string;
}

/** Payload of a `progress` event (no envelope). */
export interface ProgressPayload {
  progress_percent: number;
  status: string;
}

/** Payload of a `heatmap` event (no envelope) — the full XAI triple. */
export interface HeatmapPayload {
  frame_id: string;
  heatmap_id: string;
  /** Base64 PNG of the ORIGINAL frame. */
  original_b64?: string;
  /** Base64 PNG of the raw Grad-CAM heatmap. */
  heatmap_b64?: string;
  /** Base64 PNG of the final overlay (heatmap blended over the original). */
  overlay_b64?: string;
}

/** Payload of a `result` event (no envelope). */
export interface ResultPayload {
  score: number | null;
  label: ClassificationLabel | null;
  modalities_used: Array<"visual" | "audio">;
  inconclusive: boolean;
  status: string;
  /** Per-modality evidence behind the fused score (null when absent). */
  visual_likelihood?: number | null;
  audio_likelihood?: number | null;
  /** Fusion weights actually applied to the two modalities. */
  weights?: { visual: number; audio: number };
  threshold?: number;
}

export interface ProgressEvent extends EventEnvelope, ProgressPayload {}

export interface HeatmapEvent extends EventEnvelope, HeatmapPayload {}

export interface ResultEvent extends EventEnvelope, ResultPayload {}

export interface ErrorEvent extends EventEnvelope {
  type: "error";
  error_code?: string;
  code?: string;
  message: string;
  frame_id?: string;
}

/** Discriminated union of all server -> client stream events. */
export type StreamEvent =
  | ProgressEvent
  | HeatmapEvent
  | ResultEvent
  | ErrorEvent;

/** Client -> server resume request sent on (re)connect. */
export interface ResumeRequest {
  type: "resume";
  last_seq: number;
}