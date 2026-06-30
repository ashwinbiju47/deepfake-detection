/**
 * WebSocket message protocol for the Result_Streamer, mirroring the design's
 * "Components and Interfaces > Result_Streamer" section. All server->client
 * messages carry a monotonically increasing `seq` to support lossless,
 * gap-free replay on reconnect (Requirement 5.5).
 */
import type { ClassificationLabel } from "../api/types";

export interface ProgressEvent {
  type: "progress";
  seq: number;
  session_id: string;
  /** Completion percentage 0..100. */
  percent: number;
  stage: string;
  ts: string;
}

export interface HeatmapEvent {
  type: "heatmap";
  seq: number;
  session_id: string;
  frame_id: string;
  /** Base64 data URL of the Grad-CAM overlay PNG. */
  image: string;
  intensity_scale: [number, number];
}

export interface ResultEvent {
  type: "result";
  seq: number;
  session_id: string;
  /** Classification_Score in [0,1]; null when inconclusive. */
  score: number | null;
  label: ClassificationLabel | null;
  modalities_used: Array<"visual" | "audio">;
}

export interface ErrorEvent {
  type: "error";
  seq: number;
  session_id: string;
  code: string;
  message: string;
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
