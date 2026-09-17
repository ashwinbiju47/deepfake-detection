import React, { useCallback, useState } from "react";
import { ApiClient } from "../api/client";
import type { GroundTruth, MediaKind } from "../api/types";

interface SessionMetricsProps {
  /** Kind of media analyzed in the last session. */
  mediaKind: MediaKind | null;
  /** Original filename of the analyzed upload. */
  sourceRef: string | null;
  /** Fused score in [0,1], null when inconclusive. */
  score: number | null;
  /** Per-modality evidence from the last analysis. */
  visualLikelihood: number | null;
  audioLikelihood: number | null;
  visualState: string | null;
  audioState: string | null;
  framesAnalyzed: number | null;
  facesIsolated: number | null;
  modalitiesUsed: Array<"visual" | "audio">;
  threshold: number | null;
  /** Session id — enables the ground-truth labeling controls. */
  sessionId?: string | null;
  /** Current user-declared ground truth for this session, if any. */
  groundTruth?: GroundTruth;
  /** Called after a successful label change so live figures can refresh. */
  onLabeled?: () => void;
}

function pct(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

const STATE_BADGE: Record<string, string> = {
  OK: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  NO_VISUAL_SIGNAL: "bg-slate-500/15 text-slate-300 border-slate-500/40",
  NO_AUDIO_SIGNAL: "bg-slate-500/15 text-slate-300 border-slate-500/40",
  VISUAL_ERROR: "bg-red-500/15 text-red-300 border-red-500/40",
  AUDIO_ERROR: "bg-red-500/15 text-red-300 border-red-500/40",
};

function StateBadge({ state }: { state: string | null }) {
  if (!state) return null;
  const cls = STATE_BADGE[state] ?? "bg-slate-500/15 text-slate-300 border-slate-500/40";
  return (
    <span className={`ml-2 rounded border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${cls}`}>
      {state.replace(/_/g, " ")}
    </span>
  );
}

/**
 * Metrics of the **last analysis** (this session's own pipeline output), as
 * opposed to the constant benchmark reference tables. Every number here comes
 * from the session's persisted VisualResult / AudioResult / FusionResult.
 */
export const SessionMetrics: React.FC<SessionMetricsProps> = ({
  mediaKind,
  sourceRef,
  score,
  visualLikelihood,
  audioLikelihood,
  visualState,
  audioState,
  framesAnalyzed,
  facesIsolated,
  modalitiesUsed,
  threshold,
  sessionId,
  groundTruth: initialGroundTruth = null,
  onLabeled,
}) => {
  const kindLabel = mediaKind ? mediaKind.toUpperCase() : "—";
  const [groundTruth, setGroundTruth] = useState<GroundTruth>(initialGroundTruth);
  const [labelBusy, setLabelBusy] = useState(false);
  const [labelError, setLabelError] = useState<string | null>(null);

  const applyLabel = useCallback(
    async (value: GroundTruth) => {
      if (!sessionId || labelBusy) return;
      setLabelBusy(true);
      setLabelError(null);
      try {
        const res = await new ApiClient().setGroundTruth(sessionId, value);
        setGroundTruth(res.ground_truth);
        onLabeled?.();
      } catch (err) {
        setLabelError(err instanceof Error ? err.message : String(err));
      } finally {
        setLabelBusy(false);
      }
    },
    [sessionId, labelBusy, onLabeled]
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-sm font-semibold text-slate-200">
          Last analysis metrics
          <span className="ml-2 rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-indigo-300">
            {kindLabel}
          </span>
        </h4>
        {sourceRef && (
          <span className="max-w-[16rem] truncate font-mono text-xs text-slate-500" title={sourceRef}>
            {sourceRef}
          </span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
          <p className="uppercase tracking-wider text-slate-500">Fused score</p>
          <p className="mt-1 font-mono text-lg font-bold text-slate-100">{pct(score)}</p>
          <p className="text-[10px] text-slate-500">
            modalities: {modalitiesUsed.length ? modalitiesUsed.join(" + ") : "none"}
            {threshold != null ? ` · threshold ${threshold.toFixed(2)}` : ""}
          </p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
          <p className="uppercase tracking-wider text-slate-500">Visual likelihood</p>
          <p className="mt-1 font-mono text-lg font-bold text-sky-300">
            {pct(visualLikelihood)}
          </p>
          <p className="text-[10px] text-slate-500">
            {framesAnalyzed != null ? `${framesAnalyzed} frame(s)` : "no visual pass"}
            {facesIsolated != null ? ` · ${facesIsolated} face(s)` : ""}
          </p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
          <p className="uppercase tracking-wider text-slate-500">Audio likelihood</p>
          <p className="mt-1 font-mono text-lg font-bold text-fuchsia-300">
            {pct(audioLikelihood)}
          </p>
          <p className="text-[10px] text-slate-500">
            {audioLikelihood == null && audioState ? audioState.replace(/_/g, " ").toLowerCase() : "spectrogram scored"}
          </p>
        </div>
      </div>

      {(visualState || audioState) && (
        <p className="text-[11px] text-slate-500">
          Pipeline states:
          {visualState && (
            <span>
              {" "}visual <StateBadge state={visualState} />
            </span>
          )}
          {audioState && (
            <span>
              {" "}· audio <StateBadge state={audioState} />
            </span>
          )}
        </p>
      )}

      {sessionId && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-slate-300">
              Is this media real or a deepfake?
            </span>
            <span className="text-[10px] text-slate-500">
              labeled analyses build your own ROC curves &amp; confusion matrices
            </span>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={labelBusy}
              onClick={() => void applyLabel("real")}
              className={`rounded-lg border px-3 py-1.5 text-xs font-bold transition ${
                groundTruth === "real"
                  ? "border-emerald-400 bg-emerald-500/25 text-emerald-200"
                  : "border-slate-700 bg-slate-900 text-slate-300 hover:border-emerald-500/60"
              } disabled:opacity-50`}
            >
              REAL
            </button>
            <button
              type="button"
              disabled={labelBusy}
              onClick={() => void applyLabel("fake")}
              className={`rounded-lg border px-3 py-1.5 text-xs font-bold transition ${
                groundTruth === "fake"
                  ? "border-red-400 bg-red-500/25 text-red-200"
                  : "border-slate-700 bg-slate-900 text-slate-300 hover:border-red-500/60"
              } disabled:opacity-50`}
            >
              DEEPFAKE
            </button>
            {groundTruth && (
              <button
                type="button"
                disabled={labelBusy}
                onClick={() => void applyLabel(null)}
                className="rounded-lg border border-slate-700 px-2 py-1.5 text-[10px] text-slate-400 hover:text-slate-200 disabled:opacity-50"
              >
                clear
              </button>
            )}
            {groundTruth && (
              <span className="text-[10px] uppercase tracking-wider text-slate-500">
                labeled: {groundTruth}
              </span>
            )}
            {labelError && (
              <span className="text-[10px] text-red-300">label failed: {labelError}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SessionMetrics;
