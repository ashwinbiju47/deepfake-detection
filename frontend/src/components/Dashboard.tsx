import React, { useEffect, useState } from "react";
import { ResultStreamer } from "../ws/client";
import type { ConnectionState } from "../ws/client";
import type { HeatmapPayload, ResultPayload, StreamEvent } from "../ws/types";
import PipelineFlow, { PIPELINE_STAGES } from "./PipelineFlow";
import ResultsTable from "./ResultsTable";
import XAIPanel from "./XAIPanel";

interface DashboardProps {
  sessionId: string;
  onReset: () => void;
}

/** Map WS progress percentage onto the coarse pipeline-stage index for the flow diagram. */
function stageFromProgress(percent: number, hasResult: boolean): number {
  if (hasResult) return PIPELINE_STAGES.length - 1;
  if (percent <= 0) return 0; // Video ingested
  if (percent < 50) return 3; // Visual pipeline done (frames, faces, visual model)
  if (percent < 100) return 6; // Audio + fusion done
  return 7;
}

export const Dashboard: React.FC<DashboardProps> = ({ sessionId, onReset }) => {
  const [progress, setProgress] = useState(0);
  const [connState, setConnState] = useState<ConnectionState>("idle");
  const [result, setResult] = useState<ResultPayload | null>(null);
  const [heatmaps, setHeatmaps] = useState<HeatmapPayload[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const streamer = new ResultStreamer(sessionId);

    streamer.onStateChange((state) => {
      setConnState(state);
      if (state === "error") {
        setErrorMessage("Connection to real-time analysis server failed.");
      }
    });

    streamer.onEvent((event: StreamEvent) => {
      if (event.type === "progress") {
        const percent =
          (event as { progress_percent?: number }).progress_percent ?? 0;
        setProgress(percent);
      } else if (event.type === "result") {
        const payload = event as unknown as ResultPayload;
        setResult(payload);
        setProgress(100);
      } else if (event.type === "heatmap") {
        setHeatmaps((prev) => [...prev, event as unknown as HeatmapPayload]);
      } else if (event.type === "error") {
        const err = event as { message?: string; error_code?: string };
        setErrorMessage(err.message || "Analysis pipeline error.");
      }
    });

    streamer.connect();

    return () => {
      streamer.close();
    };
  }, [sessionId]);

  const reportUrl = `/api/analyses/${sessionId}/report`;
  const activeStage = stageFromProgress(progress, result != null);

  return (
    <div className="mx-auto max-w-4xl space-y-6 rounded-2xl border border-slate-800 bg-slate-900 p-6 text-white shadow-xl">
      <div className="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-sky-400">Analysis Session</h2>
          <p className="font-mono text-xs text-slate-400">{sessionId}</p>
        </div>
        <div className="flex items-center space-x-3">
          <span
            className={`inline-block w-3 h-3 rounded-full ${
              connState === "open"
                ? "animate-pulse bg-emerald-400"
                : connState === "connecting" || connState === "reconnecting"
                ? "animate-ping bg-amber-400"
                : "bg-red-500"
            }`}
          />
          <span className="text-xs capitalize text-slate-300">{connState}</span>
          <button
            onClick={onReset}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-700"
          >
            New Analysis
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="rounded-lg border border-red-500 bg-red-950/80 p-3 text-sm text-red-200">
          {errorMessage}
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        {/* Pipeline flow — mirrors the architecture diagram */}
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-slate-300">
            Detection Pipeline (live)
          </h4>
          <PipelineFlow activeStage={activeStage} finalLabel={result?.label ?? null} />
        </div>

        {/* Progress + result card */}
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm font-semibold">
              <span>Processing Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full bg-gradient-to-r from-sky-500 to-indigo-500 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {result && (
            <div
              className={`space-y-4 rounded-2xl border p-5 ${
                result.inconclusive
                  ? "border-slate-600/50 bg-slate-800/40"
                  : result.label === "deepfake"
                  ? "border-red-500/50 bg-red-950/40"
                  : "border-emerald-500/50 bg-emerald-950/40"
              }`}
            >
              <div className="flex justify-between items-center">
                <div>
                  <p className="text-xs uppercase tracking-wider text-slate-400">
                    Classification Outcome
                  </p>
                  <h3
                    className={`text-3xl font-extrabold capitalize ${
                      result.inconclusive
                        ? "text-slate-300"
                        : result.label === "deepfake"
                        ? "text-red-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {result.label || "Inconclusive"}
                  </h3>
                </div>
                <div className="text-right">
                  <p className="text-xs uppercase tracking-wider text-slate-400">
                    Fake Probability
                  </p>
                  <p className="text-3xl font-bold">
                    {result.score != null ? `${(result.score * 100).toFixed(1)}%` : "N/A"}
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between border-t border-slate-800 pt-3 text-xs text-slate-400">
                <span>
                  Modalities Used: {result.modalities_used?.join(", ") || "None"}
                </span>
                <a
                  href={reportUrl}
                  download
                  className="rounded-xl bg-sky-500 px-4 py-2 font-bold text-white transition hover:bg-sky-600"
                >
                  Download PDF Report
                </a>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* XAI: ORIGINAL + HEATMAP + OVERLAY triple */}
      <XAIPanel heatmaps={heatmaps} />

      {/* Results / Evaluation chapter: benchmark tables */}
      <div className="border-t border-slate-800 pt-4">
        <h3 className="mb-3 text-lg font-bold text-sky-400">
          Results / Evaluation
        </h3>
        <ResultsTable />
      </div>
    </div>
  );
};

export default Dashboard;
