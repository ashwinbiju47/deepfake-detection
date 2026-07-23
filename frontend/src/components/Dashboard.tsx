import React, { useEffect, useState } from "react";
import { ResultStreamer } from "../ws/client";
import type { ConnectionState } from "../ws/client";
import type { HeatmapPayload, ProgressPayload, ResultPayload } from "../ws/types";

interface DashboardProps {
  sessionId: string;
  onReset: () => void;
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

    streamer.onEvent((event) => {
      if (event.type === "progress") {
        const p = (event as any).progress_percent ?? (event.payload as ProgressPayload)?.progress_percent ?? 0;
        setProgress(p);
      } else if (event.type === "result") {
        setResult(event as unknown as ResultPayload);
        setProgress(100);
      } else if (event.type === "heatmap") {
        setHeatmaps((prev) => [...prev, event as unknown as HeatmapPayload]);
      } else if (event.type === "error") {
        setErrorMessage((event as any).message || "Analysis pipeline error.");
      }
    });

    streamer.connect();

    return () => {
      streamer.close();
    };
  }, [sessionId]);

  const reportUrl = `/api/analyses/${sessionId}/report`;

  return (
    <div className="bg-slate-900 border border-slate-800 text-white p-6 rounded-2xl shadow-xl max-w-3xl mx-auto space-y-6">
      <div className="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-sky-400">Analysis Session</h2>
          <p className="text-xs font-mono text-slate-400">{sessionId}</p>
        </div>
        <div className="flex items-center space-x-3">
          <span
            className={`inline-block w-3 h-3 rounded-full ${
              connState === "open"
                ? "bg-emerald-400 animate-pulse"
                : connState === "connecting" || connState === "reconnecting"
                ? "bg-amber-400 animate-ping"
                : "bg-red-500"
            }`}
          />
          <span className="text-xs capitalize text-slate-300">{connState}</span>
          <button
            onClick={onReset}
            className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700"
          >
            New Analysis
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="bg-red-950/80 border border-red-500 text-red-200 p-3 rounded-lg text-sm">
          {errorMessage}
        </div>
      )}

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm font-semibold">
          <span>Processing Pipeline</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden">
          <div
            className="bg-gradient-to-r from-sky-500 to-indigo-500 h-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Classification Result Card */}
      {result && (
        <div
          className={`p-6 rounded-2xl border ${
            result.label === "deepfake"
              ? "bg-red-950/40 border-red-500/50"
              : "bg-emerald-950/40 border-emerald-500/50"
          } space-y-4`}
        >
          <div className="flex justify-between items-center">
            <div>
              <p className="text-xs uppercase tracking-wider text-slate-400">Classification Outcome</p>
              <h3
                className={`text-3xl font-extrabold capitalize ${
                  result.label === "deepfake" ? "text-red-400" : "text-emerald-400"
                }`}
              >
                {result.label || "Inconclusive"}
              </h3>
            </div>
            <div className="text-right">
              <p className="text-xs uppercase tracking-wider text-slate-400">Confidence Score</p>
              <p className="text-3xl font-bold">
                {result.score != null ? `${(result.score * 100).toFixed(1)}%` : "N/A"}
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-400 border-t border-slate-800 pt-3">
            <span>Modalities Used: Visual {/* Audio disabled for 50% milestone */}</span>
            {/* PDF Report disabled for 50% milestone
            <a
              href={reportUrl}
              download
              className="bg-sky-500 hover:bg-sky-600 text-white px-4 py-2 rounded-xl font-bold transition text-xs"
            >
              Download PDF Report
            </a>
            */}
          </div>
        </div>
      )}

      {/* Heatmaps Overlay Stream disabled for 50% milestone
      {heatmaps.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-slate-300">Grad-CAM XAI Heatmaps</h4>
          <div className="grid grid-cols-2 gap-4">
            {heatmaps.map((h, idx) => (
              <div key={idx} className="bg-slate-800 p-2 rounded-xl border border-slate-700 space-y-1">
                <p className="text-xs font-mono text-slate-400">{h.frame_id}</p>
                {h.overlay_b64 && (
                  <img
                    src={`data:image/png;base64,${h.overlay_b64}`}
                    alt={`Heatmap ${h.frame_id}`}
                    className="w-full h-32 object-cover rounded-lg"
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      */}
    </div>
  );
};
