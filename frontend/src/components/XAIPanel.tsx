import React from "react";
import type { HeatmapPayload } from "../ws/types";

interface XAIPanelProps {
  /** Heatmap events received for this session (one per analyzed frame). */
  heatmaps: HeatmapPayload[];
}

const CAPTIONS = [
  { key: "original_b64", label: "ORIGINAL" },
  { key: "heatmap_b64", label: "GRAD-CAM HEATMAP" },
  { key: "overlay_b64", label: "FINAL OVERLAY" },
] as const;

/**
 * Explainability viewer: for each analyzed frame shows the full XAI triple —
 * the ORIGINAL frame, the raw Grad-CAM HEATMAP, and the final OVERLAY
 * (heatmap alpha-blended over the original) — never the heatmap alone.
 */
export const XAIPanel: React.FC<XAIPanelProps> = ({ heatmaps }) => {
  if (heatmaps.length === 0) return null;

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-semibold text-slate-200">
          Grad-CAM Explainability (XAI)
        </h4>
        <p className="text-xs text-slate-500">
          Red regions mark the pixels the visual model attended to when deciding
          the frame is a deepfake.
        </p>
      </div>

      {heatmaps.map((h, idx) => (
        <div
          key={h.heatmap_id ?? idx}
          className="rounded-xl border border-slate-700 bg-slate-800/60 p-3"
        >
          <p className="mb-2 font-mono text-xs text-slate-400">{h.frame_id}</p>
          <div className="grid grid-cols-3 gap-3">
            {CAPTIONS.map(({ key, label }) => {
              const b64 = h[key];
              return (
                <div key={key} className="space-y-1">
                  {b64 ? (
                    <img
                      src={`data:image/png;base64,${b64}`}
                      alt={`${label} ${h.frame_id}`}
                      className="h-32 w-full rounded-lg border border-slate-700 object-cover"
                    />
                  ) : (
                    <div className="flex h-32 w-full items-center justify-center rounded-lg border border-dashed border-slate-700 bg-slate-900 text-xs text-slate-600">
                      unavailable
                    </div>
                  )}
                  <p className="text-center text-[10px] font-bold tracking-wider text-slate-400">
                    {label}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
};

export default XAIPanel;