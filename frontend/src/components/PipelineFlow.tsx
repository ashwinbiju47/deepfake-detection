import React from "react";
import type { MediaKind } from "../api/types";

/**
 * The end-to-end multimodal detection pipeline, shown exactly as in the
 * architecture diagram:
 *
 *   Video -> Frame extraction -> Face detection / preprocessing -> Visual model
 *   -> Audio extraction -> Audio model -> Multimodal fusion -> Fake probability
 *   -> XAI explanation -> Final result
 *
 * The first stage label adapts to the analyzed media kind: image uploads skip
 * the audio branch, audio uploads skip the visual branch (their stages are
 * rendered greyed-out as "not applicable" so the full architecture stays
 * visible while the active path is unambiguous).
 */
export const PIPELINE_STAGES = [
  "Video",
  "Frame extraction",
  "Face detection / preprocessing",
  "Visual model",
  "Audio extraction",
  "Audio model",
  "Multimodal fusion",
  "Fake probability",
  "XAI explanation",
  "Final result",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];

/** Stage indices that belong to the visual branch of the pipeline. */
const VISUAL_STAGES = new Set([1, 2, 3]);
/** Stage indices that belong to the audio branch of the pipeline. */
const AUDIO_STAGES = new Set([4, 5]);

/** Which branch each media kind actually runs. */
const ACTIVE_BRANCH: Record<MediaKind, "visual" | "audio" | "both"> = {
  video: "both",
  image: "visual",
  audio: "audio",
};

/** The stage-0 label shown for each media kind. */
const INPUT_LABEL: Record<MediaKind, string> = {
  video: "Video",
  image: "Image",
  audio: "Audio",
};

interface PipelineFlowProps {
  /** Highest stage index (0-based) currently active/completed. -1 = idle. */
  activeStage?: number;
  /** Final classification label, shown on the last stage once done. */
  finalLabel?: "authentic" | "deepfake" | null;
  /** Kind of media being analyzed (adapts labels and branch highlighting). */
  mediaKind?: MediaKind | null;
}

/**
 * Vertical flow diagram of the detection pipeline. Stages up to
 * `activeStage` are highlighted; the visual/audio branches are color-coded,
 * and stages skipped for the analyzed media kind are greyed out.
 */
export const PipelineFlow: React.FC<PipelineFlowProps> = ({
  activeStage = -1,
  finalLabel = null,
  mediaKind = null,
}) => {
  const branch = mediaKind ? ACTIVE_BRANCH[mediaKind] : "both";
  const done = (i: number) => activeStage >= i;
  /** A stage participates in this analysis's active path. */
  const runs = (i: number) =>
    branch === "both" ||
    (branch === "visual" && (VISUAL_STAGES.has(i) || !AUDIO_STAGES.has(i))) ||
    (branch === "audio" && (AUDIO_STAGES.has(i) || !VISUAL_STAGES.has(i)));

  return (
    <div className="space-y-1.5 font-mono text-sm">
      {PIPELINE_STAGES.map((stage, i) => {
        const isVisual = VISUAL_STAGES.has(i);
        const isAudio = AUDIO_STAGES.has(i);
        const isFinal = i === PIPELINE_STAGES.length - 1;
        const label = i === 0 && mediaKind ? INPUT_LABEL[mediaKind] : stage;
        const skipped = mediaKind != null && !runs(i);

        const color = skipped
          ? "text-slate-600"
          : done(i)
          ? isVisual
            ? "text-sky-400"
            : isAudio
            ? "text-fuchsia-400"
            : isFinal && finalLabel
            ? finalLabel === "deepfake"
              ? "text-red-400"
              : "text-emerald-400"
            : "text-indigo-300"
          : "text-slate-500";

        const boxColor = skipped
          ? "border-slate-800/60 bg-slate-900/40"
          : done(i)
          ? isVisual
            ? "border-sky-500/60 bg-sky-950/40"
            : isAudio
            ? "border-fuchsia-500/60 bg-fuchsia-950/40"
            : "border-indigo-500/60 bg-indigo-950/40"
          : "border-slate-800 bg-slate-900/60";

        const marker = skipped ? "×" : done(i) ? "●" : "○";

        return (
          <div key={stage}>
            <div className={`flex items-center gap-3 ${i > 0 ? "ml-1" : ""}`}>
              <span
                className={`${
                  skipped
                    ? "text-slate-700"
                    : done(i)
                    ? "text-emerald-400"
                    : "text-slate-600"
                } text-xs`}
              >
                {marker}
              </span>
              <span
                className={`inline-block rounded-lg border px-3 py-1.5 ${boxColor} ${color} ${
                  skipped ? "line-through opacity-60" : ""
                }`}
                title={skipped ? "Not run for this media kind" : undefined}
              >
                {label}
                {isFinal && finalLabel ? `: ${finalLabel.toUpperCase()}` : ""}
              </span>
            </div>
            {i < PIPELINE_STAGES.length - 1 && (
              <div
                className={`ml-[13px] h-3 w-px ${
                  !skipped && done(i + 1) ? "bg-indigo-400/70" : "bg-slate-700"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
};

export default PipelineFlow;
