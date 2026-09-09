import React from "react";

/**
 * The end-to-end multimodal detection pipeline, shown exactly as in the
 * architecture diagram:
 *
 *   Video -> Frame extraction -> Face detection / preprocessing -> Visual model
 *   -> Audio extraction -> Audio model -> Multimodal fusion -> Fake probability
 *   -> XAI explanation -> Final result
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

interface PipelineFlowProps {
  /** Highest stage index (0-based) currently active/completed. -1 = idle. */
  activeStage?: number;
  /** Final classification label, shown on the last stage once done. */
  finalLabel?: "authentic" | "deepfake" | null;
}

/**
 * Vertical flow diagram of the detection pipeline. Stages up to
 * `activeStage` are highlighted; the visual/audio branches are color-coded.
 */
export const PipelineFlow: React.FC<PipelineFlowProps> = ({
  activeStage = -1,
  finalLabel = null,
}) => {
  const done = (i: number) => activeStage >= i;

  return (
    <div className="space-y-1.5 font-mono text-sm">
      {PIPELINE_STAGES.map((stage, i) => {
        const isVisual = VISUAL_STAGES.has(i);
        const isAudio = AUDIO_STAGES.has(i);
        const isFinal = i === PIPELINE_STAGES.length - 1;

        const color = done(i)
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

        const boxColor = done(i)
          ? isVisual
            ? "border-sky-500/60 bg-sky-950/40"
            : isAudio
            ? "border-fuchsia-500/60 bg-fuchsia-950/40"
            : "border-indigo-500/60 bg-indigo-950/40"
          : "border-slate-800 bg-slate-900/60";

        const marker = done(i) ? "●" : "○";

        return (
          <div key={stage}>
            <div className={`flex items-center gap-3 ${i > 0 ? "ml-1" : ""}`}>
              <span className={`${done(i) ? "text-emerald-400" : "text-slate-600"} text-xs`}>
                {marker}
              </span>
              <span
                className={`inline-block rounded-lg border px-3 py-1.5 ${boxColor} ${color}`}
              >
                {stage}
                {isFinal && finalLabel ? `: ${finalLabel.toUpperCase()}` : ""}
              </span>
            </div>
            {i < PIPELINE_STAGES.length - 1 && (
              <div className={`ml-[13px] h-3 w-px ${done(i + 1) ? "bg-indigo-400/70" : "bg-slate-700"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
};

export default PipelineFlow;