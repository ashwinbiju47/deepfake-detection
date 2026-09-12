import React, { useEffect, useState } from "react";
import { ApiClient } from "../api/client";
import type {
  AblationRow,
  ConfusionMatrixSummary,
  CurvesPayload,
  EvaluationBenchmark,
  FigurePayload,
  WeightAblation,
} from "../api/types";

const VARIANT_COLOR: Record<string, string> = {
  multimodal: "text-teal-300",
  visual_only: "text-sky-300",
  audio_only: "text-amber-300",
};

function Figure({ figure, alt }: { figure?: FigurePayload; alt: string }) {
  if (!figure) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-slate-700 bg-slate-900 text-xs text-slate-500">
        figure unavailable
      </div>
    );
  }
  return (
    <img
      src={`data:image/png;base64,${figure.png_base64}`}
      alt={alt}
      className="w-full rounded-xl border border-slate-700 bg-white"
    />
  );
}

/** One annotated confusion matrix with the TP/FP/TN/FN cells spelled out. */
export const ConfusionMatrixCard: React.FC<{ matrix: ConfusionMatrixSummary }> = ({
  matrix,
}) => {
  const cell = (
    label: string,
    value: number,
    tone: "good" | "bad" | "neutral"
  ) => (
    <div
      className={`rounded-lg border p-3 text-center ${
        tone === "good"
          ? "border-emerald-600/60 bg-emerald-950/40"
          : tone === "bad"
          ? "border-red-600/60 bg-red-950/40"
          : "border-slate-700 bg-slate-900/60"
      }`}
    >
      <p className="text-[10px] font-bold tracking-wider text-slate-400">{label}</p>
      <p className="text-2xl font-bold text-slate-100">{value}</p>
    </div>
  );

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h5 className={`text-sm font-bold ${VARIANT_COLOR[matrix.variant] ?? ""}`}>
          {matrix.label}
        </h5>
        <span className="text-xs text-slate-400">
          accuracy {(matrix.accuracy * 100).toFixed(1)}%
        </span>
      </div>
      <p className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
        ground truth rows / model prediction columns
      </p>
      <div className="grid grid-cols-2 gap-2">
        {cell("TN · authentic kept", matrix.tn, "good")}
        {cell("FP · false alarm", matrix.fp, "bad")}
        {cell("FN · fake missed", matrix.fn, "bad")}
        {cell("TP · fake caught", matrix.tp, "good")}
      </div>
      <p className="mt-2 text-xs text-slate-400">{matrix.error_summary}</p>
      <p className="mt-1 text-[11px] text-slate-500">
        precision {(matrix.precision * 100).toFixed(1)}% · recall{" "}
        {(matrix.recall * 100).toFixed(1)}% · F1 {(matrix.f1_score * 100).toFixed(1)}%
      </p>
    </div>
  );
};

/** Confusion matrices + the glossary explaining what each error means. */
export const ConfusionMatrices: React.FC<{ curves: CurvesPayload }> = ({ curves }) => (
  <div className="space-y-3">
    <h4 className="text-sm font-semibold text-slate-200">
      Confusion matrices — multimodal, visual-only, audio-only
    </h4>
    <div className="grid gap-3 md:grid-cols-3">
      {curves.confusion_matrices.map((matrix) => (
        <ConfusionMatrixCard key={matrix.variant} matrix={matrix} />
      ))}
    </div>
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 text-xs text-slate-300">
      <p className="mb-2 text-sm font-semibold text-slate-200">What the errors mean</p>
      <ul className="space-y-1">
        <li>
          <span className="font-bold text-emerald-300">False positive (FP)</span> —{" "}
          {curves.glossary.FP}
        </li>
        <li>
          <span className="font-bold text-red-300">False negative (FN)</span> —{" "}
          {curves.glossary.FN}
        </li>
        <li>
          <span className="font-bold text-slate-100">Precision</span> — {curves.glossary.precision}
        </li>
        <li>
          <span className="font-bold text-slate-100">Recall</span> — {curves.glossary.recall}
        </li>
      </ul>
    </div>
  </div>
);

/** Ablation table highlighting the chosen weights. */
export const AblationTable: React.FC<{ ablation: WeightAblation }> = ({ ablation }) => {
  const isBest = (row: AblationRow) => Math.abs(row.alpha - ablation.best_alpha) < 1e-9;
  const isChosen = (row: AblationRow) =>
    Math.abs(row.alpha - ablation.configured_alpha) < 1e-9;
  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-slate-200">
        Fusion-weight ablation — visual weight α (audio weight 1 − α)
      </h4>
      <p className="text-xs text-slate-500">
        {ablation.method} Validation size {ablation.validation_size} videos (seed{" "}
        {ablation.seed}), threshold {ablation.threshold.toFixed(2)}.
      </p>
      <div className="overflow-x-auto rounded-xl border border-slate-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-800/80 text-left text-xs uppercase tracking-wider text-slate-300">
              <th className="px-4 py-2.5">Weights (visual / audio)</th>
              <th className="px-4 py-2.5 text-right">Accuracy</th>
              <th className="px-4 py-2.5 text-right">Precision</th>
              <th className="px-4 py-2.5 text-right">Recall</th>
              <th className="px-4 py-2.5 text-right">F1</th>
              <th className="px-4 py-2.5 text-right">ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            {ablation.rows.map((row) => (
              <tr
                key={row.alpha}
                className={`border-t border-slate-800 ${
                  isBest(row) ? "bg-emerald-950/40" : "bg-slate-900/60"
                }`}
              >
                <td className="px-4 py-2 font-mono text-xs text-slate-200">
                  {row.alpha.toFixed(1)} / {row.audio_weight.toFixed(1)}
                  {isChosen(row) && (
                    <span className="ml-2 rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-bold text-emerald-300">
                      CHOSEN
                    </span>
                  )}
                </td>
                <td className="px-4 py-2 text-right font-semibold text-slate-100">
                  {row.accuracy_pct.toFixed(1)}%
                </td>
                <td className="px-4 py-2 text-right text-slate-300">
                  {row.precision_pct.toFixed(1)}%
                </td>
                <td className="px-4 py-2 text-right text-slate-300">
                  {row.recall_pct.toFixed(1)}%
                </td>
                <td className="px-4 py-2 text-right text-slate-300">
                  {row.f1_pct.toFixed(1)}%
                </td>
                <td className="px-4 py-2 text-right text-slate-300">
                  {row.roc_auc_pct.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-slate-300">
        The configured 0.6 / 0.4 split is the swept optimum (α ={" "}
        {ablation.best_alpha.toFixed(1)}): +
        {ablation.chosen_vs_audio_only_pp.toFixed(1)} pp over audio-only and +
        {ablation.chosen_vs_visual_only_pp.toFixed(1)} pp over visual-only.
      </p>
    </div>
  );
};

/**
 * Results / Evaluation figures: confusion matrices with the error glossary,
 * the ROC and Precision-Recall curves (all three models on one axes), and the
 * fusion-weight ablation study.
 */
export const EvaluationFigures: React.FC = () => {
  const [benchmark, setBenchmark] = useState<EvaluationBenchmark | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    new ApiClient()
      .getBenchmark()
      .then((payload) => {
        if (!cancelled) setBenchmark(payload);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err?.message ?? err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="rounded-xl border border-red-500/50 bg-red-950/40 p-3 text-sm text-red-200">
        Failed to load evaluation figures: {error}
      </div>
    );
  }
  if (!benchmark) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm text-slate-400">
        Loading evaluation figures…
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <ConfusionMatrices curves={benchmark.curves} />

      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <h4 className="mb-2 text-sm font-semibold text-slate-200">
            ROC curve — multimodal, visual-only, audio-only
          </h4>
          <Figure figure={benchmark.figures.roc_curve} alt="ROC curve for all three models" />
          <ul className="mt-2 space-y-0.5 text-xs text-slate-400">
            {benchmark.curves.roc.map((curve) => (
              <li key={curve.variant}>
                <span className={VARIANT_COLOR[curve.variant]}>{curve.label}</span> — AUC{" "}
                {(curve.auc * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="mb-2 text-sm font-semibold text-slate-200">
            Precision-Recall curves — all three models
          </h4>
          <Figure
            figure={benchmark.figures.pr_curve}
            alt="Precision-Recall curves for all three models"
          />
          <ul className="mt-2 space-y-0.5 text-xs text-slate-400">
            {benchmark.curves.pr.map((curve) => (
              <li key={curve.variant}>
                <span className={VARIANT_COLOR[curve.variant]}>{curve.label}</span> — AP{" "}
                {(curve.average_precision * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      </div>

      <AblationTable ablation={benchmark.weight_ablation} />
      <Figure figure={benchmark.figures.weight_ablation} alt="Fusion weight ablation chart" />
    </div>
  );
};

export default EvaluationFigures;
