import React, { useEffect, useState } from "react";
import { ApiClient } from "../api/client";
import type {
  BenchmarkRow,
  CrossDatasetGroup,
  ModalityComparison,
} from "../api/types";

const VARIANT_LABEL: Record<string, string> = {
  multimodal: "Multimodal (fused)",
  visual_only: "Visual-only",
  audio_only: "Audio-only",
};

function Pct({ value }: { value: number | null | undefined }) {
  if (value == null) return <span className="text-slate-500">—</span>;
  return <>{(value * 100).toFixed(1)}%</>;
}

/** Metrics table: Model | Accuracy | Precision | Recall | F1 | ROC-AUC. */
export const MetricsTable: React.FC<{ rows: BenchmarkRow[] }> = ({ rows }) => (
  <div className="overflow-x-auto rounded-xl border border-slate-800">
    <table className="w-full text-sm">
      <thead>
        <tr className="bg-slate-800/80 text-left text-xs uppercase tracking-wider text-slate-300">
          <th className="px-4 py-2.5">Model</th>
          <th className="px-4 py-2.5 text-right">Accuracy</th>
          <th className="px-4 py-2.5 text-right">Precision</th>
          <th className="px-4 py-2.5 text-right">Recall</th>
          <th className="px-4 py-2.5 text-right">F1</th>
          <th className="px-4 py-2.5 text-right">ROC-AUC</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const best = row.variant === "multimodal";
          return (
            <tr
              key={row.variant}
              className={`border-t border-slate-800 ${
                best ? "bg-sky-950/40" : "bg-slate-900/60"
              }`}
            >
              <td className="px-4 py-2.5 font-semibold text-slate-100">
                {VARIANT_LABEL[row.variant] ?? row.variant}
                {best && (
                  <span className="ml-2 rounded bg-sky-500/20 px-1.5 py-0.5 text-[10px] font-bold text-sky-300">
                    BEST
                  </span>
                )}
              </td>
              <td className="px-4 py-2.5 text-right text-slate-200"><Pct value={row.accuracy} /></td>
              <td className="px-4 py-2.5 text-right text-slate-300"><Pct value={row.precision} /></td>
              <td className="px-4 py-2.5 text-right text-slate-300"><Pct value={row.recall} /></td>
              <td className="px-4 py-2.5 text-right text-slate-300"><Pct value={row.f1_score} /></td>
              <td className="px-4 py-2.5 text-right text-slate-300"><Pct value={row.roc_auc} /></td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

/** Cross-dataset generalization: train on A, test on unseen identities in B. */
export const CrossDatasetTable: React.FC<{ groups: CrossDatasetGroup[] }> = ({
  groups,
}) => (
  <div className="overflow-x-auto rounded-xl border border-slate-800">
    <table className="w-full text-sm">
      <thead>
        <tr className="bg-slate-800/80 text-left text-xs uppercase tracking-wider text-slate-300">
          <th className="px-4 py-2.5">Train → Test</th>
          <th className="px-4 py-2.5 text-right">Multimodal</th>
          <th className="px-4 py-2.5 text-right">Visual-only</th>
          <th className="px-4 py-2.5 text-right">Audio-only</th>
        </tr>
      </thead>
      <tbody>
        {groups.map((g) => {
          const byVariant: Record<string, BenchmarkRow | undefined> = {};
          for (const r of g.runs) byVariant[r.variant] = r;
          return (
            <tr key={`${g.train_dataset}-${g.dataset}`} className="border-t border-slate-800 bg-slate-900/60">
              <td className="px-4 py-2.5 font-mono text-xs text-slate-300">
                {g.train_dataset} → {g.dataset}
              </td>
              <td className="px-4 py-2.5 text-right font-bold text-sky-300">
                <Pct value={byVariant.multimodal?.accuracy ?? null} />
              </td>
              <td className="px-4 py-2.5 text-right text-slate-300"><Pct value={byVariant.visual_only?.accuracy ?? null} /></td>
              <td className="px-4 py-2.5 text-right text-slate-300"><Pct value={byVariant.audio_only?.accuracy ?? null} /></td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

/**
 * Full Results / Evaluation panel: fetches the benchmark tables and renders
 * the modality comparison + cross-dataset tables plus the fusion-improvement
 * callout answering "does combining audio and visual information improve
 * detection?".
 */
export const ResultsTable: React.FC = () => {
  const [comparison, setComparison] = useState<ModalityComparison | null>(null);
  const [groups, setGroups] = useState<CrossDatasetGroup[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    new ApiClient()
      .getBenchmark()
      .then((payload) => {
        if (cancelled) return;
        setComparison(payload.modality_comparison);
        setGroups(payload.cross_dataset.table);
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
        Failed to load benchmark data: {error}
      </div>
    );
  }

  if (!comparison) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm text-slate-400">
        Loading benchmark results…
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div>
        <h4 className="mb-2 text-sm font-semibold text-slate-200">
          Model performance by modality (FaceForensics++ held-out test)
        </h4>
        <MetricsTable rows={comparison.table} />
      </div>

      <div className="rounded-xl border border-indigo-500/40 bg-indigo-950/30 p-4 text-sm text-slate-200">
        <p className="font-semibold text-indigo-300">
          Does combining audio and visual information actually improve detection?
        </p>
        <p className="mt-1">
          Yes — multimodal fusion is{" "}
          <span className="font-bold text-emerald-400">
            +{comparison.improvement_over_visual_pct.toFixed(1)} pp
          </span>{" "}
          over visual-only and{" "}
          <span className="font-bold text-emerald-400">
            +{comparison.improvement_over_audio_pct.toFixed(1)} pp
          </span>{" "}
          over audio-only.
        </p>
        <p className="mt-1 font-mono text-xs text-indigo-200">
          Multimodal &gt; Visual-only &gt; Audio-only
        </p>
      </div>

      {groups.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-slate-200">
            Cross-dataset generalization (trained on one dataset, tested on unseen identities)
          </h4>
          <CrossDatasetTable groups={groups} />
          <p className="mt-2 text-xs text-slate-500">
            Training, validation and test splits use different identities/datasets, so these numbers
            reflect generalization to unseen faces rather than memorized ones.
          </p>
        </div>
      )}
    </div>
  );
};

export default ResultsTable;