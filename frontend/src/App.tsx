/**
 * Application shell. This is intentionally a thin placeholder — the full
 * dashboard (upload form, progress charts, heatmap overlays, report download)
 * is implemented in a later task (Task 17). This scaffold only establishes the
 * tooling, the typed API client stub, and the typed WebSocket client stub.
 */
export function App() {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 flex items-center justify-center p-8">
      <div className="max-w-xl text-center space-y-4">
        <h1 className="text-3xl font-bold">Deepfake Detection Platform</h1>
        <p className="text-slate-600">
          Frontend scaffold ready. Dashboard UI is implemented in a later task.
        </p>
      </div>
    </main>
  );
}
