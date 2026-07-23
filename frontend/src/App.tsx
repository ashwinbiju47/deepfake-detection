import { useState } from "react";
import { Dashboard } from "./components/Dashboard";
import { UploadForm } from "./components/UploadForm";

export function App() {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between p-8">
      <header className="max-w-4xl mx-auto w-full flex justify-between items-center pb-8 border-b border-slate-800">
        <h1 className="text-2xl font-black bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
          Real-Time Deepfake Detection Platform
        </h1>
        <span className="text-xs bg-slate-800 text-sky-400 font-mono px-3 py-1 rounded-full border border-slate-700">
          XAI Multi-Modal Engine
        </span>
      </header>

      <div className="my-12">
        {activeSessionId ? (
          <Dashboard
            sessionId={activeSessionId}
            onReset={() => setActiveSessionId(null)}
          />
        ) : (
          <UploadForm onSessionCreated={(sessionId) => setActiveSessionId(sessionId)} />
        )}
      </div>

      <footer className="max-w-4xl mx-auto w-full text-center text-xs text-slate-500 border-t border-slate-800 pt-6">
        &copy; 2026 Deepfake Detection Platform &bull; Multi-Modal Vision & Audio Deepfake Analysis Engine
      </footer>
    </main>
  );
}
