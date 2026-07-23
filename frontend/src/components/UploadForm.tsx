import React, { useState } from "react";
import { ApiClient } from "../api/client";

interface UploadFormProps {
  onSessionCreated: (sessionId: string) => void;
}

export const UploadForm: React.FC<UploadFormProps> = ({ onSessionCreated }) => {
  const [activeTab, setActiveTab] = useState<"file" | "url">("file");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a video file to upload.");
      return;
    }
    setError(null);
    setIsSubmitting(true);
    try {
      const res = await ApiClient.submitFile(file);

      if (res.accepted && res.session_id) {
        onSessionCreated(res.session_id);
      } else {
        setError(res.message || res.error_code || "Upload failed.");
      }
    } catch (err: any) {
      setError(err.message || "Network error submitting video file.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) {
      setError("Please enter a valid video URL.");
      return;
    }
    setError(null);
    setIsSubmitting(true);
    try {
      const res = await ApiClient.submitUrl(url.trim());
      if (res.accepted && res.session_id) {
        onSessionCreated(res.session_id);
      } else {
        setError(res.message || res.error_code || "URL submission failed.");
      }
    } catch (err: any) {
      setError(err.message || "Network error submitting URL.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 text-white p-6 rounded-2xl shadow-xl max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-sky-400 mb-4">Deepfake Detection Intake</h2>
      <div className="flex border-b border-slate-700 mb-6">
        <button
          className={`py-2 px-4 font-semibold ${
            activeTab === "file"
              ? "border-b-2 border-sky-400 text-sky-400"
              : "text-slate-400 hover:text-slate-200"
          }`}
          onClick={() => setActiveTab("file")}
        >
          Upload Video File
        </button>
        <button
          className={`py-2 px-4 font-semibold ${
            activeTab === "url"
              ? "border-b-2 border-sky-400 text-sky-400"
              : "text-slate-400 hover:text-slate-200"
          }`}
          onClick={() => setActiveTab("url")}
        >
          External Video URL
        </button>
      </div>

      {error && (
        <div className="bg-red-950/80 border border-red-500 text-red-200 p-3 rounded-lg text-sm mb-4">
          {error}
        </div>
      )}

      {activeTab === "file" ? (
        <form onSubmit={handleFileSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Select Video File (MP4, AVI - Max 50MB)
            </label>
            <input
              type="file"
              accept=".mp4,.avi"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-sky-500 file:text-white hover:file:bg-sky-600 bg-slate-800 rounded-xl border border-slate-700 p-2"
            />
          </div>
          <button
            type="submit"
            disabled={isSubmitting || !file}
            className="w-full bg-sky-500 hover:bg-sky-600 disabled:opacity-50 text-white font-bold py-3 px-6 rounded-xl transition duration-200"
          >
            {isSubmitting ? "Uploading..." : "Analyze Video"}
          </button>
        </form>
      ) : (
        <form onSubmit={handleUrlSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              External Video URL (HTTP/HTTPS)
            </label>
            <input
              type="url"
              placeholder="https://example.com/video.mp4"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-xl p-3 text-white focus:outline-none focus:border-sky-400"
            />
          </div>
          <button
            type="submit"
            disabled={isSubmitting || !url.trim()}
            className="w-full bg-sky-500 hover:bg-sky-600 disabled:opacity-50 text-white font-bold py-3 px-6 rounded-xl transition duration-200"
          >
            {isSubmitting ? "Fetching URL..." : "Analyze URL"}
          </button>
        </form>
      )}
    </div>
  );
};
