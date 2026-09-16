import React, { useMemo, useRef, useState } from "react";
import { ApiClient } from "../api/client";
import type { MediaKind } from "../api/types";

interface UploadFormProps {
  onSessionCreated: (sessionId: string) => void;
}

const ACCEPTED_INPUT =
  ".mp4,.avi,.mov,.mkv,.webm,.jpg,.jpeg,.png,.webp,.bmp,.wav,.mp3,.flac,.ogg,.m4a";

const KIND_META: Record<MediaKind, { label: string; verb: string; hint: string }> = {
  video: {
    label: "Video",
    verb: "Analyze Video",
    hint: "Full multimodal pipeline: visual + audio analysis, fusion, and Grad-CAM XAI.",
  },
  image: {
    label: "Image",
    verb: "Analyze Image",
    hint: "Visual pipeline only: face detection, visual model scoring, and Grad-CAM XAI.",
  },
  audio: {
    label: "Audio",
    verb: "Analyze Audio",
    hint: "Audio pipeline only: mel-spectrogram scoring and spectrogram attention XAI.",
  },
};

function detectKind(file: File): MediaKind | null {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (["mp4", "avi", "mov", "mkv", "webm"].includes(ext)) return "video";
  if (["jpg", "jpeg", "png", "webp", "bmp"].includes(ext)) return "image";
  if (["wav", "mp3", "flac", "ogg", "m4a"].includes(ext)) return "audio";
  return null;
}

export const UploadForm: React.FC<UploadFormProps> = ({ onSessionCreated }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const kind = useMemo(() => (file ? detectKind(file) : null), [file]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const selected = e.target.files?.[0] || null;
    setFile(selected);
  };

  const clearFile = () => {
    setFile(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a video, image, or audio file to upload.");
      return;
    }
    if (!kind) {
      setError("Unsupported file type. Upload a video, image, or audio file.");
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
      setError(err.message || "Network error submitting file.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const meta = kind ? KIND_META[kind] : null;

  return (
    <div className="bg-slate-900 border border-slate-800 text-white p-6 rounded-2xl shadow-xl max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-sky-400 mb-1">
        Deepfake Detection Intake
      </h2>
      <p className="text-xs text-slate-400 mb-5">
        Upload a video, image, or audio file — the platform detects which it is
        and runs the matching analysis pipeline. URL input is not supported.
      </p>

      {error && (
        <div className="bg-red-950/80 border border-red-500 text-red-200 p-3 rounded-lg text-sm mb-4">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1">
            Select Media File (max 50MB)
          </label>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_INPUT}
            onChange={handleFileChange}
            className="w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-sky-500 file:text-white hover:file:bg-sky-600 bg-slate-800 rounded-xl border border-slate-700 p-2"
          />
        </div>

        {file && (
          <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-800/60 p-3 text-sm">
            <div className="min-w-0">
              <p className="truncate font-mono text-xs text-slate-200">{file.name}</p>
              <p className="text-xs text-slate-400">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
                {meta ? ` — detected: ${meta.label}` : " — unsupported type"}
              </p>
            </div>
            <button
              type="button"
              onClick={clearFile}
              className="ml-3 shrink-0 rounded-lg border border-slate-600 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-700"
            >
              Remove
            </button>
          </div>
        )}

        {meta && (
          <p className="rounded-lg border border-sky-500/30 bg-sky-950/30 px-3 py-2 text-xs text-sky-200">
            {meta.hint}
          </p>
        )}

        <button
          type="submit"
          disabled={isSubmitting || !file || !kind}
          className="w-full bg-sky-500 hover:bg-sky-600 disabled:opacity-50 text-white font-bold py-3 px-6 rounded-xl transition duration-200"
        >
          {isSubmitting ? "Uploading..." : meta ? meta.verb : "Analyze Media"}
        </button>
      </form>
    </div>
  );
};

export default UploadForm;
