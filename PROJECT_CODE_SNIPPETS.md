# Project code snippets

This is a curated map of the project’s important implementation points. Each
source link includes the exact line where the relevant code begins; the ranges
below are also copied from the current working tree so the behavior can be
reviewed without searching the whole repository.

## 1. System shape

The project is a Django/DRF + Celery/Redis backend and a React/Vite frontend.
The intended runtime flow is:

```text
multipart upload
  -> UploadService validation and transient storage
  -> Celery analyze_session(session_id)
  -> visual extraction/model and/or audio extraction/model
  -> deterministic FusionEngine
  -> persisted results + WebSocket progress/result events
  -> XAI artifacts and media purge
```

Useful overview: [README.md](README.md:13-52).

## 2. Upload validation and acceptance

Source: [backend/detection/services/upload.py](backend/detection/services/upload.py:213-280)

```python
def submit_file(self, upload: UploadedFile) -> UploadResult:
    media_kind = media_kind_for(upload.name)
    if media_kind is None:
        return self._reject(ERROR_UNSUPPORTED_FORMAT, ...)

    size_bytes = upload.size if upload.size is not None else 0
    if size_bytes <= 0:
        return self._reject(ERROR_EMPTY_FILE, "The uploaded file is empty.")
    if size_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
        return self._reject(ERROR_TOO_LARGE, ...)

    with tempfile.NamedTemporaryFile(
        suffix=self._suffix(upload.name), delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        self._write_upload(upload, tmp_path)
        if not self._probe_for(media_kind)(str(tmp_path)):
            return self._reject(ERROR_UNDECODABLE, ...)

        session = self._accept(upload, tmp_path, media_kind)
        if not self._enqueue_analysis(session):
            return self._reject(ERROR_ENQUEUE_FAILED, ...)
        return UploadResult(
            accepted=True,
            session_id=str(session.id),
            error_code=None,
            message=None,
            media_kind=media_kind,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
```

Session creation and transient storage are transactional: [upload.py](backend/detection/services/upload.py:328-346).

```python
with transaction.atomic():
    session = AnalysisSession.objects.create(
        source_type=AnalysisSession.SourceType.FILE,
        source_ref=Path(upload.name).name if upload.name else "upload",
        media_kind=media_kind,
        status=AnalysisSession.Status.QUEUED,
        media_state=AnalysisSession.MediaState.PRESENT,
    )
    media_dir = session_media_dir(session.id)
    media_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(staged_path, media_dir / stored_name)
```

## 3. Visual frame sampling and face isolation

Source: [backend/detection/processing/frame_extractor.py](backend/detection/processing/frame_extractor.py:232-320).

Sampling guarantees at least the configured minimum FPS by selecting every
`floor(frame_rate / min_fps)` native frame:

```python
@staticmethod
def _sampling_step(frame_rate: float, min_fps: float) -> int:
    if min_fps <= 0:
        raise ValueError("min_fps must be > 0")
    if frame_rate <= 0:
        return 1
    return max(1, int(frame_rate // min_fps))

def extract_frames(self, video_path: str, min_fps: float = 1.0,
                   max_frames: int | None = None) -> Iterator[Frame]:
    decoded = self._decoder(video_path)
    frame_rate = float(getattr(decoded, "frame_rate", 0.0) or 0.0)
    step = self._sampling_step(frame_rate, min_fps)
    for position, raw in enumerate(decoded.raw_frames()):
        if position % step != 0:
            continue
        timestamp = (position / frame_rate) if frame_rate > 0 else float(position)
        yield Frame(index=position, timestamp=timestamp,
                    width=raw.width, height=raw.height, image=raw.image)
```

Only faces occupying at least 5% of the frame are retained, then sorted for
reproducibility: [frame_extractor.py](backend/detection/processing/frame_extractor.py:287-320).

```python
frame_area = frame.area
if frame_area <= 0:
    return []

regions = []
for candidate in self._detector(frame):
    ratio = candidate.area / frame_area
    if ratio >= min_area_ratio:
        regions.append(FaceRegion(
            frame_index=frame.index,
            x=candidate.x, y=candidate.y,
            width=candidate.width, height=candidate.height,
            frame_width=frame.width, frame_height=frame.height,
            image=candidate.image,
        ))
regions.sort(key=lambda r: (r.frame_index, r.y, r.x))
return regions
```

Image uploads use the same threshold and state mapping for a single frame:
[frame_extractor.py](backend/detection/processing/frame_extractor.py:324-365).

## 4. Audio extraction and spectrogram fallback

Source: [backend/detection/processing/audio_extractor.py](backend/detection/processing/audio_extractor.py:131-226).

Non-empty audio is guaranteed to yield at least one spectrogram representation:

```python
def to_spectrograms(self, audio: DecodedAudio) -> list[SpectrogramRepresentation]:
    if audio.is_empty:
        return []
    spectrograms = self._spectrogram_generator(audio)
    if not spectrograms:
        spectrograms = [SpectrogramRepresentation(
            sample_rate=audio.sample_rate,
            n_mels=128,
            time_steps=max(1, int(audio.duration_seconds * 100)),
            data=None,
        )]
    return spectrograms
```

Extraction errors and missing audio are converted into explicit states instead
of crashing the whole analysis: [audio_extractor.py](backend/detection/processing/audio_extractor.py:184-226).

## 5. Deterministic model wrappers

Visual likelihoods are clamped and per-face predictions are aggregated by mean
or optional top-k mean: [backend/detection/ml/visual_model.py](backend/detection/ml/visual_model.py:22-28), [visual_model.py](backend/detection/ml/visual_model.py:144-174).

```python
def clamp_likelihood(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)

def aggregate(self, likelihoods: List[float], top_k: int | None = None) -> float:
    if not likelihoods:
        return 0.0
    clamped = [clamp_likelihood(v) for v in likelihoods]
    if top_k is not None and 0 < top_k < len(clamped):
        clamped = sorted(clamped, reverse=True)[:top_k]
    return clamp_likelihood(sum(clamped) / len(clamped))
```

Audio inference rejects an empty spectrogram list and clamps the backend result:
[backend/detection/ml/audio_model.py](backend/detection/ml/audio_model.py:125-145).

## 6. Multimodal fusion and classification

Source: [backend/detection/services/fusion.py](backend/detection/services/fusion.py:57-114).

```python
has_visual = visual_likelihood is not None
has_audio = audio_likelihood is not None

if not has_visual and not has_audio:
    return FusionOutcome(score=None, label=None, modalities_used=[],
                         inconclusive=True, threshold_used=threshold)

if has_visual and has_audio:
    v_val = clamp_likelihood(visual_likelihood)
    a_val = clamp_likelihood(audio_likelihood)
    total_weight = weight_visual + weight_audio
    score = ((v_val * weight_visual + a_val * weight_audio) / total_weight
             if total_weight > 0 else (v_val + a_val) / 2.0)
    modalities_used = ["visual", "audio"]
elif has_visual:
    score, modalities_used = clamp_likelihood(visual_likelihood), ["visual"]
else:
    score, modalities_used = clamp_likelihood(audio_likelihood), ["audio"]

fused_score = clamp_likelihood(score)
label = "deepfake" if fused_score >= threshold else "authentic"
return FusionOutcome(score=fused_score, label=label,
                     modalities_used=modalities_used,
                     inconclusive=False, threshold_used=threshold)
```

## 7. Celery analysis orchestrator

Source: [backend/detection/tasks.py](backend/detection/tasks.py:53-65), [tasks.py](backend/detection/tasks.py:109-251).

The task is capped at three retries and uses late acknowledgements:

```python
@shared_task(
    bind=True,
    name="detection.tasks.analyze_session",
    max_retries=3,
    acks_late=True,
    reject_on_worker_lost=True,
)
def analyze_session(self: Any, session_id: str) -> str:
    ...
```

The modality branch runs visual analysis for video/image and audio analysis for
video/audio, preserving the successful branch when the other branch fails:

```python
if media_kind != AnalysisSession.MediaKind.AUDIO:
    outcome = (extractor.extract_visual_signal_from_image(media_path)
               if media_kind == AnalysisSession.MediaKind.IMAGE
               else extractor.extract_visual_signal(media_path, ...))
    outcome.faces.sort(key=lambda f: (f.frame_index, f.y, f.x))
    if outcome.has_visual_signal and outcome.faces:
        visual_model = VisualModel()
        visual_likelihood = visual_model.aggregate(
            [visual_model.infer_face(face) for face in outcome.faces]
        )
    VisualResult.objects.update_or_create(session=session, defaults={...})

if media_kind != AnalysisSession.MediaKind.IMAGE:
    a_outcome = AudioExtractor().extract_audio_signal(media_path)
    if a_outcome.has_audio_signal and a_outcome.spectrograms:
        audio_likelihood = AudioModel().infer(a_outcome.spectrograms)
    AudioResult.objects.update_or_create(session=session, defaults={...})
```

Fusion, persistence, terminal status, and result event: [tasks.py](backend/detection/tasks.py:215-292).

```python
fusion_outcome = FusionEngine.fuse(
    visual_likelihood=visual_likelihood,
    audio_likelihood=audio_likelihood,
    weight_visual=weight_visual,
    weight_audio=weight_audio,
    threshold=threshold,
)

FusionResult.objects.update_or_create(
    session=session,
    defaults={
        "score": fusion_outcome.score,
        "label": label_choice,
        "modalities_used": ",".join(fusion_outcome.modalities_used),
        "inconclusive": fusion_outcome.inconclusive,
        "threshold_used": fusion_outcome.threshold_used,
    },
)

session.status = (AnalysisSession.Status.INCONCLUSIVE
                  if fusion_outcome.inconclusive
                  else AnalysisSession.Status.COMPLETED)
session.completed_at = timezone.now()
session.save(update_fields=["status", "completed_at"])

StreamService.publish_event(session_id, "result", {
    "score": fusion_outcome.score,
    "label": fusion_outcome.label,
    "modalities_used": fusion_outcome.modalities_used,
    "status": session.status,
    "media_kind": media_kind,
    "visual_likelihood": visual_likelihood,
    "audio_likelihood": audio_likelihood,
    "threshold": fusion_outcome.threshold_used,
})
```

After XAI generation, transient media is purged: [tasks.py](backend/detection/tasks.py:294-311).

## 8. WebSocket event persistence, broadcast, and replay

Source: [backend/detection/services/streamer.py](backend/detection/services/streamer.py:17-71).

```python
with transaction.atomic():
    session = AnalysisSession.objects.get(id=session_id)
    last_event = (StreamEvent.objects.filter(session=session)
                  .order_by("-seq").first())
    seq = (last_event.seq + 1) if last_event else 1
    full_payload = dict(payload)
    full_payload.update(seq=seq, type=event_type, session_id=str(session_id))
    event = StreamEvent.objects.create(
        session=session, seq=seq, type=event_type, payload=full_payload
    )

channel_layer = get_channel_layer()
if channel_layer:
    async_to_sync(channel_layer.group_send)(
        f"analysis_{session_id}",
        {"type": "stream.event", "payload": full_payload},
    )
return event
```

Replay reads only events after the client’s last sequence number:
[streamer.py](backend/detection/services/streamer.py:65-71).

The Channels consumer joins the session group and handles `resume`:
[backend/detection/consumers.py](backend/detection/consumers.py:15-43).

## 9. XAI heatmap triple

Heatmaps are normalized into `[0.0, 1.0]`, rendered as ORIGINAL / HEATMAP /
OVERLAY PNGs, persisted, base64 encoded, and streamed: [backend/detection/services/xai.py](backend/detection/services/xai.py:38-66), [xai.py](backend/detection/services/xai.py:83-162).

```python
raw_matrix = activation_matrix or [[0.1, 0.8], [0.3, 0.9]]
norm_matrix = normalize_heatmap(raw_matrix)
original_png, heatmap_png, overlay_png = render_artifacts(
    norm_matrix, frame_image=frame_image
)
record = FrameHeatmap.objects.create(
    session=session,
    frame_id=frame_id,
    original_png=original_png,
    heatmap_png=heatmap_png,
    overlay_png=overlay_png,
    delivered=True,
)

StreamService.publish_event(
    session_id=session_id,
    event_type="heatmap",
    payload={
        "frame_id": frame_id,
        "heatmap_id": str(record.id),
        "original_b64": base64.b64encode(original_png).decode("ascii"),
        "heatmap_b64": base64.b64encode(heatmap_png).decode("ascii"),
        "overlay_b64": base64.b64encode(overlay_png).decode("ascii"),
    },
)
```

If XAI fails, classification is retained and an error event is emitted: [xai.py](backend/detection/services/xai.py:163-171).

## 10. REST API boundary

Upload intake and typed error mapping: [backend/detection/views.py](backend/detection/views.py:64-111).

```python
upload = request.FILES.get("file")
if upload is not None:
    result = UploadService().submit_file(upload)
    if result["accepted"]:
        return Response(
            {"session_id": result["session_id"],
             "media_kind": result.get("media_kind")},
            status=status.HTTP_202_ACCEPTED,
        )
    return Response(
        {"error_code": result["error_code"], "message": result["message"]},
        status=_ERROR_STATUS.get(result["error_code"], 400),
    )
```

Persisted session details expose per-modality evidence and the fused result:
[views.py](backend/detection/views.py:114-169).

Health flags and PDF report endpoint: [views.py](backend/detection/views.py:50-61), [views.py](backend/detection/views.py:233-253).

## 11. Frontend HTTP client

Source: [frontend/src/api/client.ts](frontend/src/api/client.ts:37-60), [client.ts](frontend/src/api/client.ts:68-112).

```ts
static streamUrl(sessionId: string, origin: string = location.origin): string {
  const wsOrigin = origin.replace(/^http/, "ws");
  return `${wsOrigin}/ws/analyses/${encodeURIComponent(sessionId)}/`;
}

async submitFileInstance(file: File): Promise<UploadResult> {
  const body = new FormData();
  body.append("file", file);
  const res = await this.fetchImpl(`${this.baseUrl}/analyses`, {
    method: "POST",
    body,
  });
  return this.toUploadResult(res);
}
```

The report client preserves backend error messages so the UI can explain
disabled/incomplete reports: [client.ts](frontend/src/api/client.ts:85-100).

## 12. Frontend WebSocket reconnect and resume

Source: [frontend/src/ws/client.ts](frontend/src/ws/client.ts:38-59), [client.ts](frontend/src/ws/client.ts:88-179).

```ts
constructor(sessionId: string, options: ResultStreamerOptions = {}) {
  this.sessionId = sessionId;
  this.url = options.url ?? ApiClient.streamUrl(sessionId);
  this.maxRetries = options.maxRetries ?? 3;
  this.connectTimeoutMs = options.connectTimeoutMs ?? 3000;
  this.WebSocketImpl = options.webSocketImpl ?? WebSocket;
}

private handleMessage(raw: unknown): void {
  let event: StreamEvent;
  try {
    event = typeof raw === "string" ? JSON.parse(raw) as StreamEvent
                                     : raw as StreamEvent;
  } catch {
    return;
  }
  if (typeof event.seq === "number" && event.seq > this.lastSeq) {
    this.lastSeq = event.seq;
  }
  for (const handler of this.eventHandlers) handler(event);
}

private sendResume(): void {
  if (!this.socket || this.socket.readyState !== this.socket.OPEN) return;
  const request: ResumeRequest = { type: "resume", last_seq: this.lastSeq };
  this.socket.send(JSON.stringify(request));
}
```

Connection timeout and retry behavior are implemented in [client.ts](frontend/src/ws/client.ts:104-149).

## 13. Dashboard event handling and deduplication

Source: [frontend/src/components/Dashboard.tsx](frontend/src/components/Dashboard.tsx:120-160).

```tsx
const streamer = new ResultStreamer(sessionId);
streamer.onEvent((event: StreamEvent) => {
  if (event.type === "progress") {
    setProgress((event as { progress_percent?: number }).progress_percent ?? 0);
  } else if (event.type === "result") {
    const payload = event as unknown as ResultPayload;
    setResult(payload);
    if (payload.media_kind) setMediaKind(payload.media_kind);
    setProgress(100);
  } else if (event.type === "heatmap") {
    const incoming = event as unknown as HeatmapPayload;
    setHeatmaps((prev) => prev.some(
      (h) => h.heatmap_id === incoming.heatmap_id
    ) ? prev : [...prev, incoming]);
  } else if (event.type === "error") {
    setErrorMessage((event as { message?: string }).message
                    || "Analysis pipeline error.");
  }
});
streamer.connect();
```

The stage mapping accounts for image uploads skipping audio stages: [Dashboard.tsx](frontend/src/components/Dashboard.tsx:18-33).

## 14. PDF reports and evaluation

Report generation, incomplete-session checks, architecture diagram, and XAI
triple rendering live in [backend/detection/services/report.py](backend/detection/services/report.py:377-447) and [report.py](backend/detection/services/report.py:777-809).

Evaluation/benchmark APIs are assembled in [backend/detection/views.py](backend/detection/views.py:210-230), with metric and curve logic in [backend/detection/services/benchmark.py](backend/detection/services/benchmark.py:238-409).

The frontend renders evaluation figures and modality tables in:

- [frontend/src/components/EvaluationFigures.tsx](frontend/src/components/EvaluationFigures.tsx:1-274)
- [frontend/src/components/ResultsTable.tsx](frontend/src/components/ResultsTable.tsx:1-190)

## 15. Media purge and privacy boundary

Source: [backend/detection/services/purger.py](backend/detection/services/purger.py:33-93).

```python
class MediaPurger:
    @staticmethod
    def purge(session_id: str, max_retries: int = 3) -> PurgeRecord:
        # Deletes transient media, verifies deletion, and records SUCCESS or
        # FAILED with retry count in PurgeRecord.
        ...
```

The orchestrator invokes purge after result/XAI processing at
[backend/detection/tasks.py](backend/detection/tasks.py:304-309).

## 16. High-value tests for these paths

The contract is extensively covered by unit/property/integration tests. The
most useful starting points are:

- Upload validation: [backend/tests/unit/test_upload_errors.py](backend/tests/unit/test_upload_errors.py:1)
- Fusion determinism and threshold: [backend/tests/unit/test_detection_determinism.py](backend/tests/unit/test_detection_determinism.py:1), [backend/tests/property/test_property_fusion_threshold.py](backend/tests/property/test_property_fusion_threshold.py:1)
- Frame sampling and face area: [backend/tests/property/test_property_frame_sampling.py](backend/tests/property/test_property_frame_sampling.py:1), [backend/tests/property/test_property_face_area.py](backend/tests/property/test_property_face_area.py:1)
- Audio extraction: [backend/tests/unit/test_audio_extraction.py](backend/tests/unit/test_audio_extraction.py:1)
- End-to-end pipeline: [backend/tests/integration/test_e2e_pipeline.py](backend/tests/integration/test_e2e_pipeline.py:1)
- WebSocket integration/reconnect: [backend/tests/integration/test_ws_integration.py](backend/tests/integration/test_ws_integration.py:1), [frontend/src/test/ws_client.test.ts](frontend/src/test/ws_client.test.ts:1)
- XAI normalization and PDF reports: [backend/tests/unit/test_heatmap_xai.py](backend/tests/unit/test_heatmap_xai.py:1), [backend/tests/unit/test_pdf_report.py](backend/tests/unit/test_pdf_report.py:1)
- Media purge: [backend/tests/property/test_property_media_purge.py](backend/tests/property/test_property_media_purge.py:1)

## Quick navigation

| Concern | Primary file | Entry point |
|---|---|---|
| Intake | `backend/detection/services/upload.py` | `UploadService.submit_file` |
| Visual extraction | `backend/detection/processing/frame_extractor.py` | `FrameExtractor.extract_visual_signal` |
| Audio extraction | `backend/detection/processing/audio_extractor.py` | `AudioExtractor.extract_audio_signal` |
| Visual model | `backend/detection/ml/visual_model.py` | `VisualModel.infer_face` |
| Audio model | `backend/detection/ml/audio_model.py` | `AudioModel.infer` |
| Fusion | `backend/detection/services/fusion.py` | `FusionEngine.fuse` |
| Orchestration | `backend/detection/tasks.py` | `analyze_session` |
| Streaming | `backend/detection/services/streamer.py` | `StreamService.publish_event` |
| XAI | `backend/detection/services/xai.py` | `XAIGenerator.generate_heatmap` |
| REST API | `backend/detection/views.py` | `create_analysis`, `get_analysis` |
| WebSocket client | `frontend/src/ws/client.ts` | `ResultStreamer` |
| Dashboard | `frontend/src/components/Dashboard.tsx` | `Dashboard` |
| Reports | `backend/detection/services/report.py` | `ReportGenerator.generate` |
| Purge | `backend/detection/services/purger.py` | `MediaPurger.purge` |
