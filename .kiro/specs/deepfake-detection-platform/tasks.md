# Implementation Plan: Real-Time Deepfake Detection Platform

## Overview

This plan converts the approved design into an incremental, test-driven implementation sequence.
Work is ordered by MoSCoW priority: Must-Have (Requirements 1–9, the MVP pipeline) first, then
Should-Have (Requirements 10 URL input, 11 Grad-CAM), then Could-Have (Requirement 12 PDF report).

Each task builds on prior tasks and ends by wiring new code into the running pipeline so there is
no orphaned code. The implementation stack is fixed by the design:

- **Backend:** Python — Django + Django REST Framework + Django Channels (ASGI), Celery workers on
  a Redis broker/channel layer, PyTorch for the visual & audio models and Grad-CAM, OpenCV/FFmpeg/
  MTCNN/Librosa for media processing, PostgreSQL via the Django ORM.
- **Frontend:** TypeScript — React + TailwindCSS, Chart.js/D3 for visualizations, native WebSocket
  client.
- **Property-based testing:** [Hypothesis](https://hypothesis.readthedocs.io/) for Python logic,
  [fast-check](https://github.com/dubzzz/fast-check) for the TypeScript event-replay logic. Minimum
  100 iterations per property test. ML inference is mocked with deterministic stubs in property
  tests. Each property test is tagged with the comment:
  **Feature: deepfake-detection-platform, Property {number}: {property_text}**

Tasks marked with `*` are optional test sub-tasks and may be skipped for a faster MVP. Top-level
tasks and core implementation sub-tasks are never optional.

## Tasks

- [x] 1. Scaffold project structure, configuration, and test tooling
  - [x] 1.1 Create the Django backend project skeleton and dependencies
    - Initialize a Django project with a `detection` app; configure DRF, Channels (ASGI),
      Celery, and the Redis broker/channel layer in settings
    - Add PostgreSQL via the Django ORM and a transient media store path keyed by `session_id`
    - Define feature flags `EXTERNAL_URL_ENABLED`, `HEATMAP_ENABLED`, `REPORT_ENABLED` and the
      configurable `CELERY_WORKER_CONCURRENCY` (1–64), decision `threshold`, and fusion weights
    - Pin dependencies (Django, DRF, channels, celery, redis, torch, opencv-python, ffmpeg,
      librosa, mtcnn/mediapipe, reportlab/weasyprint, hypothesis, pytest, pytest-django)
    - _Requirements: 6.4_

  - [x] 1.2 Scaffold the React + TypeScript frontend with tooling
    - Initialize a React + TypeScript app with TailwindCSS and Chart.js/D3 dependencies
    - Install and configure fast-check and the test runner (vitest/jest) for property tests
    - Add a typed WebSocket client module stub and an API client stub pointing at the DRF endpoints
    - _Requirements: 5.1_

  - [x] 1.3 Configure the Python test harness for property-based testing
    - Configure pytest + pytest-django and Hypothesis (set default `max_examples >= 100`)
    - Add a deterministic ML-inference stub fixture (returns values in `[0,1]`) for use in
      property and unit tests so PyTorch is never invoked in logic tests
    - _Requirements: 8.2_

- [x] 2. Implement persistence layer (Django data models)
  - [x] 2.1 Define core ORM models and migrations
    - Implement `AnalysisSession`, `VisualResult`, `AudioResult`, `FusionResult`,
      `FrameHeatmap`, `StreamEvent`, `PurgeRecord`, `Report`, `ModelEvaluation`,
      `EvaluationMetrics` per the ERD, with nullable likelihood/score fields and the
      media/metadata separation (`source_ref` holds only filename/URL, no media)
    - Generate and apply migrations
    - _Requirements: 9.3, 4.4, 2.4, 3.4_

  - [x]* 2.2 Write unit tests for model constraints and state enums
    - Test status/state enum values, nullability of likelihood/score, and that sessions store no
      media bytes
    - _Requirements: 9.3_

- [x] 3. Implement file Upload_Service and validation (Must-Have, Requirement 1)
  - [x] 3.1 Implement the ordered validation pipeline and file submission
    - Implement `UploadService.submit_file` with fail-fast validation: format check (MP4/AVI),
      size check (`1..52428800`), decodability probe (FFmpeg/OpenCV); on success create exactly
      one `AnalysisSession` (status QUEUED) and return its id; on failure create no session and
      return a typed `error_code` + message (EMPTY_FILE, UNSUPPORTED_FORMAT, TOO_LARGE, UNDECODABLE)
    - Wire the `POST /api/analyses` DRF view (multipart) to call the service and return
      `202 {session_id}` or the appropriate 4xx error
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x]* 3.2 Write property test for valid uploads
    - **Property 1: Valid uploads create exactly one session and return its id**
    - **Validates: Requirements 1.1, 1.6**

  - [x]* 3.3 Write property test for invalid uploads
    - **Property 2: Invalid uploads are rejected with no session and a correct message**
    - **Validates: Requirements 1.2, 1.3, 1.4, 1.5**

  - [x]* 3.4 Write unit tests for each upload error code
    - Cover zero-byte, oversize, unsupported-format, and declared-supported-but-undecodable
      examples returning the correct code and message
    - _Requirements: 1.2, 1.3, 1.4, 1.5_

- [x] 4. Implement Frame_Extractor and face isolation (Must-Have, Requirement 2)
  - [x] 4.1 Implement frame sampling and face isolation
    - Implement `FrameExtractor.extract_frames` sampling at ≥ 1 fps of duration (OpenCV/FFmpeg)
      and `isolate_faces` keeping only regions ≥ 5% of frame area (MTCNN/MediaPipe)
    - Record the `no_visual_signal` state when no face is isolated (2.4) and the `visual_error`
      state when extraction fails before any frame is produced (2.5)
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

  - [x]* 4.2 Write property test for minimum frame sampling rate
    - **Property 3: Frame extraction meets the minimum sampling rate**
    - **Validates: Requirements 2.1**

  - [x]* 4.3 Write property test for the 5% face-area threshold
    - **Property 4: Face isolation respects the 5% area threshold**
    - **Validates: Requirements 2.2**

  - [x]* 4.4 Write unit test for extraction-failure handling
    - Inject an extraction failure and assert `visual_error` state with error indication, and that
      audio analysis still proceeds
    - _Requirements: 2.5_

- [ ] 5. Implement Audio_Extractor and spectrogram generation (Must-Have, Requirement 3)
  - [ ] 5.1 Implement audio demux and spectrogram generation
    - Implement `AudioExtractor.extract_audio` (FFmpeg demux; returns None when no audio track) and
      `to_spectrograms` (Librosa mel-spectrograms, ≥ 1 representation for non-empty audio)
    - Record `no_audio_signal` (3.4) and `audio_error` (3.5) states, retaining existing session data
    - _Requirements: 3.1, 3.2, 3.4, 3.5_

  - [ ]* 5.2 Write property test for spectrogram generation
    - **Property 7: Spectrogram generation yields at least one representation**
    - **Validates: Requirements 3.2**

  - [ ]* 5.3 Write unit tests for audio extraction edge cases
    - Sample-media extraction success (3.1), no-audio-track path (3.4), and injected
      extraction/spectrogram failure (3.5)
    - _Requirements: 3.1, 3.4, 3.5_

- [ ] 6. Implement Visual_Model and Audio_Model inference (Must-Have, Requirements 2, 3)
  - [ ] 6.1 Implement Visual_Model inference and aggregation
    - Implement `VisualModel.infer_face` returning a per-face likelihood in `[0,1]` and
      `aggregate` returning a single aggregate visual likelihood in `[0,1]` (mean with optional
      top-k weighting), clamped to range
    - _Requirements: 2.3, 2.6_

  - [ ] 6.2 Implement Audio_Model inference
    - Implement `AudioModel.infer` over spectrograms returning an audio likelihood in `[0,1]`;
      record `audio_error` on inference failure, retaining session data (3.6)
    - _Requirements: 3.3, 3.6_

  - [ ]* 6.3 Write property test for likelihood range bounds
    - **Property 5: All likelihood values fall within [0.0, 1.0]**
    - **Validates: Requirements 2.3, 2.6, 3.3**

  - [ ]* 6.4 Write unit test for audio inference-failure handling
    - Inject an inference failure and assert `audio_error` state with session data retained and
      visual analysis continuing
    - _Requirements: 3.6_

- [ ] 7. Implement Fusion_Engine (Must-Have, Requirement 4)
  - [ ] 7.1 Implement deterministic multi-modal fusion and labeling
    - Implement `FusionEngine.fuse` as a pure deterministic function: both modalities →
      weighted combination clamped to `[0,1]`; one modality → derived score with
      `modalities_used` recorded; neither → `score=None`, `inconclusive=True`; assign label
      `"authentic"` iff `score < threshold`, `"deepfake"` iff `score >= threshold`
    - Persist the `FusionResult` (score, label, modalities_used, threshold_used, inconclusive)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 7.2 Write property test for fusion determinism and modality-awareness
    - **Property 8: Fusion is deterministic, range-bounded, and modality-aware**
    - **Validates: Requirements 4.1, 4.2, 4.4**

  - [ ]* 7.3 Write property test for label threshold logic
    - **Property 9: Classification label matches the decision threshold**
    - **Validates: Requirements 4.3**

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement Task_Queue and async orchestration (Must-Have, Requirement 6)
  - [ ] 9.1 Implement the Celery analysis orchestrator and enqueue path
    - Implement the `analyze_session` orchestrator coordinating visual and audio sub-pipelines as a
      parallel group joined by a chord callback that runs fusion, then persists and (later) streams
    - Wire the Upload_Service enqueue: on success enqueue and let the request thread return within
      2s; on enqueue failure mark the session FAILED_TO_QUEUE (never PROCESSING) and return an error
    - Configure FIFO dispatch, `task_acks_late`, `task_reject_on_worker_lost`, `max_retries=3` with
      exponential backoff, and `worker_concurrency` (1–64)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 9.2 Write property test for enqueue-failure session state
    - **Property 11: Enqueue failure never leaves a session in-progress**
    - **Validates: Requirements 6.2**

  - [ ]* 9.3 Write property test for FIFO, loss-free dispatch
    - **Property 12: Queue dispatch is FIFO and loss-free**
    - **Validates: Requirements 6.3, 6.5**

  - [ ]* 9.4 Write property test for worker-capacity bound
    - **Property 13: In-progress sessions never exceed configured capacity**
    - **Validates: Requirements 6.4**

  - [ ]* 9.5 Write property test for capped retries and session integrity
    - **Property 14: Job retries are capped at 3 and preserve session integrity**
    - **Validates: Requirements 6.6**

  - [ ]* 9.6 Write property test for cross-modality resilience
    - **Property 6: A missing or failed modality does not abort the other modality**
    - **Validates: Requirements 2.4, 3.4, 3.5, 3.6**

- [ ] 10. Implement Result_Streamer WebSocket layer (Must-Have, Requirements 5, 7)
  - [ ] 10.1 Implement the Channels consumer, event log, and replay
    - Implement an `AsyncWebsocketConsumer` with per-session group `analysis_{session_id}`;
      persist a sequence-numbered append-only `StreamEvent` log in Redis (TTL ≥ 30s) and on
      `resume {last_seq}` replay all events with `seq > last_seq` before resuming live streaming
    - Wire the orchestrator to publish `progress`/`result`/`error` events: emit `progress 0%` first,
      heartbeat progress at intervals ≤ 2s, and emit the result event on score production
    - _Requirements: 5.3, 5.4, 5.5, 7.1, 7.2, 7.3, 6.6_

  - [ ]* 10.2 Write property test for lossless gap-free replay (TypeScript / fast-check)
    - **Property 10: Stream replay on reconnect is lossless and gap-free**
    - **Validates: Requirements 5.5**

  - [ ]* 10.3 Write unit test for delayed-update retry handling
    - Assert delayed updates retry up to 3 times, show an "updates delayed" indication, and retain
      session state for resumed streaming
    - _Requirements: 7.4_

- [ ] 11. Implement Media_Purger (Must-Have / GDPR, Requirement 9)
  - [ ] 11.1 Implement media purge, verification, and retry
    - Implement `MediaPurger.purge` invoked on completion, failure, or cancellation: delete the
      uploaded video and all intermediate artifacts, `verify_empty`, record a `PurgeRecord`
      (SUCCESS/FAILED), retry up to 3 times, and on persistent failure record FAILED and raise an
      operator alert; retain only score, label, and timestamps
    - Wire purge invocation into the orchestrator's terminal stages (completed/failed/canceled)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 11.2 Write property test for complete media purge across all triggers
    - **Property 15: Media purge removes all media artifacts across all triggers**
    - **Validates: Requirements 9.1, 9.2, 9.4**

  - [ ]* 11.3 Write property test for retained-metadata-only invariant
    - **Property 16: Only non-media metadata is retained after purge**
    - **Validates: Requirements 9.3**

  - [ ]* 11.4 Write unit test for purge-failure alerting
    - Inject residual artifacts and assert retry up to 3 times then a FAILED purge status naming the
      session plus an operator alert
    - _Requirements: 9.5_

- [ ] 12. Implement model evaluation and metrics persistence (Must-Have, Requirement 8)
  - [ ] 12.1 Implement metric computation, persistence, and baseline flag
    - Implement confusion-matrix-derived accuracy, precision, recall, F1 (each in `[0,1]`); persist
      `ModelEvaluation` + `EvaluationMetrics`; set `meets_baseline` iff accuracy ≥ 0.85 and, when
      below, report measured accuracy and the failed threshold; expose `GET /api/evaluations/{run_id}`
    - _Requirements: 8.2, 8.3, 8.4_

  - [ ]* 12.2 Write property test for metric relationships
    - **Property 17: Evaluation metrics satisfy their mathematical relationships**
    - **Validates: Requirements 8.2**

  - [ ]* 12.3 Write property test for metrics persistence round-trip
    - **Property 18: Evaluation metrics persist and round-trip**
    - **Validates: Requirements 8.3**

  - [ ]* 12.4 Write property test for the 85% baseline flag
    - **Property 19: Baseline flag reflects the 85% threshold**
    - **Validates: Requirements 8.1, 8.4**

  - [ ]* 12.5 Write the FaceForensics++ accuracy benchmark (smoke evaluation)
    - Run the multi-modal pipeline against the held-out FaceForensics++ test split and assert
      accuracy ≥ 85%, persisting the run's metrics
    - _Requirements: 8.1_

- [ ] 13. Checkpoint - Ensure all MVP tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Implement external video URL input (Should-Have, Requirement 10)
  - [ ] 14.1 Implement URL intake with scheme, reachability, and streaming size guard
    - Implement `UploadService.submit_url` (gated by `EXTERNAL_URL_ENABLED`): reject non-HTTP(S)
      schemes (BAD_SCHEME); retrieve with a 30s timeout (URL_UNREACHABLE on failure/no content);
      halt retrieval once 50MB is exceeded (TOO_LARGE); format-check retrieved content; on success
      create an `AnalysisSession` and return its id
    - Wire URL submission into the `POST /api/analyses` (json) path
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 14.2 Write property test for URL scheme rejection
    - **Property 21: Non-HTTP(S) URL schemes are rejected with scheme guidance**
    - **Validates: Requirements 10.3**

  - [ ]* 14.3 Write unit tests for URL retrieval edge cases
    - Cover success path (10.1, 10.2), unreachable/no-content (10.4), unsupported retrieved format
      (10.5), and oversize halt (10.6)
    - _Requirements: 10.1, 10.2, 10.4, 10.5, 10.6_

- [ ] 15. Implement Grad-CAM XAI_Generator (Should-Have, Requirement 11)
  - [ ] 15.1 Implement Grad-CAM generation, normalization, and delivery
    - Implement `XAIGenerator.grad_cam` (gated by `HEATMAP_ENABLED`) from the Visual_Model's last
      conv-layer activations/gradients, normalized to `[0,1]`, overlaid on the source frame;
      persist `FrameHeatmap` overlays as non-source-media artifacts
    - On generation failure retain the classification, omit the heatmap, emit an error for that
      frame (11.2); wire delivery into the streamer — stream within 1s if connected, otherwise
      discard without blocking and emit an error (11.3, 11.4)
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 15.2 Write property test for heatmap normalization
    - **Property 20: Grad-CAM heatmaps are normalized to [0.0, 1.0]**
    - **Validates: Requirements 11.1**

  - [ ]* 15.3 Write unit tests for heatmap failure and no-client delivery
    - Cover generation-failure omission with error (11.2) and discard-on-no-client with error (11.4)
    - _Requirements: 11.2, 11.4_

- [ ] 16. Implement downloadable PDF Report_Generator (Could-Have, Requirement 12)
  - [ ] 16.1 Implement PDF generation for completed sessions
    - Implement `ReportGenerator.generate` (gated by `REPORT_ENABLED`) producing a single PDF with
      score, label, per-modality findings, and every generated heatmap, within 10s; refuse
      incomplete sessions with the completion message (12.4); on failure deliver no partial PDF,
      return a failure message, leave session data unchanged (12.5)
    - Wire `GET /api/analyses/{id}/report` to produce and serve the PDF (12.2)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 16.2 Write property test for report content and incomplete-session refusal
    - **Property 22: Reports for completed sessions contain required content; incomplete sessions are refused**
    - **Validates: Requirements 12.1, 12.3, 12.4**

  - [ ]* 16.3 Write unit tests for report availability and failure paths
    - Cover availability after generation (12.2) and no-partial-PDF on generation failure (12.5)
    - _Requirements: 12.2, 12.5_

- [ ] 17. Implement React Dashboard frontend
  - [ ] 17.1 Implement upload/URL submission and session views
    - Implement the upload form (file + URL), display the returned `session_id`, and render
      session status/result fetched from the API
    - _Requirements: 1.6, 10.2_

  - [ ] 17.2 Implement the WebSocket client with reconnect, resume, and visualizations
    - Connect within 3s with up to 3 retries and an error indication on exhaustion (5.1, 5.2);
      on reconnect send `resume {last_seq}`; render progress (Chart.js/D3), final score/label,
      heatmap overlays, and a report-download control
    - _Requirements: 5.1, 5.2, 5.5, 11.3, 12.2_

  - [ ]* 17.3 Write unit tests for the WebSocket client reconnect/resume logic
    - Test retry-up-to-3, error indication on exhaustion, and `resume` request emission
    - _Requirements: 5.2, 5.5_

- [ ] 18. Integration tests for transport, timing, and end-to-end wiring
  - [ ]* 18.1 Write WebSocket connection and timing integration tests
    - Connection establishment < 3s (5.1) with retry (5.2); progress interval ≤ 2s (5.3, 7.2);
      first update ≤ 5s (7.1); result delivery ≤ 1s/2s (5.4, 7.3); heatmap delivery ≤ 1s (11.3)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 7.1, 7.2, 7.3, 11.3_

  - [ ]* 18.2 Write async acceptance and FIFO dispatch integration tests
    - Acceptance/ack returns < 2s without waiting for inference (6.1); FIFO dispatch within 5s of a
      worker becoming available (6.3)
    - _Requirements: 6.1, 6.3_

  - [ ]* 18.3 Write the end-to-end happy-path integration test
    - Upload → worker pipeline → streamed result → media purged within 60s (9.1, 9.2)
    - _Requirements: 9.1, 9.2_

- [ ] 19. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; core implementation
  sub-tasks are never optional.
- Each task references specific requirements (and properties where applicable) for traceability.
- MoSCoW ordering: Must-Have (Tasks 1–13) → Should-Have (Tasks 14–15) → Could-Have (Task 16),
  followed by the frontend and integration/wiring tasks.
- All 22 correctness properties are covered by single property-based tests; Property 10 uses
  fast-check (TypeScript), the rest use Hypothesis (Python). ML inference is mocked in property
  tests.
- Timing/transport criteria and the accuracy benchmark are covered by integration and smoke tests
  (Tasks 18, 12.5), per the design's Testing Strategy.
- Checkpoints (Tasks 8, 13, 19) provide incremental validation at MVP-logic, MVP-complete, and
  full-feature boundaries.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1", "4.1", "5.1", "6.1", "6.2", "7.1", "12.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4", "4.2", "4.3", "4.4", "5.2", "5.3", "6.3", "6.4", "7.2", "7.3", "12.2", "12.3", "12.4", "12.5"] },
    { "id": 4, "tasks": ["9.1"] },
    { "id": 5, "tasks": ["9.2", "9.3", "9.4", "9.5", "9.6", "10.1", "11.1", "14.1", "15.1"] },
    { "id": 6, "tasks": ["10.2", "10.3", "11.2", "11.3", "11.4", "14.2", "14.3", "15.2", "15.3", "16.1"] },
    { "id": 7, "tasks": ["16.2", "16.3", "17.1", "17.2"] },
    { "id": 8, "tasks": ["17.3", "18.1", "18.2", "18.3"] }
  ]
}
```
