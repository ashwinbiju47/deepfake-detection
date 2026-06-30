# Requirements Document

## Introduction

The Deepfake Detection Platform is a web-based system that analyzes online video content using multi-modal deep learning to detect deepfake manipulation in near real-time. The platform combines visual analysis (facial frame inspection for spatial artifacts) with audio analysis (synthetic voice anomaly detection via spectrograms), fuses the two signals into a single classification score, and streams results to a browser dashboard over WebSockets.

To build user trust, the platform integrates Explainable AI (XAI) by generating Grad-CAM heatmaps that highlight manipulated regions of analyzed frames. To satisfy GDPR obligations, the platform processes media transiently and purges all uploaded media after analysis completes.

The architecture decouples HTTP request handling from compute-intensive ML inference using an asynchronous task queue and message broker, preventing request timeouts and supporting concurrent analyses. This document defines the requirements for the Minimum Viable Product (MVP) and the prioritized enhancements, organized using MoSCoW prioritization.

## Glossary

- **Platform**: The complete Deepfake Detection Platform, including frontend, backend, task queue, and ML inference components.
- **Upload_Service**: The backend component that receives, validates, and accepts video inputs (file uploads and external URLs).
- **Frame_Extractor**: The component that extracts video frames and isolates facial regions for visual analysis.
- **Audio_Extractor**: The component that extracts the audio track from a video and produces spectrogram representations.
- **Visual_Model**: The vision deep learning model (CNN or Transformer) that classifies facial frames for spatial deepfake artifacts.
- **Audio_Model**: The deep learning model that classifies audio spectrograms for synthetic voice anomalies.
- **Fusion_Engine**: The component that combines Visual_Model and Audio_Model inferences into a final classification score.
- **Result_Streamer**: The WebSocket-based component that transmits analysis progress and results to the frontend.
- **Dashboard**: The React-based frontend that displays analysis progress, results, and explainability visualizations.
- **XAI_Generator**: The component that produces Grad-CAM heatmaps overlaying analyzed video frames.
- **Report_Generator**: The component that produces a downloadable PDF summary of an analysis.
- **Task_Queue**: The asynchronous processing layer (Celery workers backed by a Redis broker) that executes ML inference jobs.
- **Media_Purger**: The component that deletes uploaded and intermediate media after analysis completes.
- **Analysis_Session**: A single end-to-end processing job for one video input, identified by a unique session identifier.
- **Classification_Score**: A numeric value in the range 0.0 to 1.0 representing the likelihood that the analyzed video is a deepfake.
- **Supported_Format**: A video container format the Platform accepts, limited to MP4 and AVI.
- **Grad-CAM**: Gradient-weighted Class Activation Mapping, a technique that produces a heatmap indicating the regions a model used for its classification.

## Requirements

### Requirement 1: Video File Upload (FR-01, Must-Have)

**User Story:** As a user, I want to upload a video file, so that the Platform can analyze it for deepfake manipulation.

#### Acceptance Criteria

1. WHEN a user submits a video file in a Supported_Format with a size between 1 byte and 52,428,800 bytes (50MB) inclusive, THE Upload_Service SHALL accept the file and create exactly one Analysis_Session.
2. IF a user submits a file in a format other than a Supported_Format, THEN THE Upload_Service SHALL reject the file, create no Analysis_Session, and return a message identifying the supported formats (MP4 and AVI).
3. IF a user submits a file larger than 52,428,800 bytes (50MB), THEN THE Upload_Service SHALL reject the file, create no Analysis_Session, and return a message stating the 50MB size limit.
4. IF a user submits a file of 0 bytes, THEN THE Upload_Service SHALL reject the file, create no Analysis_Session, and return a message stating that the file is empty.
5. IF a user submits a file that declares a Supported_Format but cannot be decoded as a valid video, THEN THE Upload_Service SHALL reject the file, create no Analysis_Session, and return a message stating that the file could not be read as a valid video.
6. WHEN the Upload_Service accepts a video file, THE Upload_Service SHALL create exactly one Analysis_Session and return its Analysis_Session identifier to the user.

### Requirement 2: Visual Frame Analysis (FR-02, Must-Have)

**User Story:** As a user, I want the Platform to analyze facial frames in my video, so that spatial deepfake artifacts can be detected.

#### Acceptance Criteria

1. WHEN an Analysis_Session begins, THE Frame_Extractor SHALL extract video frames at a sampling rate of at least 1 frame per second of video duration.
2. WHEN the Frame_Extractor has extracted a frame, THE Frame_Extractor SHALL isolate each facial region occupying at least 5% of the frame area as a separate facial region.
3. WHERE an extracted frame contains at least one isolated facial region, THE Visual_Model SHALL produce a per-frame visual deepfake likelihood value in the inclusive range 0.0 to 1.0 for each isolated facial region within 2 seconds of receiving that frame.
4. IF the Frame_Extractor isolates no facial region across all extracted frames in an Analysis_Session, THEN THE Platform SHALL record the Analysis_Session as having no visual signal, return an indication to the caller that no visual signal was detected, and continue with audio analysis.
5. IF frame extraction fails for an Analysis_Session before any frame is produced, THEN THE Platform SHALL record the Analysis_Session as having no visual signal, return an error indication identifying the extraction failure, and continue with audio analysis.
6. WHEN per-frame visual analysis completes for all extracted frames in an Analysis_Session, THE Visual_Model SHALL produce a single aggregate visual likelihood value in the inclusive range 0.0 to 1.0.

### Requirement 3: Audio Track Analysis (FR-03, Must-Have)

**User Story:** As a user, I want the Platform to analyze the audio of my video, so that synthetic voice anomalies can be detected.

#### Acceptance Criteria

1. WHEN an Analysis_Session begins and the video contains an audio track, THE Audio_Extractor SHALL extract the audio track into a decoded audio stream.
2. WHEN the Audio_Extractor completes extraction of the audio stream, THE Audio_Extractor SHALL generate one or more spectrogram representations from the extracted audio stream.
3. WHEN spectrogram generation completes, THE Audio_Model SHALL produce an audio deepfake likelihood value in the inclusive range 0.0 to 1.0.
4. IF the video contains no audio track, THEN THE Platform SHALL record the Analysis_Session as having no audio signal and continue with visual analysis.
5. IF audio extraction or spectrogram generation fails, THEN THE Audio_Extractor SHALL record an audio analysis error indicating the failure, retain the existing Analysis_Session data, and continue with visual analysis.
6. IF the Audio_Model fails to produce a likelihood value, THEN THE Platform SHALL record an audio analysis error indicating inference failure, retain the existing Analysis_Session data, and continue with visual analysis.

### Requirement 4: Multi-Modal Fusion (FR-04, Must-Have)

**User Story:** As a user, I want the Platform to combine visual and audio findings, so that I receive a single trustworthy classification.

#### Acceptance Criteria

1. WHEN both the visual likelihood value and the audio likelihood value are available for an Analysis_Session, THE Fusion_Engine SHALL combine both values into a single Classification_Score in the inclusive range 0.0 to 1.0, such that the same pair of input likelihood values always yields an identical Classification_Score.
2. WHERE only one modality produced a likelihood value, THE Fusion_Engine SHALL derive the Classification_Score in the inclusive range 0.0 to 1.0 from the available modality and record whether the visual or the audio modality was used.
3. WHEN the Fusion_Engine produces a Classification_Score, THE Fusion_Engine SHALL assign a classification label of "authentic" when the Classification_Score is below the configured decision threshold (a value in the inclusive range 0.0 to 1.0) and "deepfake" when the Classification_Score is at or above the configured decision threshold.
4. IF neither modality produced a likelihood value for an Analysis_Session, THEN THE Fusion_Engine SHALL produce no Classification_Score, mark the Analysis_Session as inconclusive, and record that no modality signal was available.

### Requirement 5: Real-Time Result Streaming (FR-05, Must-Have)

**User Story:** As a user, I want to see analysis results as they are produced, so that I do not have to wait for the entire process before getting feedback.

#### Acceptance Criteria

1. WHEN a user opens the Dashboard for an active Analysis_Session, THE Result_Streamer SHALL establish a WebSocket connection for that Analysis_Session within 3 seconds.
2. IF the WebSocket connection cannot be established within 3 seconds, THEN THE Result_Streamer SHALL retry establishment up to 3 times and present an error indication to the Dashboard when all retries are exhausted.
3. WHILE an Analysis_Session is processing, THE Result_Streamer SHALL transmit incremental progress updates, expressed as a completion percentage from 0 to 100, to the connected Dashboard at intervals not exceeding 2 seconds.
4. WHEN the Fusion_Engine produces the Classification_Score for an Analysis_Session, THE Result_Streamer SHALL transmit the Classification_Score and classification label to the connected Dashboard within 1 second of the score being produced.
5. IF the WebSocket connection is interrupted while an Analysis_Session is processing, THEN THE Result_Streamer SHALL allow the Dashboard to reconnect within 30 seconds and resume receiving all updates produced during the interruption for the same Analysis_Session without loss.

### Requirement 6: Asynchronous Processing and Scalability (NFR-03, Must-Have)

**User Story:** As a platform operator, I want video analysis to run independently of web requests, so that the server remains responsive and does not time out.

#### Acceptance Criteria

1. WHEN the Upload_Service accepts a video input, THE Platform SHALL enqueue the analysis as a job on the Task_Queue and return an acknowledgment response to the user within 2 seconds, without waiting for inference to complete.
2. IF enqueuing a job onto the Task_Queue fails, THEN THE Platform SHALL reject the upload, return an error status to the user indicating the job could not be queued, and SHALL NOT mark the Analysis_Session as in-progress.
3. WHILE one or more analysis jobs are queued, THE Task_Queue SHALL dispatch each queued job to an available worker within 5 seconds of a worker becoming available, in first-in-first-out order.
4. THE Platform SHALL process Analysis_Sessions concurrently up to the configured worker capacity (a value between 1 and 64 concurrent workers).
5. WHILE the number of in-progress Analysis_Sessions equals the configured worker capacity, THE Platform SHALL retain additional submitted jobs in the Task_Queue in first-in-first-out order until a worker becomes available, without dropping or rejecting them.
6. IF an analysis job fails during processing, THEN THE Platform SHALL retry the job up to 3 times, and upon exhausting all retries SHALL record the failure, preserve the Analysis_Session record without partial or corrupt results, and transmit an error status indicating the analysis failed to the connected Dashboard within 5 seconds.

### Requirement 7: Near Real-Time Performance (NFR-01)

**User Story:** As a user, I want analysis feedback quickly, so that the experience feels responsive.

#### Acceptance Criteria

1. WHEN the Upload_Service accepts a video input, THE Result_Streamer SHALL transmit the first progress update to the Dashboard within 5 seconds of acceptance.
2. WHILE an Analysis_Session is processing, THE Result_Streamer SHALL transmit a progress update to the Dashboard at intervals not exceeding 2 seconds between consecutive updates.
3. WHEN an Analysis_Session completes, THE Result_Streamer SHALL transmit the final result to the Dashboard within 2 seconds of completion.
4. IF the Result_Streamer cannot transmit a progress update to the Dashboard within 2 seconds of the scheduled interval, THEN THE Result_Streamer SHALL retry transmission up to 3 times and display an error indication on the Dashboard signaling that updates are delayed, while retaining the Analysis_Session state for resumed streaming.

### Requirement 8: Model Accuracy (NFR-02)

**User Story:** As a stakeholder, I want the detection model to meet a defined accuracy baseline, so that classifications are reliable.

#### Acceptance Criteria

1. WHEN the multi-modal model is evaluated against the held-out test split of the FaceForensics++ benchmark dataset, THE Platform SHALL achieve a classification accuracy of at least 85%.
2. WHEN model evaluation completes, THE Platform SHALL record the confusion matrix, overall accuracy, precision, recall, and F1-score for the evaluation run.
3. WHEN model evaluation completes, THE Platform SHALL persist the recorded evaluation metrics such that they remain retrievable for subsequent review.
4. IF the measured classification accuracy is below 85% upon completion of an evaluation run, THEN THE Platform SHALL flag the model as not meeting the accuracy baseline and provide an indication to the requesting user identifying the measured accuracy and the failed threshold.

### Requirement 9: GDPR Media Purge (FR-09, Won't-Store)

**User Story:** As a user, I want my video to be deleted after analysis, so that my media is not retained and my privacy is protected.

#### Acceptance Criteria

1. WHEN an Analysis_Session completes, THE Media_Purger SHALL delete the uploaded video and all intermediate media artifacts derived from that video within 60 seconds of session completion.
2. IF an Analysis_Session fails or is canceled, THEN THE Media_Purger SHALL delete the uploaded video and all intermediate media artifacts derived from that video within 60 seconds of the failure or cancellation event.
3. THE Platform SHALL retain only non-media analysis metadata, namely the Classification_Score, classification label, and Analysis_Session timestamps, after media purge.
4. WHEN a media purge operation completes, THE Media_Purger SHALL verify that zero media artifacts derived from that Analysis_Session remain in storage and record a purge-completion status indicating success.
5. IF a media purge operation does not remove all media artifacts, THEN THE Media_Purger SHALL retry the purge up to 3 times, and IF media artifacts still remain after the final retry, THEN THE Media_Purger SHALL record a purge-failure status indicating which Analysis_Session was affected and raise an alert to the operator.

### Requirement 10: External Video URL Input (FR-06, Should-Have)

**User Story:** As a user, I want to submit a video by URL, so that I can analyze online content without downloading it first.

#### Acceptance Criteria

1. WHERE external URL input is enabled, WHEN a user submits an external video URL that uses the HTTP or HTTPS scheme and references a Supported_Format video of 50MB or less, THE Upload_Service SHALL retrieve the video and create an Analysis_Session.
2. WHEN the Upload_Service creates an Analysis_Session from an external video URL, THE Upload_Service SHALL return the Analysis_Session identifier to the user.
3. IF the submitted URL does not use the HTTP or HTTPS scheme, THEN THE Upload_Service SHALL reject the input and return an error message identifying the accepted URL schemes.
4. IF the submitted URL cannot be reached within 30 seconds or returns no retrievable content, THEN THE Upload_Service SHALL reject the input and return an error message indicating that the video could not be retrieved from the URL.
5. IF the retrieved content is not in a Supported_Format, THEN THE Upload_Service SHALL reject the input and return a message identifying the supported formats.
6. IF a retrieved video exceeds the 50MB size limit, THEN THE Upload_Service SHALL halt further retrieval, reject the input, and return a message stating the 50MB size limit.

### Requirement 11: Grad-CAM Explainability Heatmaps (FR-07, Should-Have)

**User Story:** As a user, I want to see which areas of the video influenced the classification, so that I can understand and trust the result.

#### Acceptance Criteria

1. WHERE heatmap generation is enabled, WHEN the Visual_Model classifies a facial frame, THE XAI_Generator SHALL produce a Grad-CAM heatmap overlaying that frame within 2 seconds of the classification, applying a normalized activation intensity scale from 0.0 (no influence) to 1.0 (maximum influence) across all pixels of the frame.
2. WHERE heatmap generation is enabled, IF the XAI_Generator fails to produce a Grad-CAM heatmap for a classified frame, THEN THE XAI_Generator SHALL retain the original classification result, omit the heatmap overlay for that frame, and emit an error indication identifying the affected frame.
3. WHEN a Grad-CAM heatmap is produced, THE Result_Streamer SHALL transmit the heatmap overlay to the connected Dashboard for display within 1 second of receiving the heatmap.
4. IF the Dashboard is not connected when a Grad-CAM heatmap is produced, THEN THE Result_Streamer SHALL discard the heatmap overlay without blocking transmission of subsequent results and emit an error indication that the heatmap could not be delivered.

### Requirement 12: Downloadable PDF Report (FR-08, Could-Have)

**User Story:** As a user, I want to download a report of the analysis, so that I can save and share the findings.

#### Acceptance Criteria

1. WHERE report generation is enabled, WHEN a user requests a report for a completed Analysis_Session, THE Report_Generator SHALL produce a single PDF document containing the Classification_Score, the classification label, and the per-modality findings for every modality analyzed in that Analysis_Session, and SHALL complete generation within 10 seconds.
2. WHERE report generation is enabled, WHEN the PDF document for a completed Analysis_Session has been produced, THE Report_Generator SHALL make the PDF document available to the requesting user for download.
3. WHERE Grad-CAM heatmaps were generated for an Analysis_Session, WHEN the Report_Generator produces the PDF document, THE Report_Generator SHALL include every generated heatmap image in the PDF document.
4. IF a user requests a report for an Analysis_Session that is not complete, THEN THE Report_Generator SHALL NOT produce a PDF document and SHALL return a message indicating that the report is available only after analysis completes.
5. IF report generation fails after a valid request for a completed Analysis_Session, THEN THE Report_Generator SHALL NOT deliver a partial PDF document and SHALL return a message indicating that report generation failed, leaving the Analysis_Session data unchanged.
