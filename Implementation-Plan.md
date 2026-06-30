# Project Requirements: Real-Time Deepfake Detection in Online Video Content

## 1. Project Overview
A web-based platform that utilizes multi-modal deep learning (audio and visual analysis) to detect deepfake videos in near real-time. The system integrates Explainable AI (XAI) to provide users with transparent reasoning for its classifications via visual heatmaps and feature highlighting.

## 2. Technology Stack
* **Frontend:** React.js, TailwindCSS, Chart.js/D3.js (for real-time graphs)
* **Backend:** Python, Django, Django REST Framework, Django Channels (WebSockets)
* **Task Queue:** Celery, Redis (for async video processing)
* **Machine Learning:** PyTorch or TensorFlow/Keras
* **Data Processing:** OpenCV, FFmpeg, Librosa, MTCNN/MediaPipe

## 3. Functional Requirements
* **FR-01 (Input):** The system shall allow users to upload video files (MP4, AVI) up to a specified size limit.
* **FR-02 (Input):** The system shall accept external video URLs (e.g., YouTube) for processing.
* **FR-03 (Vision Analysis):** The system shall extract facial frames and analyze them for spatial and temporal deepfake artifacts.
* **FR-04 (Audio Analysis):** The system shall extract the audio track, convert it to spectrograms, and analyze it for synthetic generation artifacts.
* **FR-05 (Multi-Modal Fusion):** The system shall combine vision and audio inferences to generate a final confidence score (0-100% Fake).
* **FR-06 (XAI - Vision):** The system shall generate Grad-CAM heatmaps overlaying the original video frames to show the spatial areas that contributed to the classification.
* **FR-07 (Real-Time Feedback):** The system shall stream analysis results back to the user interface via WebSockets as the video is being processed.
* **FR-08 (Reporting):** The system shall generate a downloadable PDF report summarizing the deepfake analysis and XAI justifications.

## 4. Non-Functional Requirements
* **NFR-01 (Performance/Real-Time):** Frame-by-frame processing latency should not exceed 500ms per frame to maintain a near real-time user experience.
* **NFR-02 (Accuracy):** The underlying multi-modal model should achieve a minimum baseline accuracy of 85% on standard datasets (e.g., FaceForensics++, Deepfake Detection Challenge Dataset).
* **NFR-03 (Scalability):** The backend architecture must decouple HTTP requests from ML inference using message brokers (Celery/Redis) to prevent server timeouts during heavy processing.
* **NFR-04 (Usability):** The XAI outputs must be easily interpretable by a non-technical user (e.g., red areas indicate high likelihood of manipulation).

## 5. Dataset Requirements
* **Training Data:** The models will be trained on established datasets containing both pristine and manipulated (Deepfakes, FaceSwap, Face2Face, NeuralTextures) content.
* **Data Privacy:** All uploaded user videos must be deleted from the server storage immediately after the session concludes to ensure GDPR compliance.