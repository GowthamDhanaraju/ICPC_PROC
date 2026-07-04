# Batch Video Proctoring Pipeline (S3-sourced)

A robust, high-throughput, S3-event-driven batch video proctoring pipeline. This application processes finished exam video recordings to output a highly structured JSON observation log containing timestamped anomalies, audio slice clips, and a 3D-visualized video overlay.

---

## 1. High-Level Architecture

```
                    ┌─────────────────────────────┐
  Host app writes   │  MinIO / S3 (incoming/)     │
  video directly ──▶│  (client's bucket, or yours)│
                    └──────────────┬───────────────┘
                                   │ S3 Event Notification
                                   ▼
                    ┌─────────────────────────────┐
                    │   SQS Queue (job messages)  │
                    └──────────────┬───────────────┘
                                   ▼
                     ┌────────────────────────────────────┐
                     │        Worker Pool (autoscaled)    │
                     │  ┌───────────────┐  ┌─────────────┐│
                     │  │  Preprocessor │  │ Preprocessor││
                     │  │ ffmpeg (video)│  │ audio split ││
                     │  └───────┬───────┘  └──────┬──────┘│
                     │          ▼                 ▼       │
                     │  ┌───────────────┐  ┌─────────────┐│
                     │  │ Visual detect │  │ Audio detect││
                     │  │ (batched GPU) │  │  (CPU/GPU)  ││
                     │  └───────┬───────┘  └──────┬──────┘│
                     │          └────────┬────────┘       │
                     │                   ▼                │
                     │     Observation Report Builder     │
                     │       (JSON + MP4 Overlay)         │
                     └───────────────────┬────────────────┘
                                         ▼
                    ┌────────────────────────────────────┐
                    │ MinIO / S3 (results/ prefix)       │
                    │ - `_result.json` (Full Data)       │
                    │ - `_result.mp4` (Visuals)          │
                    │ - `_secondary_voice.wav` (Audio)   │
                    └────────────────────┬───────────────┘
                                         ▼
                         Webhook → Host App with JSON URL
```

### Key Processing Stages

1. **Ingestion & Idempotency**: Jobs are triggered either by an **S3 Event Notification** landing in an SQS Queue, or explicitly via a REST API. 
2. **Preprocessing**: 
   - Demuxes the video file into a WAV audio track (16kHz mono).
   - Video frames are extracted via **Adaptive Sampling**.
   - Extracted frames are downscaled to 640px long edge for fast visual inference.
3. **Detection Modules**:
   - **Face count**: YOLOv8 face model counts faces (0 = absence, 2+ = collusion).
   - **Identity verification**: Cosine similarity check between ArcFace embeddings of detected face and enrollment photo.
   - **Gaze / Head pose**: 6DRepNet calculates yaw/pitch rotation and tracks exact seconds spent looking Straight, Left, and Right.
   - **Gadget detection**: YOLOv8 detects prohibited items (phones, laptops, books).
   - **Audio VAD**: Silero VAD detects human speech vs background noise.
   - **Speaker diarization**: Resemblyzer speaker diarization flags multiple distinct voices.
4. **Data Extraction & Delivery**:
   - **Gaze 3D Overlay**: Renders a 3D-projected red arrow directly out of the candidate's face on the output MP4 to easily visualize their exact head pose.
   - **Audio Slicing**: Uses `ffmpeg` to precisely slice out any flagged secondary voices as standalone `.wav` clips for auditing.
   - **JSON Reporting**: Aggregates all anomalies into a structured JSON observation log (`_result.json`), uploads it to MinIO, and fires a Webhook to the host app.

---

## 2. Directory Layout

```
.
├── app/
│   ├── __init__.py
│   ├── config.py              # Settings & threshold configs
│   ├── database.py            # SQLite/Postgres DB models (SQLAlchemy)
│   ├── main.py                # FastAPI endpoints & worker start/stop
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── queue.py           # In-memory queues & SQS listeners
│   │   └── worker.py          # Main job pipeline orchestration
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── media.py           # Video frame sampling, Audio Slicing, MinIO I/O
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── face.py            # Face detection & identity matching
│   │   ├── gaze.py            # Gaze/head pose tracking (6DRepNet)
│   │   ├── gadget.py          # Prohibited item checks (YOLO)
│   │   ├── audio_vad.py       # Speech/silence detection
│   │   └── diarization.py     # Second voice presence (Resemblyzer)
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── report.py          # JSON log builder & time aggregation
│   │   └── overlay.py         # 3D Gaze line and bounding box video renderer
│   └── sdk/
│       ├── __init__.py
│       └── client.py          # SDK wrapper client
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # Pipeline & API unit tests
├── requirements.txt           # Python dependency specifications
├── run.py                     # CLI helper runner
└── README.md
```

---

## 3. Setup & Installation

### Prerequisite Libraries

Make sure Python (3.9+) and `ffmpeg` are installed on the host system.

```bash
# Clone the repository and install requirements
pip install -r requirements.txt
```

*Note: For Windows/macOS/Linux environments without system-level FFmpeg installed, the python package `imageio-ffmpeg` is included in `requirements.txt` and will automatically provide precompiled static FFmpeg binaries.*

### Configuration (.env)

Create a `.env` file in the root directory to customize parameters (defaults are stored in `app/config.py`):

```env
DATABASE_URL=sqlite:///./proctoring.db
MOCK_ML_MODELS=true # Set to false to run actual PyTorch/ONNX models
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=your_key
MINIO_SECRET_KEY=your_secret
MINIO_BUCKET_NAME=proctortest
```

---

## 4. Run Scripts

The project provides a unified entry point, `run.py`, to easily run the system.

### A. Run API Server (Including Background Workers)
This launches the FastAPI server and starts background worker threads to automatically pick up submitted sessions.
```bash
python run.py api
```
Access Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### B. Run Standalone SQS Worker Fleet
Launches the worker queue processing engine and subscribes directly to SQS messages (triggered by MinIO/S3 events).
```bash
python run.py worker
```

---

## 5. Thin Client SDK Usage

Host applications can integrate the proctoring engine with a few lines of code:

```python
from app.sdk.client import ProctoringClient

# Initialize client
client = ProctoringClient(base_url="http://localhost:8000")

# Submit a job
response = client.submit_session(
    candidate_id="candidate_abc_123",
    video_s3_uri="s3://proctortest/exam_123.mp4",
    enrollment_photo_s3_uri="s3://proctortest/enrollments/candidate_abc_123.jpg",
    webhook_url="https://hostapp.com/api/proctoring-webhook"
)

job_id = response["job_id"]
print(f"Submitted proctoring job. ID: {job_id}")

# Wait for completion (optional polling)
result = client.poll_session_until_complete(job_id, interval=2.0, timeout=180.0)
print(f"Job Status: {result['status']}")
print(f"Final Report Location: {result['report_path']}")
```
