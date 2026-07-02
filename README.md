# Batch Video Proctoring Pipeline (S3-sourced)

A robust, high-throughput, S3-event-driven batch video proctoring pipeline. This application processes finished exam video recordings to output a timestamped list of proctoring violations and calculate an overall candidate fairness score.

---

## 1. High-Level Architecture

```
                    ┌─────────────────────────────┐
  Host app writes   │  S3 bucket (incoming/ prefix)│
  video directly ──▶│  (client's bucket, or yours) │
                    └──────────────┬───────────────┘
                                   │ S3 Event Notification
                                   ▼
                    ┌─────────────────────────────┐
                    │   SQS Queue (job messages)   │
                    └──────────────┬───────────────┘
                                   ▼
                     ┌────────────────────────────────────┐
                     │        Worker Pool (autoscaled)      │
                     │  ┌───────────────┐  ┌─────────────┐ │
                     │  │  Preprocessor │  │  Preprocessor│ │
                     │  │ ffmpeg (range/ │  │  audio split │ │
                     │  │  streamed read)│  │             │ │
                     │  └───────┬───────┘  └──────┬──────┘ │
                     │          ▼                  ▼        │
                     │  ┌───────────────┐  ┌─────────────┐ │
                     │  │ Visual detect │  │ Audio detect │ │
                     │  │ (batched GPU) │  │  (CPU/GPU)   │ │
                     │  └───────┬───────┘  └──────┬──────┘ │
                     │          └────────┬─────────┘        │
                     │                   ▼                  │
                     │       Event Aggregator + Scorer       │
                     └────────────────────┬───────────────────┘
                                          ▼
                    ┌────────────────────────────────────┐
                    │  Results Store (Postgres / SQLite)   │
                    │  + Evidence frames → S3 results/ prefix│
                    └────────────────────┬───────────────────┘
                                         ▼
                         Webhook → Host App   (+ optional poll API)
```

### Key Processing Stages

1. **Ingestion & Idempotency**: Jobs are triggered either by an **S3 Event Notification** landing in an SQS Queue, or explicitly via a REST API. Idempotency is enforced on the database layer via unique source video URIs to prevent double-processing.
2. **Preprocessing**: 
   - Demuxes the video file into a WAV audio track (16kHz mono).
   - Video frames are extracted via **Adaptive Sampling** (sampling densifies down to 0.4s during high motion/scene change, and sparsifies up to 4.0s during low motion).
   - Extracted frames are downscaled to 640px long edge for fast visual inference.
3. **Detection Modules**:
   - **Face count**: YOLOv8 face model counts faces (0 = absence, 2+ = collusion).
   - **Identity verification**: Cosine similarity check between ArcFace embeddings of detected face and enrollment photo.
   - **Gaze / Head pose**: MediaPipe face landmarker calculates yaw/pitch rotation.
   - **Gadget detection**: YOLOv8 detects prohibited items (phones, laptops, books, etc.).
   - **Audio VAD**: Silero VAD detects human speech vs background noise.
   - **Speaker diarization**: Pyannote speaker diarization flags multiple voices in speaker segments.
   - **Whisper transcript**: Transcribes flagged diarized segments for suspicious keywords.
4. **Aggregation & Debounce**:
   - Smooths raw detections. A flag must persist for a minimum duration (debounce) to count as a violation.
   - Merges consecutive violations of the same type within a specific time window.
5. **Scoring Engine**:
   - Computes a candidate fairness score starting from 100.
   - Subtracts weighted severity penalties with a **diminishing marginal penalty** (geometric decay for count-based flags, sub-linear square root scaling for duration-based flags).
6. **Delivery**: Writes results to database, uploads violation evidence frames, and delivers webhook notifications to the host app with exponential backoff.

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
│   │   └── media.py           # Video frame sampling & audio split
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── face.py            # Face detection & identity matching
│   │   ├── gaze.py            # Gaze/head pose tracking
│   │   ├── gadget.py          # Prohibited item checks
│   │   ├── audio_vad.py       # Speech/silence detection
│   │   ├── diarization.py     # Second voice presence
│   │   └── whisper_transcription.py
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── aggregator.py      # Timeline debounce & merge rules
│   │   └── scorer.py          # Weighted severity scoring engine
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

Create a `.env` file in the root directory to customize parameters (defaults are stored in [app/config.py](file:///d:/ICPC_PROC/app/config.py)):

```env
DATABASE_URL=sqlite:///./proctoring.db
MOCK_ML_MODELS=true # Set to false to run actual PyTorch/ONNX models
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/proctoring-queue
```

---

## 4. Run Scripts

The project provides a unified entry point, [run.py](file:///d:/ICPC_PROC/run.py), to easily run the system.

### A. Run API Server (Including Background Workers)
This launches the FastAPI server and starts background worker threads to automatically pick up submitted sessions.
```bash
python run.py api
```
Access Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### B. Run Standalone SQS Worker Fleet
Launches the worker queue processing engine and subscribes directly to SQS messages (triggered by S3 bucket events).
```bash
python run.py worker
```

### C. Run Local End-to-End Test Simulation
This generates a mock video, starts the API server in a background thread, submits a job using the Client SDK, processes it through the pipeline, receives the webhook callback on a test receiver endpoint, and displays the score details and timeline violations.
```bash
python run.py test-e2e
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
    video_s3_uri="s3://exam-bucket/recordings/exam_123.mp4",
    enrollment_photo_s3_uri="s3://exam-bucket/enrollments/candidate_abc_123.jpg",
    webhook_url="https://hostapp.com/api/proctoring-webhook"
)

job_id = response["job_id"]
print(f"Submitted proctoring job. ID: {job_id}")

# Wait for completion (optional polling)
result = client.poll_session_until_complete(job_id, interval=2.0, timeout=180.0)
print(f"Fairness score: {result['overall_score']}")
print(f"Violations timeline: {result['violations']}")
```
