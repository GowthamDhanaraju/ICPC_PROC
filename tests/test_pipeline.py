"""
Test suite for the Batch Video Proctoring Pipeline.

Test isolation:
- Uses a separate SQLite test database (test_proctoring.db).
- The DB is initialized before each module and cleaned up after.
- TESTING_MODE=True (set in pytest.ini env) ensures /test/* routes are mounted.
"""
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Override DATABASE_URL before any app imports touch the engine
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_proctoring.db")
os.environ.setdefault("MOCK_ML_MODELS", "True")
os.environ.setdefault("TESTING_MODE", "True")

from app.config import settings
from app.database import init_db, Base, engine, SessionLocal, Job, Violation
from app.preprocessing.media import calculate_motion_score, find_nearest_frame
from app.scoring.aggregator import EventAggregator
from app.scoring.scorer import ScoringEngine
from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Create all tables before the test module runs, drop them after."""
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)
    # Dispose engine connections before attempting to delete the file (required on Windows)
    engine.dispose()
    db_path = settings.DATABASE_URL.replace("sqlite:///", "").replace("./", "")
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            pass  # Windows may still hold a handle; file will be cleaned on next run


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def api_client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

def test_config():
    """Settings loaded successfully from environment."""
    assert settings.DATABASE_URL is not None
    assert settings.ADAPTIVE_SAMPLING_BASELINE_INTERVAL == 1.6
    assert settings.GAZE_AWAY_YAW_THRESHOLD == 20.0
    assert settings.MOCK_ML_MODELS is True


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

def test_motion_score_identical():
    """Identical frames should produce zero motion."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert calculate_motion_score(img, img) == 0.0


def test_motion_score_different():
    """Black vs white frame should produce ~1.0 motion score."""
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    white = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert pytest.approx(calculate_motion_score(black, white), abs=0.05) == 1.0


def test_find_nearest_frame_bisect():
    """find_nearest_frame should use binary search to find the nearest timestamp."""
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    frames = [(0.0, dummy), (1.6, dummy), (3.2, dummy), (4.8, dummy)]
    result = find_nearest_frame(frames, 3.0)
    assert result is not None  # Should return frame at 3.2 (nearest)

    result_end = find_nearest_frame(frames, 10.0)
    assert result_end is not None  # Should return last frame

    result_empty = find_nearest_frame([], 5.0)
    assert result_empty is None


# ---------------------------------------------------------------------------
# Aggregator Tests
# ---------------------------------------------------------------------------

def test_aggregator_debounce_removes_brief_event():
    """Single-frame violations below min_duration should be filtered out."""
    agg = EventAggregator()
    frame_detections = [{
        "timestamp": 10.0,
        "faces": [
            {"box": [0, 0, 10, 10], "confidence": 0.9},
            {"box": [20, 20, 30, 30], "confidence": 0.8},
        ],
        "gaze": {"yaw": 0.0, "pitch": 0.0},
        "gadgets": [],
    }]
    violations = agg.aggregate(frame_detections, [])
    assert len(violations) == 0


def test_aggregator_keeps_sustained_event():
    """Events spanning >= min_duration should be preserved as violations."""
    agg = EventAggregator()
    frame_detections = [
        {"timestamp": t, "faces": [
            {"box": [0, 0, 10, 10], "confidence": 0.9},
            {"box": [20, 20, 30, 30], "confidence": 0.8},
        ]}
        for t in [10.0, 11.0, 12.0]
    ]
    violations = agg.aggregate(frame_detections, [])
    assert len(violations) == 1
    assert violations[0]["type"] == "MULTIPLE_FACES"
    assert violations[0]["start_ts"] == 10.0
    assert violations[0]["end_ts"] == 12.0
    assert pytest.approx(violations[0]["confidence"]) == 0.85


# ---------------------------------------------------------------------------
# Scoring Tests
# ---------------------------------------------------------------------------

def test_scorer_no_violations():
    assert ScoringEngine().calculate_score([]) == 100.0


def test_scorer_count_based_penalty():
    scorer = ScoringEngine()
    v1 = [{"type": "MULTIPLE_FACES", "duration": 2.0, "confidence": 0.9}]
    assert scorer.calculate_score(v1) == 75.0  # 100 - 25

    v2 = v1 * 2
    assert scorer.calculate_score(v2) == 62.5  # 100 - 25 - 12.5


def test_scorer_duration_based_penalty():
    scorer = ScoringEngine()
    # sqrt(5/5) = 1.0; penalty = 10 * 1.0 = 10
    v3 = [{"type": "GAZE_AWAY_SUSTAINED", "duration": 5.0, "confidence": 0.8}]
    assert scorer.calculate_score(v3) == 90.0

    # sqrt(20/5) = 2.0; penalty = 10 * 2.0 = 20
    v4 = [{"type": "GAZE_AWAY_SUSTAINED", "duration": 20.0, "confidence": 0.8}]
    assert scorer.calculate_score(v4) == 80.0


# ---------------------------------------------------------------------------
# API Tests
# ---------------------------------------------------------------------------

def test_health_endpoint(api_client):
    """Health probe returns 200 with status=ok."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_ready_endpoint(api_client):
    """Readiness probe returns 200 when DB is reachable."""
    resp = api_client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["database"] == "ok"


def test_submit_and_poll_session(api_client, db_session):
    """Submitted session is persisted in DB and queryable."""
    payload = {
        "candidate_id": "pytest_candidate",
        "video_s3_uri": "s3://proctoring-incoming/pytest/session.mp4",
        "webhook_url": "http://localhost:8000/test/webhook-target",
    }
    resp = api_client.post("/v1/sessions", json=payload)
    assert resp.status_code == 201

    data = resp.json()
    assert "job_id" in data
    job_id = data["job_id"]

    status_resp = api_client.get(f"/v1/sessions/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["candidate_id"] == "pytest_candidate"
    assert status_data["status"] == "QUEUED"

    # Idempotency: re-submitting the same job should return the existing one
    resp2 = api_client.post("/v1/sessions", json=payload)
    assert resp2.status_code == 201
    assert resp2.json()["job_id"] == job_id

    # Cleanup
    job = db_session.query(Job).filter(Job.id == job_id).first()
    if job:
        db_session.delete(job)
        db_session.commit()


def test_get_nonexistent_session(api_client):
    """Querying a non-existent job returns 404."""
    resp = api_client.get("/v1/sessions/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
