import os
import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.database import init_db, SessionLocal, Job, Violation
from app.preprocessing.media import calculate_motion_score
from app.scoring.aggregator import EventAggregator
from app.scoring.scorer import ScoringEngine
from app.main import app

# Initialize test DB
init_db()

@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_config():
    """Verify settings loaded configuration successfully."""
    assert settings.DATABASE_URL is not None
    assert settings.ADAPTIVE_SAMPLING_BASELINE_INTERVAL == 1.6
    assert settings.GAZE_AWAY_YAW_THRESHOLD == 20.0

def test_motion_calculation():
    """Verify motion calculation computes expected values for same and different images."""
    import numpy as np
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Identical images should have 0 motion score
    score_same = calculate_motion_score(img1, img2)
    assert score_same == 0.0
    
    # Completely different images (black vs white) should have 1.0 motion score
    img3 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    score_diff = calculate_motion_score(img1, img3)
    assert pytest.approx(score_diff, 0.05) == 1.0

def test_aggregator_debounce():
    """Verify event aggregator filters out events below the debounce threshold."""
    agg = EventAggregator()
    
    # We simulate brief visual flags (e.g. 1 frame at 10.0s)
    # The debounce threshold for MULTIPLE_FACES is 1.5 seconds.
    # A single frame flag at 10.0s has 0 duration, so it should be debounced/discarded.
    frame_detections = [
        {
            "timestamp": 10.0,
            "faces": [{"box": [0,0,10,10], "confidence": 0.9}, {"box": [20,20,30,30], "confidence": 0.8}],
            "gaze": {"yaw": 0.0, "pitch": 0.0},
            "gadgets": []
        }
    ]
    
    violations = agg.aggregate(frame_detections, [])
    # Should be empty since it lasted only 0 seconds (1 frame)
    assert len(violations) == 0

def test_aggregator_merge_and_debounce_keep():
    """Verify event aggregator merges and keeps flags that exceed the threshold."""
    agg = EventAggregator()
    
    # Simulate MULTIPLE_FACES active from 10s to 12s (2 seconds duration >= 1.5s threshold)
    frame_detections = [
        {
            "timestamp": 10.0,
            "faces": [{"box": [0,0,10,10], "confidence": 0.9}, {"box": [20,20,30,30], "confidence": 0.8}]
        },
        {
            "timestamp": 11.0,
            "faces": [{"box": [0,0,10,10], "confidence": 0.9}, {"box": [20,20,30,30], "confidence": 0.8}]
        },
        {
            "timestamp": 12.0,
            "faces": [{"box": [0,0,10,10], "confidence": 0.9}, {"box": [20,20,30,30], "confidence": 0.8}]
        }
    ]
    
    violations = agg.aggregate(frame_detections, [])
    assert len(violations) == 1
    assert violations[0]["type"] == "MULTIPLE_FACES"
    assert violations[0]["start_ts"] == 10.0
    assert violations[0]["end_ts"] == 12.0
    assert violations[0]["duration"] == 2.0
    assert pytest.approx(violations[0]["confidence"]) == 0.85

def test_scoring_engine_diminishing_marginal_penalty():
    """Verify that scoring applies penalties and scales down marginal repeats geometrically/sub-linearly."""
    scorer = ScoringEngine()
    
    # 1. Base test: 0 violations = 100 score
    assert scorer.calculate_score([]) == 100.0
    
    # 2. Count-based: 1 MULTIPLE_FACES event (base penalty is 25.0)
    v1 = [{"type": "MULTIPLE_FACES", "duration": 2.0, "confidence": 0.9}]
    assert scorer.calculate_score(v1) == 75.0  # 100 - 25 = 75
    
    # 3. Repeat count-based: 2 MULTIPLE_FACES events (should decay geometrically: 25 + 12.5 = 37.5)
    v2 = [
        {"type": "MULTIPLE_FACES", "duration": 2.0, "confidence": 0.9},
        {"type": "MULTIPLE_FACES", "duration": 2.0, "confidence": 0.9}
    ]
    assert scorer.calculate_score(v2) == 62.5  # 100 - 37.5 = 62.5
    
    # 4. Duration-based: GAZE_AWAY_SUSTAINED (base is 10.0)
    # For duration of 5.0 seconds, scaling is sqrt(5 / 5) = 1.0. Penalty should be exactly 10.0.
    v3 = [{"type": "GAZE_AWAY_SUSTAINED", "duration": 5.0, "confidence": 0.8}]
    assert scorer.calculate_score(v3) == 90.0  # 100 - 10 = 90
    
    # 5. Long duration scaling: 20 seconds. sqrt(20 / 5) = 2.0. Penalty is 10.0 * 2 = 20.0
    v4 = [{"type": "GAZE_AWAY_SUSTAINED", "duration": 20.0, "confidence": 0.8}]
    assert scorer.calculate_score(v4) == 80.0  # 100 - 20 = 80

def test_api_submit_and_poll(db_session):
    """Verify submitted session writes to DB and can be queried."""
    client = TestClient(app)
    
    # Submit job
    payload = {
        "candidate_id": "pytest_candidate",
        "video_s3_uri": "s3://proctoring-incoming/pytest_candidate/session.mp4",
        "webhook_url": "http://localhost:8000/test/webhook-target"
    }
    response = client.post("/v1/sessions", json=payload)
    assert response.status_code == 201
    
    data = response.json()
    assert "job_id" in data
    job_id = data["job_id"]
    
    # Query status
    status_response = client.get(f"/v1/sessions/{job_id}")
    assert status_response.status_code == 200
    
    status_data = status_response.json()
    assert status_data["job_id"] == job_id
    assert status_data["candidate_id"] == "pytest_candidate"
    assert status_data["status"] == "QUEUED"
    
    # Clean up test DB record
    job_record = db_session.query(Job).filter(Job.id == job_id).first()
    if job_record:
        db_session.delete(job_record)
        db_session.commit()
