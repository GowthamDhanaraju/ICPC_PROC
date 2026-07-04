import os
import sys
import argparse
import time
import tempfile
import threading
import cv2
import numpy as np
import uvicorn
from app.config import settings
from app.database import init_db
from app.sdk.client import ProctoringClient

def generate_dummy_video(path: str, duration_sec: float = 15.0, fps: float = 10.0):
    """
    Programmatically creates a dummy MP4 video using OpenCV.
    Ensures that preprocessing can read frames and calculate motion correctly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = 640, 480
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        ts = i / fps
        cv2.putText(frame, f"Time: {ts:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Batch Proctoring E2E Demo Video", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        x_pos = int((ts * 30) % (width - 100))
        cv2.rectangle(frame, (x_pos, 200), (x_pos + 100, 300), (255, 0, 0), -1)
        out.write(frame)
        
    out.release()
    print(f"Successfully generated a {duration_sec}s dummy MP4 video at: {path}")

def run_api_server(host: str = "127.0.0.1"):
    """Runs the FastAPI server using Uvicorn."""
    uvicorn.run("app.main:app", host=host, port=8000, log_level="info")

def wait_for_server(url: str = "http://127.0.0.1:8000/health", timeout: float = 60.0):
    """
    Waits for the background FastAPI server to open port 8000 and boot up.
    """
    import requests
    print("Waiting for API server to boot up...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code == 200:
                print(f"API server ready in {time.time() - start_time:.1f}s.")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    print("Error: API server failed to boot within timeout limit.")
    sys.exit(1)

def run_sqs_worker():
    """Runs a standalone worker listening to SQS events."""
    from app.orchestration.queue import SQSListener, global_queue
    
    init_db()
    global_queue.start(num_workers=2)
    
    sqs_url = os.getenv("SQS_QUEUE_URL")
    if not sqs_url:
        print("Error: SQS_QUEUE_URL environment variable is not set. Standalone worker exiting.")
        sys.exit(1)
        
    listener = SQSListener(sqs_url)
    listener.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping SQS worker...")
        listener.stop()
        global_queue.stop()

def run_e2e_test():
    """
    Executes a complete local end-to-end integration flow:
    1. Generates a dummy MP4 video in a system temp directory.
    2. Spins up FastAPI server on a background thread.
    3. Uses ProctoringClient SDK to submit a job (local abs path as URI).
    4. Polls job status until completion.
    5. Displays score and timeline violations.
    """
    print("=== STARTING END-TO-END PROCTORING PIPELINE TEST ===")

    settings.TESTING_MODE = True
    init_db()
    
    # Write the test video to a system temp dir (not ./storage/)
    test_temp_dir = tempfile.mkdtemp(prefix="proctoring_e2e_")
    local_video_path = os.path.join(test_temp_dir, "exam_video.mp4")
    
    generate_dummy_video(local_video_path, duration_sec=120.0, fps=10.0)
    
    server_thread = threading.Thread(target=run_api_server, daemon=True)
    server_thread.start()
    wait_for_server()
    
    client = ProctoringClient("http://127.0.0.1:8000")
    
    # Submit the absolute local path directly — get_local_path() returns it as-is
    # because os.path.exists() is True.  No MinIO download needed for tests.
    webhook_url = "http://127.0.0.1:8000/test/webhook-target"
    
    print(f"Submitting job via Client SDK for {local_video_path}...")
    res = client.submit_session(
        candidate_id="candidate_123",
        video_s3_uri=local_video_path,   # absolute path, passes schema validator
        webhook_url=webhook_url
    )
    job_id = res["job_id"]
    print(f"Job submitted! Job ID: {job_id}")
    
    print("Polling job status until execution completes...")
    try:
        final_result = client.poll_session_until_complete(job_id, interval=1.0, timeout=120.0)
        print("\n=== PIPELINE EXECUTION COMPLETED ===")
        print(f"Job Status: {final_result['status']}")
        print(f"Overall Fairness Score: {final_result['overall_score']}/100")
        print("\nDetected Violations Timeline:")
        for v in final_result["violations"]:
            print(f" - {v['type']}: {v['start_ts']} -> {v['end_ts']} ({v['duration']:.1f}s, Conf: {v['confidence']:.2f})")
            if v['evidence_frame_s3_uri']:
                print(f"   Evidence S3 URI: {v['evidence_frame_s3_uri']}")
                
        import requests
        hooks_res = requests.get("http://127.0.0.1:8000/test/webhook-received")
        if hooks_res.status_code == 200 and len(hooks_res.json()) > 0:
            print("\n[SUCCESS] Webhook was dispatched and received successfully!")
            print(f"Webhook Payload: {hooks_res.json()[-1]}")
        else:
            print("\n[WARNING] Webhook was not received by the test endpoint.")
            
    except Exception as e:
        print(f"E2E test failed with error: {e}")
        sys.exit(1)
    finally:
        import shutil
        shutil.rmtree(test_temp_dir, ignore_errors=True)
        
    print("\n=== E2E TEST COMPLETED SUCCESSFULLY ===")

import re
import requests

def parse_gdrive_id(url: str) -> str:
    """Parses Google Drive share URL and extracts the file ID."""
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    if re.match(r'^[a-zA-Z0-9_-]{25,}$', url):
        return url
    raise ValueError(f"Could not extract Google Drive File ID from: {url}")

def download_gdrive_file(file_id: str, dest_path: str):
    """Downloads a publicly accessible file from Google Drive using gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Connecting and downloading file ID: {file_id} via gdown...")
    gdown.download(url, dest_path, quiet=False, use_cookies=False)

def run_gdrive_test(url: str, candidate_id: str):
    """Downloads a Google Drive video and runs the local proctoring pipeline on it."""
    if not url:
        print("Error: --url parameter is required when running test-gdrive mode.")
        sys.exit(1)
        
    try:
        file_id = parse_gdrive_id(url)
    except Exception as e:
        print(f"URL Parsing Error: {e}")
        sys.exit(1)

    # Download to a system temp dir
    test_temp_dir = tempfile.mkdtemp(prefix=f"proctoring_gdrive_{file_id[:8]}_")
    local_video_path = os.path.join(test_temp_dir, "video.mp4")
    
    if not os.path.exists(local_video_path):
        try:
            download_gdrive_file(file_id, local_video_path)
        except Exception as e:
            print(f"Failed to download Google Drive video: {e}")
            sys.exit(1)
    else:
        print(f"Google Drive video already at: {local_video_path}")
        
    settings.TESTING_MODE = True
    init_db()
    
    server_thread = threading.Thread(target=run_api_server, daemon=True)
    server_thread.start()
    wait_for_server()
    
    client = ProctoringClient("http://127.0.0.1:8000")
    webhook_url = "http://127.0.0.1:8000/test/webhook-target"
    
    print(f"Submitting job via Client SDK for {local_video_path}...")
    res = client.submit_session(
        candidate_id=candidate_id,
        video_s3_uri=local_video_path,
        webhook_url=webhook_url
    )
    job_id = res["job_id"]
    print(f"Job submitted! Job ID: {job_id}")
    
    print("Polling until pipeline completes...")
    try:
        final_result = client.poll_session_until_complete(job_id, interval=2.0, timeout=600.0)
        print("\n=== PIPELINE EXECUTION COMPLETED ===")
        print(f"Job Status: {final_result['status']}")
        print(f"Overall Fairness Score: {final_result['overall_score']}/100")
        print("\nDetected Violations Timeline:")
        for v in final_result["violations"]:
            print(f" - {v['type']}: {v['start_ts']} -> {v['end_ts']} ({v['duration']:.1f}s, Conf: {v['confidence']:.2f})")
            if v['evidence_frame_s3_uri']:
                print(f"   Evidence S3 URI: {v['evidence_frame_s3_uri']}")
    except Exception as e:
        print(f"Gdrive test run failed: {e}")
        sys.exit(1)
    finally:
        import shutil
        shutil.rmtree(test_temp_dir, ignore_errors=True)

def run_local_test(video_path: str, candidate_id: str):
    """
    Runs the proctoring pipeline directly on a local video file.
    No temp dir management needed — the file is already on disk.
    """
    if not video_path:
        print("Error: --file parameter is required for test-local mode.")
        sys.exit(1)

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"=== LOCAL FILE PROCTORING TEST ===")
    print(f"Video: {video_path} ({file_size_mb:.1f} MB)")
    print(f"Candidate: {candidate_id}")

    settings.TESTING_MODE = True
    init_db()

    server_thread = threading.Thread(target=run_api_server, daemon=True)
    server_thread.start()
    wait_for_server()

    client = ProctoringClient("http://127.0.0.1:8000")
    webhook_url = "http://127.0.0.1:8000/test/webhook-target"

    print(f"Submitting job for {video_path}...")
    res = client.submit_session(
        candidate_id=candidate_id,
        video_s3_uri=video_path,   # absolute path passed directly
        webhook_url=webhook_url
    )
    job_id = res["job_id"]
    print(f"Job submitted! ID: {job_id}")

    print("Polling until pipeline completes (this may take several minutes for long videos)...")
    try:
        final = client.poll_session_until_complete(job_id, interval=3.0, timeout=7200.0)
        print("\n=== PIPELINE COMPLETED ===")
        print(f"Job Status : {final['status']}")
        print(f"Fairness Score: {final['overall_score']}/100")
        print("\nViolation Timeline:")
        for v in final["violations"]:
            print(f"  [{v['type']}] {v['start_ts']} → {v['end_ts']} "
                  f"({v['duration']:.1f}s, conf={v['confidence']:.2f})")
            if v.get('evidence_frame_s3_uri'):
                print(f"    Evidence: {v['evidence_frame_s3_uri']}")
        if not final["violations"]:
            print("  No violations detected.")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proctoring Pipeline Control CLI")
    parser.add_argument("mode", choices=["api", "worker", "test-e2e", "test-gdrive", "test-local"],
                        help="Execution mode")
    parser.add_argument("--url", help="Google Drive shared file URL (required for test-gdrive)")
    parser.add_argument("--file", help="Local video file path (required for test-local)")
    parser.add_argument("--candidate", default="test_candidate", help="Candidate ID")
    args = parser.parse_args()

    if args.mode == "api":
        run_api_server(host="0.0.0.0")
    elif args.mode == "worker":
        run_sqs_worker()
    elif args.mode == "test-e2e":
        run_e2e_test()
    elif args.mode == "test-gdrive":
        run_gdrive_test(args.url, args.candidate)
    elif args.mode == "test-local":
        run_local_test(args.file, args.candidate)
