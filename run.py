import json
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


def _print_report_summary(report_path: str):
    """
    Reads a saved JSON report and prints a human-readable summary to stdout.
    Exits gracefully if the file cannot be read.
    """
    if not report_path or not os.path.exists(report_path):
        print(f"[WARNING] Report file not found: {report_path}")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    meta = report.get("meta", {})
    summary = report.get("summary", {})
    events = report.get("events", {})

    print("\n" + "=" * 60)
    print(" PIPELINE DETECTION REPORT")
    print("=" * 60)
    print(f"  Job ID        : {meta.get('job_id', 'N/A')}")
    print(f"  Candidate     : {meta.get('candidate_id', 'N/A')}")
    print(f"  Video         : {meta.get('video_source', 'N/A')}")
    print(f"  Duration      : {meta.get('video_duration_s', 'N/A')} s")
    print(f"  Frames sampled: {meta.get('frames_sampled', 'N/A')}")
    print(f"  Generated at  : {meta.get('generated_at', 'N/A')}")

    # --- Face ---
    fs = summary.get("face", {})
    print("\n[ FACE DETECTION ]")
    print(f"  Frames with no face      : {fs.get('frames_with_no_face', 0)}")
    print(f"  Frames with one face     : {fs.get('frames_with_one_face', 0)}")
    print(f"  Frames with multiple faces: {fs.get('frames_with_multiple_faces', 0)}")
    print(f"  Max faces in one frame   : {fs.get('max_faces_in_single_frame', 0)}")

    # --- Gaze ---
    gs = summary.get("gaze", {})
    print("\n[ GAZE / HEAD POSE ]")
    print(f"  Frames with gaze data    : {gs.get('frames_with_gaze_data', 0)}")
    print(f"  Frames looking away      : {gs.get('frames_looking_away', 0)}")

    # --- Gadgets ---
    gd = summary.get("gadget", {})
    print("\n[ GADGET / OBJECT DETECTION ]")
    print(f"  Frames with gadget       : {gd.get('frames_with_gadget_detected', 0)}")
    classes = gd.get('unique_classes_detected', [])
    print(f"  Device classes detected  : {', '.join(classes) if classes else 'None'}")

    # --- Identity ---
    ids = summary.get("identity", {})
    print("\n[ IDENTITY VERIFICATION ]")
    print(f"  Enrollment photo used    : {ids.get('enrollment_photo_provided', False)}")
    print(f"  Frames checked           : {ids.get('frames_checked', 0)}")
    print(f"  Frames flagged (mismatch): {ids.get('frames_flagged_as_mismatch', 0)}")

    # --- Audio ---
    aud = summary.get("audio", {})
    print("\n[ AUDIO ANALYSIS ]")
    if aud.get("available"):
        print(f"  Total speech duration    : {aud.get('total_speech_s', 'N/A')} s")
        print(f"  Distinct speakers        : {aud.get('distinct_speakers', 'N/A')}")
        print(f"  Non-primary segments     : {aud.get('non_primary_segments', 0)}")
    else:
        print("  No audio track in video.")

    # --- Events ---
    print("\n[ DETECTED EVENT WINDOWS ]")
    for etype, edata in events.items():
        windows = edata.get("windows", [])
        if windows:
            print(f"  {etype} ({len(windows)} window(s)):")
            for w in windows:
                print(
                    f"    {w['start_ts']:.1f}s → {w['end_ts']:.1f}s "
                    f"| dur={w['duration_s']:.1f}s "
                    f"| conf={w['avg_confidence']:.2f}"
                )

    print("\n" + "-" * 60)
    print(f"  Full report saved to: {report_path}")
    print("=" * 60 + "\n")

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

def run_e2e_test(output_dir: str = "reports"):
    """
    Executes a complete local end-to-end integration flow:
    1. Generates a dummy MP4 video in a system temp directory.
    2. Spins up FastAPI server on a background thread.
    3. Uses ProctoringClient SDK to submit a job (local abs path as URI).
    4. Polls job status until completion.
    5. Prints report summary and saves JSON report to output_dir.
    """
    print("=== STARTING END-TO-END PROCTORING PIPELINE TEST ===")

    settings.TESTING_MODE = True
    init_db()

    test_temp_dir = tempfile.mkdtemp(prefix="proctoring_e2e_")
    local_video_path = os.path.join(test_temp_dir, "exam_video.mp4")

    generate_dummy_video(local_video_path, duration_sec=120.0, fps=10.0)

    server_thread = threading.Thread(target=run_api_server, daemon=True)
    server_thread.start()
    wait_for_server()

    client = ProctoringClient("http://127.0.0.1:8000")
    webhook_url = "http://127.0.0.1:8000/test/webhook-target"

    print(f"Submitting job via Client SDK for {local_video_path}...")
    res = client.submit_session(
        candidate_id="candidate_123",
        video_s3_uri=local_video_path,
        webhook_url=webhook_url,
    )
    job_id = res["job_id"]
    print(f"Job submitted! Job ID: {job_id}")

    print("Polling job status until execution completes...")
    try:
        final_result = client.poll_session_until_complete(job_id, interval=1.0, timeout=120.0)
        print(f"\nJob Status: {final_result['status']}")

        # Report is saved by the worker; show its path and summary
        report_path = os.path.join(
            os.path.abspath(output_dir), f"{job_id}.json"
        )
        _print_report_summary(report_path)

        import requests as _req
        hooks_res = _req.get("http://127.0.0.1:8000/test/webhook-received")
        if hooks_res.status_code == 200 and len(hooks_res.json()) > 0:
            print("[SUCCESS] Webhook was dispatched and received.")
        else:
            print("[WARNING] Webhook was not received by the test endpoint.")

    except Exception as e:
        print(f"E2E test failed: {e}")
        sys.exit(1)
    finally:
        import shutil
        shutil.rmtree(test_temp_dir, ignore_errors=True)

    print("\n=== E2E TEST COMPLETED SUCCESSFULLY ===")




def run_local_test(video_path: str, candidate_id: str, output_dir: str = "reports"):
    """
    Runs the proctoring pipeline directly on a local video file.
    No temp dir management needed — the file is already on disk.
    The pipeline saves a JSON report to output_dir/<job_id>.json.
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
    print(f"Video     : {video_path} ({file_size_mb:.1f} MB)")
    print(f"Candidate : {candidate_id}")
    print(f"Output dir: {os.path.abspath(output_dir)}")

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
        video_s3_uri=video_path,
        webhook_url=webhook_url,
    )
    job_id = res["job_id"]
    print(f"Job submitted! ID: {job_id}")

    print("Polling until pipeline completes (this may take several minutes for long videos)...")
    try:
        final = client.poll_session_until_complete(job_id, interval=3.0, timeout=7200.0)
        print(f"\nJob Status: {final['status']}")
        report_path = os.path.join(os.path.abspath(output_dir), f"{job_id}.json")
        _print_report_summary(report_path)
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proctoring Pipeline Control CLI")
    parser.add_argument(
        "mode",
        choices=["api", "worker", "test-e2e", "test-local"],
        help="Execution mode",
    )
    parser.add_argument("--file", help="Local video file path (required for test-local)")
    parser.add_argument("--candidate", default="test_candidate", help="Candidate ID")
    parser.add_argument(
        "--output",
        default="reports",
        help="Directory where the JSON report will be saved (default: ./reports)",
    )
    args = parser.parse_args()

    if args.mode == "api":
        run_api_server(host="0.0.0.0")
    elif args.mode == "worker":
        run_sqs_worker()
    elif args.mode == "test-e2e":
        run_e2e_test(output_dir=args.output)
    elif args.mode == "test-local":
        run_local_test(args.file, args.candidate, output_dir=args.output)
