import os
import sys
import argparse
import time
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
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw target details
        ts = i / fps
        cv2.putText(frame, f"Time: {ts:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Batch Proctoring E2E Demo Video", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Simulate some visual movement to check adaptive sampling (scene motion)
        # Shift a drawn white rectangle across the screen
        x_pos = int((ts * 30) % (width - 100))
        cv2.rectangle(frame, (x_pos, 200), (x_pos + 100, 300), (255, 0, 0), -1)
        
        out.write(frame)
        
    out.release()
    print(f"Successfully generated a {duration_sec}s dummy MP4 video at: {path}")

def run_api_server():
    """Runs the FastAPI server using Uvicorn."""
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="info")

def run_sqs_worker():
    """Runs a standalone worker listening to SQS events."""
    from app.orchestration.queue import SQSListener, global_queue
    
    init_db()
    # Start internal queue workers
    global_queue.start(num_workers=2)
    
    # Configure SQS listener if queue URL is set
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
    1. Generates a dummy MP4 video.
    2. Spins up FastAPI server on a background thread.
    3. Uses ProctoringClient SDK to submit a job.
    4. Polls job status until completion.
    5. Displays score and timeline violations.
    6. Shuts down and exits.
    """
    print("=== STARTING END-TO-END PROCTORING PIPELINE TEST ===")
    
    # Ensure database is initialized
    init_db()
    
    # Define file paths
    video_key = "test_candidate/exam_video.mp4"
    local_video_path = os.path.join(settings.LOCAL_STORAGE_DIR, settings.SOURCE_S3_BUCKET, video_key)
    
    # Generate dummy video
    generate_dummy_video(local_video_path, duration_sec=120.0, fps=10.0) # 120s video covers all mock violation ranges
    
    # Spin up server in background thread
    server_thread = threading.Thread(target=run_api_server, daemon=True)
    server_thread.start()
    
    # Wait for server to boot
    time.sleep(3.0)
    
    # Initialize client and submit job
    client = ProctoringClient("http://127.0.0.1:8000")
    
    s3_uri = f"s3://{settings.SOURCE_S3_BUCKET}/{video_key}"
    webhook_url = "http://127.0.0.1:8000/test/webhook-target"
    
    print(f"Submitting job via Client SDK for {s3_uri}...")
    res = client.submit_session(
        candidate_id="candidate_123",
        video_s3_uri=s3_uri,
        webhook_url=webhook_url
    )
    job_id = res["job_id"]
    print(f"Job submitted! Job ID: {job_id}")
    
    # Poll status
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
                
        # Query received webhooks on the test endpoint to verify webhook delivery
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
        
    print("\n=== E2E TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proctoring Pipeline Control CLI")
    parser.add_argument("mode", choices=["api", "worker", "test-e2e"], help="Execution mode")
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api_server()
    elif args.mode == "worker":
        run_sqs_worker()
    elif args.mode == "test-e2e":
        run_e2e_test()
