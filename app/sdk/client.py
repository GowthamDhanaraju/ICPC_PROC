import time
import requests
from typing import Optional, Dict, Any

class ProctoringClient:
    """
    A thin Python SDK wrapper for the Batch Video Proctoring Pipeline API.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def submit_session(
        self,
        candidate_id: str,
        video_s3_uri: str,
        enrollment_photo_s3_uri: Optional[str] = None,
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submits a video proctoring session for batch processing.
        """
        url = f"{self.base_url}/v1/sessions"
        payload = {
            "candidate_id": candidate_id,
            "video_s3_uri": video_s3_uri,
            "enrollment_photo_s3_uri": enrollment_photo_s3_uri,
            "webhook_url": webhook_url
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_session_status(self, job_id: str) -> Dict[str, Any]:
        """
        Fetches the status and results of a proctoring job.
        """
        url = f"{self.base_url}/v1/sessions/{job_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def poll_session_until_complete(
        self,
        job_id: str,
        interval: float = 1.0,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Polls the status of a proctoring job until it reaches COMPLETED or FAILED state.
        Raises TimeoutError if processing exceeds timeout threshold.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_session_status(job_id)
            status = result.get("status")
            
            if status in ("COMPLETED", "FAILED"):
                return result
                
            time.sleep(interval)
            
        raise TimeoutError(f"Job {job_id} processing timed out after {timeout} seconds.")
