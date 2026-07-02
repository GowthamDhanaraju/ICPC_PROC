import os
from app.config import settings

def transcribe_segment(wav_path: str, start: float, end: float) -> str:
    """
    Transcribes a specific audio segment (start to end seconds) using Whisper.
    Only run on segments flagged with diarization / multi-speaker to conserve compute.
    """
    if settings.MOCK_ML_MODELS or not wav_path or not os.path.exists(wav_path):
        # Mock transcription text
        return "the answer to question five is option B"

    # Real whisper model transcription logic
    # In production:
    # import whisper
    # model = whisper.load_model("base")
    # ...
    
    return "[Transcription placeholder]"
