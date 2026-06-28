import os
import json
from datetime import datetime


def get_all_videos(input_folder: str) -> list[str]:

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: '{input_folder}'")

    return [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if f.lower().endswith(".mp4")
    ]


def get_video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def make_output_folder(base: str, video_name: str) -> str:
    folder = os.path.join(base, video_name)
    os.makedirs(folder, exist_ok=True)
    return folder


# ── STEP-LEVEL CHECKPOINTING ──────────────────────────────────

STATUS_FILE = "status.json"

# All pipeline steps in order.
# Add new step keys here when you build them.

PIPELINE_STEPS = [
    "step1_extract",
    "step2_yolo",
    "step3_tracking",       
    "step4_vision",     
    "step5_chromadb",    
    "step6_alerts"     

]


def _read_status(output_folder: str) -> dict:
    path = os.path.join(output_folder, STATUS_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _write_status(output_folder: str, status: dict) -> None:
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, STATUS_FILE)
    with open(path, "w") as f:
        json.dump(status, f, indent=2)


def is_step_done(output_folder: str, step: str) -> bool:
    status = _read_status(output_folder)
    return status.get("steps", {}).get(step, False)


def mark_step_done(output_folder: str, step: str, summary: dict = {}) -> None:
    status = _read_status(output_folder)
    if "steps" not in status:
        status["steps"] = {}
    if "summaries" not in status:
        status["summaries"] = {}
    status["steps"][step]     = True
    status["summaries"][step] = {
        "completed_at": datetime.now().isoformat(),
        **summary
    }
    _write_status(output_folder, status)


def get_processing_status(output_folder: str) -> dict | None:
    status = _read_status(output_folder)
    return status if status else None


def clear_step(output_folder: str, step: str) -> None:
    status = _read_status(output_folder)
    if "steps" in status:
        status["steps"][step] = False
    _write_status(output_folder, status)


def clear_all_steps(output_folder: str) -> None:
    path = os.path.join(output_folder, STATUS_FILE)
    if os.path.exists(path):
        os.remove(path)
 