"""
utils/job_tracker.py
─────────────────────
Tracks progress of background pipeline jobs.

Why this exists:
    Pipeline is slow — YOLO + Claude + embeddings per frame.
    We can't block the HTTP response for 5+ minutes.

    So instead:
        POST /api/process
            → creates job, returns job_id IMMEDIATELY
            → pipeline runs in background thread
            → client can continue doing other things

        GET /api/job/{job_id}
            → client polls this to check progress
            → returns current step, % complete, results

Flow:
    create_job()     → called when POST /api/process received
         │
         ▼
    set_step()       → called inside pipeline as each step runs
         │              updates current_video, current_step, progress
         ▼
    finish_job()     → called when all videos complete
         │
         ▼
    get_job()        → called by GET /api/job/{id} to return status

Storage:
    In-memory Python dict — fine for portfolio project.
    In production: use Redis so jobs survive server restarts.

Progress percentages per step:
    step1_extract    →  5%
    step2_yolo       → 20%
    step3_tracking   → 40%
    step4_vision     → 55%
    step5_embeddings → 70%
    step6_storage    → 85%
    done             → 100%
"""

import uuid
from datetime import datetime
from typing import Optional


# ── STORAGE ───────────────────────────────────────────────────
# In-memory store: { job_id: job_dict }
# Resets when server restarts — acceptable for this project
_jobs: dict[str, dict] = {}


# ── PROGRESS MAP ──────────────────────────────────────────────
# Maps step name → progress percentage
# Used to show meaningful progress to the client
# IMPORTANT: these keys must exactly match what pipeline passes to set_step()
STEP_PROGRESS = {
    "step1_extract":    5,
    "step2_yolo":      20,
    "step3_tracking":  40,
    "step4_vision":    55,
    "step5_embeddings":70,
    "step6_storage":   85,   # ← pipeline must call set_step(..., "step6_storage", ...)
    "done":           100
}


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────

def create_job(video_names: list[str]) -> str:
    """
    Create a new job entry and return its ID.

    Called by POST /api/process before starting background thread.

    Args:
        video_names: list of video names being processed
                     e.g. ["video1", "video2", "video3"]

    Returns:
        job_id: short unique ID e.g. "a1b2c3d4"

    Example response to client:
        {
          "job_id": "a1b2c3d4",
          "status": "queued",
          "message": "Poll /api/job/a1b2c3d4 for progress"
        }
    """

    # Short UUID — 8 chars is enough for a portfolio project
    job_id = str(uuid.uuid4())[:8]

    _jobs[job_id] = {
        "job_id":        job_id,
        "status":        "queued",      # queued → running → done/failed
        "videos":        video_names,   # all videos in this job
        "current_video": None,          # which video is processing now
        "current_step":  None,          # which step is running now
        "progress_pct":  0,             # 0 → 100
        "started_at":    datetime.now().isoformat(),
        "finished_at":   None,
        "error":         None,          # set if job fails
        "results":       []             # filled when job completes
    }

    print(f"  ✅ Job created: {job_id} | Videos: {video_names}")
    return job_id


def set_step(
    job_id:       str,
    video_name:   str,
    step:         str,
    progress_pct: int
) -> None:
    """
    Update current step and progress percentage.

    Called INSIDE the pipeline as each step starts.
    Client polls GET /api/job/{id} to see this update.

    Args:
        job_id:       job to update
        video_name:   which video is currently processing
        step:         step key e.g. "step2_yolo"
        progress_pct: 0-100 percentage complete

    Example:
        set_step("a1b2c3d4", "video1", "step3_tracking", 40)
    """

    if job_id not in _jobs:
        # Log clearly — silent failure is very hard to debug
        print(f"  ⚠️  set_step called with unknown job_id: '{job_id}'")
        print(f"      Known jobs: {list(_jobs.keys())}")
        return

    _jobs[job_id].update({
        "status":        "running",
        "current_video": video_name,
        "current_step":  step,
        "progress_pct":  progress_pct
    })

    # Log every step update — helps debug pipeline flow
    print(f"  📊 Job {job_id} | {video_name} | {step} | {progress_pct}%")


def finish_job(job_id: str, results: list) -> None:
    """
    Mark job as successfully completed.

    Called after ALL videos have been processed.

    Args:
        job_id:  job to mark complete
        results: list of VideoResult dicts from pipeline
    """

    if job_id not in _jobs:
        print(f"  ⚠️  finish_job called with unknown job_id: '{job_id}'")
        return

    _jobs[job_id].update({
        "status":        "done",
        "progress_pct":  100,
        "current_step":  None,
        "current_video": None,
        "finished_at":   datetime.now().isoformat(),
        "results":       results
    })

    print(f"  ✅ Job {job_id} finished successfully | Results: {len(results)} videos")


def fail_job(job_id: str, error: str) -> None:
    """
    Mark job as failed with error message.

    Called if any unhandled exception occurs in the pipeline.

    Args:
        job_id: job to mark failed
        error:  error message string
    """

    if job_id not in _jobs:
        print(f"  ⚠️  fail_job called with unknown job_id: '{job_id}'")
        return

    _jobs[job_id].update({
        "status":        "failed",
        "progress_pct":  _jobs[job_id].get("progress_pct", 0),  # keep last known %
        "error":         error,
        "finished_at":   datetime.now().isoformat()
    })

    print(f"  ❌ Job {job_id} failed: {error}")


def get_job(job_id: str) -> Optional[dict]:
    """
    Return job dict or None if not found.

    Called by GET /api/job/{job_id} endpoint.

    Returns full job dict:
        {
          "job_id":        "a1b2c3d4",
          "status":        "running",
          "current_video": "video1",
          "current_step":  "step3_tracking",
          "progress_pct":  40,
          ...
        }
    """
    job = _jobs.get(job_id)

    if job is None:
        print(f"  ⚠️  get_job: job_id '{job_id}' not found")
        print(f"      Known jobs: {list(_jobs.keys())}")

    return job


def get_all_jobs() -> list[dict]:
    """
    Return all jobs sorted newest first.

    Called by GET /api/jobs endpoint.
    Useful for monitoring all pipeline runs.
    """
    return sorted(
        _jobs.values(),
        key=lambda j: j["started_at"],
        reverse=True
    )
