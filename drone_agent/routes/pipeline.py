"""
routes/pipeline.py
───────────────────
Full pipeline — all 7 steps wired together with:

- Background tasks  → user gets job_id instantly, no waiting
- Checkpointing     → only new/changed steps rerun
- Job tracking      → poll /api/job/{id} for live progress

Steps:
    1. Extract frames     (OpenCV)
    2. Object detection   (YOLOv8)
    3. Object tracking    (ByteTrack)
    4. Frame description  (Claude Haiku via Bedrock)
    5. Embeddings         (OpenCLIP)
    6. Store              (ChromaDB + MongoDB)
    7. Alert agent        (LangGraph + Claude Sonnet)
"""

import os
import json
from fastapi  import APIRouter, BackgroundTasks
from pydantic import BaseModel

from agents.extract_frames    import extract_frames
from agents.object_detection  import run_object_detection
from agents.object_tracking   import run_object_tracking
from agents.frame_description import run_frame_description
from agents.embedding         import run_embeddings
from agents.security_agent    import run_alert_agent
from database.chroma_store    import store_embeddings
from database.mongo_store     import store_video, store_frames

from utils.file_helpers import (
    get_all_videos, get_video_name, make_output_folder,
    is_step_done, mark_step_done,
    get_processing_status, clear_step, clear_all_steps
)

from utils.job_tracker import (
    create_job, set_step, finish_job, fail_job,
    get_job, get_all_jobs,
    STEP_PROGRESS   # ← import so pipeline and tracker always in sync
)

INPUT_FOLDER    = "input_folder"
OUTPUT_BASE     = "output"
EVERY_NTH_FRAME = 6
TOTAL_STEPS     = 7

# ── ALL STEPS ─────────────────────────────────────────────────
# Single source of truth for step order
# Used for: checkpointing, redo-step clearing, status endpoint
# MUST match keys in STEP_PROGRESS in job_tracker.py
ALL_STEPS = [
    "step1_extract",
    "step2_yolo",
    "step3_tracking",
    "step4_vision",
    "step5_embeddings",
    "step6_storage",    # ← fixed: was "step6_store", must match job_tracker.py
    "step7_alerts",
]

router = APIRouter()


# ── RESPONSE MODEL ────────────────────────────────────────────

class ProcessResponse(BaseModel):
    job_id:  str
    status:  str
    message: str
    videos:  list[str]


# ── PIPELINE ──────────────────────────────────────────────────

def run_full_pipeline(job_id: str, videos: list[str]) -> None:
    """
    Runs all 7 steps for every video in sequence.
    Called by FastAPI BackgroundTasks — never directly by the user.

    Progress reporting:
        - set_step() called BEFORE step starts  → shows "starting X"
        - set_step() called AFTER step completes → shows correct % 
        - finish_job() called at very end        → sets 100%

    Checkpointing logic per step:
        - is_step_done() → True  → load saved result, skip heavy work
        - is_step_done() → False → run step, save result, mark done
    """
    try:
        all_results = []

        for video_path in videos:
            name          = get_video_name(video_path)
            output_folder = make_output_folder(
                os.path.join(OUTPUT_BASE, "extracted_frames"), name
            )

            print(f"\n{'─'*50}")
            print(f"  Processing: {name}")
            print(f"{'─'*50}")

            # Intermediate results passed between steps
            frames       = []
            detections   = []
            tracks       = []
            descriptions = []
            embedded     = []

            # ── STEP 1: Extract frames ─────────────────────────
            # Report BEFORE step so UI shows "extracting..."
            set_step(job_id, name, "step1_extract", STEP_PROGRESS["step1_extract"])

            if is_step_done(output_folder, "step1_extract"):
                print(f"  [{name}] Step 1 — skipped (cached)")
                path = os.path.join(output_folder, "frames_metadata.json")
                if os.path.exists(path):
                    with open(path) as f:
                        frames = json.load(f)
                print(f"  [{name}] Step 1 — loaded {len(frames)} frames from cache")
            else:
                print(f"  [{name}] Step 1 — extracting frames...")
                frames = extract_frames(video_path, output_folder, EVERY_NTH_FRAME)
                mark_step_done(output_folder, "step1_extract", {
                    "frames_saved": len(frames)
                })
                print(f"  [{name}] Step 1 ✅ — {len(frames)} frames extracted")

            # ── STEP 2: YOLO detection ─────────────────────────
            set_step(job_id, name, "step2_yolo", STEP_PROGRESS["step2_yolo"])

            if is_step_done(output_folder, "step2_yolo"):
                print(f"  [{name}] Step 2 — skipped (cached)")
                path = os.path.join(output_folder, "detections.json")
                if os.path.exists(path):
                    with open(path) as f:
                        detections = json.load(f)
                print(f"  [{name}] Step 2 — loaded {len(detections)} detections from cache")
            else:
                print(f"  [{name}] Step 2 — running YOLO detection...")
                detections = run_object_detection(frames)
                labels     = list({
                    lbl
                    for fr in detections
                    for lbl in fr.get("objects_found", [])
                })
                mark_step_done(output_folder, "step2_yolo", {
                    "total_detections": sum(f["total_objects"] for f in detections),
                    "objects_found":    labels
                })
                print(f"  [{name}] Step 2 ✅ — {len(detections)} frames processed, labels: {labels}")

            # ── STEP 3: ByteTrack tracking ─────────────────────
            set_step(job_id, name, "step3_tracking", STEP_PROGRESS["step3_tracking"])

            if is_step_done(output_folder, "step3_tracking"):
                print(f"  [{name}] Step 3 — skipped (cached)")
                path = os.path.join(output_folder, "tracking.json")
                if os.path.exists(path):
                    with open(path) as f:
                        tracks = json.load(f)
                print(f"  [{name}] Step 3 — loaded tracks from cache")
            else:
                print(f"  [{name}] Step 3 — running ByteTrack...")
                tracks  = run_object_tracking(detections)
                all_ids = {
                    tid
                    for fr in tracks
                    for tid in fr.get("unique_track_ids", [])
                }
                mark_step_done(output_folder, "step3_tracking", {
                    "unique_tracks": len(all_ids)
                })
                print(f"  [{name}] Step 3 ✅ — {len(all_ids)} unique objects tracked")

            # ── STEP 4: Claude Vision via Bedrock ──────────────
            set_step(job_id, name, "step4_vision", STEP_PROGRESS["step4_vision"])

            if is_step_done(output_folder, "step4_vision"):
                print(f"  [{name}] Step 4 — skipped (cached)")
                path = os.path.join(output_folder, "descriptions.json")
                if os.path.exists(path):
                    with open(path) as f:
                        descriptions = json.load(f)
                print(f"  [{name}] Step 4 — loaded {len(descriptions)} descriptions from cache")
            else:
                print(f"  [{name}] Step 4 — describing frames with Claude (Bedrock)...")
                descriptions = run_frame_description(frames, tracks)
                mark_step_done(output_folder, "step4_vision", {
                    "frames_described": len(descriptions)
                })
                print(f"  [{name}] Step 4 ✅ — {len(descriptions)} frames described")

            # ── STEP 5: OpenCLIP embeddings ────────────────────
            # Note: embeddings are NOT saved to disk
            # (512 floats × N frames × videos = too large)
            # We always regenerate them in memory — takes ~seconds not minutes
            set_step(job_id, name, "step5_embeddings", STEP_PROGRESS["step5_embeddings"])

            if is_step_done(output_folder, "step5_embeddings"):
                # ← FIXED: was running embeddings in BOTH branches
                # Still need to generate in memory for step 6
                print(f"  [{name}] Step 5 — regenerating embeddings in memory (not stored on disk)")
                embedded = run_embeddings(descriptions)
                print(f"  [{name}] Step 5 — {len(embedded)} embeddings ready (from cache path)")
            else:
                print(f"  [{name}] Step 5 — generating embeddings...")
                embedded = run_embeddings(descriptions)
                mark_step_done(output_folder, "step5_embeddings", {
                    "frames_embedded": len(embedded)
                })
                print(f"  [{name}] Step 5 ✅ — {len(embedded)} embeddings generated")

            # ── STEP 6: Store in ChromaDB + MongoDB ────────────
            # ← FIXED: was "step6_store", now "step6_storage" to match job_tracker.py
            set_step(job_id, name, "step6_storage", STEP_PROGRESS["step6_storage"])

            if is_step_done(output_folder, "step6_storage"):
                print(f"  [{name}] Step 6 — skipped (cached)")
            else:
                print(f"  [{name}] Step 6 — storing in ChromaDB + MongoDB...")
                store_embeddings(name, embedded)
                store_video(name, video_path, len(frames))
                store_frames(name, embedded)
                mark_step_done(output_folder, "step6_storage", {
                    "frames_stored": len(embedded)
                })
                print(f"  [{name}] Step 6 ✅ — {len(embedded)} frames stored")

            # ── STEP 7: LangGraph Alert Agent ──────────────────
            set_step(job_id, name, "step7_alerts", STEP_PROGRESS.get("step7_alerts", 92))

            if is_step_done(output_folder, "step7_alerts"):
                print(f"  [{name}] Step 7 — skipped (cached)")
            else:
                print(f"  [{name}] Step 7 — running LangGraph alert agent...")
                alerts = run_alert_agent(name)
                mark_step_done(output_folder, "step7_alerts", {
                    "alerts_generated": len(alerts)
                })
                print(f"  [{name}] Step 7 ✅ — {len(alerts)} alerts generated")

            # ── Collect results summary ────────────────────────
            # Read back from saved status file — single source of truth
            saved     = get_processing_status(output_folder) or {}
            summaries = saved.get("summaries", {})

            all_results.append({
                "video_name":       name,
                "frames_saved":     summaries.get("step1_extract",    {}).get("frames_saved",     0),
                "total_detections": summaries.get("step2_yolo",       {}).get("total_detections", 0),
                "unique_tracks":    summaries.get("step3_tracking",   {}).get("unique_tracks",    0),
                "frames_described": summaries.get("step4_vision",     {}).get("frames_described", 0),
                "frames_stored":    summaries.get("step6_storage",    {}).get("frames_stored",    0),
                "alerts_generated": summaries.get("step7_alerts",     {}).get("alerts_generated", 0),
            })

            print(f"\n  [{name}] All steps complete ✅")

        # Mark entire job as done — sets progress to 100%
        finish_job(job_id, all_results)
        print(f"\n{'═'*50}")
        print(f"  Job {job_id} complete ✅ — {len(all_results)} video(s) processed")
        print(f"{'═'*50}\n")

    except Exception as e:
        import traceback
        fail_job(job_id, str(e))
        print(f"\n{'═'*50}")
        print(f"  Job {job_id} FAILED ❌")
        print(f"  Error: {e}")
        print(f"  Traceback:\n{traceback.format_exc()}")
        print(f"{'═'*50}\n")
        raise


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@router.post("/process", response_model=ProcessResponse)
def start_pipeline(background_tasks: BackgroundTasks):
    """
    Start the full pipeline in the background.
    Returns job_id immediately — no waiting.

    Then poll:  GET /api/job/{job_id}
    """
    videos = get_all_videos(INPUT_FOLDER)

    if not videos:
        return ProcessResponse(
            job_id  = "none",
            status  = "error",
            message = "No .mp4 files found in input_folder/",
            videos  = []
        )

    video_names = [get_video_name(v) for v in videos]
    job_id      = create_job(video_names)

    background_tasks.add_task(run_full_pipeline, job_id, videos)

    response = ProcessResponse(
        job_id  = job_id,
        status  = "started",
        message = f"Pipeline started. Poll /api/job/{job_id} for progress.",
        videos  = video_names
    )

    print(f"\n  🚀 Pipeline started | job_id={job_id} | videos={video_names}")
    return response


@router.get("/job/{job_id}")
def get_job_status(job_id: str):
    """
    Live progress of a running or completed job.

    Called by Streamlit every 3 seconds while pipeline runs.

    Response while running:
        {
          "status":        "running",
          "current_video": "video1",
          "current_step":  "step4_vision",
          "progress_pct":  55
        }

    Response when done:
        {
          "status":      "done",
          "progress_pct": 100,
          "results":     [...]
        }
    """
    job = get_job(job_id)

    # ← FIXED: was returning all jobs instead of raising 404
    if not job:
        from fastapi import HTTPException
        raise HTTPException(
            status_code = 404,
            detail      = f"Job '{job_id}' not found. Server may have restarted."
        )

    return job


@router.get("/jobs")
def list_jobs():
    """
    All jobs sorted newest first.

    Useful for monitoring / debugging all pipeline runs.
    """
    # ← FIXED: was on /job/{id} route, conflicting with get_job_status
    return {"jobs": get_all_jobs()}


@router.get("/status")
def get_status():
    """
    Step-by-step completion status for every video in input_folder.

    Used by Streamlit Pipeline page to show the status grid.
    """
    videos = get_all_videos(INPUT_FOLDER) if os.path.exists(INPUT_FOLDER) else []
    result = []

    for vp in videos:
        name          = get_video_name(vp)
        output_folder = os.path.join(OUTPUT_BASE, "extracted_frames", name)
        saved         = get_processing_status(output_folder)
        steps_done    = saved.get("steps", {}) if saved else {}

        result.append({
            "video": name,
            "steps": {s: steps_done.get(s, False) for s in ALL_STEPS}
        })

    return {"videos": result}


@router.post("/reprocess/{video_name}")
def reprocess_video(video_name: str, background_tasks: BackgroundTasks):
    """
    Force rerun ALL 7 steps for one video.
    Clears all cached step results first.
    """
    output_folder = os.path.join(OUTPUT_BASE, "extracted_frames", video_name)
    clear_all_steps(output_folder)

    video_path = os.path.join(INPUT_FOLDER, f"{video_name}.mp4")
    if not os.path.exists(video_path):
        return {"error": f"{video_name}.mp4 not found in input_folder/"}

    job_id = create_job([video_name])
    background_tasks.add_task(run_full_pipeline, job_id, [video_path])

    print(f"  🔄 Reprocessing all steps for {video_name} | job_id={job_id}")
    return {
        "job_id":  job_id,
        "message": f"Reprocessing all 7 steps for {video_name}"
    }


@router.post("/redo-step/{video_name}/{step}")
def redo_step(video_name: str, step: str, background_tasks: BackgroundTasks):
    """
    Redo one step AND all steps after it for one video.

    Example — changed YOLO confidence threshold:
        POST /api/redo-step/video1/step2_yolo
        → Step 1 stays cached
        → Steps 2 through 7 rerun

    Example — regenerate alerts only:
        POST /api/redo-step/video1/step7_alerts
        → Steps 1-6 stay cached
        → Only Step 7 reruns
    """
    output_folder = os.path.join(OUTPUT_BASE, "extracted_frames", video_name)

    if not os.path.exists(output_folder):
        return {"error": f"No output folder for '{video_name}'. Run /process first."}

    if step not in ALL_STEPS:
        return {
            "error":        f"Unknown step '{step}'.",
            "valid_steps":  ALL_STEPS
        }

    # Clear this step and everything after it
    idx            = ALL_STEPS.index(step)
    steps_cleared  = ALL_STEPS[idx:]

    for s in steps_cleared:
        clear_step(output_folder, s)

    video_path = os.path.join(INPUT_FOLDER, f"{video_name}.mp4")
    if not os.path.exists(video_path):
        return {"error": f"{video_name}.mp4 not found in input_folder/"}

    job_id = create_job([video_name])
    background_tasks.add_task(run_full_pipeline, job_id, [video_path])

    print(f"  ↩️  Redo from {step} for {video_name} | job_id={job_id} | clearing: {steps_cleared}")
    return {
        "job_id":        job_id,
        "message":       f"Rerunning from {step} onwards for {video_name}",
        "steps_cleared": steps_cleared
    }