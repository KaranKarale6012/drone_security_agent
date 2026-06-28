"""
streamlit_app.py
─────────────────
Frontend UI for the Drone Security Agent.

Pages:
    🎬 Pipeline  — upload video, run pipeline, track job progress
    🚨 Alerts    — view security alerts from footage
    💬 Chat      — ask questions about footage in plain English

How progress tracking works:
    1. User uploads video → saved to input_folder/
    2. User clicks "Start Pipeline"
    3. POST /api/process → returns job_id immediately
    4. job_id saved to st.session_state
    5. Page polls GET /api/job/{job_id} every 3 seconds
    6. Progress bar updates until status == "done" or "failed"
"""

import os
import shutil
import streamlit as st
import requests
import time

API          = "http://localhost:8000/api"
INPUT_FOLDER = "input_folder"   # must match pipeline.py INPUT_FOLDER


# ── STEP LABELS ───────────────────────────────────────────────
# Maps internal step keys → human readable labels for UI
# Keys must match STEP_PROGRESS keys in job_tracker.py
STEP_LABELS = {
    "step1_extract":    "1️⃣  Extract Frames",
    "step2_yolo":       "2️⃣  YOLO Detection",
    "step3_tracking":   "3️⃣  Object Tracking",
    "step4_vision":     "4️⃣  Vision Analysis (Claude)",
    "step5_embeddings": "5️⃣  Generating Embeddings",
    "step6_storage":    "6️⃣  Storing Results",
    "done":             "✅  Complete",
}

# Maps step key → progress percentage
# Must stay in sync with job_tracker.py STEP_PROGRESS
STEP_PROGRESS = {
    "step1_extract":    5,
    "step2_yolo":      20,
    "step3_tracking":  40,
    "step4_vision":    55,
    "step5_embeddings":70,
    "step6_storage":   85,
    "done":           100,
}


# ── PAGE CONFIG ───────────────────────────────────────────────

st.set_page_config(
    page_title = "Drone Security Agent",
    page_icon  = "🚁",
    layout     = "wide"
)


# ── SIDEBAR ───────────────────────────────────────────────────

st.sidebar.title("🚁 Drone Security Agent")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🎬 Pipeline", "🚨 Alerts", "💬 Chat"]
)

# ── Server status in sidebar ──────────────────────────────────
st.sidebar.divider()
try:
    requests.get(f"{API}/status", timeout=2)
    st.sidebar.success("✅ Server connected")
except Exception:
    st.sidebar.error("❌ Server offline")
    st.sidebar.code("python main.py", language="bash")


# ══════════════════════════════════════════════════════════════
# HELPER — clear input folder
# ══════════════════════════════════════════════════════════════

def clear_input_folder() -> int:
    """
    Delete all files inside input_folder/.
    Creates the folder if it doesn't exist.

    Returns:
        count: number of files deleted

    Why:
        We want one clean video per run.
        Old videos left in input_folder would be reprocessed
        every time the user starts the pipeline.
    """
    # Create folder if it doesn't exist yet
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    deleted = 0
    for filename in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted += 1
                print(f"  🗑️  Deleted: {file_path}")
            elif os.path.isdir(file_path):
                # Remove subdirectories too (shouldn't exist but just in case)
                shutil.rmtree(file_path)
                deleted += 1
                print(f"  🗑️  Deleted dir: {file_path}")
        except Exception as e:
            print(f"  ⚠️  Could not delete {file_path}: {e}")

    print(f"  ✅ Cleared input_folder — {deleted} file(s) removed")
    return deleted


def save_uploaded_video(uploaded_file) -> str:
    """
    Save Streamlit UploadedFile to input_folder/.

    Args:
        uploaded_file: st.file_uploader result object

    Returns:
        saved_path: full path where file was saved
                    e.g. "input_folder/my_video.mp4"

    Why write in chunks:
        Streamlit uploads can be large (drone footage = GB+).
        Writing in 8MB chunks avoids loading entire file into RAM.
    """
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    saved_path = os.path.join(INPUT_FOLDER, uploaded_file.name)

    # Write in chunks — safe for large video files
    with open(saved_path, "wb") as f:
        chunk_size = 8 * 1024 * 1024   # 8 MB chunks
        while True:
            chunk = uploaded_file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
    print(f"  💾 Saved: {saved_path} ({file_size_mb:.1f} MB)")
    return saved_path


# ══════════════════════════════════════════════════════════════
# HELPER — render job progress
# ══════════════════════════════════════════════════════════════

def render_job_progress(job_id: str) -> None:
    """
    Fetch job status and render progress bar + step info.

    Called on every rerun while a job is active.
    Handles three states:
        running  → show progress bar + auto-refresh
        done     → show results + clear button
        failed   → show error + clear button

    Args:
        job_id: job ID stored in st.session_state["job_id"]
    """

    st.subheader(f"Job Progress — `{job_id}`")

    try:
        # ── Fetch current job state ───────────────────────
        resp = requests.get(f"{API}/job/{job_id}", timeout=5)

        # Handle 404 — job not found on server
        if resp.status_code == 404:
            st.error(f"Job `{job_id}` not found on server.")
            st.caption("The server may have restarted (jobs are in-memory).")
            if st.button("🗑️ Clear Job", key="clear_404"):
                del st.session_state["job_id"]
                st.rerun()
            return

        job    = resp.json()
        status = job.get("status", "unknown")
        pct    = job.get("progress_pct", 0)

        # ── Progress bar ──────────────────────────────────
        st.progress(
            value = pct / 100,
            text  = f"{pct}% complete"
        )

        # ── Step timeline ─────────────────────────────────
        current_step = job.get("current_step") or ""
        cols         = st.columns(len(STEP_LABELS) - 1)   # exclude "done"
        step_keys    = [k for k in STEP_LABELS if k != "done"]

        for i, key in enumerate(step_keys):
            step_pct = STEP_PROGRESS.get(key, 0)
            label    = STEP_LABELS[key]

            with cols[i]:
                if pct >= step_pct:
                    st.success(label, icon="✅")
                elif key == current_step:
                    st.warning(label, icon="⏳")
                else:
                    st.info(label, icon="⬜")

        st.divider()

        # ── Status-specific rendering ─────────────────────

        if status == "done":
            # ── DONE ─────────────────────────────────────
            st.success(
                f"✅ Pipeline complete! "
                f"Finished at {job.get('finished_at', '')[:19]}"
            )

            results = job.get("results", [])
            if results:
                st.subheader("📊 Results Summary")
                for r in results:
                    with st.container(border=True):
                        st.markdown(f"### 📹 {r.get('video_name', 'Unknown')}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Frames Saved",     r.get("frames_saved",     0))
                        c2.metric("Total Detections", r.get("total_detections", 0))
                        c3.metric("Unique Tracks",    r.get("unique_tracks",    0))
                        c4.metric("Alerts Generated", r.get("alerts_generated", 0))
            else:
                st.info("Pipeline finished but no results returned.")

            # Time taken
            started  = job.get("started_at",  "")
            finished = job.get("finished_at", "")
            if started and finished:
                try:
                    from datetime import datetime
                    elapsed = (
                        datetime.fromisoformat(finished) -
                        datetime.fromisoformat(started)
                    )
                    minutes = int(elapsed.total_seconds() // 60)
                    seconds = int(elapsed.total_seconds() %  60)
                    st.caption(f"⏱️ Total time: {minutes}m {seconds}s")
                except Exception:
                    pass

            if st.button("🗑️ Clear Job", key="clear_done"):
                del st.session_state["job_id"]
                st.session_state.pop("refresh_count", None)
                st.rerun()

        elif status == "failed":
            # ── FAILED ───────────────────────────────────
            st.error("❌ Pipeline failed!")
            st.code(job.get("error", "No error message"), language="text")
            st.caption("Check your server logs for the full traceback.")

            if st.button("🗑️ Clear Job", key="clear_failed"):
                del st.session_state["job_id"]
                st.session_state.pop("refresh_count", None)
                st.rerun()

        else:
            # ── RUNNING / QUEUED ─────────────────────────
            current_video = job.get("current_video", "...")
            current_step  = job.get("current_step",  "...")
            step_label    = STEP_LABELS.get(current_step, current_step)

            st.info(
                f"⏳ **Status: {status.upper()}**  \n"
                f"📹 **Video:** `{current_video}`  \n"
                f"⚙️ **Step:** `{step_label}`"
            )

            # Refresh counter — reassures user polling is working
            st.session_state.setdefault("refresh_count", 0)
            st.session_state["refresh_count"] += 1

            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.caption(
                    f"🔄 Auto-refreshing every 3s "
                    f"(poll #{st.session_state['refresh_count']})"
                )
            with col_b:
                if st.button("🔄 Refresh Now", key="manual_refresh"):
                    st.rerun()

            # Render first → sleep → rerun
            # This ensures progress bar is visible BEFORE page reloads
            time.sleep(3)
            st.rerun()

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach server. Is it still running?")
        st.code("python main.py", language="bash")

    except Exception as e:
        st.error(f"Error fetching job status: {e}")
        st.caption("Try refreshing the page manually.")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PIPELINE
# ══════════════════════════════════════════════════════════════

if page == "🎬 Pipeline":
    st.title("🎬 Pipeline")
    st.caption("Upload a drone video and run the full AI pipeline")

    # ══════════════════════════════════════════════════════════
    # SECTION 1 — VIDEO UPLOAD
    # ══════════════════════════════════════════════════════════

    st.subheader("📤 Upload Video")
  

    # ── Show what's currently in input_folder ─────────────────
    existing_files = []
    if os.path.exists(INPUT_FOLDER):
        existing_files = [
            f for f in os.listdir(INPUT_FOLDER)
            if os.path.isfile(os.path.join(INPUT_FOLDER, f))
        ]

    if existing_files:
        st.info(
            f"`{'`, `'.join(existing_files)}`  \n"   
        )
    else:
        st.info("📂 **input_folder** is empty — ready for upload.")

    # ── Upload widget ─────────────────────────────────────────
    uploaded_file = st.file_uploader(
        label       = "Choose a video file",
        type        = ["mp4"],
        help        = "Only .mp4 files supported. Large files may take a moment to upload.",
        # key ensures widget resets after we clear it via session state
        key         = st.session_state.get("uploader_key", "video_uploader")
    )

    # ── Handle upload ─────────────────────────────────────────
    if uploaded_file is not None:

        # Show file info before saving
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("File Name", uploaded_file.name)
            c2.metric("File Size", f"{file_size_mb:.1f} MB")
            c3.metric("Type",      uploaded_file.type or "video/mp4")

        st.write("")   # small spacer

        # Confirm button — user must explicitly confirm before we clear folder
        col_confirm, col_cancel = st.columns([2, 1])

        with col_confirm:
            confirm_upload = st.button(
                f"✅ save '{uploaded_file.name}'",
                type                = "primary",
                use_container_width = True,
                help                = (
                    f"This will delete {len(existing_files)} existing file(s) "
                    f"from input_folder, then save your video."
                ) if existing_files else "Save video to input_folder"
            )

        with col_cancel:
            if st.button(
                "❌ Cancel",
                use_container_width = True
            ):
                # Reset uploader by changing its key
                st.session_state["uploader_key"] = f"video_uploader_{time.time()}"
                st.rerun()

        # ── Perform the upload ────────────────────────────────
        if confirm_upload:
            with st.status(
                "Saving video...",
                expanded = True
            ) as upload_status:

                # Step 1 — Clear input folder
                st.write("🗑️ Clearing input_folder...")
                deleted_count = clear_input_folder()
                st.write(f"   ✅ Cleared {deleted_count} file(s)")

                # Step 2 — Save uploaded file
                st.write(f"💾 Saving `{uploaded_file.name}`...")
                saved_path = save_uploaded_video(uploaded_file)
                st.write(f"   ✅ Saved to `{saved_path}`")

                # Step 3 — Verify file exists and has correct size
                st.write("🔍 Verifying...")
                if os.path.exists(saved_path):
                    actual_size = os.path.getsize(saved_path) / (1024 * 1024)
                    st.write(f"   ✅ Verified — {actual_size:.1f} MB on disk")
                    upload_status.update(
                        label    = f"✅ '{uploaded_file.name}' ready!",
                        state    = "complete",
                        expanded = False
                    )
                    # Save to session state so we can show it below
                    st.session_state["uploaded_video"] = uploaded_file.name

                    # Reset uploader widget so user can upload again fresh
                    st.session_state["uploader_key"] = f"video_uploader_{time.time()}"
                else:
                    st.write("   ❌ File not found after save — check permissions")
                    upload_status.update(
                        label = "❌ Upload failed",
                        state = "error"
                    )

            st.rerun()

    # ── Show currently saved video ────────────────────────────
    # Read directly from disk — always accurate
    current_videos = []
    if os.path.exists(INPUT_FOLDER):
        current_videos = [
            f for f in os.listdir(INPUT_FOLDER)
            if f.endswith(".mp4")
        ]

    if current_videos:
        st.success(
            f"✅ **Ready to process:** "
            f"`{'`, `'.join(current_videos)}`"
        )
    else:
        st.warning(
            "⚠️ No video in input_folder yet.  \n"
            "Upload a `.mp4` file above before starting the pipeline."
        )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SECTION 2 — VIDEO STATUS OVERVIEW
    # ══════════════════════════════════════════════════════════

    st.subheader("📂 Video Processing Status")

    try:
        status_resp = requests.get(f"{API}/status", timeout=5).json()
        videos      = status_resp.get("videos", [])

        if not videos:
            st.info(
                "No processed videos found.  \n"
                "Upload a video and run the pipeline."
            )
        else:
            total_videos = len(videos)
            fully_done   = sum(
                1 for v in videos
                if all(v["steps"].values())
            )
            st.caption(
                f"Found **{total_videos}** video(s) — "
                f"**{fully_done}** fully processed"
            )

            for v in videos:
                steps = v["steps"]
                done  = sum(1 for s in steps.values() if s)
                total = len(steps)

                icon = "✅" if done == total else ("⏳" if done > 0 else "⬜")

                with st.expander(
                    f"{icon} **{v['video']}**  —  {done}/{total} steps done"
                ):
                    labels = {
                        "step1_extract":    "1. Extract",
                        "step2_yolo":       "2. YOLO",
                        "step3_tracking":   "3. Track",
                        "step4_vision":     "4. Vision",
                        "step5_embeddings": "5. Embed",
                        "step6_storage":    "6. Store",
                        "step7_alerts":     "7. Alerts",
                    }
                    cols = st.columns(len(labels))
                    for i, (key, label) in enumerate(labels.items()):
                        with cols[i]:
                            if steps.get(key):
                                st.success(label)
                            else:
                                st.warning(label)

    except Exception as e:
        st.error(f"Could not load status: {e}")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # SECTION 3 — RUN PIPELINE
    # ══════════════════════════════════════════════════════════

    st.subheader("▶️ Run Pipeline")

    # Disable if no video uploaded
    no_video    = len(current_videos) == 0

    # Disable if job already running
    job_running = (
        "job_id" in st.session_state
        and st.session_state.get("job_status", "") not in ("done", "failed", "")
    )

    col1, col2 = st.columns(2)

    with col1:
        if no_video:
            st.button(
                "▶️ Start Pipeline",
                type                = "primary",
                use_container_width = True,
                disabled            = True,
                help                = "Upload a video first"
            )
            st.caption("⚠️ Upload a video before starting.")

        elif job_running:
            st.button(
                "▶️ Start Pipeline",
                type                = "primary",
                use_container_width = True,
                disabled            = True,
                help                = "Wait for current job to finish"
            )
            st.caption("⏳ Pipeline already running.")

        else:
            if st.button(
                "▶️ Start Pipeline",
                type                = "primary",
                use_container_width = True
            ):
                try:
                    resp   = requests.post(f"{API}/process", timeout=10).json()
                    status = resp.get("status")

                    if status == "error":
                        st.error(resp.get("message", "Unknown error"))
                    else:
                        job_id = resp.get("job_id")
                        st.session_state["job_id"]     = job_id
                        st.session_state["job_status"] = "queued"
                        st.session_state.pop("refresh_count", None)
                        st.success(f"✅ Pipeline started — Job ID: `{job_id}`")
                        time.sleep(0.5)
                        st.rerun()

                except Exception as e:
                    st.error(f"Failed to start pipeline: {e}")

    with col2:
        if st.button(
            "🔄 Refresh Status",
            use_container_width = True,
            help                = "Reload video status from server"
        ):
            st.rerun()

    # ── Job progress ──────────────────────────────────────────
    if "job_id" in st.session_state:
        st.divider()
        render_job_progress(st.session_state["job_id"])

    st.divider()

# ══════════════════════════════════════════════════════════════
# PAGE 2 — ALERTS
# ══════════════════════════════════════════════════════════════

elif page == "🚨 Alerts":
    st.title("🚨 Security Alerts")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        filter_video = st.text_input(
            "Filter by video",
            placeholder = "Leave empty for all videos"
        )
    with col2:
        filter_severity = st.selectbox(
            "Filter by severity",
            ["All", "critical", "high", "medium", "low"]
        )
    with col3:
        st.write("")
        st.write("")
        st.button("🔄 Refresh")

    try:
        url    = f"{API}/alerts/{filter_video}" if filter_video else f"{API}/alerts"
        data   = requests.get(url, timeout=5).json()
        alerts = data.get("alerts", [])

        if filter_severity != "All":
            alerts = [
                a for a in alerts
                if a.get("severity") == filter_severity
            ]

        total    = len(alerts)
        critical = sum(1 for a in alerts if a.get("severity") == "critical")
        high     = sum(1 for a in alerts if a.get("severity") == "high")
        medium   = sum(1 for a in alerts if a.get("severity") == "medium")
        low      = sum(1 for a in alerts if a.get("severity") == "low")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total",       total)
        m2.metric("🔴 Critical", critical)
        m3.metric("🟠 High",     high)
        m4.metric("🟡 Medium",   medium)
        m5.metric("🟢 Low",      low)

        st.divider()

        if not alerts:
            st.info(
                "No alerts found.  \n"
                "Run the pipeline first, or try a different filter."
            )
        else:
            severity_icons = {
                "critical": "🔴",
                "high":     "🟠",
                "medium":   "🟡",
                "low":      "🟢",
            }

            for alert in reversed(alerts):
                severity = alert.get("severity", "low")
                icon     = severity_icons.get(severity, "⚪")

                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(
                            f"{icon} **[{severity.upper()}]** "
                            f"{alert.get('description', 'No description')}"
                        )
                        st.caption(
                            f"📹 {alert.get('video_name', '?')}  |  "
                            f"🎞️ Frame: {alert.get('frame_id', '?')}  |  "
                            f"⏱️ {alert.get('timestamp_sec', 0):.1f}s  |  "
                            f"🏷️ {alert.get('alert_type', '?')}"
                        )
                    with c2:
                        st.caption(str(alert.get("created_at", ""))[:19])

    except Exception as e:
        st.error(f"Could not load alerts: {e}")

    st.divider()

    st.subheader("⚡ Generate Alerts for a Video")
    st.caption(
        "Runs the LangGraph alert agent manually on a specific video.  \n"
        "Use this if alerts weren't generated during the main pipeline run."
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        alert_video = st.text_input(
            "Video name",
            placeholder = "video1",
            key         = "alert_video"
        )
    with c2:
        st.write("")
        st.write("")
        if st.button("🚨 Generate Alerts", type="primary", use_container_width=True):
            if alert_video:
                with st.spinner("Running alert agent..."):
                    try:
                        resp  = requests.post(
                            f"{API}/alerts/run/{alert_video}",
                            timeout=120
                        ).json()
                        count = resp.get("alerts_generated", 0)
                        st.success(f"✅ Generated {count} alert(s) for `{alert_video}`")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate alerts: {e}")
            else:
                st.warning("⚠️ Enter a video name first")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CHAT
# ══════════════════════════════════════════════════════════════

elif page == "💬 Chat":
    st.title("💬 Chat with Your Footage")
    st.caption(
        "Ask questions in plain English — "
        "powered by Claude Sonnet + LangGraph RAG pipeline"
    )

    chat_video = st.text_input(
        "Limit search to one video (optional)",
        placeholder = "Leave empty to search across all videos"
    )

    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state["chat_history"]:
        st.markdown("**💡 Example questions to get started:**")
        examples = [
            "Were there any suspicious people detected?",
            "How many vehicles appeared in the footage?",
            "What happened near the entrance gate?",
            "Were there any loitering alerts?",
            "Describe the most critical security events.",
            "Were any people seen at night?",
        ]
        rows = [examples[:3], examples[3:]]
        for row in rows:
            cols = st.columns(3)
            for i, question in enumerate(row):
                if cols[i].button(question, key=f"example_{question}"):
                    st.session_state["prefill"] = question

    user_input = st.chat_input("Ask about the footage...")

    if "prefill" in st.session_state:
        user_input = st.session_state.pop("prefill")

    if user_input:
        st.session_state["chat_history"].append({
            "role":    "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching footage and generating answer..."):
                try:
                    resp = requests.post(
                        f"{API}/chat",
                        json    = {
                            "question":   user_input,
                            "video_name": chat_video or None
                        },
                        timeout = 60
                    ).json()

                    answer = resp.get("answer", "Could not generate an answer.")
                    st.markdown(answer)

                    sources = resp.get("sources", [])
                    if sources:
                        with st.expander(f"📎 {len(sources)} source frame(s) used"):
                            for s in sources:
                                st.caption(
                                    f"📹 {s.get('video_name', '?')} — "
                                    f"🎞️ {s.get('frame_id', '?')} — "
                                    f"⏱️ {s.get('timestamp_sec', '?')}s"
                                )

                    st.session_state["chat_history"].append({
                        "role":    "assistant",
                        "content": answer
                    })

                except requests.exceptions.Timeout:
                    err = "⏱️ Request timed out. The server may be busy."
                    st.error(err)
                    st.session_state["chat_history"].append({
                        "role": "assistant", "content": err
                    })

                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state["chat_history"].append({
                        "role": "assistant", "content": err
                    })

    if st.session_state["chat_history"]:
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state["chat_history"] = []
            st.rerun()