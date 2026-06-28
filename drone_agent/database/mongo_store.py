"""
Step 6b: Store structured frame metadata in MongoDB
"""

import os
import numpy as np
from datetime import datetime
import re
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

from agents.embedding import embed_text


# ── ENV SETUP ────────────────────────────────────────────────
load_dotenv("env")


# ── LOITERING CONCEPT (SEMANTIC DETECTION) ──────────────────
LOITERING_CONCEPT = """
A person staying in the same location for an extended period of time without a clear purpose,
showing minimal movement or repeatedly appearing in the same area,
waiting, lingering, or observing surroundings in a suspicious or unusual manner.
Includes behavior such as standing idle near restricted areas, slowly pacing,
hesitating, repeatedly approaching and leaving a spot, or monitoring activity
without engaging in a normal task.
Often associated with surveillance, reconnaissance, or intent to perform suspicious activity.
"""

LOITERING_EMBED = embed_text(LOITERING_CONCEPT)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =============================
# DATABASE CONNECTION
# =============================
def get_db():
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    dbname = os.getenv("MONGODB_DATABASE", "drone_agent")

    client = MongoClient(uri, serverSelectionTimeoutMS=3000)

    try:
        client.admin.command("ping")
    except Exception as e:
        print("MongoDB connection failed")
        raise e

    return client[dbname]


# =============================
# INIT DATABASE + INDEXES
# =============================
def init_database() -> None:

    try:
        db = get_db()

        db.frames.create_index(
            [("video_name", ASCENDING), ("frame_id", ASCENDING)],
            unique=True
        )

        db.frames.create_index([("video_name", ASCENDING)])
        db.frames.create_index([("has_person", ASCENDING)])
        db.frames.create_index([("has_vehicle", ASCENDING)])
        db.frames.create_index([("loitering", ASCENDING)])
        db.frames.create_index([("timestamp_sec", ASCENDING)])

        db.frames.create_index([
            ("video_name", ASCENDING),
            ("timestamp_sec", ASCENDING)
        ])

        db.frames.create_index([
            ("video_name", ASCENDING),
            ("loitering", ASCENDING)
        ])

        db.frames.create_index([
            ("video_name", ASCENDING),
            ("is_suspicious", ASCENDING)
        ])

        db.alerts.create_index([("video_name", ASCENDING)])
        db.alerts.create_index([("severity", ASCENDING)])
        db.alerts.create_index([
            ("video_name", ASCENDING),
            ("timestamp_sec", ASCENDING)
        ])

        print("✅ MongoDB indexes ready.")

    except ConnectionFailure as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise


# =============================
# STORE VIDEO
# =============================
# =============================
# STORE VIDEO
# =============================
def store_video(video_name: str, file_path: str, total_frames: int) -> None:
    db = get_db()

    db.videos.update_one(
        {"video_name": video_name},
        {
            "$set": {
                "video_name": video_name,
                "file_path": file_path,
                "total_frames": total_frames,
                "updated_at": datetime.now()
            }
        },
        upsert=True
    )


def _extract_objects(description: str) -> list[str]:
    """
    Robustly extract object types from frame description text.
    Uses whole-word matching to avoid false positives.
    """
    objects = []
    desc_lower = description.lower()

    # ── Person detection ──────────────────────────────────────
    person_patterns = [
        r'\bperson\b', r'\bindividual\b', r'\bpeople\b',
        r'\bman\b', r'\bwoman\b', r'\bsomeone\b',
        r'\bsubject\b', r'\bsuspect\b', r'\bfigure\b',
        r'\bpedestrian\b', r'\bwalker\b'
    ]
    if any(re.search(p, desc_lower) for p in person_patterns):
        objects.append("person")

    # ── Vehicle detection ─────────────────────────────────────
    vehicle_patterns = [
        r'\bcar\b', r'\btruck\b', r'\bbus\b',
        r'\bmotorcycle\b', r'\bvehicle\b', r'\bvan\b',
        r'\bsuv\b', r'\bautomobile\b', r'\bsedан\b'
    ]
    if any(re.search(p, desc_lower) for p in vehicle_patterns):
        objects.append("vehicle")

    return objects

# =============================
# HELPER: DETECT SUSPICION
# =============================
def _detect_suspicion(description: str) -> bool:
    """
    Robustly detect suspicion level from description text.
    Handles multiple formats the LLM might produce.
    """
    desc_lower = description.lower()

    # ── Check explicit suspicion level markers first ──────────
    # Pattern: "suspicion level: suspicious" or "suspicion level: non-suspicious"
    match = re.search(
        r'suspicion\s+level\s*[:\-]\s*(non[-\s]?suspicious|suspicious)',
        desc_lower
    )
    if match:
        verdict = match.group(1).replace(" ", "").replace("-", "")
        return verdict == "suspicious"

    # ── Fallback: check for explicit NON-suspicious markers ───
    non_suspicious_patterns = [
        r'\bnon[-\s]?suspicious\b',
        r'\bnot\s+suspicious\b',
        r'\bno\s+suspicious\b',
        r'\bnormal\s+activity\b',
        r'\blegitimate\s+activity\b',
    ]
    if any(re.search(p, desc_lower) for p in non_suspicious_patterns):
        return False

    # ── Fallback: check for suspicious markers ─────────────────
    suspicious_patterns = [
        r'\bsuspicious\b',
        r'\bsuspect\b',
        r'\bcriminal\b',
        r'\bmalicious\b',
        r'\billicit\b',
        r'\bunauthorized\b',
        r'\bnefarious\b',
    ]
    if any(re.search(p, desc_lower) for p in suspicious_patterns):
        return True

    return False


# =============================
# HELPER: DETECT LOITERING
# =============================
def _detect_loitering(description: str, desc_embed: list, threshold: float = 0.72) -> tuple[bool, float]:
    """
    Dual-method loitering detection:
    1. Semantic similarity against loitering concept embedding
    2. Keyword fallback for explicit mentions
    """
    # ── Method 1: Embedding similarity ────────────────────────
    score = cosine_sim(desc_embed, LOITERING_EMBED)
    if score > threshold:
        return True, float(score)

    # ── Method 2: Keyword detection ───────────────────────────
    desc_lower = description.lower()
    loitering_keywords = [
        r'\bloitering\b', r'\blingering\b', r'\bloiter\b',
        r'\bprolonged\s+presence\b', r'\bextended\s+period\b',
        r'\bstanding\s+idle\b', r'\bpacing\b',
        r'\brepeated(?:ly)?\s+(?:approach|appear|return)\b',
    ]
    if any(re.search(p, desc_lower) for p in loitering_keywords):
        # Boost score slightly if keyword found but embedding missed
        return True, max(float(score), 0.65)

    return False, float(score)
# =============================
# STORE FRAMES (FIXED)
# =============================
def store_frames(video_name: str, embedded_frames: list[dict]) -> int:
    if not embedded_frames:
        return 0

    db = get_db()
    operations = []
    skipped = 0

    for frame in embedded_frames:

        # ── Validate required fields ──────────────────────────
        if not frame.get("description"):
            print(f"⚠️  Skipping frame {frame.get('frame_id', '?')} — missing description")
            skipped += 1
            continue

        if not frame.get("embedding"):
            print(f"⚠️  Skipping frame {frame.get('frame_id', '?')} — missing embedding")
            skipped += 1
            continue

        description = frame["description"]
        desc_embed = frame["embedding"]

        # ── Extract objects ────────────────────────────────────
        objects = _extract_objects(description)

        # ── Detect suspicion ───────────────────────────────────
        is_suspicious = _detect_suspicion(description)

        # ── Detect loitering ───────────────────────────────────
        loitering, loitering_score = _detect_loitering(description, desc_embed)

        # ── Build document ─────────────────────────────────────
        doc = {
            "video_name":      video_name,
            "frame_id":        frame["frame_id"],
            "timestamp_sec":   frame.get("timestamp_sec", 0.0),
            "filepath":        frame.get("filepath", ""),
            "description":     description,

            "objects":         objects,
            "has_person":      "person"  in objects,
            "has_vehicle":     "vehicle" in objects,

            "loitering":       loitering,
            "loitering_score": loitering_score,
            "is_suspicious":   is_suspicious,

            "updated_at":      datetime.now()
        }

        operations.append(
            UpdateOne(
                {
                    "video_name": video_name,
                    "frame_id":   frame["frame_id"]
                },
                {"$set": doc},
                upsert=True
            )
        )

    if not operations:
        print(f"⚠️  No valid frames to store (skipped {skipped})")
        return 0

    result = db.frames.bulk_write(operations, ordered=False)
    count = result.upserted_count + result.modified_count

    print(f"✅ Stored/updated {count} frames | skipped {skipped} | "
          f"upserted {result.upserted_count} | modified {result.modified_count}")
    return count


# =============================
# QUERY FRAMES (FIXED)
# =============================
def get_frames_by_filter(
    video_name:    str   = None,
    has_person:    bool  = None,
    has_vehicle:   bool  = None,
    loitering:     bool  = None,
    is_suspicious: bool  = None,
    min_time:      float = None,
    max_time:      float = None,
    skip:          int   = 0,
    limit:         int   = 500          # ← raised default; 50 was truncating results
) -> list[dict]:

    db = get_db()
    query = {}

    # ── Debug logging (fixed set literal bug) ─────────────────
    print(f"[DB Query] video_name={video_name!r} has_person={has_person!r} "
          f"has_vehicle={has_vehicle!r} loitering={loitering!r} "
          f"is_suspicious={is_suspicious!r} "
          f"min_time={min_time!r} max_time={max_time!r} "
          f"skip={skip} limit={limit}")

    # ── Build query ────────────────────────────────────────────
    if video_name is not None:
        query["video_name"] = video_name

    if has_person is not None:
        query["has_person"] = bool(has_person)

    if has_vehicle is not None:
        query["has_vehicle"] = bool(has_vehicle)

    if loitering is not None:
        query["loitering"] = bool(loitering)

    if is_suspicious is not None:
        query["is_suspicious"] = bool(is_suspicious)

    if min_time is not None or max_time is not None:
        query["timestamp_sec"] = {}
        if min_time is not None:
            query["timestamp_sec"]["$gte"] = float(min_time)
        if max_time is not None:
            query["timestamp_sec"]["$lte"] = float(max_time)

    # ── Execute query ──────────────────────────────────────────
    projection = {
        "_id":          0,
        "video_name":   1,
        "frame_id":     1,
        "timestamp_sec":1,
        "has_person":   1,
        "has_vehicle":  1,
        "loitering":    1,
        "loitering_score": 1,
        "is_suspicious":1,
        "filepath":     1,
        "description":  1,
        "objects":      1,
    }

    cursor = (
        db.frames
        .find(query, projection)
        .sort("timestamp_sec", ASCENDING)
        .skip(skip)
        .limit(limit)
    )

    results = list(cursor)
    print(f"[DB Query] → {len(results)} frames returned for query: {query}")
    return results


# =============================
# STORE ALERT
# =============================
VALID_SEVERITY = {"low", "medium", "high", "critical"}


def store_alert(
    video_name:    str,
    frame_id:      str,
    alert_type:    str,
    severity:      str,
    description:   str,
    timestamp_sec: float
) -> None:

    if severity not in VALID_SEVERITY:
        raise ValueError(f"Invalid severity '{severity}'. Must be one of {VALID_SEVERITY}")

    db = get_db()

    db.alerts.insert_one({
        "video_name":    video_name,
        "frame_id":      frame_id,
        "alert_type":    alert_type,
        "severity":      severity,
        "description":   description,
        "timestamp_sec": float(timestamp_sec),
        "created_at":    datetime.now()
    })


# =============================
# GET ALERTS
# =============================
def get_alerts(
    video_name: str = None,
    severity:   str = None
) -> list[dict]:

    db = get_db()
    query = {}

    if video_name is not None:
        query["video_name"] = video_name

    if severity is not None:
        if severity not in VALID_SEVERITY:
            raise ValueError(f"Invalid severity '{severity}'. Must be one of {VALID_SEVERITY}")
        query["severity"] = severity

    # ── Full diagnostics before query ─────────────────────────
    total_alerts      = db.alerts.count_documents({})
    total_frames      = db.frames.count_documents({})
    suspicious_frames = db.frames.count_documents({"is_suspicious": True})

    print(f"\n{'='*50}")
    print(f"[DB Alerts] query            : {query}")
    print(f"[DB Alerts] total alerts     : {total_alerts}")
    print(f"[DB Alerts] total frames     : {total_frames}")
    print(f"[DB Alerts] suspicious frames: {suspicious_frames}")

    if video_name:
        v_frames     = db.frames.count_documents({"video_name": video_name})
        v_suspicious = db.frames.count_documents({
            "video_name":    video_name,
            "is_suspicious": True
        })
        v_alerts = db.alerts.count_documents({"video_name": video_name})
        print(f"[DB Alerts] frames  for '{video_name}': {v_frames}")
        print(f"[DB Alerts] suspicious '{video_name}': {v_suspicious}")
        print(f"[DB Alerts] alerts  for '{video_name}': {v_alerts}")

    # ── Show sample documents for debugging ───────────────────
    if total_alerts == 0:
        print("[DB Alerts] ⚠️  alerts collection is EMPTY")
        print("[DB Alerts]    → store_alert() was never called")
        print("[DB Alerts]    → Check AlertAgent.run() is executing")

        # Show sample frame to verify fields
        sample_frame = db.frames.find_one(
            {"video_name": video_name} if video_name else {},
            {"_id": 0, "frame_id": 1, "is_suspicious": 1,
             "has_person": 1, "loitering": 1, "timestamp_sec": 1}
        )
        print(f"[DB Alerts] Sample frame doc : {sample_frame}")

    else:
        sample_alert = db.alerts.find_one({}, {"_id": 0})
        print(f"[DB Alerts] Sample alert doc : {sample_alert}")

    print(f"{'='*50}\n")

    # ── Execute query ──────────────────────────────────────────
    results = list(
        db.alerts
        .find(query, {"_id": 0})
        .sort("timestamp_sec", ASCENDING)
    )
    print(f"[DB Alerts] → returning {len(results)} alerts")
    return results