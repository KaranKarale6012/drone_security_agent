"""
Security Agents:
- AlertAgent  : detects & stores alerts from suspicious frames
- ChatAgent   : answers questions using Chroma + Mongo + Alerts
"""

import json
import re
import boto3
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from database.mongo_store import (
    get_alerts,
    get_frames_by_filter,
    store_alert
)
from database.chroma_store import semantic_search

load_dotenv("env")

MODEL_ID = "us.anthropic.claude-3-haiku-20240307-v1:0"


# ════════════════════════════════════════════════
# BEDROCK HELPERS
# ════════════════════════════════════════════════

def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )


def call_claude(
    client,
    prompt: str,
    max_tokens: int = 500,
    retries: int = 2
) -> str:
    """
    Call Claude via Bedrock with retry logic.
    Returns raw text response.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    last_error = None

    for attempt in range(retries + 1):
        try:
            response = client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"].strip()

        except Exception as e:
            last_error = e
            print(f"[Claude] Attempt {attempt + 1} failed: {e}")

    raise RuntimeError(f"Claude failed after {retries + 1} attempts: {last_error}")


def parse_json_response(response: str) -> dict:
    """
    Robustly parse JSON from Claude response.
    Handles markdown code blocks and extra text.
    """
    # ── Remove markdown code fences ───────────────────────────
    # Handles ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?\s*", "", response)
    cleaned = cleaned.replace("```", "").strip()

    # ── Try direct parse ──────────────────────────────────────
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── Extract first JSON object with regex ──────────────────
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response:\n{response[:300]}")


# ════════════════════════════════════════════════
# ALERT AGENT — STATE
# ════════════════════════════════════════════════

class AlertState(TypedDict):
    video_name:        str
    frames:            list[dict]
    suspicious_frames: list[dict]
    alerts_generated:  list[dict]


# ════════════════════════════════════════════════
# ALERT AGENT — NODE 1: Load Frames
# ════════════════════════════════════════════════

def load_frames_node(state: AlertState) -> AlertState:
    """
    Load ALL frames for the video from MongoDB.
    Uses high limit to avoid truncation.
    """
    print(f"\n[AlertAgent] Loading frames for: {state['video_name']!r}")

    # ── Load all frames (no filter yet) ───────────────────────
    frames = get_frames_by_filter(
        video_name=state["video_name"],
        limit=1000                          # ← was 100, raised to avoid truncation
    )

    print(f"[AlertAgent] Loaded {len(frames)} frames")

    # ── Debug: show field distribution ────────────────────────
    if frames:
        suspicious_count = sum(1 for f in frames if f.get("is_suspicious"))
        loitering_count  = sum(1 for f in frames if f.get("loitering"))
        person_count     = sum(1 for f in frames if f.get("has_person"))

        print(f"[AlertAgent] is_suspicious : {suspicious_count}")
        print(f"[AlertAgent] loitering     : {loitering_count}")
        print(f"[AlertAgent] has_person    : {person_count}")
    else:
        print(f"[AlertAgent] ⚠️  No frames found — "
              f"was store_frames() called before this agent?")

    return {**state, "frames": frames}


# ════════════════════════════════════════════════
# ALERT AGENT — NODE 2: Find Suspicious Frames
# ════════════════════════════════════════════════

def find_suspicious_node(state: AlertState) -> AlertState:
    """
    Flag frames as suspicious based on multiple signals.
    Fixed timestamp logic — video timestamps are cumulative seconds,
    NOT time-of-day, so hour-of-day check is removed.
    """
    suspicious = []
    seen_frame_ids = set()  # ← prevent duplicates

    for frame in state["frames"]:

        # ── Skip duplicates ────────────────────────────────────
        frame_id = frame.get("frame_id")
        if frame_id in seen_frame_ids:
            continue
        seen_frame_ids.add(frame_id)

        reasons = []

        # ── Signal 1: LLM marked suspicious ───────────────────
        if frame.get("is_suspicious") is True:
            reasons.append("LLM marked as suspicious")

        # ── Signal 2: Loitering detected ──────────────────────
        if frame.get("loitering") is True:
            reasons.append("loitering detected")

        # ── Signal 3: High loitering score ────────────────────
        loitering_score = float(frame.get("loitering_score", 0.0))
        if loitering_score > 0.72:
            reasons.append(
                f"high loitering score ({loitering_score:.2f})"
            )

        # ── Signal 4: Person + vehicle together ───────────────
        if frame.get("has_person") and frame.get("has_vehicle"):
            reasons.append("person with vehicle detected")

        # ── Only flag if at least one reason ──────────────────
        if reasons:
            frame["_reasons"] = reasons
            suspicious.append(frame)

    print(f"[AlertAgent] Found {len(suspicious)} suspicious frames "
          f"from {len(state['frames'])} total")

    # ── Show reason distribution ───────────────────────────────
    if suspicious:
        all_reasons = [r for f in suspicious for r in f.get("_reasons", [])]
        reason_counts = {}
        for r in all_reasons:
            # Normalize reason key for counting
            key = r.split("(")[0].strip()
            reason_counts[key] = reason_counts.get(key, 0) + 1
        print(f"[AlertAgent] Reason distribution: {reason_counts}")

    return {**state, "suspicious_frames": suspicious}


# ════════════════════════════════════════════════
# ALERT AGENT — CONDITION
# ════════════════════════════════════════════════

def has_suspicious(state: AlertState) -> str:
    if not state["suspicious_frames"]:
        print("[AlertAgent] No suspicious frames — skipping alert generation")
        return "end"
    print(f"[AlertAgent] Proceeding to analyze "
          f"{len(state['suspicious_frames'])} frames")
    return "analyze"


# ════════════════════════════════════════════════
# ALERT AGENT — NODE 3: Analyze & Generate Alerts
# ════════════════════════════════════════════════

# Valid values — must match store_alert() validation
VALID_SEVERITY   = {"low", "medium", "high", "critical"}
VALID_ALERT_TYPE = {
    "loitering", "trespassing",
    "suspicious_vehicle", "unusual_activity",
    "suspicious_person", "atm_tampering"
}


def analyze_and_alert_node(state: AlertState) -> AlertState:
    """
    Send each suspicious frame to Claude for final decision.
    Stores confirmed alerts in MongoDB.
    """
    client = get_bedrock_client()
    alerts_generated = []

    # ── Deduplicate by frame_id ────────────────────────────────
    seen = set()
    unique_frames = []
    for frame in state["suspicious_frames"]:
        fid = frame.get("frame_id")
        if fid not in seen:
            seen.add(fid)
            unique_frames.append(frame)

    print(f"\n[AlertAgent] Analyzing {len(unique_frames)} unique frames...")

    for i, frame in enumerate(unique_frames):

        print(f"[AlertAgent] Frame {i+1}/{len(unique_frames)}: "
              f"{frame.get('frame_id')} @ {frame.get('timestamp_sec', 0):.1f}s")

        prompt = f"""
You are a security analyst reviewing surveillance footage.

Frame details:
- Video      : {state['video_name']}
- Frame ID   : {frame.get('frame_id')}
- Timestamp  : {frame.get('timestamp_sec', 0):.1f} seconds
- Has Person : {frame.get('has_person')}
- Has Vehicle: {frame.get('has_vehicle')}
- Loitering  : {frame.get('loitering')} (score: {frame.get('loitering_score', 0):.2f})
- Reasons    : {', '.join(frame.get('_reasons', []))}
- Description: {str(frame.get('description', ''))[:400]}

Respond with ONLY a valid JSON object (no markdown, no explanation):
{{
  "is_alert": true or false,
  "severity": "low" or "medium" or "high" or "critical",
  "alert_type": "loitering" or "trespassing" or "suspicious_vehicle" or "unusual_activity" or "suspicious_person" or "atm_tampering",
  "reason": "one sentence explanation"
}}
"""

        try:
            # ── Call Claude ────────────────────────────────────
            raw_response = call_claude(client, prompt, max_tokens=200)
            print(f"[AlertAgent] Raw response: {raw_response[:150]}")

            # ── Parse JSON safely ──────────────────────────────
            decision = parse_json_response(raw_response)

            # ── Validate fields ────────────────────────────────
            is_alert   = bool(decision.get("is_alert", False))
            severity   = str(decision.get("severity", "low")).lower()
            alert_type = str(decision.get("alert_type", "unusual_activity")).lower()
            reason     = str(decision.get("reason", "Suspicious activity detected"))

            # ── Sanitize severity & alert_type ────────────────
            if severity not in VALID_SEVERITY:
                print(f"[AlertAgent] ⚠️  Invalid severity '{severity}' → defaulting to 'medium'")
                severity = "medium"

            if alert_type not in VALID_ALERT_TYPE:
                print(f"[AlertAgent] ⚠️  Invalid alert_type '{alert_type}' → defaulting to 'unusual_activity'")
                alert_type = "unusual_activity"

            # ── Store if Claude confirmed alert ────────────────
            if is_alert:
                alert = {
                    "video_name":    state["video_name"],
                    "frame_id":      frame["frame_id"],
                    "alert_type":    alert_type,
                    "severity":      severity,
                    "description":   reason,
                    "timestamp_sec": float(frame.get("timestamp_sec", 0.0))
                }

                store_alert(**alert)
                alerts_generated.append(alert)

                print(f"[AlertAgent] 🚨 [{severity.upper():8s}] "
                      f"{alert_type} — {reason[:80]}")
            else:
                print(f"[AlertAgent] ✅ Frame {frame.get('frame_id')} "
                      f"— Claude determined NOT an alert")

        except Exception as e:
            print(f"[AlertAgent] ❌ Failed on {frame.get('frame_id')}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[AlertAgent] ✅ Total alerts generated: {len(alerts_generated)}")
    return {**state, "alerts_generated": alerts_generated}


# ════════════════════════════════════════════════
# ALERT AGENT — BUILD & RUN
# ════════════════════════════════════════════════

def build_alert_agent():
    graph = StateGraph(AlertState)

    graph.add_node("load_frames",    load_frames_node)
    graph.add_node("find_suspicious", find_suspicious_node)
    graph.add_node("analyze",        analyze_and_alert_node)

    graph.set_entry_point("load_frames")

    graph.add_edge("load_frames", "find_suspicious")

    graph.add_conditional_edges(
        "find_suspicious",
        has_suspicious,
        {
            "analyze": "analyze",
            "end":     END
        }
    )

    graph.add_edge("analyze", END)

    return graph.compile()


def run_alert_agent(video_name: str) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"[AlertAgent] Starting pipeline for: {video_name!r}")
    print(f"{'='*60}")

    agent = build_alert_agent()

    result = agent.invoke({
        "video_name":        video_name,
        "frames":            [],
        "suspicious_frames": [],
        "alerts_generated":  []
    })

    alerts = result.get("alerts_generated", [])
    print(f"\n[AlertAgent] Pipeline complete — {len(alerts)} alerts stored")
    return alerts


# ════════════════════════════════════════════════
# CHAT AGENT — STATE
# ════════════════════════════════════════════════

class ChatState(TypedDict):
    question:       str
    video_name:     Optional[str]
    chroma_results: list[dict]
    mongo_results:  list[dict]
    final_answer:   str


# ════════════════════════════════════════════════
# CHAT AGENT — NODE 1: Chroma Search
# ════════════════════════════════════════════════

def search_chroma_node(state: ChatState) -> ChatState:
    """
    Semantic search in ChromaDB.
    Safely handles varying result structures.
    """
    try:
        results = semantic_search(
            query=state["question"],
            video_name=state.get("video_name"),
            n_results=5
        )
        print(f"[ChatAgent] Chroma results: {len(results)}")
    except Exception as e:
        print(f"[ChatAgent] Chroma search failed: {e}")
        results = []

    return {**state, "chroma_results": results}


# ════════════════════════════════════════════════
# CHAT AGENT — NODE 2: MongoDB Search
# ════════════════════════════════════════════════

def search_mongo_node(state: ChatState) -> ChatState:
    """
    Structured search in MongoDB.
    Fixed: loitering and suspicious are separate flags,
    not combined into one filter.
    """
    q = state["question"].lower()

    # ── Parse intent from question ────────────────────────────
    want_person    = "person"   in q or "people"   in q or "individual" in q
    want_vehicle   = "vehicle"  in q or "car"       in q or "truck"      in q
    want_loitering = "loiter"   in q or "lingering" in q or "standing"   in q
    want_suspicious = "suspicious" in q or "alert"  in q or "threat"     in q

    print(f"[ChatAgent] Query intent — "
          f"person={want_person} vehicle={want_vehicle} "
          f"loitering={want_loitering} suspicious={want_suspicious}")

    try:
        results = get_frames_by_filter(
            video_name  = state.get("video_name"),
            has_person  = True if want_person   else None,
            has_vehicle = True if want_vehicle  else None,
            loitering   = True if want_loitering else None,
            # ← Don't pass is_suspicious=True for general "suspicious" queries
            # because that combines with other filters and may return 0 results
            is_suspicious = True if want_suspicious and not (
                want_person or want_vehicle or want_loitering
            ) else None,
            limit=10
        )
        print(f"[ChatAgent] Mongo results: {len(results)}")
    except Exception as e:
        print(f"[ChatAgent] Mongo search failed: {e}")
        results = []

    return {**state, "mongo_results": results}


# ════════════════════════════════════════════════
# CHAT AGENT — NODE 3: Synthesize Answer
# ════════════════════════════════════════════════

def synthesize_answer_node(state: ChatState) -> ChatState:
    """
    Combine Chroma + Mongo + Alerts context and call Claude.
    Fixed: safe access to chroma result fields.
    """
    client = get_bedrock_client()

    # ── Build Chroma context safely ────────────────────────────
    chroma_lines = []
    for r in state["chroma_results"]:
        # Handle both flat and nested metadata structures
        if isinstance(r, dict):
            meta      = r.get("metadata", r)          # fallback to r itself
            ts        = meta.get("timestamp_sec", "?")
            desc      = r.get("description") or meta.get("description", "")
            chroma_lines.append(f"- [t={ts}s] {str(desc)[:150]}")

    chroma_ctx = "\n".join(chroma_lines) or "No semantic results found."

    # ── Build Mongo context safely ─────────────────────────────
    mongo_lines = []
    for f in state["mongo_results"]:
        ts   = f.get("timestamp_sec", "?")
        desc = str(f.get("description", ""))[:150]
        flags = []
        if f.get("has_person"):    flags.append("person")
        if f.get("has_vehicle"):   flags.append("vehicle")
        if f.get("loitering"):     flags.append("loitering")
        if f.get("is_suspicious"): flags.append("suspicious")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        mongo_lines.append(f"- [t={ts}s]{flag_str} {desc}")

    mongo_ctx = "\n".join(mongo_lines) or "No structured results found."

    # ── Build Alerts context ───────────────────────────────────
    try:
        alerts = get_alerts(video_name=state.get("video_name"))
    except Exception as e:
        print(f"[ChatAgent] Failed to load alerts: {e}")
        alerts = []

    alert_lines = []
    for a in alerts:
        alert_lines.append(
            f"- [{a.get('severity', '?').upper()}] "
            f"{a.get('alert_type', '?')} @ "
            f"t={a.get('timestamp_sec', '?')}s — "
            f"{a.get('description', '')[:120]}"
        )

    alert_ctx = "\n".join(alert_lines) or "No alerts on record."

    print(f"[ChatAgent] Context — "
          f"chroma={len(chroma_lines)} mongo={len(mongo_lines)} "
          f"alerts={len(alert_lines)}")

    # ── Build prompt ───────────────────────────────────────────
    prompt = f"""
You are a security analyst assistant reviewing drone surveillance footage.

User question: {state['question']}

Semantic search results (ChromaDB):
{chroma_ctx}

Structured frame data (MongoDB):
{mongo_ctx}

Active alerts:
{alert_ctx}

Instructions:
- Answer the question directly and concisely (max 5 sentences)
- Reference specific timestamps if relevant
- If no relevant data found, say so clearly
- Do not speculate beyond the provided data
"""

    try:
        answer = call_claude(client, prompt, max_tokens=400)
    except Exception as e:
        answer = f"Unable to generate answer: {e}"
        print(f"[ChatAgent] Claude failed: {e}")

    print(f"[ChatAgent] Answer: {answer[:100]}...")
    return {**state, "final_answer": answer}


# ════════════════════════════════════════════════
# CHAT AGENT — BUILD & RUN
# ════════════════════════════════════════════════

def build_chat_agent():
    graph = StateGraph(ChatState)

    graph.add_node("search_chroma", search_chroma_node)
    graph.add_node("search_mongo",  search_mongo_node)
    graph.add_node("synthesize",    synthesize_answer_node)

    graph.set_entry_point("search_chroma")

    graph.add_edge("search_chroma", "search_mongo")
    graph.add_edge("search_mongo",  "synthesize")
    graph.add_edge("synthesize",    END)

    return graph.compile()


def run_chat_agent(
    question:   str,
    video_name: Optional[str] = None
) -> str:
    print(f"\n[ChatAgent] Question: {question!r}")

    agent = build_chat_agent()

    result = agent.invoke({
        "question":       question,
        "video_name":     video_name,
        "chroma_results": [],
        "mongo_results":  [],
        "final_answer":   ""
    })

    answer = result.get("final_answer", "No answer generated.")
    print(f"[ChatAgent] Done")
    return answer