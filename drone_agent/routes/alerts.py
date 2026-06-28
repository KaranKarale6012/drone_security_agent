from fastapi  import APIRouter
from agents.security_agent import run_alert_agent
from database.mongo_store  import get_alerts

AlertRouter = APIRouter()


@AlertRouter.post("/run/{video_name}")
def generate_alerts(video_name: str):
    """
    Runs the LangGraph alert agent for one video.
    Reads frames from MongoDB, finds suspicious ones,
    sends to Claude Sonnet, saves alerts back to MongoDB.
    """
    print(f"\nRunning alert agent for: {video_name}")
    alerts = run_alert_agent(video_name)


    return {
        "video_name":       video_name,
        "alerts_generated": len(alerts),
        "alerts":           alerts
    }


@AlertRouter.get("")
def get_all_alerts():
    """Returns all alerts across all videos."""
    alerts = get_alerts()
    return {
    "total":   len(alerts),
    "alerts":  alerts
    }


@AlertRouter.get("/{video_name}")
def get_video_alerts(video_name: str):
    """Returns alerts for one specific video."""
    alerts = get_alerts(video_name=video_name)
    return {
    "video_name": video_name,
    "total":      len(alerts),
    "alerts":     alerts
    }