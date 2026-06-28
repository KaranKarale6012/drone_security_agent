import boto3
import json
import base64
from pathlib import Path
import os
from dotenv import load_dotenv

# Load env
load_dotenv('.env')

# Bedrock client
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ID = "us.anthropic.claude-3-haiku-20240307-v1:0"
MAX_TOKENS = 200


def _image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_single_frame(client, frame: dict, tracks: list[dict]) -> str:
    image_path = frame["filepath"]

    if not os.path.exists(image_path):
        return "Image file not found"

    image_data = _image_to_base64(image_path)

    # ✅ Build tracking context
    if tracks:
        track_lines = []
        for t in tracks:
            line = f"- {t['label']} (Track ID={t['track_id']}, seen for {t['frames_seen']} frames)"
            if t.get('frames_seen', 0) >= 10:
                line += " -> possible loitering"
            track_lines.append(line)

        track_context = "Tracked objects:\n" + "\n".join(track_lines)
    else:
        track_context = "No tracked objects"

    # ✅ Proper prompt
    prompt = f"""
            You are a drone security analyst reviewing surveillance footage.

            Timestamp: {frame['timestamp_sec']} seconds into the video.
            {track_context}

            STRICT INSTRUCTIONS:
            - You MUST classify activity into one of the following:
            1. NORMAL
            2. SUSPICIOUS
            3. HIGHLY SUSPICIOUS

            - Do NOT say "cannot determine" or "insufficient evidence".
            - Always make the BEST possible judgment based on visible cues.

            Describe:

            1. People Behavior:
            - What are they doing?
            - Any unusual or suspicious actions?

            2. Vehicles:
            - Type, movement, entry/exit patterns

            3. Security Concerns:
            - Explicitly list why this may be suspicious
            - Mention behaviors like:
                • loitering
                • repeated movement
                • hiding identity
                • object exchange
                • perimeter activity
                • following/monitoring

            4. Environment:
            - Lighting (day/night)
            - Visibility conditions

            FINAL OUTPUT FORMAT:

            Suspicion Level: <NORMAL / SUSPICIOUS / HIGHLY SUSPICIOUS>

            Reason:
            - Bullet points explaining why you classified it
            """

    # ✅ Correct Bedrock request format
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"].strip()


def run_frame_description(frames: list[dict], tracking_results: list[dict]) -> list[dict]:

    if not frames:
        print("No frames to describe.")
        return []

    # Build lookup: frame_id → tracks
    track_map = {
        t["frame_id"]: t.get("tracks", [])
        for t in tracking_results
    }

    results = []
    frames_dir = str(Path(frames[0]["filepath"]).parent)
    total = len(frames)

    for i, frame in enumerate(frames):
        frame_id = frame["frame_id"]
        tracks = track_map.get(frame_id, [])

        print(f"Describing {frame_id} ({i+1}/{total})...", end=" ", flush=True)

        description = describe_single_frame(bedrock_client, frame, tracks)

        print("done")
        results.append({**frame, "description": description})

    # Save output
    output_path = os.path.join(frames_dir, "descriptions.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDescriptions saved → {output_path}")

    return results