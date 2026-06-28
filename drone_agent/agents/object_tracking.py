"""
Run YOLOv8 on every extracted frames and return what was detected.

what this file does:
    - Takes the list of frame dicts produced by the extract_frames.py
    - Run YOLOv8 on every frame image
    - Return a new list of dicts -one person frame - with detection added

"""

from ultralytics import YOLO
from pathlib import Path
import os 
import json
import cv2



MODEL_PATH = "yolov8s.pt"

#confidence threshold : only keep detections above this score
# 0.4 means the "at least 40%" confident
# lower = more detection(more noise)  higher = fewer (might miss things)
CONNFIDENCE_THRESHOLD = 0.4

#Classes we care about for security monitoring 
#YOLO can detect the 80 classes. but we filter only relevent ones.
#none means keep all classes

SECURITY_CLASSES={
    "person","bicycle","car","motorcycle","bus","truck","boat","backpack","handbag","suitcase","weapon","sickle","gun","iron_rod","knife"
}


CLASS_COLORS = {
    "person":     (0, 0, 255),      # red
    "car":        (255, 0, 0),      # blue
    "truck":      (0, 165, 255),    # orange
    "motorcycle": (0, 255, 255),    # yellow
    "bicycle":    (255, 255, 0),    # cyan
    "bus":        (128, 0, 128),    # purple
    "boat":       (255, 128, 0),    # light blue
    "backpack":   (0, 128, 255),    # amber
 
}
DEFAULT_COLOR = (0, 255, 0)         # green for anything not in the list


def draw_boxes(image_path:str , detections: list[dict], output_path: str) -> None:
    """
    Draw bounding  boxes on the copy of the images and save it.
    """

    image =cv2.imread(image_path)
    if image is None:
        print(f"could not read the image : {image_path}")
        return
    

    for det in detections:
        label = det["label"]
        confidence=  det["confidence"]
        x1,y1,x2,y2 = [int(v) for v in det["bbox"]]
        color = CLASS_COLORS.get(label,DEFAULT_COLOR)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        # ── Draw label background ──────────────────────────────
        # A filled rectangle behind the text makes it readable
        # even on busy backgrounds
        label_text = f"{label} {confidence:.0%}"
        # Calculate text size so background box fits perfectly
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            thickness=1
        )
        # Draw filled rectangle as text background
        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            thickness=-1    # -1 = filled
        )
        # ── Draw label text ────────────────────────────────────
        cv2.putText(
            image,
            label_text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.55,
            color     = (255, 255, 255),   # white text
            thickness = 1,
            lineType  = cv2.LINE_AA        # anti-aliased = smoother text
        )
    cv2.imwrite(output_path, image)


def run_object_detection(frames: list[dict]) -> list[dict]:
    """
        Runs YOLOv8 on each frame and returns detections results

        Args:
            frames: list of frame dicts from extract_frames.py
            each dict has rame_id, filepath, timestamp_sec, etc

        Returns:
            list of dicts - same frames but with detections added.
            also save the detection.json file next to the frames

        Flow:
        frame image -> YOLO -> raw results -> filter by class -> confidence -> structure dict -> saved to JSON

    """

    if not frames:
        print(" No frames to process")
        return []

    print(" Loading YOLO mdoel...............")
    model = YOLO(MODEL_PATH)


    frames_dir     = str(Path(frames[0]["filepath"]).parent)
    annotated_dir  = os.path.join(frames_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
 
    results = []
    total_detected= 0

    for frame in frames:
        frame_id = frame["frame_id"]
        filepath = frame["filepath"]
        timestamp_sec= frame["timestamp_sec"]


        #skip if the image file doesn't exxist
        if not os.path.exists(filepath):
            print(f"skipping the {frame_id} - file not found:{filepath}")
            continue

        #Run the yolo on the images
        yolo_output = model(filepath, verbose=False)[0]          #verbose= False -> silence the yolo's own print output
        print(f"print the yolo output : {yolo_output}")
        
        #parse the yolo results
        detections =[]

        for box in yolo_output.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            #keep only high confidence detections
            if confidence < CONNFIDENCE_THRESHOLD:
                continue

            # bounding box : [x_min,y_min,x-max,y_max] i pixels
            bbox = [round(float(v),1) for v in box.xyxy[0]]

            detections.append({
                "label": label,
                "confidence": round(confidence,3),
                "bbox" : bbox
            })


        annotated_path = os.path.join(annotated_dir, Path(filepath).name)
        draw_boxes(filepath,detections,annotated_path)
        # collect unique object label for quick referece
        unique_labels = list ({d["label"] for d in detections})

        frame_result = {
            "frame_id": frame_id,
            "filepath": filepath,
            "timestamp_sec":timestamp_sec,
            "annotated_path":annotated_path,
            "detections": detections,
            "object_found": unique_labels,
            "has_person" : "person" in unique_labels,
            "total_objects": len(detections)

        }

        results.append(frame_result)
        total_detected += len(detections)

        # print the summary of each frame
        if detections:
            labels_str = ", ".join(
                f"{d['label']}({d['confidence']:.0%})" for d in detections
            )
            print(f"    {frame_id} [{timestamp_sec}s] → {labels_str}")
        else:
            print(f"    {frame_id} [{timestamp_sec}s] → nothing detected")


   # ── Save all detections to JSON ───────────────────────────
   # We save next to the frames so everything for one video stays together
    if results:
        output_dir       = str(Path(results[0]["filepath"]).parent)
        detections_path  = os.path.join(output_dir, "detections.json")
        with open(detections_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n    Detections saved → {detections_path}")
    print(f"    Total objects detected across all frames: {total_detected}")
    return results