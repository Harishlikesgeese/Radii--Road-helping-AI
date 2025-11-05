# road_hazard_master.py
import os
import cv2
from ultralytics import YOLO

# ===== MODEL SETUP =====
# Use your own trained YOLOv8 model if available
MODEL_PATH = "yolov8n.pt"  # replace with custom model if needed
model = YOLO(MODEL_PATH)
SCORE_THRESH = 0.35

# ===== HAZARD DEFINITIONS AND FIXES =====
ROAD_HAZARDS = {
    "person": "Pedestrian on road — possible jaywalking risk.",
    "car": "Vehicle detected — normal unless parked oddly.",
    "truck": "Large vehicle — risk if stationary or sideways.",
    "bus": "Bus detected — check lane position.",
    "motorcycle": "Two-wheeler — possible instability or weaving.",
    "bicycle": "Cyclist detected — maintain safe distance.",
    "stop sign": "Intersection ahead — ensure stop compliance.",
    "traffic light": "Traffic signal ahead — obey lights.",
    "fire": "🔥 Fire detected — possible accident or hazard zone!",
    "smoke": "⚠️ Smoke — possible fire or mechanical issue!",
    "pothole": "⚠️ Pothole detected — road damage risk!",
    "debris": "⚠️ Debris detected — obstacle hazard!",
    "crash": "🚨 Accident detected!",
    "crowd": "Crowd on road — possible obstruction or event."
}

# Suggested FIXES for hazards
FIX_MAP = {
    "Pedestrian on road — possible jaywalking risk.": [
        "Install pedestrian barriers or crossings.",
        "Enhance road lighting for pedestrian visibility."
    ],
    "Vehicle detected — normal unless parked oddly.": [
        "Ensure parked cars are in safe zones.",
        "Add no-parking signs in unsafe areas."
    ],
    "Large vehicle — risk if stationary or sideways.": [
        "Add reflective markers and clear lane boundaries.",
        "Alert traffic officers for lane blockage."
    ],
    "Two-wheeler — possible instability or weaving.": [
        "Install rumble strips or slow lanes for bikes."
    ],
    "Cyclist detected — maintain safe distance.": [
        "Add dedicated bike lanes or warning boards."
    ],
    "⚠️ Pothole detected — road damage risk!": [
        "Schedule road repair immediately.",
        "Install temporary warning sign."
    ],
    "⚠️ Debris detected — obstacle hazard!": [
        "Remove debris and inspect area.",
        "Place cones or caution tape temporarily."
    ],
    "🚨 Accident detected!": [
        "Call emergency services immediately.",
        "Divert traffic and clear wreckage."
    ],
    "Crowd on road — possible obstruction or event.": [
        "Redirect traffic or deploy crowd control."
    ],
    "🔥 Fire detected — possible accident or hazard zone!": [
        "Alert fire department and evacuate area."
    ],
    "⚠️ Smoke — possible fire or mechanical issue!": [
        "Investigate source and warn drivers."
    ],
}

# ===== IMAGE ANALYSIS =====
def analyze_image(image_path):
    if not os.path.exists(image_path):
        return f"Error: {image_path} not found."

    results = model(image_path)
    detected = []
    for r in results:
        for c in r.boxes.cls:
            label = model.names[int(c)]
            if label in ROAD_HAZARDS:
                hazard = ROAD_HAZARDS[label]
                detected.append({
                    "label": label,
                    "hazard": hazard,
                    "fix": FIX_MAP.get(hazard, ["No suggested fix available."])
                })
    if not detected:
        return {"status": "No hazards found", "message": "Road appears clear."}
    return {"status": "Hazards detected", "details": detected}


# ===== VIDEO ANALYSIS =====
def analyze_video(video_path, frame_skip=15):
    if not os.path.exists(video_path):
        return f"Error: {video_path} not found."

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    hazards = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % frame_skip != 0:
            continue

        results = model(frame, conf=SCORE_THRESH)
        for r in results:
            for c in r.boxes.cls:
                label = model.names[int(c)]
                if label in ROAD_HAZARDS:
                    hazard = ROAD_HAZARDS[label]
                    hazards[hazard] = FIX_MAP.get(hazard, ["No suggested fix available."])

    cap.release()
    if not hazards:
        return {"status": "No hazards found", "summary": "Video appears clear."}
    return {"status": "Hazards detected", "details": hazards}


# ===== LIVE CAMERA (REAL-TIME) ANALYSIS =====
def live_detection(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam or camera source.")

    print("Press 'q' to quit live detection.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=SCORE_THRESH)
        for r in results:
            if r.boxes is None:
                continue
            for box_obj in r.boxes:
                conf = float(box_obj.conf.cpu().numpy())
                cls_id = int(box_obj.cls.cpu().numpy())
                box = box_obj.xyxy.cpu().numpy()[0]
                label = r.names[cls_id]
                if label in ROAD_HAZARDS:
                    hazard = ROAD_HAZARDS[label]
                    fix = FIX_MAP.get(hazard, ["No fix available."])[0]
                    # Draw box + label
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {hazard}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Fix: {fix}", (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("Live Road Hazard Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===== MAIN TEST =====
if __name__ == "__main__":
    # Test on image
    print("\n--- IMAGE TEST ---")
    print(analyze_image("road_test_image.jpg"))

    # Test on video
    print("\n--- VIDEO TEST ---")
    print(analyze_video("road_footage.mp4"))

    # Live camera
    print("\n--- STARTING LIVE DETECTION ---")
    live_detection()
