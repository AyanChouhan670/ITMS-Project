from flask import Flask, render_template, Response, jsonify
import cv2
import time
import threading
import os
from ultralytics import YOLO
import urllib.request
from collections import deque

# Disable ffmpeg/OpenCV internal threading to avoid assertion crashes
os.environ["OPENCV_FFMPEG_THREAD_NUMBER"] = "1"
try:
    cv2.setNumThreads(1)
except Exception:
    pass

app = Flask(__name__)

# Fast mode streams raw frames but still counts with YOLO at a lower frequency
FAST_MODE = os.environ.get("FAST_MODE", "1") == "1"
INFER_EVERY = int(os.environ.get("INFER_EVERY", "8" if FAST_MODE else "3"))  # run YOLO every N frames per lane

# Ensure YOLO model exists
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv8 model...")
    url = "https://github.com/ultralytics/ultralytics/releases/download/v8.0.50/yolov8n.pt"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Vehicle class IDs (COCO): car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Resolve video paths with fallback (Video/ or templates/Video/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_DIRS = [
    os.path.join(BASE_DIR, "Video"),
    os.path.join(BASE_DIR, "templates", "Video"),
]
LANE_FILES = ["lane3.mp4", "lane2.mp4", "lane1.mp4", "lane4.mp4"]

def resolve_video_path(filename):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

def open_captures():
    captures = []
    for idx, name in enumerate(LANE_FILES):
        path = resolve_video_path(name)
        if path is None:
            print(f"Error: video file not found for lane {idx}: tried 'Video/{name}' and 'templates/Video/{name}'")
            captures.append(None)
            continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Warning: failed to open video for lane {idx}: {path}")
            captures.append(None)
        else:
            print(f"Opened lane {idx}: {path}")
            captures.append(cap)
    return captures

# Open video streams with fallback
videos = open_captures()

traffic_lights = ["RED", "RED", "RED", "RED"]
lane_counts = [0] * 4
signal_times = [10] * 4
current_timer = [0]
lock = threading.Lock()

# New globals for summary
total_vehicles = 0
priority_lane = 0  # 0-based index

# Latest frames shared by all generators (avoid multiple reads)
latest_frames = [None, None, None, None]
# Per-lane frame index to throttle inference
lane_frame_index = [0, 0, 0, 0]
# Store latest detections (list of boxes per lane) to draw rectangles on frames between inferences
latest_boxes = [[], [], [], []]  # each item: list of (x1,y1,x2,y2)
# Smoothed counts (EMA) per lane for stable UI
smoothed_counts = [0.0, 0.0, 0.0, 0.0]
EMA_ALPHA = float(os.environ.get("COUNT_SMOOTH_ALPHA", "0.4"))  # 0..1; higher=more responsive

# Analytics: rolling history of totals and per-lane counts (seconds)
ANALYTICS_MAXLEN = int(os.environ.get("ANALYTICS_POINTS", "180"))  # last 3 minutes by default
analytics_history = deque(maxlen=ANALYTICS_MAXLEN)  # items: {ts, total, lanes:[c0..c3]}

# Target processed resolution (slightly larger for better display)
PROC_W, PROC_H = 480, 360


def process_lane(i, cap):
    if cap is None:
        return None, int(round(smoothed_counts[i]))

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            return None, int(round(smoothed_counts[i]))

    try:
        frame = cv2.resize(frame, (PROC_W, PROC_H))
    except Exception:
        return None, int(round(smoothed_counts[i]))

    # Decide whether to run YOLO this frame
    run_inference = (lane_frame_index[i] % INFER_EVERY == 0)
    lane_frame_index[i] += 1

    detected_count = int(round(smoothed_counts[i]))
    if run_inference:
        try:
            # Filter by vehicle classes for accuracy and speed
            results = model(frame, imgsz=PROC_W, conf=0.35, iou=0.5, classes=list(VEHICLE_CLASS_IDS), verbose=False)
        except Exception:
            results = []
        boxes = []
        try:
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                for idx_box, b in enumerate(results[0].boxes.xyxy):
                    cls_id = None
                    try:
                        cls_id = int(results[0].boxes.cls[idx_box]) if results[0].boxes.cls is not None else None
                    except Exception:
                        cls_id = None
                    if cls_id is None or cls_id in VEHICLE_CLASS_IDS:
                        x1, y1, x2, y2 = map(int, b[:4])
                        boxes.append((x1, y1, x2, y2))
                detected_count = len(boxes)
        except Exception:
            pass
        latest_boxes[i] = boxes

    # Smooth the count (EMA) for stable UI
    smoothed_counts[i] = (1.0 - EMA_ALPHA) * smoothed_counts[i] + EMA_ALPHA * float(detected_count)
    stable_count = int(round(smoothed_counts[i]))

    # Draw rectangles from latest boxes (even if no fresh inference)
    try:
        for (x1, y1, x2, y2) in latest_boxes[i][:60]:  # cap drawing for performance
            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
    except Exception:
        pass

    return frame, stable_count


def detect_vehicles():
    global lane_counts, signal_times, total_vehicles, priority_lane
    results_list = [0] * len(videos)
    frames = [None] * len(videos)

    # Process lanes sequentially (single reader)
    for i, cap in enumerate(videos):
        frame, count = process_lane(i, cap)
        frames[i] = frame
        results_list[i] = count

    with lock:
        lane_counts[:] = results_list
        total_vehicles = sum(lane_counts)
        # Estimated time proportional to counts, bounded [10, 60]
        signal_times[:] = [max(10, min(int(count / total_vehicles * 60), 60)) if total_vehicles > 0 else 10 for count in lane_counts]
        # Determine priority lane (max vehicles)
        priority_lane = max(range(len(lane_counts)), key=lambda i: lane_counts[i]) if len(lane_counts) > 0 else 0

    return frames


def processing_loop():
    # Single loop to update latest_frames for all lanes
    while True:
        frames = detect_vehicles()
        with lock:
            for i in range(len(latest_frames)):
                latest_frames[i] = None if frames is None else frames[i]
        time.sleep(0.03)


def analytics_loop():
    # Record one snapshot per second for analytics
    while True:
        with lock:
            snapshot = {
                "ts": int(time.time()),
                "total": int(total_vehicles),
                "lanes": list(map(int, lane_counts))
            }
            analytics_history.append(snapshot)
        time.sleep(1)


def traffic_light_control():
    global traffic_lights, current_timer
    while True:
        time.sleep(3)
        with lock:
            sorted_lanes = sorted(range(4), key=lambda i: lane_counts[i], reverse=True)
        for i in sorted_lanes:
            with lock:
                for j in range(4):
                    traffic_lights[j] = "RED"
                traffic_lights[i] = "GREEN"
                current_timer[0] = signal_times[i]
            while current_timer[0] > 0:
                time.sleep(1)
                current_timer[0] -= 1


def generate_frames(lane_id):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 72]
    while True:
        with lock:
            frame = latest_frames[lane_id]
        if frame is None:
            time.sleep(0.02)
            continue
        try:
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                time.sleep(0.01)
                continue
            frame_bytes = buffer.tobytes()
        except Exception:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', lights=traffic_lights, lane_counts=lane_counts, signal_times=signal_times, timer=current_timer)

@app.route('/traffic_data')
def traffic_data():
    with lock:
        return {
            "lights": traffic_lights,
            "signal_times": signal_times,
            "lane_counts": lane_counts,
            "timer": current_timer[0],
            "total_vehicles": total_vehicles,
            "priority_lane": priority_lane
        }

@app.route('/analytics_data')
def analytics_data():
    # Return recent analytics with relative timestamps (seconds ago)
    with lock:
        items = list(analytics_history)
    if not items:
        return jsonify({"times": [], "total": [], "lanes": [[], [], [], []], "avg_total": 0, "peak_lane": 0, "peak_total": 0})
    now = int(time.time())
    times = [now - it["ts"] for it in items]  # seconds ago
    totals = [it["total"] for it in items]
    lanes_series = [
        [it["lanes"][0] for it in items],
        [it["lanes"][1] for it in items],
        [it["lanes"][2] for it in items],
        [it["lanes"][3] for it in items],
    ]
    avg_total = int(sum(totals) / max(1, len(totals)))
    # Determine peak in the latest snapshot
    last = items[-1]
    peak_lane = max(range(4), key=lambda i: last["lanes"][i])
    peak_total = last["lanes"][peak_lane]
    return jsonify({
        "times": times,  # seconds ago
        "total": totals,
        "lanes": lanes_series,
        "avg_total": avg_total,
        "peak_lane": peak_lane,
        "peak_total": peak_total
    })

@app.route('/video_feed/<int:lane_id>')
def video_feed(lane_id):
    return Response(generate_frames(lane_id), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print(f"FAST_MODE={'ON' if FAST_MODE else 'OFF'}; INFER_EVERY={INFER_EVERY}; SIZE={PROC_W}x{PROC_H}; EMA_ALPHA={EMA_ALPHA}")
    threading.Thread(target=processing_loop, daemon=True).start()
    threading.Thread(target=analytics_loop, daemon=True).start()
    threading.Thread(target=traffic_light_control, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
