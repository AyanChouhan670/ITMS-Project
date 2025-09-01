from flask import Flask, render_template, Response
import cv2
import time
import threading
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video streams
videos = [
    cv2.VideoCapture("Video/lane3.mp4"),
    cv2.VideoCapture("Video/lane2.mp4"), 
    cv2.VideoCapture("Video/lane1.mp4"),
    cv2.VideoCapture("Video/lane4.mp4"),
]

traffic_lights = ["RED", "RED", "RED", "RED"]
lane_counts = [0] * 4
signal_times = [10] * 4
current_timer = [0]
lock = threading.Lock()

def detect_vehicles():
    """Detect vehicles in each lane, draw bounding boxes, and update traffic logic."""
    global lane_counts, signal_times
    results_list = [0] * len(videos)

    def process_lane(i, cap):
        ret, frame = cap.read()
        if not ret:
            return None, 0
        
        frame = cv2.resize(frame, (400, 400))
        results = model(frame)

        vehicle_count = len(results[0].boxes)
        results_list[i] = vehicle_count

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame, vehicle_count
    
    threads = []
    frames = [None] * len(videos)

    for i, cap in enumerate(videos):
        thread = threading.Thread(target=lambda idx, cap: (frames.__setitem__(idx, process_lane(idx, cap)[0]), results_list.__setitem__(idx, process_lane(idx, cap)[1])), args=(i, cap))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    with lock:
        lane_counts[:] = results_list
        total_vehicles = sum(lane_counts)
        signal_times[:] = [max(10, min(int(count / total_vehicles * 60), 60)) if total_vehicles > 0 else 10 for count in lane_counts]

    return frames

        
def traffic_light_control():
    """Control traffic lights based on priority of vehicle count."""
    global traffic_lights, current_timer
    while True:
        detect_vehicles()
        time.sleep(5)

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
    """Generate video frames with bounding boxes and vehicle count."""
    while True:
        frames = detect_vehicles()
        if frames is None or frames[lane_id] is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frames[lane_id])
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', lights=traffic_lights, lane_counts=lane_counts, signal_times=signal_times, timer=current_timer)

@app.route('/traffic_data')
def traffic_data():
    return {"lights": traffic_lights, "signal_times": signal_times, "lane_counts": lane_counts, "timer": current_timer[0]}

@app.route('/video_feed/<int:lane_id>')
def video_feed(lane_id):
    return Response(generate_frames(lane_id), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    threading.Thread(target=traffic_light_control, daemon=True).start()
    app.run(debug=True, threaded=True)
