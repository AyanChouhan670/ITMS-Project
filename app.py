from flask import Flask, render_template, Response, jsonify
import cv2
import time
import os
import gc
import traceback
from ultralytics import YOLO
import urllib.request

app = Flask(__name__)

# Global variables with safe defaults
traffic_light = "RED"
vehicle_count = 0
signal_time = 10
current_timer = 0
model = None
cap = None
yolo_available = False

def safe_yolo_load():
    """Safely load YOLO model with error handling"""
    global model, yolo_available
    try:
        # Ensure YOLO model exists
        MODEL_PATH = "yolov8n.pt"
        if not os.path.exists(MODEL_PATH):
            print("Downloading YOLOv8 model...")
            url = "https://github.com/ultralytics/ultralytics/releases/download/v8.0.50/yolov8n.pt"
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("Download complete!")
        
        # Load YOLO model with memory optimization
        model = YOLO(MODEL_PATH)
        yolo_available = True
        print("YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        print("Running in demo mode without vehicle detection")
        yolo_available = False
        return False

def safe_video_load():
    """Safely load video with error handling"""
    global cap
    try:
        video_path = "Video/lane1.mp4"
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file")
            return False
        
        print("Video loaded successfully")
        return True
    except Exception as e:
        print(f"Failed to load video: {e}")
        return False

def detect_vehicles_safe():
    """Safe vehicle detection with fallback"""
    global vehicle_count, signal_time, current_timer
    
    try:
        if cap is None or not cap.isOpened():
            return None, 0
        
        ret, frame = cap.read()
        if not ret:
            # Reset video when it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return None, 0

        # Resize frame for better performance and memory usage
        frame = cv2.resize(frame, (320, 240))  # Smaller size for memory efficiency
        
        # Only run YOLO if available and not too frequently
        if yolo_available and model is not None:
            try:
                # Run YOLO detection with memory management
                results = model(frame, verbose=False, conf=0.3)  # Lower confidence threshold
                
                # Count vehicles (class 2 is car, class 3 is motorcycle, class 5 is bus, class 7 is truck)
                vehicle_count = 0
                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.cls is not None and len(box.cls) > 0:
                            class_id = int(box.cls[0])
                            if class_id in [2, 3, 5, 7]:  # Vehicle classes
                                vehicle_count += 1
                
                # Draw bounding boxes on frame
                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Clean up results to free memory
                del results
                gc.collect()
                
            except Exception as e:
                print(f"YOLO detection failed: {e}")
                vehicle_count = max(0, vehicle_count - 1) if vehicle_count > 0 else 0
        else:
            # Demo mode - simulate vehicle detection
            vehicle_count = (vehicle_count + 1) % 10  # Cycle through 0-9 for demo
        
        # Calculate signal time based on vehicle count
        signal_time = max(10, min(vehicle_count * 5, 60))
        
        # Add vehicle count text
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add mode indicator
        mode_text = "YOLO Mode" if yolo_available else "Demo Mode"
        cv2.putText(frame, mode_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame, vehicle_count
        
    except Exception as e:
        print(f"Error in detect_vehicles_safe: {e}")
        traceback.print_exc()
        return None, 0

def generate_frames_safe():
    """Safe frame generation with error handling"""
    frame_count = 0
    while True:
        try:
            frame_count += 1
            
            # Process every 3rd frame to reduce CPU usage
            if frame_count % 3 != 0:
                time.sleep(0.1)
                continue
            
            frame, _ = detect_vehicles_safe()
            if frame is None:
                time.sleep(0.5)
                continue

            # Compress frame with lower quality for memory efficiency
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in generate_frames_safe: {e}")
            time.sleep(1)
            continue

@app.route('/')
def index():
    return render_template('index.html', 
                         light=traffic_light, 
                         vehicle_count=vehicle_count, 
                         signal_time=signal_time, 
                         timer=current_timer,
                         yolo_available=yolo_available)

@app.route('/traffic_data')
def traffic_data():
    return {
        "light": traffic_light, 
        "signal_time": signal_time, 
        "vehicle_count": vehicle_count, 
        "timer": current_timer,
        "yolo_available": yolo_available
    }

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_safe(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_light')
def toggle_light():
    global traffic_light, current_timer
    try:
        if traffic_light == "RED":
            traffic_light = "GREEN"
            current_timer = signal_time
        else:
            traffic_light = "RED"
            current_timer = 0
        return {"light": traffic_light, "timer": current_timer}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_timer')
def update_timer():
    global current_timer
    try:
        if traffic_light == "GREEN" and current_timer > 0:
            current_timer -= 1
            if current_timer <= 0:
                traffic_light = "RED"
        return {"light": traffic_light, "timer": current_timer}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy", "yolo_available": yolo_available}

@app.route('/restart')
def restart():
    """Restart YOLO model if it failed"""
    global yolo_available
    try:
        if safe_yolo_load():
            return {"status": "YOLO model restarted successfully"}
        else:
            return {"status": "Failed to restart YOLO model"}
    except Exception as e:
        return {"status": f"Error restarting: {str(e)}"}

if __name__ == '__main__':
    print("Starting Smart Traffic System...")
    
    # Initialize components safely
    print("Loading YOLO model...")
    safe_yolo_load()
    
    print("Loading video...")
    safe_video_load()
    
    print("Starting Flask app...")
    port = int(os.environ.get("PORT", 5000))  # Render dynamic port
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=False)
    except Exception as e:
        print(f"Failed to start Flask app: {e}")
        traceback.print_exc()
