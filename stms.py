import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open four different videos
videos = [
    cv2.VideoCapture("Video/lane1.mp4"),
    cv2.VideoCapture("Video/lane2.mp4"),
    cv2.VideoCapture("Video/lane3.mp4"),
    cv2.VideoCapture("Video/lane4.mp4"),
]

# Process videos frame by frame
while True:
    lane_counts = []  

    for i, cap in enumerate(videos):
        ret, frame = cap.read()
        if not ret:
            continue  

        # Resize frame 
        frame = cv2.resize(frame, (400, 400))

        # Perform vehicle detection
        results = model(frame)
        vehicle_count = 0  
        
        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  
            label = results[0].names[int(detection.cls[0])] 
            confidence = detection.conf[0] 

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            vehicle_count += 1  
        
        lane_counts.append(vehicle_count)

        # Draw vehicle count on the frame
        cv2.putText(frame, f"Lane {i+1}: {vehicle_count} vehicles", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow(f"Lane {i+1}", frame)

    # Calculate dynamic signal times
    total_vehicles = sum(lane_counts)
    signal_times = [(count / total_vehicles * 60) if total_vehicles > 0 else 10 for count in lane_counts]
    signal_times = [max(10, min(t, 60)) for t in signal_times] 

    print(f"Signal Times: {signal_times}") 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in videos:
    cap.release()
cv2.destroyAllWindows()
