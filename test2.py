import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt") 

# Define lane zones (manually set based on the video frame)
LANE_ZONES = [(50, 200, 300, 400), (350, 200, 600, 400)]  # (x1, y1, x2, y2) for lanes

# Open video
cap = cv2.VideoCapture("Video/lane2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (400, 400))
    # Perform object detection
    results = model(frame)
    
    # Initialize vehicle count per lane
    lane_counts = [0] * len(LANE_ZONES)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            
            # Check which lane the vehicle is in
            for i, (lx1, ly1, lx2, ly2) in enumerate(LANE_ZONES):
                if lx1 < x1 < lx2 and ly1 < y1 < ly2:  # If vehicle is in the lane
                    lane_counts[i] += 1

    # Display lane vehicle counts
    for i, (lx1, ly1, lx2, ly2) in enumerate(LANE_ZONES):
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
        cv2.putText(frame, f"Count: {lane_counts[i]}", (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Traffic Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()