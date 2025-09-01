import cv2
from ultralytics import YOLO

# Load YOLO model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")

# Open video stream (0 for webcam or path to video file)
cap = cv2.VideoCapture("Video/lane2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (400, 400))
    # Perform object detection
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


