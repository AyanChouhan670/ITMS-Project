# ITMS-Project

**Intelligent Traffic Management System (ITMS)**

This project detects vehicles in four lanes using **YOLOv8** and dynamically controls traffic lights based on vehicle counts. It is built with **Python, Flask, OpenCV, and Ultralytics YOLO**.

## Features
- Real-time vehicle detection from multiple video streams
- Counts vehicles in each lane
- Dynamically adjusts traffic light timings based on traffic density
- Live video feed with bounding boxes for detected vehicles
- Traffic data available via Flask web interface

## Folder Structure
```
ITMS-Project/
│
├─ app.py                 # Main Flask application
├─ requirements.txt       # Python dependencies
├─ README.md              # Project description
├─ Video/                 # Lane videos (lane1.mp4, lane2.mp4, etc.)
├─ templates/             # HTML templates (index.html)
└─ static/                # Optional CSS/images
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/AyanChouhan670/ITMS-Project.git
cd ITMS-Project
```

2. Create a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project
1. Make sure the `Video/` folder contains your lane videos (`lane1.mp4`, `lane2.mp4`, `lane3.mp4`, `lane4.mp4`).  
2. Run the Flask app:
```bash
python app.py
```

3. Open your browser and go to:
```
http://127.0.0.1:5000
```

You will see live video
