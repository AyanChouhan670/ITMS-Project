# Smart Traffic Management System - Demo

A single-lane traffic monitoring system using YOLO object detection for vehicle counting and traffic light control.

## Features

- 🚦 **Single Lane Monitoring**: Focused on one traffic lane for demo purposes
- 🤖 **YOLO Vehicle Detection**: Uses YOLOv8 nano model for real-time vehicle detection
- 📊 **Real-time Statistics**: Live vehicle count and signal timing
- 🎮 **Interactive Controls**: Manual traffic light toggle and auto mode
- 🌐 **Web Interface**: Clean, responsive web dashboard
- 🚀 **Render Ready**: Optimized for free-tier deployment

## Demo Mode

This version is optimized for demonstration and deployment:
- **Single video stream** (lane1.mp4)
- **Single-threaded** for Render free-tier compatibility
- **Memory efficient** - won't crash on limited resources
- **Full demo** available locally for recruiters

## Local Development

### Prerequisites
- Python 3.11+
- OpenCV
- YOLO model (automatically downloaded)

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd ITMS-Project

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Local Demo
For full multi-lane demo with recruiters:
1. Modify `app.py` to use multiple video streams
2. Enable threading for better performance
3. Use all available video files

## Deployment on Render

### Free Tier Deployment
1. **Connect Repository**: Link your GitHub repo to Render
2. **Auto-deploy**: Render will use `render.yaml` for configuration
3. **Environment**: Python 3.11 with single worker/thread
4. **Memory**: Optimized for 512MB free tier

### Manual Deployment
```bash
# Build command
pip install -r requirements.txt

# Start command
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1
```

### Deployment Fix
If you encounter build errors:
- **Python Version**: Use Python 3.11 (specified in `runtime.txt`)
- **Dependencies**: Simplified requirements.txt for better compatibility
- **Build Issues**: Check Render logs for specific error messages

## Technical Details

### Architecture
- **Flask Backend**: Lightweight web framework
- **OpenCV**: Video processing and frame handling
- **YOLOv8 Nano**: Efficient object detection model
- **Single Thread**: Memory-efficient processing

### Performance Optimizations
- Frame resizing (320x240) for faster processing
- Disabled verbose logging in production
- Single video stream processing
- Efficient memory management

### API Endpoints
- `/`: Main dashboard
- `/video_feed`: Live video stream
- `/traffic_data`: Real-time traffic statistics
- `/toggle_light`: Manual traffic light control
- `/update_timer`: Timer management
- `/health`: Health check for monitoring
- `/restart`: Restart YOLO model if needed

## File Structure

```
ITMS-Project/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies (simplified)
├── runtime.txt         # Python version specification
├── render.yaml        # Render deployment config
├── yolov8n.pt         # YOLO model (kept in repo)
├── Video/
│   └── lane1.mp4      # Demo video (kept in repo)
└── templates/
    └── index.html     # Web interface
```

## Usage

1. **Access Dashboard**: Open the web interface
2. **Monitor Traffic**: Watch real-time vehicle detection
3. **Control Lights**: Use toggle button for manual control
4. **Auto Mode**: Enable automatic traffic light timing
5. **View Stats**: Monitor vehicle count and signal duration

## Troubleshooting

### Common Issues
- **Build Failures**: Ensure Python 3.11 is specified
- **Memory Errors**: App automatically falls back to demo mode
- **Video Issues**: Check if lane1.mp4 exists in Video folder

### Render Deployment
- **Python Version**: 3.11.7 (specified in render.yaml)
- **Dependencies**: Simplified requirements.txt
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1`

## Notes for Recruiters

- **Local Demo**: For full multi-lane experience, run locally
- **Production Ready**: Single-lane version deployed on Render
- **Scalable**: Architecture supports expansion to multiple lanes
- **Professional**: Clean code with proper error handling

## License

This project is for demonstration purposes.
