# Render Deployment Guide

## Quick Deploy Steps

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Add analytics and responsive UI"
   git push origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Sign in with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your `ITMS-Project` repository
   - Render will auto-detect the `render.yaml` configuration

3. **Deployment Settings** (auto-configured):
   - **Name**: smart-traffic-system
   - **Environment**: Python 3.11.7
   - **Plan**: Free
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1`

## Environment Variables (Auto-set)

- `FAST_MODE=1` - Enables fast streaming mode
- `INFER_EVERY=8` - Run YOLO every 8 frames (performance optimization)
- `COUNT_SMOOTH_ALPHA=0.4` - Smoothing factor for vehicle counts
- `ANALYTICS_POINTS=60` - Keep 60 data points (1 minute of history)

## Performance Optimizations for Free Tier

- **Single worker/thread** to avoid memory issues
- **Fast mode enabled** - streams raw frames with periodic YOLO detection
- **Reduced analytics history** - 60 points instead of 180
- **Optimized frame processing** - every 8th frame for YOLO
- **Lower JPEG quality** (72%) for faster streaming

## Expected Build Time

- **First deploy**: 3-5 minutes (downloads YOLO model)
- **Subsequent deploys**: 1-2 minutes

## Monitoring

- Check build logs for any errors
- Monitor memory usage in Render dashboard
- If crashes occur, the app will auto-restart

## Local Testing Before Deploy

```bash
# Test with same settings as Render
$env:FAST_MODE="1"; $env:INFER_EVERY="8"; $env:PORT="5000"
python app.py
```

## Troubleshooting

- **Build fails**: Check Python version compatibility
- **App crashes**: Reduce `INFER_EVERY` to 12 or 16
- **Slow performance**: Increase `INFER_EVERY` to 6 or 4
- **Memory issues**: Set `ANALYTICS_POINTS=30`

Your app will be available at: `https://smart-traffic-system.onrender.com`
