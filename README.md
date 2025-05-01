# Hand Gesture Video Recorder with Overlay Icons 🎥✋

A Python application using OpenCV and MediaPipe that detects specific finger gestures through a webcam and overlays transparent PNG icons on the fingers. The app also allows video recording using a simple keyboard shortcut.

## 🔧 Features

- Detects index, middle, and ring fingers using MediaPipe
- Displays a custom icon above each lifted finger
- Press `R` to start/stop recording video
- Press `ESC` to exit the application
- Automatically saves the recorded video with timestamped filename

## 📦 Requirements

Make sure you have the following installed:

```bash
pip install opencv-python mediapipe numpy
```

## ▶️ How to Run

```bash
python main.py
```

