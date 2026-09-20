import cv2
from pathlib import Path


# Find the project directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Build the path to the input video.
VIDEO_PATH = PROJECT_DIR / "data" / "input" / "rgb.avi"


# Open the video.
video = cv2.VideoCapture(str(VIDEO_PATH))


# Check whether OpenCV managed to open it.
if not video.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


# Read video properties.
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


# Calculate duration.
duration = frame_count / fps if fps > 0 else 0


print("Video inspection")
print("----------------")
print(f"File:       {VIDEO_PATH}")
print(f"Resolution: {width} x {height}")
print(f"FPS:        {fps:.2f}")
print(f"Frames:     {frame_count}")
print(f"Duration:   {duration:.2f} seconds")


# Release the video resource.
video.release()