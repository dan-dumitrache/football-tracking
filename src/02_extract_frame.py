import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_DIR / "data" / "input" / "rgb.avi"
OUTPUT_PATH = PROJECT_DIR / "data" / "output" / "frame_00100.jpg"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_number = 100

video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
success, frame = video.read()

video.release()

if not success:
    raise RuntimeError(f"Could not read frame {frame_number}")

cv2.imwrite(str(OUTPUT_PATH), frame)

print(f"Saved frame {frame_number} to:")
print(OUTPUT_PATH)