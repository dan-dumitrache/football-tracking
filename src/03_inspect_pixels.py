import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_DIR / "data" / "output" / "frame_00100.jpg"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError(f"Could not open image: {IMAGE_PATH}")

# OpenCV loads images as BGR, not RGB.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("Image inspection")
print("----------------")
print(f"Width:  {image.shape[1]}")
print(f"Height: {image.shape[0]}")

# Example: inspect the pixel at the approximate ball location.
x = 700
y = 150

bgr_pixel = image[y, x]
hsv_pixel = hsv[y, x]

print()
print(f"Pixel at (x={x}, y={y})")
print(f"BGR: {bgr_pixel}")
print(f"HSV: {hsv_pixel}")