import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_DIR / "data" / "output" / "frame_00100.jpg"
OUTPUT_PATH = PROJECT_DIR / "data" / "output" / "frame_00100_red_mask.png"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError(f"Could not open image: {IMAGE_PATH}")

# Convert BGR → HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Red appears at both ends of OpenCV's HSV hue range.
lower_red_1 = (0, 100, 80)
upper_red_1 = (10, 255, 255)

lower_red_2 = (170, 100, 80)
upper_red_2 = (179, 255, 255)

mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

mask = cv2.bitwise_or(mask_1, mask_2)

cv2.imwrite(str(OUTPUT_PATH), mask)

print("Red mask created.")
print(f"Saved to: {OUTPUT_PATH}")