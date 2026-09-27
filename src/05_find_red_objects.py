import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_DIR / "data" / "output" / "frame_00100.jpg"
OUTPUT_PATH = PROJECT_DIR / "data" / "output" / "frame_00100_candidates.jpg"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError(f"Could not open image: {IMAGE_PATH}")

# Convert BGR → HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Red can appear at both ends of OpenCV's HSV hue range.
lower_red_1 = (0, 100, 80)
upper_red_1 = (10, 255, 255)

lower_red_2 = (170, 100, 80)
upper_red_2 = (179, 255, 255)

mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

mask = cv2.bitwise_or(mask_1, mask_2)

# Find connected regions.
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Found {len(contours)} red regions.")
print()

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:
        circularity = 4 * 3.14159 * area / (perimeter * perimeter)
    else:
        circularity = 0

    x, y, width, height = cv2.boundingRect(contour)

    print(f"Region {i}")
    print(f"  Area:        {area:.1f}")
    print(f"  Bounding box: x={x}, y={y}, w={width}, h={height}")
    print(f"  Circularity: {circularity:.3f}")
    print()

    # Draw the candidate boundary.
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"#{i}",
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

cv2.imwrite(str(OUTPUT_PATH), image)

print(f"Saved candidate image to:")
print(OUTPUT_PATH)