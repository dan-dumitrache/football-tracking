import cv2
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_DIR / "data" / "input" / "rgb.avi"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "candidate_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Frames we want to inspect.
FRAME_NUMBERS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]


# Red thresholds in OpenCV HSV.
LOWER_RED_1 = (0, 100, 80)
UPPER_RED_1 = (10, 255, 255)

LOWER_RED_2 = (170, 100, 80)
UPPER_RED_2 = (179, 255, 255)


video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


for frame_number in FRAME_NUMBERS:

    # Jump to the requested frame.
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = video.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    # Convert BGR → HSV.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the two red masks.
    mask_1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask_2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)

    # Combine them.
    mask = cv2.bitwise_or(mask_1, mask_2)

    # Find connected red regions.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore extremely tiny regions.
        if area < 10:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * 3.14159 * area
            / (perimeter * perimeter)
        )

        x, y, width, height = cv2.boundingRect(contour)

        aspect_ratio = width / height if height > 0 else 0

        candidates.append(
            {
                "contour": contour,
                "area": area,
                "circularity": circularity,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
            }
        )

    # Sort candidates by circularity.
    candidates.sort(
        key=lambda candidate: candidate["circularity"],
        reverse=True
    )

    print()
    print("=" * 60)
    print(f"FRAME {frame_number}")
    print("=" * 60)

    if not candidates:
        print("No candidates found.")
        continue

    # Show the five most circular candidates.
    for index, candidate in enumerate(candidates[:5]):

        print(
            f"Candidate {index}: "
            f"area={candidate['area']:.1f}, "
            f"size={candidate['width']}x{candidate['height']}, "
            f"aspect={candidate['aspect_ratio']:.2f}, "
            f"circularity={candidate['circularity']:.3f}, "
            f"position=({candidate['x']}, {candidate['y']})"
        )

        # Draw candidate rectangle.
        x = candidate["x"]
        y = candidate["y"]
        width = candidate["width"]
        height = candidate["height"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # Label candidate.
        cv2.putText(
            frame,
            f"#{index}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    output_path = (
        OUTPUT_DIR
        / f"frame_{frame_number:04d}_candidates.jpg"
    )

    cv2.imwrite(str(output_path), frame)

    print(f"Saved: {output_path}")


video.release()

print()
print("Inspection complete.")