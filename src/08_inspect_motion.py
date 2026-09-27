import cv2
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_DIR / "data" / "input" / "rgb.avi"


LOWER_RED_1 = (0, 100, 80)
UPPER_RED_1 = (10, 255, 255)

LOWER_RED_2 = (170, 100, 80)
UPPER_RED_2 = (179, 255, 255)


# Focus on the kick and the reappearance.
FRAME_RANGES = [
    range(140, 221, 5),
    range(370, 462, 5),
]


video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


for frame_range in FRAME_RANGES:

    for frame_number in frame_range:

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = video.read()

        if not success:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_1 = cv2.inRange(
            hsv,
            LOWER_RED_1,
            UPPER_RED_1
        )

        mask_2 = cv2.inRange(
            hsv,
            LOWER_RED_2,
            UPPER_RED_2
        )

        mask = cv2.bitwise_or(mask_1, mask_2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 50:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            circularity = (
                4 * 3.14159 * area
                / (perimeter * perimeter)
            )

            x, y, width, height = cv2.boundingRect(contour)

            aspect_ratio = width / height

            if not 0.5 <= aspect_ratio <= 2.0:
                continue

            if circularity < 0.25:
                continue

            center_x = x + width / 2
            center_y = y + height / 2

            candidates.append(
                (
                    area,
                    circularity,
                    width,
                    height,
                    center_x,
                    center_y
                )
            )

        candidates.sort(reverse=True)

        print(
            f"Frame {frame_number:3d}: ",
            end=""
        )

        if not candidates:
            print("NO CANDIDATES")
            continue

        for candidate in candidates[:3]:

            (
                area,
                circularity,
                width,
                height,
                center_x,
                center_y
            ) = candidate

            print(
                f"[center=({center_x:.1f}, {center_y:.1f}), "
                f"size={width}x{height}, "
                f"area={area:.1f}, "
                f"C={circularity:.3f}] ",
                end=""
            )

        print()


video.release()