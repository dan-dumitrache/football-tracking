import cv2
import csv
import math
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_DIR
    / "data"
    / "input"
    / "rgb.avi"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUTPUT_DIR / "ball_detections.csv"


# ---------------------------------------------------------
# Red HSV thresholds
# ---------------------------------------------------------
#
# OpenCV HSV hue range is 0-179.
#
# Red wraps around the HSV hue range, therefore we use
# two ranges.
# ---------------------------------------------------------

LOWER_RED_1 = (0, 100, 80)
UPPER_RED_1 = (10, 255, 255)

LOWER_RED_2 = (170, 100, 80)
UPPER_RED_2 = (179, 255, 255)


# ---------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------

MIN_AREA = 30
MAX_AREA = 10000

MIN_CIRCULARITY = 0.20

MIN_ASPECT_RATIO = 0.40
MAX_ASPECT_RATIO = 2.50


# ---------------------------------------------------------
# Open video
# ---------------------------------------------------------

video = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not video.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = video.get(cv2.CAP_PROP_FPS)

width = int(
    video.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    video.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

frame_count = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)


print("Automatic ball detection")
print("------------------------")
print(f"Resolution: {width} x {height}")
print(f"FPS:        {fps:.2f}")
print(f"Frames:     {frame_count}")


# ---------------------------------------------------------
# CSV output
# ---------------------------------------------------------

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "frame",
        "time_seconds",
        "detected",
        "center_x",
        "center_y",
        "width",
        "height",
        "area",
        "circularity",
        "aspect_ratio"
    ])


    # -----------------------------------------------------
    # Process every frame
    # -----------------------------------------------------

    for frame_number in range(frame_count):

        success, frame = video.read()

        if not success:
            print(
                f"Could not read frame {frame_number}"
            )
            break


        # -------------------------------------------------
        # Convert to HSV
        # -------------------------------------------------

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )


        # -------------------------------------------------
        # Create red mask
        # -------------------------------------------------

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

        mask = cv2.bitwise_or(
            mask_1,
            mask_2
        )


        # -------------------------------------------------
        # Find contours
        # -------------------------------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        candidates = []


        # -------------------------------------------------
        # Analyze every contour
        # -------------------------------------------------

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < MIN_AREA:
                continue

            if area > MAX_AREA:
                continue


            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue


            circularity = (
                4.0
                * math.pi
                * area
                / (perimeter * perimeter)
            )


            if circularity < MIN_CIRCULARITY:
                continue


            x, y, w, h = cv2.boundingRect(
                contour
            )


            if h == 0:
                continue


            aspect_ratio = w / h


            if aspect_ratio < MIN_ASPECT_RATIO:
                continue

            if aspect_ratio > MAX_ASPECT_RATIO:
                continue


            center_x = x + w / 2.0
            center_y = y + h / 2.0


            # ---------------------------------------------
            # Reject objects touching the top border.
            #
            # We already observed a large red region from
            # the man's clothing around the top of the image.
            # ---------------------------------------------

            touches_top = y <= 2

            if touches_top:
                continue


            # ---------------------------------------------
            # Candidate score
            #
            # This is deliberately simple for now.
            #
            # Circularity is useful, but area also matters.
            # We use log(area) so a huge region doesn't
            # completely dominate the score.
            # ---------------------------------------------

            score = (
                3.0 * circularity
                + 0.5 * math.log(area)
            )


            candidates.append({
                "contour": contour,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "center_x": center_x,
                "center_y": center_y,
                "area": area,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "score": score
            })


        # -------------------------------------------------
        # Select best candidate
        # -------------------------------------------------

        candidates.sort(
            key=lambda c: c["score"],
            reverse=True
        )


        time_seconds = (
            frame_number / fps
            if fps > 0
            else 0
        )


        if candidates:

            best = candidates[0]

            writer.writerow([
                frame_number,
                f"{time_seconds:.4f}",
                1,
                f"{best['center_x']:.2f}",
                f"{best['center_y']:.2f}",
                best["w"],
                best["h"],
                f"{best['area']:.2f}",
                f"{best['circularity']:.4f}",
                f"{best['aspect_ratio']:.4f}"
            ])

        else:

            writer.writerow([
                frame_number,
                f"{time_seconds:.4f}",
                0,
                "",
                "",
                "",
                "",
                "",
                "",
                ""
            ])


video.release()


print()
print("Detection complete.")
print(f"Saved measurements to:")
print(CSV_PATH)