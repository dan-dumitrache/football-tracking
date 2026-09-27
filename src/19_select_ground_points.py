"""
19_select_ground_points.py

Interactive selection of four points on a floor tile.

The selected four image points are assumed to correspond to
the four corners of a single approximately square floor tile.

Working assumption:
    tile width  = 0.60 m
    tile height = 0.60 m

IMPORTANT:
    0.60 m is a working hypothesis, NOT a verified measurement.

The script saves the selected image points and their assumed
world coordinates to:

    calibration/ground_points.json
"""

from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "rgb.avi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "calibration"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "ground_points.json"
)

# Frame where the football was previously confirmed absent.
FRAME_NUMBER = 300

# Working tile-size hypothesis.
TILE_WIDTH_M = 0.60
TILE_HEIGHT_M = 0.60


# ============================================================
# GLOBAL STATE
# ============================================================

selected_points = []

display_image = None


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    global selected_points
    global display_image

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Maximum four points.
    if len(selected_points) >= 4:
        return

    selected_points.append(
        (int(x), int(y))
    )

    print(
        f"Point {len(selected_points)}: "
        f"({x}, {y})"
    )

    # Redraw image.
    redraw_image()


# ============================================================
# REDRAW
# ============================================================

def redraw_image():

    global display_image

    display_image = original_image.copy()

    # --------------------------------------------------------
    # Draw selected points.
    # --------------------------------------------------------

    for i, (x, y) in enumerate(
        selected_points
    ):

        cv2.circle(
            display_image,
            (x, y),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            display_image,
            f"P{i + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Draw polygon after four points.
    # --------------------------------------------------------

    if len(selected_points) >= 2:

        pts = np.array(
            selected_points,
            dtype=np.int32,
        )

        cv2.polylines(
            display_image,
            [pts],
            False,
            (255, 0, 0),
            2,
        )

    if len(selected_points) == 4:

        pts = np.array(
            selected_points,
            dtype=np.int32,
        )

        cv2.polylines(
            display_image,
            [pts],
            True,
            (0, 255, 0),
            3,
        )

    # --------------------------------------------------------
    # Instructions.
    # --------------------------------------------------------

    cv2.putText(
        display_image,
        "Click four tile corners: "
        "P1 P2 P3 P4",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display_image,
        "ENTER = save    R = reset    ESC = quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(
        "Ground Plane Calibration",
        display_image,
    )


# ============================================================
# SAVE
# ============================================================

def save_points():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Image coordinates.
    # --------------------------------------------------------

    image_points = [
        {
            "name": f"P{i + 1}",
            "x_px": int(x),
            "y_px": int(y),
        }
        for i, (x, y)
        in enumerate(selected_points)
    ]

    # --------------------------------------------------------
    # World coordinates.
    #
    # We use:
    #
    # P1 = (0,   0)
    # P2 = (0.6, 0)
    # P3 = (0.6, 0.6)
    # P4 = (0,   0.6)
    #
    # X = sideways
    # Y = forward along the floor
    # --------------------------------------------------------

    world_points = [
        {
            "name": "P1",
            "x_m": 0.0,
            "y_m": 0.0,
        },
        {
            "name": "P2",
            "x_m": TILE_WIDTH_M,
            "y_m": 0.0,
        },
        {
            "name": "P3",
            "x_m": TILE_WIDTH_M,
            "y_m": TILE_HEIGHT_M,
        },
        {
            "name": "P4",
            "x_m": 0.0,
            "y_m": TILE_HEIGHT_M,
        },
    ]

    data = {
        "frame_number": FRAME_NUMBER,

        "assumption": {
            "tile_width_m": TILE_WIDTH_M,
            "tile_height_m": TILE_HEIGHT_M,
            "description": (
                "Working estimate based on visual comparison "
                "with a standard football. Tile dimensions "
                "are not independently verified."
            ),
        },

        "image_points": image_points,

        "world_points": world_points,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
        )

    print()
    print("=" * 70)
    print("GROUND CALIBRATION SAVED")
    print("=" * 70)

    print(
        f"File:\n{OUTPUT_JSON}"
    )

    print()

    print("Image points:")

    for point in image_points:

        print(
            f"  {point['name']}: "
            f"({point['x_px']}, "
            f"{point['y_px']})"
        )

    print()

    print("Assumed world coordinates:")

    for point in world_points:

        print(
            f"  {point['name']}: "
            f"({point['x_m']:.2f}, "
            f"{point['y_m']:.2f}) m"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global original_image

    print("=" * 70)
    print("STEP 19 - GROUND PLANE CALIBRATION")
    print("=" * 70)

    print()
    print(
        f"Video: {VIDEO_PATH}"
    )

    print(
        f"Frame: {FRAME_NUMBER}"
    )

    print(
        f"Working tile size: "
        f"{TILE_WIDTH_M:.2f} x "
        f"{TILE_HEIGHT_M:.2f} m"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The 60 cm tile size is only a working assumption."
    )

    print()
    print(
        "Select FOUR corners of ONE floor tile."
    )

    print(
        "Use this order:"
    )

    print(
        "  P1 -> top-left"
    )

    print(
        "  P2 -> top-right"
    )

    print(
        "  P3 -> bottom-right"
    )

    print(
        "  P4 -> bottom-left"
    )

    print()
    print(
        "Keyboard:"
    )

    print(
        "  ENTER = save"
    )

    print(
        "  R     = reset"
    )

    print(
        "  ESC   = quit"
    )

    # --------------------------------------------------------
    # Open video.
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if FRAME_NUMBER < 0 or FRAME_NUMBER >= total_frames:

        cap.release()

        raise ValueError(
            f"Frame {FRAME_NUMBER} is outside "
            f"video range 0 -> {total_frames - 1}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_NUMBER,
    )

    success, frame = cap.read()

    cap.release()

    if not success:

        raise RuntimeError(
            f"Could not read frame {FRAME_NUMBER}"
        )

    original_image = frame.copy()

    # --------------------------------------------------------
    # Display.
    # --------------------------------------------------------

    window_name = (
        "Ground Plane Calibration"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        1280,
        720,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    redraw_image()

    # --------------------------------------------------------
    # Interaction loop.
    # --------------------------------------------------------

    while True:

        key = cv2.waitKey(20) & 0xFF

        # ESC
        if key == 27:

            print()
            print(
                "Calibration cancelled."
            )

            break

        # R
        elif key in (
            ord("r"),
            ord("R"),
        ):

            selected_points.clear()

            print()
            print(
                "Points reset."
            )

            redraw_image()

        # ENTER
        elif key in (
            13,
            10,
        ):

            if len(selected_points) != 4:

                print()
                print(
                    f"You selected "
                    f"{len(selected_points)} points."
                )

                print(
                    "Exactly 4 points are required."
                )

                continue

            save_points()

            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()