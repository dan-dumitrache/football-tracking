"""
20_ground_homography.py

Calculate and validate a metric ground-plane homography.

INPUT
-----
calibration/ground_points.json
data/input/rgb.avi

OUTPUT
------
calibration/ground_homography.json

data/output/ground_plane/
    ground_plane_warp.png
    ground_plane_grid.png
    homography_validation.png

PURPOSE
-------
Map image pixels lying on the floor plane into a metric
ground-plane coordinate system.

The current working assumption is that the selected tile
measures:

    0.60 m x 0.60 m

IMPORTANT
---------
This metric scale is an assumption and has not been physically
measured.

The homography itself describes the planar geometry. The
absolute metric scale depends on the assumed tile dimensions.

Coordinate convention:

    X = horizontal direction across the tile
    Y = direction from P1/P2 toward P4/P3

World points:

    P1 -> (0.0, 0.0)
    P2 -> (0.6, 0.0)
    P3 -> (0.6, 0.6)
    P4 -> (0.0, 0.6)

The script also creates a perspective-warped image in which
the selected tile should appear approximately rectangular.
"""


from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "rgb.avi"
)

GROUND_POINTS_JSON = (
    PROJECT_ROOT
    / "calibration"
    / "ground_points.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "ground_plane"
)

HOMOGRAPHY_JSON = (
    PROJECT_ROOT
    / "calibration"
    / "ground_homography.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# The calibration frame saved in Step 19.
# ------------------------------------------------------------

FRAME_NUMBER = 300


# ------------------------------------------------------------
# Bird's-eye output resolution.
# ------------------------------------------------------------
#
# 1000 pixels per meter gives:
#
# 0.60 m × 0.60 m tile
#
# = 600 × 600 pixels.
#
# ------------------------------------------------------------

PIXELS_PER_METER = 1000


# ------------------------------------------------------------
# Ground-plane output extent.
#
# We show:
#
# X = -1.0 -> +3.0 m
# Y = -1.0 -> +4.0 m
#
# This gives some space around the calibration tile.
# ------------------------------------------------------------

X_MIN_M = -1.0
X_MAX_M = 3.0

Y_MIN_M = -1.0
Y_MAX_M = 4.0


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_ground_points():
    """
    Load the image/world point correspondences produced by
    Step 19.
    """

    if not GROUND_POINTS_JSON.exists():

        raise FileNotFoundError(
            "\nCould not find:\n"
            f"{GROUND_POINTS_JSON}\n\n"
            "Run first:\n"
            "python src\\19_select_ground_points.py"
        )

    with open(
        GROUND_POINTS_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    image_points = data.get(
        "image_points"
    )

    world_points = data.get(
        "world_points"
    )

    if image_points is None:

        raise ValueError(
            "ground_points.json does not contain "
            "'image_points'."
        )

    if world_points is None:

        raise ValueError(
            "ground_points.json does not contain "
            "'world_points'."
        )

    if len(image_points) != 4:

        raise ValueError(
            "Exactly four image points are required."
        )

    if len(world_points) != 4:

        raise ValueError(
            "Exactly four world points are required."
        )

    image_array = np.array(
        [
            [
                point["x_px"],
                point["y_px"],
            ]
            for point in image_points
        ],
        dtype=np.float32,
    )

    world_array = np.array(
        [
            [
                point["x_m"],
                point["y_m"],
            ]
            for point in world_points
        ],
        dtype=np.float32,
    )

    return (
        data,
        image_array,
        world_array,
    )


# ============================================================
# CALCULATE HOMOGRAPHY
# ============================================================

def calculate_homography(
    image_points,
    world_points,
):
    """
    Calculate:

        image -> ground-plane

    homography.

    H maps homogeneous image coordinates to homogeneous
    ground-plane coordinates.

        [X]
        [Y]
        [1]

        ~ H

        [u]
        [v]
        [1]
    """

    H, mask = cv2.findHomography(
        image_points,
        world_points,
        method=0,
    )

    if H is None:

        raise RuntimeError(
            "OpenCV could not calculate the homography."
        )

    if mask is not None:

        inliers = int(
            mask.sum()
        )

    else:

        inliers = 0

    # Normalize so H[2,2] = 1 when possible.
    if abs(H[2, 2]) > 1e-12:

        H = H / H[2, 2]

    return H, inliers


# ============================================================
# TEST HOMOGRAPHY
# ============================================================

def transform_points(
    H,
    image_points,
):
    """
    Transform image points into ground-plane coordinates.
    """

    points = (
        image_points
        .reshape(-1, 1, 2)
    )

    transformed = cv2.perspectiveTransform(
        points,
        H,
    )

    return transformed.reshape(
        -1,
        2,
    )


# ============================================================
# HOMOGRAPHY ERROR
# ============================================================

def calculate_reprojection_error(
    H,
    image_points,
    world_points,
):
    """
    Calculate the Euclidean error between the known world
    coordinates and the coordinates predicted by H.
    """

    predicted = transform_points(
        H,
        image_points,
    )

    errors = np.linalg.norm(
        predicted - world_points,
        axis=1,
    )

    return (
        predicted,
        errors,
    )


# ============================================================
# SAVE HOMOGRAPHY
# ============================================================

def save_homography(
    calibration_data,
    H,
    errors,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HOMOGRAPHY_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    assumption = calibration_data.get(
        "assumption",
        {},
    )

    output = {
        "description": (
            "Image-to-ground-plane homography "
            "calculated from four floor-tile corners."
        ),

        "frame_number": FRAME_NUMBER,

        "coordinate_system": {
            "X": (
                "horizontal direction from P1/P4 "
                "toward P2/P3"
            ),
            "Y": (
                "ground direction from P1/P2 "
                "toward P4/P3"
            ),
            "units": "meters",
        },

        "scale_assumption": assumption,

        "homography_image_to_ground": (
            H.tolist()
        ),

        "reprojection_error_m": [
            float(error)
            for error in errors
        ],

        "mean_reprojection_error_m": float(
            np.mean(errors)
        ),

        "max_reprojection_error_m": float(
            np.max(errors)
        ),
    }

    with open(
        HOMOGRAPHY_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    print()
    print(
        f"Saved homography:\n"
        f"{HOMOGRAPHY_JSON}"
    )


# ============================================================
# LOAD VIDEO FRAME
# ============================================================

def load_frame():

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_NUMBER,
    )

    success, frame = cap.read()

    cap.release()

    if not success:

        raise RuntimeError(
            f"Could not read frame "
            f"{FRAME_NUMBER}."
        )

    return frame


# ============================================================
# CREATE BIRD'S-EYE TRANSFORM
# ============================================================

def create_ground_transform():
    """
    Create the pixel coordinate system for the bird's-eye
    output image.

    Ground coordinates:

        X_MIN -> X_MAX
        Y_MIN -> Y_MAX

    are converted to output image pixels.

    Image coordinates have:

        origin = top-left
        x = right
        y = down

    Therefore increasing world Y is represented downward
    in the generated bird's-eye image.

    We intentionally keep this simple and explicit.
    """

    width = int(
        round(
            (X_MAX_M - X_MIN_M)
            * PIXELS_PER_METER
        )
    )

    height = int(
        round(
            (Y_MAX_M - Y_MIN_M)
            * PIXELS_PER_METER
        )
    )

    return width, height


# ============================================================
# WARP FRAME TO GROUND PLANE
# ============================================================

def warp_to_ground(
    frame,
    H,
):
    """
    Warp the camera image into metric ground-plane
    coordinates.
    """

    width, height = (
        create_ground_transform()
    )

    # --------------------------------------------------------
    # We have H:
    #
    # image -> world meters
    #
    # We need:
    #
    # image -> output pixels
    #
    # Construct a scale/translation matrix:
    #
    # pixel_x = (X - X_MIN) * ppm
    # pixel_y = (Y - Y_MIN) * ppm
    # --------------------------------------------------------

    ground_to_pixel = np.array(
        [
            [
                PIXELS_PER_METER,
                0.0,
                -X_MIN_M
                * PIXELS_PER_METER,
            ],
            [
                0.0,
                PIXELS_PER_METER,
                -Y_MIN_M
                * PIXELS_PER_METER,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    image_to_pixel = (
        ground_to_pixel
        @ H
    )

    warped = cv2.warpPerspective(
        frame,
        image_to_pixel,
        (
            width,
            height,
        ),
        flags=cv2.INTER_LINEAR,
    )

    return warped


# ============================================================
# DRAW GROUND GRID
# ============================================================

def draw_grid(
    image,
):
    """
    Draw a metric grid over the bird's-eye image.
    """

    result = image.copy()

    width = result.shape[1]
    height = result.shape[0]

    # --------------------------------------------------------
    # Grid every 0.10 m.
    # --------------------------------------------------------

    x_values = np.arange(
        X_MIN_M,
        X_MAX_M + 0.001,
        0.10,
    )

    y_values = np.arange(
        Y_MIN_M,
        Y_MAX_M + 0.001,
        0.10,
    )

    # --------------------------------------------------------
    # Vertical grid lines.
    # --------------------------------------------------------

    for x_m in x_values:

        px = int(
            round(
                (x_m - X_MIN_M)
                * PIXELS_PER_METER
            )
        )

        if (
            px < 0
            or px >= width
        ):
            continue

        cv2.line(
            result,
            (px, 0),
            (px, height - 1),
            (120, 120, 120),
            1,
        )

        # Label every 0.5 m.
        if abs(
            (x_m * 10)
            % 5
        ) < 1e-6:

            cv2.putText(
                result,
                f"{x_m:.1f}m",
                (
                    px + 4,
                    20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                result,
                f"{x_m:.1f}m",
                (
                    px + 4,
                    20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Horizontal grid lines.
    # --------------------------------------------------------

    for y_m in y_values:

        py = int(
            round(
                (y_m - Y_MIN_M)
                * PIXELS_PER_METER
            )
        )

        if (
            py < 0
            or py >= height
        ):
            continue

        cv2.line(
            result,
            (0, py),
            (width - 1, py),
            (120, 120, 120),
            1,
        )

        if abs(
            (y_m * 10)
            % 5
        ) < 1e-6:

            cv2.putText(
                result,
                f"{y_m:.1f}m",
                (
                    5,
                    py - 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                result,
                f"{y_m:.1f}m",
                (
                    5,
                    py - 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return result


# ============================================================
# DRAW CALIBRATION TILE
# ============================================================

def draw_reference_tile(
    image,
):
    """
    Draw the assumed 0.60 x 0.60 m calibration tile.
    """

    result = image.copy()

    world_tile = np.array(
        [
            [0.0, 0.0],
            [0.60, 0.0],
            [0.60, 0.60],
            [0.0, 0.60],
        ],
        dtype=np.float32,
    )

    pixels = []

    for x_m, y_m in world_tile:

        px = int(
            round(
                (x_m - X_MIN_M)
                * PIXELS_PER_METER
            )
        )

        py = int(
            round(
                (y_m - Y_MIN_M)
                * PIXELS_PER_METER
            )
        )

        pixels.append(
            [px, py]
        )

    pixels = np.array(
        pixels,
        dtype=np.int32,
    )

    cv2.polylines(
        result,
        [pixels],
        True,
        (0, 255, 0),
        4,
    )

    for i, (px, py) in enumerate(
        pixels
    ):

        cv2.circle(
            result,
            (int(px), int(py)),
            7,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            result,
            f"P{i + 1}",
            (
                int(px) + 10,
                int(py) - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        result,
        "REFERENCE TILE = 0.60 x 0.60 m (ASSUMED)",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        "REFERENCE TILE = 0.60 x 0.60 m (ASSUMED)",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


# ============================================================
# VALIDATION IMAGE
# ============================================================

def create_validation_image(
    frame,
    image_points,
):
    """
    Draw the selected tile on the original image.
    """

    result = frame.copy()

    points = np.round(
        image_points
    ).astype(
        np.int32
    )

    cv2.polylines(
        result,
        [points],
        True,
        (0, 255, 0),
        4,
    )

    for i, (x, y) in enumerate(
        points
    ):

        cv2.circle(
            result,
            (
                int(x),
                int(y),
            ),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            result,
            f"P{i + 1}",
            (
                int(x) + 10,
                int(y) - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        result,
        "Ground-plane calibration points",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        "Ground-plane calibration points",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 20 - GROUND-PLANE HOMOGRAPHY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load calibration.
    # --------------------------------------------------------

    (
        calibration_data,
        image_points,
        world_points,
    ) = load_ground_points()

    print()
    print("IMAGE POINTS")

    for i, point in enumerate(
        image_points
    ):

        print(
            f"  P{i + 1}: "
            f"({point[0]:.1f}, "
            f"{point[1]:.1f}) px"
        )

    print()
    print("WORLD POINTS")

    for i, point in enumerate(
        world_points
    ):

        print(
            f"  P{i + 1}: "
            f"({point[0]:.2f}, "
            f"{point[1]:.2f}) m"
        )

    # --------------------------------------------------------
    # Calculate homography.
    # --------------------------------------------------------

    H, inliers = calculate_homography(
        image_points,
        world_points,
    )

    print()
    print("=" * 70)
    print("HOMOGRAPHY")
    print("=" * 70)

    print()
    print(H)

    print()
    print(
        f"Homography inliers: "
        f"{inliers} / 4"
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    predicted_world, errors = (
        calculate_reprojection_error(
            H,
            image_points,
            world_points,
        )
    )

    print()
    print("=" * 70)
    print("REPROJECTION CHECK")
    print("=" * 70)

    for i in range(4):

        print(
            f"P{i + 1}: "
            f"known=("
            f"{world_points[i, 0]:.4f}, "
            f"{world_points[i, 1]:.4f}) m "
            f"predicted=("
            f"{predicted_world[i, 0]:.4f}, "
            f"{predicted_world[i, 1]:.4f}) m "
            f"error="
            f"{errors[i]:.6f} m"
        )

    print()
    print(
        f"Mean error: "
        f"{np.mean(errors):.6f} m"
    )

    print(
        f"Maximum error: "
        f"{np.max(errors):.6f} m"
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    save_homography(
        calibration_data,
        H,
        errors,
    )

    # --------------------------------------------------------
    # Load frame.
    # --------------------------------------------------------

    frame = load_frame()

    # --------------------------------------------------------
    # Create original-image validation.
    # --------------------------------------------------------

    validation = create_validation_image(
        frame,
        image_points,
    )

    validation_path = (
        OUTPUT_DIR
        / "homography_validation.png"
    )

    cv2.imwrite(
        str(validation_path),
        validation,
    )

    # --------------------------------------------------------
    # Warp image.
    # --------------------------------------------------------

    warped = warp_to_ground(
        frame,
        H,
    )

    warped_grid = draw_grid(
        warped,
    )

    warped_grid = draw_reference_tile(
        warped_grid,
    )

    warp_path = (
        OUTPUT_DIR
        / "ground_plane_warp.png"
    )

    cv2.imwrite(
        str(warp_path),
        warped,
    )

    grid_path = (
        OUTPUT_DIR
        / "ground_plane_grid.png"
    )

    cv2.imwrite(
        str(grid_path),
        warped_grid,
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Validation image:\n"
        f"{validation_path}"
    )

    print()

    print(
        f"Warped ground plane:\n"
        f"{warp_path}"
    )

    print()

    print(
        f"Metric grid:\n"
        f"{grid_path}"
    )

    print()
    print("=" * 70)
    print("STEP 20 COMPLETE")
    print("=" * 70)

    print()
    print(
        "The image-to-ground-plane homography has been "
        "calculated."
    )

    print(
        "The selected floor tile has been mapped into a "
        "metric bird's-eye coordinate system."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Absolute metric scale currently depends on the "
        "assumed 0.60 x 0.60 m tile size."
    )

    print()
    print(
        "Next: apply this ground-plane geometry to the "
        "football trajectory."
    )


if __name__ == "__main__":
    main()