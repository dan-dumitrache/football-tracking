"""
STEP 33 - FINAL CAMERA-COMPENSATED INTEGRATED OUTPUT

Creates a NEW final integrated video using the validated camera-motion
compensated ground trajectory.

The original final_integrated_video.mp4 is NOT modified.

Panels
------
LEFT  : RGB video + football detection + 2D trajectory
RIGHT : fixed/reference top-view + camera + compensated trajectory

Additional on-screen information
--------------------------------
- frame / time
- 3D X, Y, Z relative to camera
- distance to camera
- 3D confidence based on the empirical 20 px diameter threshold
- top-view ground X/Y

Important
---------
The metric top-view scale still uses the working 0.60 m tile assumption.
The 3D coordinates use the existing smoothed reconstruction and therefore
retain its documented nominal-intrinsics / known-ball-size limitations.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "rgb.avi"

RECONSTRUCTION_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "reconstruction_3d"
    / "smoothed_ball_3d.csv"
)

QUALITY_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "3d_quality"
    / "reconstructed_ball_3d_quality.csv"
)

COMPENSATED_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "camera_compensated_top_view"
    / "camera_compensated_contact_trajectory.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "final"

OUTPUT_VIDEO = OUTPUT_DIR / "final_integrated_video_camera_compensated.mp4"
SUMMARY_JSON = OUTPUT_DIR / "final_integrated_camera_compensated_summary.json"

# Empirical Step 28 threshold.
MIN_RELIABLE_DIAMETER_PX = 20.0

# Reference frame used by Step 31.
REFERENCE_FRAME = 300

# Top-view display area.
TOP_X_MIN = -2.0
TOP_X_MAX = 2.5
TOP_Y_MIN = -1.0
TOP_Y_MAX = 6.5

TOP_MARGIN = 55

VIDEO_CODEC = "mp4v"


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    normalized = {normalize_name(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    for col in df.columns:
        n = normalize_name(col)
        for candidate in candidates:
            c = normalize_name(candidate)
            if c in n or n in c:
                return col

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    track = find_column(df, ["track_id", "track"], "track ID")
    frame = find_column(df, ["frame", "frame_number"], "frame")
    time = find_column(
        df,
        ["time_seconds", "time_s", "timestamp"],
        "time",
    )
    diameter = find_column(
        df,
        ["diameter_px", "diameter_pixels", "ball_diameter_px"],
        "football diameter",
    )

    pixel_x = find_column(
        df,
        ["pixel_x", "image_x", "u"],
        "pixel X",
    )

    pixel_y = find_column(
        df,
        ["pixel_y", "image_y", "v"],
        "pixel Y",
    )

    x = find_column(
        df,
        ["X_m_smoothed", "x_m_smoothed", "X_m", "x_m"],
        "3D X",
    )
    y = find_column(
        df,
        ["Y_m_smoothed", "y_m_smoothed", "Y_m", "y_m"],
        "3D Y",
    )
    z = find_column(
        df,
        ["depth_m_smoothed", "depth_m", "Z_m", "z_m"],
        "3D Z/depth",
    )

    distance = None
    for candidates in [
        ["distance_m_smoothed"],
        ["distance_m"],
        ["distance"],
    ]:
        try:
            distance = find_column(
                df,
                candidates,
                "3D distance",
            )
            break
        except ValueError:
            pass

    if distance is None:
        distance_values = np.sqrt(
            pd.to_numeric(df[x], errors="coerce") ** 2
            + pd.to_numeric(df[y], errors="coerce") ** 2
            + pd.to_numeric(df[z], errors="coerce") ** 2
        )
    else:
        distance_values = pd.to_numeric(
            df[distance],
            errors="coerce",
        )

    out = pd.DataFrame(
        {
            "track_id": pd.to_numeric(
                df[track],
                errors="coerce",
            ),
            "frame": pd.to_numeric(
                df[frame],
                errors="coerce",
            ),
            "time_seconds": pd.to_numeric(
                df[time],
                errors="coerce",
            ),
            "diameter_px": pd.to_numeric(
                df[diameter],
                errors="coerce",
            ),
            "pixel_x": pd.to_numeric(
                df[pixel_x],
                errors="coerce",
            ),
            "pixel_y": pd.to_numeric(
                df[pixel_y],
                errors="coerce",
            ),
            "X_m": pd.to_numeric(df[x], errors="coerce"),
            "Y_m": pd.to_numeric(df[y], errors="coerce"),
            "Z_m": pd.to_numeric(df[z], errors="coerce"),
            "distance_m": distance_values,
        }
    )

    return out.dropna(
        subset=[
            "track_id",
            "frame",
            "diameter_px",
            "pixel_x",
            "pixel_y",
            "X_m",
            "Y_m",
            "Z_m",
            "distance_m",
        ]
    ).copy()


def normalize_quality(df: pd.DataFrame) -> pd.DataFrame:
    frame = find_column(df, ["frame", "frame_number"], "quality frame")

    quality_col = None
    for candidates in [
        ["quality"],
        ["quality_class"],
        ["3d_quality"],
        ["confidence_class"],
    ]:
        try:
            quality_col = find_column(
                df,
                candidates,
                "3D quality class",
            )
            break
        except ValueError:
            pass

    diameter = None
    for candidates in [
        ["diameter_px"],
        ["diameter_pixels"],
    ]:
        try:
            diameter = find_column(
                df,
                candidates,
                "quality diameter",
            )
            break
        except ValueError:
            pass

    result = pd.DataFrame(
        {
            "frame": pd.to_numeric(
                df[frame],
                errors="coerce",
            ),
        }
    )

    if quality_col is not None:
        result["quality"] = (
            df[quality_col].astype(str).str.upper()
        )
    elif diameter is not None:
        d = pd.to_numeric(
            df[diameter],
            errors="coerce",
        )
        result["quality"] = np.where(
            d >= MIN_RELIABLE_DIAMETER_PX,
            "ACCEPTABLE_3D_CONFIDENCE",
            "LOW_3D_CONFIDENCE",
        )
    else:
        result["quality"] = "UNKNOWN"

    return result.dropna(subset=["frame"])


def normalize_compensated(df: pd.DataFrame) -> pd.DataFrame:
    track = find_column(
        df,
        ["track_id", "track"],
        "compensated track ID",
    )
    frame = find_column(
        df,
        ["frame", "frame_number"],
        "compensated frame",
    )

    time = None
    for candidates in [
        ["time_seconds"],
        ["time_s"],
        ["timestamp"],
    ]:
        try:
            time = find_column(
                df,
                candidates,
                "compensated time",
            )
            break
        except ValueError:
            pass

    x = find_column(
        df,
        [
            "compensated_ground_x_m",
            "compensated_x_m",
            "ground_x_m_compensated",
            "ground_x_m",
            "world_x_m",
        ],
        "compensated ground X",
    )
    y = find_column(
        df,
        [
            "compensated_ground_y_m",
            "compensated_y_m",
            "ground_y_m_compensated",
            "ground_y_m",
            "world_y_m",
        ],
        "compensated ground Y",
    )

    if time is None:
        time_values = pd.to_numeric(frame, errors="coerce") / 30.0
    else:
        time_values = pd.to_numeric(
            df[time],
            errors="coerce",
        )

    out = pd.DataFrame(
        {
            "track_id": pd.to_numeric(
                df[track],
                errors="coerce",
            ),
            "frame": pd.to_numeric(
                df[frame],
                errors="coerce",
            ),
            "time_seconds": time_values,
            "ground_x_m": pd.to_numeric(
                df[x],
                errors="coerce",
            ),
            "ground_y_m": pd.to_numeric(
                df[y],
                errors="coerce",
            ),
        }
    )

    return out.dropna(
        subset=[
            "track_id",
            "frame",
            "ground_x_m",
            "ground_y_m",
        ]
    ).copy()


def create_top_view(
    width: int,
    height: int,
) -> np.ndarray:
    img = np.full(
        (height, width, 3),
        245,
        dtype=np.uint8,
    )

    # Grid.
    grid_step = 0.5

    x_values = np.arange(
        np.ceil(TOP_X_MIN / grid_step) * grid_step,
        TOP_X_MAX + 0.001,
        grid_step,
    )
    y_values = np.arange(
        np.ceil(TOP_Y_MIN / grid_step) * grid_step,
        TOP_Y_MAX + 0.001,
        grid_step,
    )

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = TOP_MARGIN + (
            (x - TOP_X_MIN)
            / (TOP_X_MAX - TOP_X_MIN)
            * (width - 2 * TOP_MARGIN)
        )

        py = height - TOP_MARGIN - (
            (y - TOP_Y_MIN)
            / (TOP_Y_MAX - TOP_Y_MIN)
            * (height - 2 * TOP_MARGIN)
        )

        return int(round(px)), int(round(py))

    for x in x_values:
        p1 = to_px(float(x), TOP_Y_MIN)
        p2 = to_px(float(x), TOP_Y_MAX)

        cv2.line(
            img,
            p1,
            p2,
            (220, 220, 220),
            1,
        )

        cv2.putText(
            img,
            f"{x:.1f}",
            (p1[0] - 15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

    for y in y_values:
        p1 = to_px(TOP_X_MIN, float(y))
        p2 = to_px(TOP_X_MAX, float(y))

        cv2.line(
            img,
            p1,
            p2,
            (220, 220, 220),
            1,
        )

        cv2.putText(
            img,
            f"{y:.1f}",
            (5, p1[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

    # Axes.
    cv2.line(
        img,
        to_px(0, TOP_Y_MIN),
        to_px(0, TOP_Y_MAX),
        (120, 120, 120),
        2,
    )
    cv2.line(
        img,
        to_px(TOP_X_MIN, 0),
        to_px(TOP_X_MAX, 0),
        (120, 120, 120),
        2,
    )

    cv2.putText(
        img,
        "X ground (m)",
        (width // 2 - 45, height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        "Y ground (m)",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    # Camera/reference position is the origin.
    camera_px = to_px(0.0, 0.0)

    cv2.drawMarker(
        img,
        camera_px,
        (0, 0, 0),
        cv2.MARKER_CROSS,
        20,
        3,
    )

    cv2.putText(
        img,
        "CAMERA / REFERENCE",
        (
            camera_px[0] + 10,
            camera_px[1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return img


def draw_polyline(
    img: np.ndarray,
    points: list[tuple[int, int]],
    thickness: int = 3,
) -> None:
    if len(points) < 2:
        return

    for a, b in zip(points[:-1], points[1:]):
        cv2.line(
            img,
            a,
            b,
            (40, 40, 40),
            thickness,
            cv2.LINE_AA,
        )


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header(
        "STEP 33 - FINAL CAMERA-MOTION-COMPENSATED INTEGRATED OUTPUT"
    )

    print()
    print("This creates a NEW final video.")
    print("The original final_integrated_video.mp4 is NOT modified.")

    # ---------------------------------------------------------------
    # CHECK INPUTS
    # ---------------------------------------------------------------

    print_header("INPUTS")

    for path in [
        VIDEO_PATH,
        RECONSTRUCTION_CSV,
        COMPENSATED_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    print(f"Video:         {VIDEO_PATH}")
    print(f"3D:            {RECONSTRUCTION_CSV}")
    print(f"Compensated:   {COMPENSATED_CSV}")

    # ---------------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------------

    print_header("LOADING DATA")

    recon = normalize_reconstruction(
        pd.read_csv(RECONSTRUCTION_CSV)
    )

    compensated = normalize_compensated(
        pd.read_csv(COMPENSATED_CSV)
    )

    if QUALITY_CSV.exists():
        quality = normalize_quality(
            pd.read_csv(QUALITY_CSV)
        )
    else:
        quality = pd.DataFrame(
            columns=["frame", "quality"]
        )

    print(f"3D observations:          {len(recon)}")
    print(f"Compensated observations: {len(compensated)}")
    print(f"Quality observations:     {len(quality)}")

    # Keep one observation per track/frame.
    recon = recon.sort_values(
        ["frame", "track_id"]
    ).drop_duplicates(
        ["frame", "track_id"],
        keep="last",
    )

    compensated = compensated.sort_values(
        ["frame", "track_id"]
    ).drop_duplicates(
        ["frame", "track_id"],
        keep="last",
    )

    # ---------------------------------------------------------------
    # CREATE LOOKUP TABLES
    # ---------------------------------------------------------------

    recon_lookup = {
        (int(row.frame), int(row.track_id)): row
        for row in recon.itertuples(index=False)
    }

    comp_lookup = {
        (int(row.frame), int(row.track_id)): row
        for row in compensated.itertuples(index=False)
    }

    quality_lookup = {}

    if not quality.empty:
        for row in quality.itertuples(index=False):
            quality_lookup[int(row.frame)] = str(row.quality)

    # Precompute trajectory histories.
    tracks = {}

    for track_id in sorted(
        set(
            int(v)
            for v in compensated["track_id"].tolist()
        )
    ):
        rows = compensated[
            compensated["track_id"] == track_id
        ].sort_values("frame")

        tracks[track_id] = rows

    # ---------------------------------------------------------------
    # OPEN VIDEO
    # ---------------------------------------------------------------

    print_header("CREATING FINAL VIDEO")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1920 x 720 for the standard 1280 + 640 layout.
    top_width = width // 2

    output_width = width + top_width
    output_height = height

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps if fps > 0 else 30.0,
        (output_width, output_height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video:\n{OUTPUT_VIDEO}"
        )

    print(
        f"Output resolution: {output_width} x {output_height}"
    )
    print(f"FPS:               {fps:.2f}")
    print(f"Frames:            {frame_count}")

    # ---------------------------------------------------------------
    # FRAME LOOP
    # ---------------------------------------------------------------

    for frame_idx in range(frame_count):

        ok, frame = cap.read()

        if not ok:
            break

        rgb_panel = frame.copy()

        # Header.
        cv2.rectangle(
            rgb_panel,
            (0, 0),
            (width, 125),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            rgb_panel,
            "2D FOOTBALL DETECTION / TRACKING",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        time_seconds = (
            frame_idx / fps
            if fps > 0
            else 0.0
        )

        cv2.putText(
            rgb_panel,
            f"Frame: {frame_idx}/{frame_count - 1}    "
            f"Time: {time_seconds:.2f} s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            rgb_panel,
            "Camera-motion compensated top view -> reference frame 300",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # -----------------------------------------------------------
        # DRAW BALL + 2D TRAJECTORY
        # -----------------------------------------------------------

        current_recon = [
            (
                key,
                value,
            )
            for key, value in recon_lookup.items()
            if key[0] == frame_idx
        ]

        current_track_ids = []

        for (f, track_id), row in current_recon:

            current_track_ids.append(track_id)

            x = int(round(float(row.pixel_x)))
            y = int(round(float(row.pixel_y)))

            diameter = float(row.diameter_px)
            radius = max(5, int(round(diameter / 2.0)))

            cv2.circle(
                rgb_panel,
                (x, y),
                radius,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.drawMarker(
                rgb_panel,
                (x, y),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                max(10, radius),
                2,
            )

            cv2.putText(
                rgb_panel,
                f"BALL T{track_id}",
                (x + radius + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Draw a short history for each track using the original
        # image-space football centers.
        for track_id, rows in tracks.items():

            visible = rows[
                rows["frame"] <= frame_idx
            ]

            if visible.empty:
                continue

            # Only retain the currently visible segment.
            frames = visible["frame"].to_numpy()

            points = []

            for f in frames:
                key = (int(f), int(track_id))

                if key not in recon_lookup:
                    continue

                pixel_row = recon_lookup[key]

                px = float(pixel_row.pixel_x)
                py = float(pixel_row.pixel_y)

                points.append(
                    (
                        int(round(px)),
                        int(round(py)),
                    )
                )

            if len(points) >= 2:

                # Use only recent trajectory history.
                points = points[-80:]

                for a, b in zip(
                    points[:-1],
                    points[1:],
                ):
                    cv2.line(
                        rgb_panel,
                        a,
                        b,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # -----------------------------------------------------------
        # TOP VIEW
        # -----------------------------------------------------------

        top = create_top_view(
            top_width,
            height,
        )

        # Draw compensated trajectory histories.
        for track_id, rows in tracks.items():

            visible = rows[
                rows["frame"] <= frame_idx
            ]

            if visible.empty:
                continue

            visible = visible.tail(120)

            top_points = []

            for row in visible.itertuples(index=False):

                x = float(row.ground_x_m)
                y = float(row.ground_y_m)

                px = TOP_MARGIN + (
                    (x - TOP_X_MIN)
                    / (TOP_X_MAX - TOP_X_MIN)
                    * (top_width - 2 * TOP_MARGIN)
                )

                py = height - TOP_MARGIN - (
                    (y - TOP_Y_MIN)
                    / (TOP_Y_MAX - TOP_Y_MIN)
                    * (height - 2 * TOP_MARGIN)
                )

                top_points.append(
                    (
                        int(round(px)),
                        int(round(py)),
                    )
                )

            if len(top_points) >= 2:
                draw_polyline(
                    top,
                    top_points,
                    thickness=4,
                )

            if top_points:
                cv2.circle(
                    top,
                    top_points[-1],
                    9,
                    (0, 0, 0),
                    -1,
                )

                cv2.circle(
                    top,
                    top_points[-1],
                    6,
                    (255, 255, 255),
                    -1,
                )

        # Top-view title.
        cv2.rectangle(
            top,
            (0, 0),
            (top_width, 70),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            top,
            "2D TOP VIEW - FIXED REFERENCE FRAME",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            top,
            "Camera motion compensated",
            (15, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # -----------------------------------------------------------
        # CURRENT 3D INFORMATION
        # -----------------------------------------------------------

        current = []

        for (f, track_id), row in recon_lookup.items():
            if f == frame_idx:
                current.append(
                    (track_id, row)
                )

        if current:
            row = current[0][1]

            diameter = float(row.diameter_px)

            if diameter >= MIN_RELIABLE_DIAMETER_PX:
                quality_text = "ACCEPTABLE_3D_CONFIDENCE"
            else:
                quality_text = "LOW_3D_CONFIDENCE"

            # Prefer explicit Step 28 quality when available.
            if frame_idx in quality_lookup:
                quality_text = quality_lookup[frame_idx]

            lines = [
                f"3D X: {float(row.X_m):+.2f} m",
                f"3D Y: {float(row.Y_m):+.2f} m",
                f"3D Z: {float(row.Z_m):+.2f} m",
                f"Distance: {float(row.distance_m):.2f} m",
                f"Ball diameter: {diameter:.1f} px",
                f"3D: {quality_text}",
            ]

            # Bottom-left information box.
            box_x = 20
            box_y = height - 220
            box_w = 390
            box_h = 200

            overlay = rgb_panel.copy()

            cv2.rectangle(
                overlay,
                (box_x, box_y),
                (box_x + box_w, box_y + box_h),
                (0, 0, 0),
                -1,
            )

            cv2.addWeighted(
                overlay,
                0.78,
                rgb_panel,
                0.22,
                0,
                rgb_panel,
            )

            cv2.putText(
                rgb_panel,
                "3D BALL ESTIMATE",
                (box_x + 15, box_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for i, text in enumerate(lines):
                cv2.putText(
                    rgb_panel,
                    text,
                    (
                        box_x + 15,
                        box_y + 60 + i * 23,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (225, 225, 225),
                    1,
                    cv2.LINE_AA,
                )

        # -----------------------------------------------------------
        # COMPENSATED CURRENT GROUND POSITION
        # -----------------------------------------------------------

        comp_current = compensated[
            compensated["frame"] == frame_idx
        ]

        if not comp_current.empty:

            row = comp_current.iloc[0]

            text = (
                f"Ground: "
                f"X={float(row.ground_x_m):+.2f} m  "
                f"Y={float(row.ground_y_m):+.2f} m"
            )

            cv2.putText(
                top,
                text,
                (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )

        # -----------------------------------------------------------
        # COMBINE
        # -----------------------------------------------------------

        combined = np.hstack(
            [
                rgb_panel,
                top,
            ]
        )

        writer.write(combined)

        if frame_idx % 30 == 0:
            print(
                f"Rendered frame "
                f"{frame_idx}/{frame_count - 1}"
            )

    cap.release()
    writer.release()

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    summary = {
        "step": 33,
        "description": (
            "Final integrated video using validated "
            "camera-motion-compensated ground trajectory."
        ),
        "input_video": str(VIDEO_PATH),
        "reconstruction": str(RECONSTRUCTION_CSV),
        "compensated_trajectory": str(COMPENSATED_CSV),
        "output_video": str(OUTPUT_VIDEO),
        "reference_frame": REFERENCE_FRAME,
        "working_tile_size_m": 0.60,
        "minimum_reliable_diameter_px": (
            MIN_RELIABLE_DIAMETER_PX
        ),
        "input_resolution": [
            width,
            height,
        ],
        "output_resolution": [
            output_width,
            output_height,
        ],
        "fps": fps,
        "frames": frame_count,
        "reconstruction_rows": len(recon),
        "compensated_rows": len(compensated),
        "tracks": sorted(
            [
                int(v)
                for v in compensated["track_id"].unique()
            ]
        ),
        "note": (
            "This video is a new candidate final output. "
            "The original final_integrated_video.mp4 "
            "was not modified."
        ),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print_header("STEP 33 COMPLETE")

    print()
    print("Created:")
    print(OUTPUT_VIDEO)
    print(SUMMARY_JSON)

    print()
    print("IMPORTANT:")
    print(
        "The original final integrated video remains unchanged."
    )
    print(
        "This new video uses the camera-motion-compensated "
        "top-view trajectory validated in Step 32."
    )


if __name__ == "__main__":
    main()
