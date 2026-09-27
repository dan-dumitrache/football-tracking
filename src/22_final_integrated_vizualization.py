"""
22_final_integrated_visualization.py

FINAL INTEGRATED FOOTBALL TRACKING VISUALIZATION

Combines:

1. Original RGB video
2. 2D football detection/tracking
3. Monocular 3D reconstruction
4. Estimated distance from camera
5. Estimated 3D speed
6. Ground-contact point
7. Ground-plane trajectory
8. Reference calibration tile

IMPORTANT
---------

The reconstructed 3D position represents approximately the
CENTER of the football.

The top-view trajectory represents an estimated
GROUND-CONTACT POINT.

The ground metric scale depends on the assumed 0.60 x 0.60 m
calibration tile.

The football is assumed to remain in contact with the ground
for this experiment.

OUTPUT
------

data/output/final/

    final_integrated_video.mp4

    final_summary.png

    frame_0020.png
    frame_0100.png
    frame_0200.png
    frame_0390.png
    frame_0450.png
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "rgb.avi"
)

TRAJECTORY_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "top_view"
    / "top_view_contact_trajectory.csv"
)

RECONSTRUCTION_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "reconstruction_3d"
    / "reconstructed_ball_3d.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "final"
)


# ============================================================
# VIDEO CONFIGURATION
# ============================================================

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 720

FPS_FALLBACK = 30.0


# ============================================================
# TOP-VIEW CONFIGURATION
# ============================================================

GROUND_X_MIN = -1.5
GROUND_X_MAX = 3.0

GROUND_Y_MIN = -1.0
GROUND_Y_MAX = 7.0


# Reference tile
TILE_SIZE = 0.60


# ============================================================
# COLORS - BGR FOR OPENCV
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RED = (0, 0, 255)
GREEN = (0, 200, 0)
BLUE = (255, 120, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
PURPLE = (200, 80, 180)
GRAY = (150, 150, 150)

TRACK_COLORS = {
    0: (255, 140, 0),     # orange/blue-ish
    1: (0, 200, 255),     # orange
}


# ============================================================
# GENERAL DRAWING HELPERS
# ============================================================

def put_text(
    image,
    text,
    position,
    scale=0.7,
    color=WHITE,
    thickness=2,
):

    cv2.putText(
        image,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel_title(
    image,
    text,
):

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 45),
        BLACK,
        -1,
    )

    put_text(
        image,
        text,
        (18, 31),
        scale=0.75,
        color=WHITE,
        thickness=2,
    )


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_trajectory():

    print("=" * 70)
    print("LOADING GROUND-CONTACT TRAJECTORY")
    print("=" * 70)

    if not TRAJECTORY_CSV.exists():

        raise FileNotFoundError(
            f"Could not find:\n{TRAJECTORY_CSV}"
        )

    df = pd.read_csv(
        TRAJECTORY_CSV
    )

    print(
        f"CSV:\n{TRAJECTORY_CSV}"
    )

    print(
        f"Rows: {len(df)}"
    )

    required = [
        "track_id",
        "frame",
        "time_seconds",
        "pixel_center_x",
        "pixel_center_y",
        "diameter_px",
        "ground_contact_x_m",
        "ground_contact_y_m",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required trajectory columns:\n"
            + ", ".join(missing)
        )

    df.sort_values(
        [
            "track_id",
            "frame",
        ],
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    return df


# ============================================================
# LOAD 3D RECONSTRUCTION
# ============================================================

def load_reconstruction():

    print()
    print("=" * 70)
    print("LOADING 3D RECONSTRUCTION")
    print("=" * 70)

    if not RECONSTRUCTION_CSV.exists():

        raise FileNotFoundError(
            f"Could not find:\n{RECONSTRUCTION_CSV}"
        )

    df = pd.read_csv(
        RECONSTRUCTION_CSV
    )

    required = [
        "track_id",
        "frame",
        "distance_m",
        "speed_mps",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing reconstruction columns:\n"
            + ", ".join(missing)
        )

    return df


# ============================================================
# MERGE DATA
# ============================================================

def merge_data(
    trajectory,
    reconstruction,
):

    keys = [
        "track_id",
        "frame",
    ]

    # Only bring across the physical quantities that
    # aren't already in the contact trajectory.
    physical_columns = [
        "track_id",
        "frame",
        "distance_m",
        "speed_mps",
    ]

    reconstruction_small = reconstruction[
        physical_columns
    ].copy()

    merged = trajectory.merge(
        reconstruction_small,
        on=keys,
        how="left",
    )

    merged.sort_values(
        [
            "track_id",
            "frame",
        ],
        inplace=True,
    )

    return merged


# ============================================================
# INDEX DATA BY FRAME
# ============================================================

def create_frame_lookup(
    df,
):

    lookup = {}

    for frame, group in df.groupby(
        "frame"
    ):

        # A frame should normally contain zero or one
        # observation per track.
        lookup[int(frame)] = group.copy()

    return lookup


# ============================================================
# GROUND -> PANEL PIXEL
# ============================================================

def ground_to_panel(
    x,
    y,
    panel_width,
    panel_height,
):

    # Keep a small margin around the plot.
    margin_left = 65
    margin_right = 25
    margin_top = 70
    margin_bottom = 45

    plot_width = (
        panel_width
        - margin_left
        - margin_right
    )

    plot_height = (
        panel_height
        - margin_top
        - margin_bottom
    )

    # Horizontal coordinate.
    px = (
        margin_left
        + (
            (x - GROUND_X_MIN)
            / (
                GROUND_X_MAX
                - GROUND_X_MIN
            )
        )
        * plot_width
    )

    # Vertical coordinate.
    # Ground Y increases upward, image Y downward.
    py = (
        margin_top
        + (
            1.0
            - (
                (y - GROUND_Y_MIN)
                / (
                    GROUND_Y_MAX
                    - GROUND_Y_MIN
                )
            )
        )
        * plot_height
    )

    return (
        int(round(px)),
        int(round(py)),
    )


# ============================================================
# DRAW TOP-VIEW AXES
# ============================================================

def draw_ground_axes(
    panel,
):

    h, w = panel.shape[:2]

    # Plot boundaries.
    left = 65
    right = w - 25
    top = 70
    bottom = h - 45

    cv2.rectangle(
        panel,
        (left, top),
        (right, bottom),
        (100, 100, 100),
        1,
    )

    # --------------------------------------------------------
    # Vertical grid lines.
    # --------------------------------------------------------

    x_values = np.arange(
        np.ceil(GROUND_X_MIN),
        GROUND_X_MAX + 0.001,
        0.5,
    )

    for x in x_values:

        px, _ = ground_to_panel(
            x,
            0,
            w,
            h,
        )

        cv2.line(
            panel,
            (px, top),
            (px, bottom),
            (50, 50, 50),
            1,
        )

        put_text(
            panel,
            f"{x:.1f}",
            (px - 15, bottom + 23),
            scale=0.45,
            color=GRAY,
            thickness=1,
        )

    # --------------------------------------------------------
    # Horizontal grid lines.
    # --------------------------------------------------------

    y_values = np.arange(
        np.ceil(GROUND_Y_MIN),
        GROUND_Y_MAX + 0.001,
        1.0,
    )

    for y in y_values:

        _, py = ground_to_panel(
            0,
            y,
            w,
            h,
        )

        cv2.line(
            panel,
            (left, py),
            (right, py),
            (50, 50, 50),
            1,
        )

        put_text(
            panel,
            f"{y:.1f}",
            (8, py + 5),
            scale=0.45,
            color=GRAY,
            thickness=1,
        )

    # Axis labels.
    put_text(
        panel,
        "Ground X (m)",
        (w // 2 - 55, h - 8),
        scale=0.5,
        color=WHITE,
        thickness=1,
    )

    # Y label is kept horizontal for readability.
    put_text(
        panel,
        "Ground Y (m)",
        (8, 58),
        scale=0.5,
        color=WHITE,
        thickness=1,
    )


# ============================================================
# DRAW REFERENCE TILE
# ============================================================

def draw_reference_tile(
    panel,
):

    points = []

    corners = [
        (0.0, 0.0),
        (TILE_SIZE, 0.0),
        (TILE_SIZE, TILE_SIZE),
        (0.0, TILE_SIZE),
    ]

    for x, y in corners:

        points.append(
            ground_to_panel(
                x,
                y,
                panel.shape[1],
                panel.shape[0],
            )
        )

    points.append(
        points[0]
    )

    for i in range(
        len(points) - 1
    ):

        cv2.line(
            panel,
            points[i],
            points[i + 1],
            GREEN,
            2,
        )

    put_text(
        panel,
        "0.60 x 0.60 m assumed tile",
        (
            points[0][0] + 5,
            points[0][1] - 10,
        ),
        scale=0.45,
        color=GREEN,
        thickness=1,
    )


# ============================================================
# DRAW COMPLETE TRAJECTORY HISTORY
# ============================================================

def draw_trajectory_history(
    panel,
    data,
    current_frame,
):

    # Only draw points that have already happened.
    visible = data[
        data["frame"] <= current_frame
    ]

    for track_id, group in visible.groupby(
        "track_id"
    ):

        group = group.sort_values(
            "frame"
        )

        color = TRACK_COLORS.get(
            int(track_id),
            WHITE,
        )

        previous = None

        for _, row in group.iterrows():

            x = row[
                "ground_contact_x_m"
            ]

            y = row[
                "ground_contact_y_m"
            ]

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):

                previous = None
                continue

            current = ground_to_panel(
                x,
                y,
                panel.shape[1],
                panel.shape[0],
            )

            if previous is not None:

                cv2.line(
                    panel,
                    previous,
                    current,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            previous = current


# ============================================================
# DRAW CURRENT FOOTBALL ON TOP VIEW
# ============================================================

def draw_current_ground_position(
    panel,
    current_rows,
):

    for _, row in current_rows.iterrows():

        x = row[
            "ground_contact_x_m"
        ]

        y = row[
            "ground_contact_y_m"
        ]

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):

            continue

        color = TRACK_COLORS.get(
            int(row["track_id"]),
            WHITE,
        )

        px, py = ground_to_panel(
            x,
            y,
            panel.shape[1],
            panel.shape[0],
        )

        cv2.circle(
            panel,
            (px, py),
            9,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            panel,
            (px, py),
            13,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        put_text(
            panel,
            f"T{int(row['track_id'])}",
            (px + 12, py - 8),
            scale=0.55,
            color=color,
            thickness=2,
        )


# ============================================================
# DRAW TOP VIEW
# ============================================================

def create_top_view(
    data,
    current_rows,
    frame_number,
):

    panel_width = (
        OUTPUT_WIDTH
        // 2
    )

    panel_height = OUTPUT_HEIGHT

    panel = np.zeros(
        (
            panel_height,
            panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    draw_panel_title(
        panel,
        "GROUND-PLANE TOP VIEW — FOOTBALL CONTACT POINT",
    )

    draw_ground_axes(
        panel
    )

    draw_reference_tile(
        panel
    )

    draw_trajectory_history(
        panel,
        data,
        frame_number,
    )

    draw_current_ground_position(
        panel,
        current_rows,
    )

    return panel


# ============================================================
# DRAW BALL ON ORIGINAL IMAGE
# ============================================================

def annotate_original(
    frame,
    current_rows,
    frame_number,
    fps,
):

    output = frame.copy()

    # Title.
    draw_panel_title(
        output,
        "ORIGINAL RGB VIDEO — FOOTBALL TRACKING",
    )

    if len(current_rows) == 0:

        put_text(
            output,
            "NO BALL TRACK",
            (25, 80),
            scale=0.9,
            color=RED,
            thickness=3,
        )

        put_text(
            output,
            "Football not visible / not confidently tracked",
            (25, 115),
            scale=0.55,
            color=WHITE,
            thickness=2,
        )

        return output

    for _, row in current_rows.iterrows():

        x = float(
            row["pixel_center_x"]
        )

        y = float(
            row["pixel_center_y"]
        )

        diameter = float(
            row["diameter_px"]
        )

        track_id = int(
            row["track_id"]
        )

        color = TRACK_COLORS.get(
            track_id,
            RED,
        )

        radius = max(
            5,
            int(
                round(
                    diameter / 2.0
                )
            ),
        )

        center = (
            int(round(x)),
            int(round(y)),
        )

        # Detection circle.
        cv2.circle(
            output,
            center,
            radius,
            color,
            3,
            cv2.LINE_AA,
        )

        # Center.
        cv2.circle(
            output,
            center,
            4,
            RED,
            -1,
            cv2.LINE_AA,
        )

        # Track label.
        put_text(
            output,
            f"BALL / TRACK {track_id}",
            (
                center[0] + radius + 8,
                center[1] - radius - 10,
            ),
            scale=0.55,
            color=color,
            thickness=2,
        )

        # Contact point approximation.
        contact_y = int(
            round(
                y + diameter / 2.0
            )
        )

        cv2.circle(
            output,
            (
                int(round(x)),
                contact_y,
            ),
            5,
            GREEN,
            -1,
            cv2.LINE_AA,
        )

        cv2.line(
            output,
            center,
            (
                int(round(x)),
                contact_y,
            ),
            GREEN,
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Information box.
        # ----------------------------------------------------

        distance = row.get(
            "distance_m",
            np.nan,
        )

        speed = row.get(
            "speed_mps",
            np.nan,
        )

        time_seconds = row.get(
            "time_seconds",
            frame_number / fps,
        )

        lines = [
            f"Track: {track_id}",
            f"Time: {time_seconds:.2f} s",
            f"Diameter: {diameter:.1f} px",
        ]

        if np.isfinite(
            distance
        ):

            lines.append(
                f"Distance: {distance:.2f} m"
            )

        else:

            lines.append(
                "Distance: n/a"
            )

        if np.isfinite(
            speed
        ):

            lines.append(
                f"3D speed: {speed:.2f} m/s"
            )

        else:

            lines.append(
                "3D speed: n/a"
            )

        box_x = 20
        box_y = 70

        box_w = 300
        box_h = (
            25
            + 30 * len(lines)
        )

        cv2.rectangle(
            output,
            (
                box_x,
                box_y,
            ),
            (
                box_x + box_w,
                box_y + box_h,
            ),
            BLACK,
            -1,
        )

        cv2.rectangle(
            output,
            (
                box_x,
                box_y,
            ),
            (
                box_x + box_w,
                box_y + box_h,
            ),
            color,
            2,
        )

        for i, line in enumerate(
            lines
        ):

            put_text(
                output,
                line,
                (
                    box_x + 12,
                    box_y + 30
                    + i * 30,
                ),
                scale=0.58,
                color=WHITE,
                thickness=2,
            )

    return output


# ============================================================
# COMBINE PANELS
# ============================================================

def combine_panels(
    original,
    top_view,
):

    target_height = OUTPUT_HEIGHT

    # Original video should be resized to exactly half
    # of final output width.

    panel_width = (
        OUTPUT_WIDTH
        // 2
    )

    original_resized = cv2.resize(
        original,
        (
            panel_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    top_view_resized = cv2.resize(
        top_view,
        (
            panel_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    combined = np.hstack(
        [
            original_resized,
            top_view_resized,
        ]
    )

    # Divider.
    cv2.line(
        combined,
        (
            panel_width,
            0,
        ),
        (
            panel_width,
            OUTPUT_HEIGHT,
        ),
        WHITE,
        2,
    )

    return combined


# ============================================================
# CREATE VIDEO
# ============================================================

def create_video(
    trajectory,
):

    print()
    print("=" * 70)
    print("CREATING FINAL INTEGRATED VIDEO")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not np.isfinite(fps) or fps <= 0:

        fps = FPS_FALLBACK

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Input resolution: "
        f"{width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {frame_count}"
    )

    # --------------------------------------------------------
    # Video writer.
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "final_integrated_video.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
        ),
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )

    frame_lookup = create_frame_lookup(
        trajectory
    )

    frame_number = 0

    while True:

        ok, frame = cap.read()

        if not ok:

            break

        current_rows = frame_lookup.get(
            frame_number,
            pd.DataFrame(
                columns=trajectory.columns
            ),
        )

        original_panel = annotate_original(
            frame,
            current_rows,
            frame_number,
            fps,
        )

        top_panel = create_top_view(
            trajectory,
            current_rows,
            frame_number,
        )

        combined = combine_panels(
            original_panel,
            top_panel,
        )

        writer.write(
            combined
        )

        frame_number += 1

        if frame_number % 50 == 0:

            print(
                f"Processed "
                f"{frame_number}/{frame_count} "
                f"frames..."
            )

    cap.release()
    writer.release()

    print()
    print(
        f"Saved:\n{output_path}"
    )

    return output_path


# ============================================================
# SAVE IMPORTANT DIAGNOSTIC FRAMES
# ============================================================

def save_diagnostic_frames(
    trajectory,
):

    print()
    print("=" * 70)
    print("CREATING DIAGNOSTIC FRAMES")
    print("=" * 70)

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open input video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not np.isfinite(fps) or fps <= 0:

        fps = FPS_FALLBACK

    frame_lookup = create_frame_lookup(
        trajectory
    )

    target_frames = [
        20,
        50,
        100,
        180,
        200,
        389,
        410,
        450,
    ]

    targets = set(
        target_frames
    )

    frame_number = 0

    while True:

        ok, frame = cap.read()

        if not ok:

            break

        if frame_number in targets:

            current_rows = frame_lookup.get(
                frame_number,
                pd.DataFrame(
                    columns=trajectory.columns
                ),
            )

            original_panel = annotate_original(
                frame,
                current_rows,
                frame_number,
                fps,
            )

            top_panel = create_top_view(
                trajectory,
                current_rows,
                frame_number,
            )

            combined = combine_panels(
                original_panel,
                top_panel,
            )

            output_path = (
                OUTPUT_DIR
                / f"frame_{frame_number:04d}.png"
            )

            cv2.imwrite(
                str(output_path),
                combined,
            )

            print(
                f"Saved: {output_path}"
            )

        frame_number += 1

    cap.release()


# ============================================================
# FINAL SUMMARY IMAGE
# ============================================================

def create_summary(
    trajectory,
):

    print()
    print("=" * 70)
    print("CREATING FINAL SUMMARY")
    print("=" * 70)

    fig, ax = plt.subplots(
        figsize=(10, 12)
    )

    for track_id, group in trajectory.groupby(
        "track_id"
    ):

        group = group.sort_values(
            "frame"
        )

        color = None

        if int(track_id) == 0:
            color = "tab:blue"
        elif int(track_id) == 1:
            color = "tab:orange"

        ax.plot(
            group[
                "ground_contact_x_m"
            ],
            group[
                "ground_contact_y_m"
            ],
            linewidth=2.5,
            label=f"Track {track_id}",
            color=color,
        )

        first = group.iloc[0]
        last = group.iloc[-1]

        ax.scatter(
            first[
                "ground_contact_x_m"
            ],
            first[
                "ground_contact_y_m"
            ],
            s=100,
            marker="o",
            color=color,
        )

        ax.scatter(
            last[
                "ground_contact_x_m"
            ],
            last[
                "ground_contact_y_m"
            ],
            s=120,
            marker="X",
            color=color,
        )

    # Reference tile.
    tile_x = [
        0,
        TILE_SIZE,
        TILE_SIZE,
        0,
        0,
    ]

    tile_y = [
        0,
        0,
        TILE_SIZE,
        TILE_SIZE,
        0,
    ]

    ax.plot(
        tile_x,
        tile_y,
        linestyle="--",
        linewidth=2,
        label="0.60 m assumed tile",
    )

    ax.scatter(
        0,
        0,
        marker="^",
        s=120,
        label="Ground reference P1",
    )

    ax.set_xlim(
        GROUND_X_MIN,
        GROUND_X_MAX,
    )

    ax.set_ylim(
        GROUND_Y_MIN,
        GROUND_Y_MAX,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "Ground X (m)"
    )

    ax.set_ylabel(
        "Ground Y (m)"
    )

    ax.set_title(
        "Final Football Ground-Contact Trajectory"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "final_summary.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)

    print(
        f"Saved:\n{output_path}"
    )


# ============================================================
# FINAL STATISTICS
# ============================================================

def print_final_statistics(
    trajectory,
):

    print()
    print("=" * 70)
    print("FINAL PROJECT STATISTICS")
    print("=" * 70)

    print()

    print(
        f"Total tracked observations: "
        f"{len(trajectory)}"
    )

    print(
        f"Tracks: "
        f"{trajectory['track_id'].nunique()}"
    )

    print()

    for track_id, group in trajectory.groupby(
        "track_id"
    ):

        group = group.sort_values(
            "frame"
        )

        print(
            f"Track {track_id}"
        )

        print(
            f"  Frames: "
            f"{int(group['frame'].min())} -> "
            f"{int(group['frame'].max())}"
        )

        print(
            f"  Observations: "
            f"{len(group)}"
        )

        x_min = group[
            "ground_contact_x_m"
        ].min()

        x_max = group[
            "ground_contact_x_m"
        ].max()

        y_min = group[
            "ground_contact_y_m"
        ].min()

        y_max = group[
            "ground_contact_y_m"
        ].max()

        print(
            f"  Contact X: "
            f"{x_min:.2f} -> {x_max:.2f} m"
        )

        print(
            f"  Contact Y: "
            f"{y_min:.2f} -> {y_max:.2f} m"
        )

        if "distance_m" in group.columns:

            distance = group[
                "distance_m"
            ]

            print(
                f"  Distance: "
                f"{distance.min():.2f} -> "
                f"{distance.max():.2f} m"
            )

        if "speed_mps" in group.columns:

            speed = group[
                "speed_mps"
            ].dropna()

            if len(speed) > 0:

                print(
                    f"  Maximum speed: "
                    f"{speed.max():.2f} m/s"
                )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 22 - FINAL INTEGRATED VISUALIZATION")
    print("=" * 70)

    trajectory = load_trajectory()

    reconstruction = load_reconstruction()

    merged = merge_data(
        trajectory,
        reconstruction,
    )

    print_final_statistics(
        merged
    )

    create_video(
        merged
    )

    save_diagnostic_frames(
        merged
    )

    create_summary(
        merged
    )

    print()
    print("=" * 70)
    print("STEP 22 COMPLETE")
    print("=" * 70)

    print()
    print(
        "The complete football tracking pipeline has been "
        "integrated into a single visualization."
    )

    print()
    print(
        "Left panel:"
    )

    print(
        "  RGB football detection and 3D measurements."
    )

    print()
    print(
        "Right panel:"
    )

    print(
        "  Ground-contact trajectory on the calibrated "
        "floor plane."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  The ground metric scale depends on the assumed "
        "0.60 x 0.60 m tile."
    )

    print()
    print(
        "  The 3D reconstruction describes the football "
        "center, while the top-view trajectory describes "
        "the estimated ground-contact point."
    )


if __name__ == "__main__":
    main()