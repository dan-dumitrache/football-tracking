"""
21_top_view_trajectory.py

Project the tracked football image coordinates onto the
calibrated ground plane using the homography calculated
in Step 20.

IMPORTANT CONCEPT
-----------------
The homography maps an image point to the location where its
camera ray intersects the calibrated floor plane.

Therefore, for an AIRBORNE football this is NOT the actual
3D center of the football.

It represents the football's ground-plane projection /
footprint.

This is useful for the required top-view map.

The actual airborne 3D position remains in:

    reconstructed_ball_3d.csv

INPUT
-----
data/output/reconstruction_3d/reconstructed_ball_3d.csv

calibration/ground_homography.json

OUTPUT
------
data/output/top_view/

    top_view_trajectory.csv
    top_view_trajectory.png
    top_view_trajectory_with_direction.png

The metric scale depends on the assumed 0.60 x 0.60 m
reference tile.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "reconstruction_3d"
    / "reconstructed_ball_3d.csv"
)

HOMOGRAPHY_JSON = (
    PROJECT_ROOT
    / "calibration"
    / "ground_homography.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "top_view"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Ground-plane plotting limits.
#
# These are deliberately broad enough to contain the
# reconstructed trajectory.
#
X_MIN_M = -1.5
X_MAX_M = 3.0

Y_MIN_M = -1.0
Y_MAX_M = 16.0


# ============================================================
# LOAD DATA
# ============================================================

def load_reconstruction():

    print()
    print("=" * 70)
    print("INPUT TRAJECTORY")
    print("=" * 70)

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"\nCSV:\n{INPUT_CSV}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        "\nColumns:"
    )

    print(
        list(df.columns)
    )

    required = [
        "track_id",
        "frame",
        "time_seconds",
        "pixel_x",
        "pixel_y",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Keep only valid pixel observations.
    # --------------------------------------------------------

    df = df.copy()

    df = df[
        np.isfinite(
            df["pixel_x"]
        )
        & np.isfinite(
            df["pixel_y"]
        )
    ].copy()

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

    print(
        f"\nValid observations: "
        f"{len(df)}"
    )

    for track_id, group in df.groupby(
        "track_id"
    ):

        print(
            f"Track {track_id}: "
            f"{len(group)} observations, "
            f"frames "
            f"{int(group['frame'].min())} -> "
            f"{int(group['frame'].max())}"
        )

    return df


# ============================================================
# LOAD HOMOGRAPHY
# ============================================================

def load_homography():

    print()
    print("=" * 70)
    print("GROUND HOMOGRAPHY")
    print("=" * 70)

    if not HOMOGRAPHY_JSON.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n{HOMOGRAPHY_JSON}"
        )

    with open(
        HOMOGRAPHY_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    H = np.asarray(
        data[
            "homography_image_to_ground"
        ],
        dtype=np.float64,
    )

    if H.shape != (3, 3):

        raise ValueError(
            "Homography must be a 3x3 matrix."
        )

    print(
        f"\nLoaded:\n{HOMOGRAPHY_JSON}"
    )

    print(
        "\nH ="
    )

    print(H)

    print()

    print(
        "Mean calibration reprojection error: "
        f"{data.get('mean_reprojection_error_m', np.nan):.8f} m"
    )

    print(
        "Maximum calibration reprojection error: "
        f"{data.get('max_reprojection_error_m', np.nan):.8f} m"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Metric scale depends on the assumed "
        "0.60 x 0.60 m tile."
    )

    return H


# ============================================================
# PROJECT PIXELS TO GROUND
# ============================================================

def project_to_ground(
    df,
    H,
):

    print()
    print(
        "=" * 70
    )
    print(
        "PROJECTING FOOTBALL TO GROUND PLANE"
    )
    print(
        "=" * 70
    )

    image_points = np.column_stack(
        [
            df["pixel_x"].to_numpy(
                dtype=np.float32
            ),
            df["pixel_y"].to_numpy(
                dtype=np.float32
            ),
        ]
    )

    image_points = (
        image_points
        .reshape(-1, 1, 2)
    )

    ground_points = (
        cv2.perspectiveTransform(
            image_points,
            H,
        )
        .reshape(-1, 2)
    )

    result = df.copy()

    result[
        "ground_x_m"
    ] = ground_points[:, 0]

    result[
        "ground_y_m"
    ] = ground_points[:, 1]

    return result


# ============================================================
# BASIC STATISTICS
# ============================================================

def print_statistics(df):

    print()
    print(
        "=" * 70
    )
    print(
        "GROUND-PROJECTED TRAJECTORY"
    )
    print(
        "=" * 70
    )

    for track_id, group in df.groupby(
        "track_id"
    ):

        print()
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

        print(
            f"  Ground X range: "
            f"{group['ground_x_m'].min():.3f} -> "
            f"{group['ground_x_m'].max():.3f} m"
        )

        print(
            f"  Ground Y range: "
            f"{group['ground_y_m'].min():.3f} -> "
            f"{group['ground_y_m'].max():.3f} m"
        )

        # ----------------------------------------------------
        # Total ground-projected path length.
        #
        # IMPORTANT:
        # calculate separately within each track.
        # ----------------------------------------------------

        x = group[
            "ground_x_m"
        ].to_numpy()

        y = group[
            "ground_y_m"
        ].to_numpy()

        dx = np.diff(x)
        dy = np.diff(y)

        path_length = np.sum(
            np.sqrt(
                dx ** 2
                + dy ** 2
            )
        )

        print(
            f"  Projected path length: "
            f"{path_length:.3f} m"
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "top_view_trajectory.csv"
    )

    columns = [
        "track_id",
        "frame",
        "time_seconds",
        "pixel_x",
        "pixel_y",
        "ground_x_m",
        "ground_y_m",
    ]

    # Add useful 3D fields if available.
    optional_columns = [
        "depth_m",
        "X_m",
        "Y_m",
        "distance_m",
    ]

    for column in optional_columns:

        if column in df.columns:

            columns.append(
                column
            )

    output = df[
        columns
    ].copy()

    output.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved:\n{output_path}"
    )

    return output_path


# ============================================================
# TOP VIEW PLOT
# ============================================================

def create_top_view_plot(
    df,
):

    print()
    print(
        "Creating top-view trajectory plot..."
    )

    fig, ax = plt.subplots(
        figsize=(11, 14)
    )

    for track_id, group in df.groupby(
        "track_id"
    ):

        group = group.sort_values(
            "frame"
        )

        ax.plot(
            group[
                "ground_x_m"
            ],
            group[
                "ground_y_m"
            ],
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=f"Track {track_id}",
        )

        # ----------------------------------------------------
        # Start point.
        # ----------------------------------------------------

        first = group.iloc[0]

        ax.scatter(
            first["ground_x_m"],
            first["ground_y_m"],
            marker="o",
            s=80,
        )

        ax.annotate(
            f"Start T{track_id}",
            (
                first["ground_x_m"],
                first["ground_y_m"],
            ),
            xytext=(8, 8),
            textcoords="offset points",
        )

        # ----------------------------------------------------
        # End point.
        # ----------------------------------------------------

        last = group.iloc[-1]

        ax.scatter(
            last["ground_x_m"],
            last["ground_y_m"],
            marker="X",
            s=100,
        )

        ax.annotate(
            f"End T{track_id}",
            (
                last["ground_x_m"],
                last["ground_y_m"],
            ),
            xytext=(8, 8),
            textcoords="offset points",
        )

    # --------------------------------------------------------
    # Ground origin / calibration tile.
    # --------------------------------------------------------

    tile_x = [
        0.0,
        0.6,
        0.6,
        0.0,
        0.0,
    ]

    tile_y = [
        0.0,
        0.0,
        0.6,
        0.6,
        0.0,
    ]

    ax.plot(
        tile_x,
        tile_y,
        linewidth=2,
        linestyle="--",
        label="Calibration tile",
    )

    # --------------------------------------------------------
    # Camera reference.
    #
    # This is NOT the actual camera 3D position.
    #
    # It is simply the origin of our selected ground
    # coordinate system.
    # --------------------------------------------------------

    ax.scatter(
        0.0,
        0.0,
        marker="^",
        s=140,
        label="Ground reference P1",
    )

    # --------------------------------------------------------
    # Axis.
    # --------------------------------------------------------

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_xlim(
        X_MIN_M,
        X_MAX_M,
    )

    ax.set_ylim(
        Y_MIN_M,
        Y_MAX_M,
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
        "Football Ground-Plane Projection"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "top_view_trajectory.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(fig)

    print(
        f"Saved:\n{output_path}"
    )


# ============================================================
# DIRECTION PLOT
# ============================================================

def create_direction_plot(
    df,
):

    print(
        "Creating direction plot..."
    )

    fig, ax = plt.subplots(
        figsize=(11, 14)
    )

    for track_id, group in df.groupby(
        "track_id"
    ):

        group = group.sort_values(
            "frame"
        )

        x = group[
            "ground_x_m"
        ].to_numpy()

        y = group[
            "ground_y_m"
        ].to_numpy()

        ax.plot(
            x,
            y,
            linewidth=2,
            label=f"Track {track_id}",
        )

        # ----------------------------------------------------
        # Add direction arrows every ~15 frames.
        # ----------------------------------------------------

        if len(group) >= 3:

            indices = np.arange(
                0,
                len(group) - 1,
                15,
            )

            for i in indices:

                dx = (
                    x[i + 1]
                    - x[i]
                )

                dy = (
                    y[i + 1]
                    - y[i]
                )

                magnitude = np.sqrt(
                    dx ** 2
                    + dy ** 2
                )

                if magnitude < 1e-8:

                    continue

                # Normalize arrow.
                dx /= magnitude
                dy /= magnitude

                arrow_length = 0.20

                ax.arrow(
                    x[i],
                    y[i],
                    dx * arrow_length,
                    dy * arrow_length,
                    head_width=0.06,
                    head_length=0.10,
                    length_includes_head=True,
                )

    # --------------------------------------------------------
    # Calibration tile.
    # --------------------------------------------------------

    ax.plot(
        [
            0.0,
            0.6,
            0.6,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.6,
            0.6,
            0.0,
        ],
        linestyle="--",
        linewidth=2,
        label="0.60 m reference tile",
    )

    ax.scatter(
        0,
        0,
        marker="^",
        s=140,
        label="Ground reference P1",
    )

    ax.set_xlim(
        X_MIN_M,
        X_MAX_M,
    )

    ax.set_ylim(
        Y_MIN_M,
        Y_MAX_M,
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
        "Football Top-View Trajectory and Direction"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "top_view_trajectory_with_direction.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(fig)

    print(
        f"Saved:\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 21 - TOP-VIEW FOOTBALL TRAJECTORY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load.
    # --------------------------------------------------------

    df = load_reconstruction()

    H = load_homography()

    # --------------------------------------------------------
    # Project.
    # --------------------------------------------------------

    result = project_to_ground(
        df,
        H,
    )

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    print_statistics(
        result
    )

    # --------------------------------------------------------
    # Save CSV.
    # --------------------------------------------------------

    save_csv(
        result
    )

    # --------------------------------------------------------
    # Plots.
    # --------------------------------------------------------

    create_top_view_plot(
        result
    )

    create_direction_plot(
        result
    )

    # --------------------------------------------------------
    # Final.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 21 COMPLETE")
    print("=" * 70)

    print()
    print(
        "The football trajectory has been projected "
        "onto the calibrated ground plane."
    )

    print()
    print(
        "IMPORTANT INTERPRETATION:"
    )

    print(
        "For airborne portions of the trajectory, "
        "the top-view point represents where the "
        "camera ray through the ball intersects the "
        "ground plane."
    )

    print()
    print(
        "It is therefore a ground-plane projection, "
        "not the actual 3D center of the airborne ball."
    )

    print()
    print(
        "The absolute metric scale depends on the "
        "assumed 0.60 x 0.60 m tile size."
    )

    print()
    print(
        "Next: combine the ground projection with "
        "the reconstructed 3D trajectory."
    )


if __name__ == "__main__":
    main()