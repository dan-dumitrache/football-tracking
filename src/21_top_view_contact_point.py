"""
21_top_view_contact_point.py

Corrected ground-plane projection for a football that is
rolling / remaining in contact with the floor.

IMPORTANT GEOMETRIC DISTINCTION
--------------------------------

The reconstructed 3D position represents approximately the
CENTER of the football.

For the ground-plane/top-view map, however, we want the
football's CONTACT POINT with the floor.

Given:

    center = (u, v)
    diameter = d pixels

we approximate the contact point as:

    u_contact = u
    v_contact = v + d/2

because image Y increases downward.

This is an approximation. The most accurate version would use
the actual segmentation mask and estimate the contact point
from the bottom of the ball silhouette.

The current CSV does not contain the complete mask, so the
diameter-based approximation is the appropriate next step.

INPUT
-----

data/output/reconstruction_3d/reconstructed_ball_3d.csv

calibration/ground_homography.json


OUTPUT
------

data/output/top_view/

    top_view_contact_trajectory.csv
    center_vs_contact_projection.png
    contact_point_trajectory.png
    contact_point_direction.png
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

X_MIN_M = -1.5
X_MAX_M = 3.0

Y_MIN_M = -1.0
Y_MAX_M = 16.0


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_data():

    print("=" * 70)
    print("STEP 21.1 - FOOTBALL CONTACT-POINT PROJECTION")
    print("=" * 70)

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Could not find:\n{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print()
    print(
        f"CSV:\n{INPUT_CSV}"
    )

    print(
        f"Rows: {len(df)}"
    )

    required = [
        "track_id",
        "frame",
        "time_seconds",
        "pixel_x",
        "pixel_y",
        "diameter_px",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()

    df = df[
        np.isfinite(
            df["pixel_x"]
        )
        &
        np.isfinite(
            df["pixel_y"]
        )
        &
        np.isfinite(
            df["diameter_px"]
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

    print()
    print(
        f"Valid observations: {len(df)}"
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

    if not HOMOGRAPHY_JSON.exists():

        raise FileNotFoundError(
            f"Could not find:\n{HOMOGRAPHY_JSON}"
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
            "Homography must be 3x3."
        )

    print()
    print("=" * 70)
    print("GROUND HOMOGRAPHY")
    print("=" * 70)

    print()
    print(H)

    return H


# ============================================================
# PROJECT IMAGE POINTS
# ============================================================

def project_points(
    points,
    H,
):

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    points = points.reshape(
        -1,
        1,
        2,
    )

    projected = cv2.perspectiveTransform(
        points,
        H,
    )

    return projected.reshape(
        -1,
        2,
    )


# ============================================================
# CALCULATE CONTACT POINT
# ============================================================

def calculate_contact_points(
    df,
):

    result = df.copy()

    # --------------------------------------------------------
    # Football center.
    # --------------------------------------------------------

    result[
        "pixel_center_x"
    ] = result[
        "pixel_x"
    ]

    result[
        "pixel_center_y"
    ] = result[
        "pixel_y"
    ]

    # --------------------------------------------------------
    # Approximate ground-contact point.
    #
    # The football diameter is in pixels.
    #
    # Therefore:
    #
    # radius ≈ diameter / 2
    #
    # Image coordinates:
    #
    # x → right
    # y → down
    #
    # so the bottom point is approximately:
    #
    # (x, y + radius)
    # --------------------------------------------------------

    result[
        "pixel_contact_x"
    ] = result[
        "pixel_x"
    ]

    result[
        "pixel_contact_y"
    ] = (
        result["pixel_y"]
        + result["diameter_px"] / 2.0
    )

    return result


# ============================================================
# PROJECT CENTER AND CONTACT
# ============================================================

def project_center_and_contact(
    df,
    H,
):

    result = calculate_contact_points(
        df
    )

    # --------------------------------------------------------
    # Center projection.
    # --------------------------------------------------------

    center_points = np.column_stack(
        [
            result[
                "pixel_center_x"
            ],
            result[
                "pixel_center_y"
            ],
        ]
    )

    center_ground = project_points(
        center_points,
        H,
    )

    result[
        "ground_center_x_m"
    ] = center_ground[:, 0]

    result[
        "ground_center_y_m"
    ] = center_ground[:, 1]

    # --------------------------------------------------------
    # Contact projection.
    # --------------------------------------------------------

    contact_points = np.column_stack(
        [
            result[
                "pixel_contact_x"
            ],
            result[
                "pixel_contact_y"
            ],
        ]
    )

    contact_ground = project_points(
        contact_points,
        H,
    )

    result[
        "ground_contact_x_m"
    ] = contact_ground[:, 0]

    result[
        "ground_contact_y_m"
    ] = contact_ground[:, 1]

    return result


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    df,
):

    print()
    print("=" * 70)
    print("CONTACT-POINT TRAJECTORY")
    print("=" * 70)

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
            "  Center ground X: "
            f"{group['ground_center_x_m'].min():.3f} -> "
            f"{group['ground_center_x_m'].max():.3f} m"
        )

        print(
            "  Center ground Y: "
            f"{group['ground_center_y_m'].min():.3f} -> "
            f"{group['ground_center_y_m'].max():.3f} m"
        )

        print(
            "  Contact ground X: "
            f"{group['ground_contact_x_m'].min():.3f} -> "
            f"{group['ground_contact_x_m'].max():.3f} m"
        )

        print(
            "  Contact ground Y: "
            f"{group['ground_contact_y_m'].min():.3f} -> "
            f"{group['ground_contact_y_m'].max():.3f} m"
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    df,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "top_view_contact_trajectory.csv"
    )

    columns = [
        "track_id",
        "frame",
        "time_seconds",

        "pixel_center_x",
        "pixel_center_y",

        "diameter_px",

        "pixel_contact_x",
        "pixel_contact_y",

        "ground_center_x_m",
        "ground_center_y_m",

        "ground_contact_x_m",
        "ground_contact_y_m",
    ]

    optional = [
        "depth_m",
        "X_m",
        "Y_m",
        "distance_m",
        "speed_mps",
    ]

    for column in optional:

        if column in df.columns:

            columns.append(
                column
            )

    df[
        columns
    ].to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved:\n{output_path}"
    )


# ============================================================
# CENTER VS CONTACT PLOT
# ============================================================

def plot_center_vs_contact(
    df,
):

    fig, ax = plt.subplots(
        figsize=(11, 14)
    )

    for track_id, group in df.groupby(
        "track_id"
    ):

        group = group.sort_values(
            "frame"
        )

        # Center
        ax.plot(
            group[
                "ground_center_x_m"
            ],
            group[
                "ground_center_y_m"
            ],
            linestyle="--",
            linewidth=1.2,
            label=(
                f"Track {track_id} "
                "center"
            ),
        )

        # Contact
        ax.plot(
            group[
                "ground_contact_x_m"
            ],
            group[
                "ground_contact_y_m"
            ],
            linewidth=2,
            label=(
                f"Track {track_id} "
                "contact"
            ),
        )

    # Calibration tile.
    ax.plot(
        [
            0,
            0.6,
            0.6,
            0,
            0,
        ],
        [
            0,
            0,
            0.6,
            0.6,
            0,
        ],
        linestyle=":",
        linewidth=2,
        label="0.60 m reference tile",
    )

    ax.scatter(
        0,
        0,
        marker="^",
        s=120,
        label="P1",
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
        "Football Center vs Ground-Contact Projection"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "center_vs_contact_projection.png"
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
# CONTACT TRAJECTORY
# ============================================================

def plot_contact_trajectory(
    df,
):

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
            "ground_contact_x_m"
        ]

        y = group[
            "ground_contact_y_m"
        ]

        ax.plot(
            x,
            y,
            linewidth=2,
            label=f"Track {track_id}",
        )

        # Start
        first = group.iloc[0]

        ax.scatter(
            first[
                "ground_contact_x_m"
            ],
            first[
                "ground_contact_y_m"
            ],
            marker="o",
            s=80,
        )

        # End
        last = group.iloc[-1]

        ax.scatter(
            last[
                "ground_contact_x_m"
            ],
            last[
                "ground_contact_y_m"
            ],
            marker="X",
            s=100,
        )

    # Calibration tile.
    ax.plot(
        [
            0,
            0.6,
            0.6,
            0,
            0,
        ],
        [
            0,
            0,
            0.6,
            0.6,
            0,
        ],
        linestyle="--",
        linewidth=2,
        label="0.60 m reference tile",
    )

    ax.scatter(
        0,
        0,
        marker="^",
        s=120,
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
        "Football Ground-Contact Trajectory"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "contact_point_trajectory.png"
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
# DIRECTION
# ============================================================

def plot_direction(
    df,
):

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
            "ground_contact_x_m"
        ].to_numpy()

        y = group[
            "ground_contact_y_m"
        ].to_numpy()

        ax.plot(
            x,
            y,
            linewidth=2,
            label=f"Track {track_id}",
        )

        # ----------------------------------------------------
        # Direction arrows.
        # ----------------------------------------------------

        for i in range(
            0,
            len(x) - 1,
            15,
        ):

            dx = (
                x[i + 1]
                - x[i]
            )

            dy = (
                y[i + 1]
                - y[i]
            )

            magnitude = np.sqrt(
                dx * dx
                + dy * dy
            )

            if magnitude < 1e-8:

                continue

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

    # Calibration tile.
    ax.plot(
        [
            0,
            0.6,
            0.6,
            0,
            0,
        ],
        [
            0,
            0,
            0.6,
            0.6,
            0,
        ],
        linestyle="--",
        linewidth=2,
        label="0.60 m reference tile",
    )

    ax.scatter(
        0,
        0,
        marker="^",
        s=120,
        label="P1",
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
        "Football Ground-Contact Trajectory and Direction"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "contact_point_direction.png"
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

    df = load_data()

    H = load_homography()

    print()
    print("=" * 70)
    print("CALCULATING BALL CONTACT POINT")
    print("=" * 70)

    result = project_center_and_contact(
        df,
        H,
    )

    print_statistics(
        result
    )

    save_csv(
        result
    )

    print()
    print("=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    plot_center_vs_contact(
        result
    )

    plot_contact_trajectory(
        result
    )

    plot_direction(
        result
    )

    print()
    print("=" * 70)
    print("STEP 21.1 COMPLETE")
    print("=" * 70)

    print()
    print(
        "The football center and estimated ground-contact "
        "point have both been projected onto the floor plane."
    )

    print()
    print(
        "For the top-view map, the contact-point trajectory "
        "should be preferred."
    )

    print()
    print(
        "The contact point is currently estimated from "
        "the ball center and detected pixel diameter."
    )

    print()
    print(
        "The 60 cm tile dimension remains an assumed metric "
        "scale rather than a measured value."
    )


if __name__ == "__main__":
    main()