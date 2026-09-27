"""
18_smooth_3d.py

Robust temporal filtering of the reconstructed 3D football trajectory.

INPUT
-----
data/output/reconstruction_3d/reconstructed_ball_3d.csv

OUTPUT
------
data/output/reconstruction_3d/smoothed_ball_3d.csv

The script keeps the RAW reconstruction intact and creates
additional SMOOTHED quantities.

RAW:
    depth_m
    X_m
    Y_m
    distance_m
    speed_mps

SMOOTHED:
    depth_m_smoothed
    X_m_smoothed
    Y_m_smoothed
    distance_m_smoothed
    Vx_mps_smoothed
    Vy_mps_smoothed
    Vz_mps_smoothed
    speed_mps_smoothed

Important:
------------
Track 0 and Track 1 are processed independently.

The gap between the two visible portions of the video is
never interpolated.

Processing pipeline:

    raw 3D
       |
       v
    median filter
       |
       v
    robust temporal smoothing
       |
       v
    smoothed 3D position
       |
       v
    velocity calculation
       |
       v
    diagnostic plots

This is a filtering step, not a replacement for actual
depth sensing.
"""

from pathlib import Path

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "reconstruction_3d"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "smoothed_ball_3d.csv"
)


# ============================================================
# FILTER CONFIGURATION
# ============================================================

# Video frame rate.
FPS = 30.0


# ------------------------------------------------------------
# Median filter
# ------------------------------------------------------------
#
# Median filtering is particularly useful for removing
# isolated spikes while preserving the overall trajectory.
#
# Must be odd.
#
MEDIAN_WINDOW = 5


# ------------------------------------------------------------
# Rolling median window
# ------------------------------------------------------------
#
# Used for robust smoothing after the initial median filter.
#
# Must be odd.
#
SMOOTHING_WINDOW = 9


# ------------------------------------------------------------
# Maximum physically reasonable frame-to-frame depth change
# ------------------------------------------------------------
#
# This is NOT used as a hard physical speed limit.
#
# It is only used for identifying extreme measurement
# outliers before smoothing.
#
# At 30 FPS:
#
# 10 m/s corresponds to ~0.33 m per frame.
#
# We deliberately use a generous threshold.
#
MAX_DEPTH_CHANGE_PER_FRAME_M = 0.75


# ------------------------------------------------------------
# Maximum change in ball diameter between consecutive frames
# ------------------------------------------------------------
#
# Again, this is only an OUTLIER DETECTION criterion.
#
MAX_DIAMETER_CHANGE_PX = 12.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def rolling_median(
    series,
    window,
):
    """
    Apply centered rolling median filtering.
    """

    return (
        series
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .median()
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_reconstruction():
    """
    Load the Step 17 reconstruction.
    """

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            "\nCould not find Step 17 output:\n"
            f"{INPUT_CSV}\n\n"
            "Run first:\n"
            "python src\\17_reconstruct_3d.py"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print("=" * 70)
    print("INPUT")
    print("=" * 70)

    print(
        f"CSV: {INPUT_CSV}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print()
    print("Columns:")
    print(list(df.columns))

    required = [
        "track_id",
        "frame",
        "time_seconds",
        "pixel_x",
        "pixel_y",
        "diameter_px",
        "depth_m",
        "X_m",
        "Y_m",
        "distance_m",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "frame",
        "time_seconds",
        "pixel_x",
        "pixel_y",
        "diameter_px",
        "depth_m",
        "X_m",
        "Y_m",
        "distance_m",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "track_id",
            "frame",
            "time_seconds",
            "diameter_px",
            "depth_m",
            "X_m",
            "Y_m",
            "distance_m",
        ]
    )

    df["frame"] = (
        df["frame"]
        .astype(int)
    )

    df = df.sort_values(
        [
            "track_id",
            "frame",
        ]
    ).reset_index(drop=True)

    print()
    print(
        f"Valid observations: {len(df)}"
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        print(
            f"Track {track_id}: "
            f"{len(group)} observations, "
            f"frames "
            f"{group['frame'].min()} -> "
            f"{group['frame'].max()}"
        )

    return df


# ============================================================
# OUTLIER DETECTION
# ============================================================

def identify_measurement_outliers(group):
    """
    Identify suspicious diameter/depth measurements.

    We do NOT immediately delete these measurements.

    Instead, they receive a boolean flag and the smoother
    gets a chance to suppress their influence.
    """

    group = group.copy()

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Diameter change
    # --------------------------------------------------------

    diameter_change = (
        group["diameter_px"]
        .diff()
        .abs()
    )

    group["diameter_change_px"] = (
        diameter_change
    )

    group["diameter_outlier"] = (
        diameter_change
        > MAX_DIAMETER_CHANGE_PX
    )

    # --------------------------------------------------------
    # Depth change
    # --------------------------------------------------------

    depth_change = (
        group["depth_m"]
        .diff()
        .abs()
    )

    group["depth_change_m"] = (
        depth_change
    )

    group["depth_outlier"] = (
        depth_change
        > MAX_DEPTH_CHANGE_PER_FRAME_M
    )

    # --------------------------------------------------------
    # Combined outlier flag
    # --------------------------------------------------------

    group["measurement_outlier"] = (
        group["diameter_outlier"]
        | group["depth_outlier"]
    )

    return group


# ============================================================
# ROBUST TRACK SMOOTHING
# ============================================================

def smooth_track(group):
    """
    Smooth one continuous track.

    Important:
    -----------
    No information from another track is used.

    The raw values remain untouched.
    """

    group = group.copy()

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 1. Preserve raw values explicitly
    # --------------------------------------------------------

    group["depth_m_raw"] = (
        group["depth_m"]
    )

    group["X_m_raw"] = (
        group["X_m"]
    )

    group["Y_m_raw"] = (
        group["Y_m"]
    )

    group["distance_m_raw"] = (
        group["distance_m"]
    )

    group["diameter_px_raw"] = (
        group["diameter_px"]
    )

    # --------------------------------------------------------
    # 2. Identify suspicious measurements
    # --------------------------------------------------------

    group = identify_measurement_outliers(
        group
    )

    # --------------------------------------------------------
    # 3. Robust median filtering
    # --------------------------------------------------------
    #
    # Median filtering is less sensitive to isolated spikes
    # than a normal moving average.
    # --------------------------------------------------------

    group["depth_m_median"] = rolling_median(
        group["depth_m"],
        MEDIAN_WINDOW,
    )

    group["X_m_median"] = rolling_median(
        group["X_m"],
        MEDIAN_WINDOW,
    )

    group["Y_m_median"] = rolling_median(
        group["Y_m"],
        MEDIAN_WINDOW,
    )

    # --------------------------------------------------------
    # 4. Second smoothing stage
    # --------------------------------------------------------
    #
    # Apply another centered rolling median to produce the
    # final robust trajectory.
    # --------------------------------------------------------

    group["depth_m_smoothed"] = rolling_median(
        group["depth_m_median"],
        SMOOTHING_WINDOW,
    )

    group["X_m_smoothed"] = rolling_median(
        group["X_m_median"],
        SMOOTHING_WINDOW,
    )

    group["Y_m_smoothed"] = rolling_median(
        group["Y_m_median"],
        SMOOTHING_WINDOW,
    )

    # --------------------------------------------------------
    # 5. Recalculate distance from smoothed coordinates
    # --------------------------------------------------------

    group["distance_m_smoothed"] = np.sqrt(
        group["X_m_smoothed"] ** 2
        + group["Y_m_smoothed"] ** 2
        + group["depth_m_smoothed"] ** 2
    )

    return group


# ============================================================
# PROCESS ALL TRACKS
# ============================================================

def smooth_all_tracks(df):
    """
    Process Track 0 and Track 1 independently.
    """

    groups = []

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        print()
        print(
            f"Smoothing Track {track_id}..."
        )

        processed = smooth_track(
            group
        )

        # Explicitly preserve ID.
        processed["track_id"] = track_id

        groups.append(
            processed
        )

    if not groups:

        raise ValueError(
            "No tracks available for smoothing."
        )

    result = pd.concat(
        groups,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "track_id",
            "frame",
        ]
    ).reset_index(drop=True)

    return result


# ============================================================
# VELOCITY FROM SMOOTHED TRAJECTORY
# ============================================================

def calculate_smoothed_velocity(group):
    """
    Calculate velocity from the SMOOTHED 3D trajectory.

    This is deliberately performed after filtering.

    That is important because numerical differentiation
    amplifies measurement noise.
    """

    group = group.copy()

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    n = len(group)

    vx = np.full(
        n,
        np.nan,
        dtype=float,
    )

    vy = np.full(
        n,
        np.nan,
        dtype=float,
    )

    vz = np.full(
        n,
        np.nan,
        dtype=float,
    )

    if n >= 2:

        time = (
            group["time_seconds"]
            .to_numpy(
                dtype=float
            )
        )

        x = (
            group["X_m_smoothed"]
            .to_numpy(
                dtype=float
            )
        )

        y = (
            group["Y_m_smoothed"]
            .to_numpy(
                dtype=float
            )
        )

        z = (
            group["depth_m_smoothed"]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Finite differences using actual timestamps.
        # ----------------------------------------------------

        vx = np.gradient(
            x,
            time,
        )

        vy = np.gradient(
            y,
            time,
        )

        vz = np.gradient(
            z,
            time,
        )

    group["Vx_mps_smoothed"] = vx
    group["Vy_mps_smoothed"] = vy
    group["Vz_mps_smoothed"] = vz

    group["speed_mps_smoothed"] = np.sqrt(
        vx ** 2
        + vy ** 2
        + vz ** 2
    )

    return group


def calculate_all_velocities(df):
    """
    Calculate smoothed velocity independently for every track.
    """

    groups = []

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        print(
            f"Calculating smoothed velocity "
            f"for Track {track_id}..."
        )

        processed = calculate_smoothed_velocity(
            group
        )

        processed["track_id"] = track_id

        groups.append(
            processed
        )

    result = pd.concat(
        groups,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "track_id",
            "frame",
        ]
    ).reset_index(drop=True)

    return result


# ============================================================
# SAVE
# ============================================================

def save_results(df):
    """
    Save raw + smoothed 3D reconstruction.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Saved:\n{OUTPUT_CSV}"
    )


# ============================================================
# PLOT — DEPTH COMPARISON
# ============================================================

def plot_depth_comparison(df):
    """
    Raw vs smoothed depth.
    """

    plt.figure(
        figsize=(13, 7)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        plt.plot(
            group["time_seconds"],
            group["depth_m_raw"],
            linestyle="--",
            alpha=0.45,
            label=f"Track {track_id} raw",
        )

        plt.plot(
            group["time_seconds"],
            group["depth_m_smoothed"],
            linewidth=2,
            label=f"Track {track_id} smoothed",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Depth Z (m)"
    )

    plt.title(
        "Raw vs Smoothed Ball Depth"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "depth_raw_vs_smoothed.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# PLOT — DISTANCE COMPARISON
# ============================================================

def plot_distance_comparison(df):
    """
    Raw vs smoothed camera-to-ball distance.
    """

    plt.figure(
        figsize=(13, 7)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        plt.plot(
            group["time_seconds"],
            group["distance_m_raw"],
            linestyle="--",
            alpha=0.45,
            label=f"Track {track_id} raw",
        )

        plt.plot(
            group["time_seconds"],
            group["distance_m_smoothed"],
            linewidth=2,
            label=f"Track {track_id} smoothed",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Distance from camera (m)"
    )

    plt.title(
        "Raw vs Smoothed Ball Distance"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "distance_raw_vs_smoothed.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# PLOT — SPEED COMPARISON
# ============================================================

def plot_speed_comparison(df):
    """
    Compare the original raw velocity against the velocity
    calculated after trajectory smoothing.
    """

    plt.figure(
        figsize=(13, 7)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        # Raw speed from Step 17.
        if "speed_mps" in group.columns:

            plt.plot(
                group["time_seconds"],
                group["speed_mps"],
                linestyle="--",
                alpha=0.35,
                label=f"Track {track_id} raw",
            )

        # Smoothed speed.
        plt.plot(
            group["time_seconds"],
            group["speed_mps_smoothed"],
            linewidth=2,
            label=f"Track {track_id} smoothed",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Speed (m/s)"
    )

    plt.title(
        "Raw vs Smoothed 3D Ball Speed"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "speed_raw_vs_smoothed.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# PLOT — X/Z TRAJECTORY
# ============================================================

def plot_xz_comparison(df):
    """
    Compare raw and smoothed X/Z trajectory.
    """

    plt.figure(
        figsize=(11, 8)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        plt.plot(
            group["X_m_raw"],
            group["depth_m_raw"],
            linestyle="--",
            alpha=0.35,
            label=f"Track {track_id} raw",
        )

        plt.plot(
            group["X_m_smoothed"],
            group["depth_m_smoothed"],
            linewidth=2,
            label=f"Track {track_id} smoothed",
        )

    plt.xlabel(
        "X (m)"
    )

    plt.ylabel(
        "Z / Depth (m)"
    )

    plt.title(
        "Raw vs Smoothed X-Z Ball Trajectory"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "trajectory_xz_raw_vs_smoothed.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# PLOT — OUTLIERS
# ============================================================

def plot_outliers(df):
    """
    Visualize measurements identified as suspicious.
    """

    plt.figure(
        figsize=(13, 7)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        normal = (
            ~group["measurement_outlier"]
        )

        outlier = (
            group["measurement_outlier"]
        )

        plt.scatter(
            group.loc[
                normal,
                "time_seconds"
            ],
            group.loc[
                normal,
                "depth_m"
            ],
            s=12,
            label=f"Track {track_id} normal",
        )

        plt.scatter(
            group.loc[
                outlier,
                "time_seconds"
            ],
            group.loc[
                outlier,
                "depth_m"
            ],
            s=45,
            marker="x",
            label=f"Track {track_id} flagged",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Raw depth (m)"
    )

    plt.title(
        "Measurements Flagged as Potential Outliers"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "depth_outliers.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(df):
    """
    Print raw vs smoothed statistics.
    """

    print()
    print("=" * 70)
    print("SMOOTHING RESULTS")
    print("=" * 70)

    total_outliers = int(
        df["measurement_outlier"]
        .sum()
    )

    print(
        f"Flagged measurements: "
        f"{total_outliers} / {len(df)}"
    )

    if len(df) > 0:

        print(
            f"Outlier percentage: "
            f"{100.0 * total_outliers / len(df):.2f}%"
        )

    print()

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        print(
            f"Track {track_id}"
        )

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        raw_depth = (
            group["depth_m_raw"]
        )

        smooth_depth = (
            group["depth_m_smoothed"]
        )

        print(
            f"  Raw depth:             "
            f"{raw_depth.min():.2f} -> "
            f"{raw_depth.max():.2f} m"
        )

        print(
            f"  Smoothed depth:        "
            f"{smooth_depth.min():.2f} -> "
            f"{smooth_depth.max():.2f} m"
        )

        # ----------------------------------------------------
        # Speed
        # ----------------------------------------------------

        raw_speed = (
            group["speed_mps"]
            .dropna()
        )

        smooth_speed = (
            group["speed_mps_smoothed"]
            .dropna()
        )

        if len(raw_speed) > 0:

            print(
                f"  Raw median speed:      "
                f"{raw_speed.median():.2f} m/s"
            )

            print(
                f"  Raw maximum speed:     "
                f"{raw_speed.max():.2f} m/s"
            )

        if len(smooth_speed) > 0:

            print(
                f"  Smooth median speed:   "
                f"{smooth_speed.median():.2f} m/s"
            )

            print(
                f"  Smooth maximum speed:  "
                f"{smooth_speed.max():.2f} m/s"
            )

        # ----------------------------------------------------
        # Position change
        # ----------------------------------------------------

        raw_range = (
            raw_depth.max()
            - raw_depth.min()
        )

        smooth_range = (
            smooth_depth.max()
            - smooth_depth.min()
        )

        print(
            f"  Raw depth variation:   "
            f"{raw_range:.2f} m"
        )

        print(
            f"  Smooth depth variation:"
            f" {smooth_range:.2f} m"
        )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 18 - ROBUST 3D SMOOTHING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load Step 17 reconstruction.
    # --------------------------------------------------------

    df = load_reconstruction()

    # --------------------------------------------------------
    # 2. Smooth each track independently.
    # --------------------------------------------------------

    df = smooth_all_tracks(
        df
    )

    # --------------------------------------------------------
    # 3. Calculate velocity AFTER smoothing.
    # --------------------------------------------------------

    df = calculate_all_velocities(
        df
    )

    # --------------------------------------------------------
    # 4. Save.
    # --------------------------------------------------------

    save_results(
        df
    )

    # --------------------------------------------------------
    # 5. Generate diagnostic plots.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    plots = [
        plot_depth_comparison(df),
        plot_distance_comparison(df),
        plot_speed_comparison(df),
        plot_xz_comparison(df),
        plot_outliers(df),
    ]

    for plot in plots:

        print(
            f"Saved: {plot}"
        )

    # --------------------------------------------------------
    # 6. Statistics.
    # --------------------------------------------------------

    print_statistics(
        df
    )

    # --------------------------------------------------------
    # Finished.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 18 COMPLETE")
    print("=" * 70)

    print()
    print(
        "The raw reconstruction has been preserved."
    )

    print(
        "A smoothed 3D trajectory and smoothed velocity "
        "have been generated."
    )

    print()
    print(
        "Next: inspect the raw-vs-smoothed plots before "
        "moving to the ground-plane/top-view reconstruction."
    )


if __name__ == "__main__":
    main()