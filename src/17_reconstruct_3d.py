"""
17_reconstruct_3d.py

Football 3D reconstruction from the refined 2D ball tracks.

INPUT
-----
data/output/refined_ball_track.csv

The current tracker CSV contains:

    track_id
    frame
    time_seconds
    detected
    measurement_x
    measurement_y
    diameter
    area
    circularity
    aspect_ratio
    quality
    vx_pixels_per_second
    vy_pixels_per_second

METHOD
------
We estimate the ball's 3D position from its apparent size in the RGB image.

For a known object diameter:

        Z = f * D / d

where:

    Z = distance/depth from camera [m]
    f = focal length [pixels]
    D = real football diameter [m]
    d = apparent football diameter [pixels]

Then:

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

where:

    u, v = ball center in image pixels
    cx, cy = principal point
    fx, fy = focal lengths in pixels

IMPORTANT
---------
This is a MONOCULAR geometric estimate.

It is NOT the same thing as direct RealSense depth.

We currently estimate camera intrinsics from the nominal D435i RGB
field of view. If the original RealSense calibration or .bag file
becomes available later, those values should replace the estimates.

The two visible ball segments are deliberately treated independently:

    Track 0: frames ~19 -> 205
    Track 1: frames ~389 -> 461

There is a genuine visibility gap between them, so no interpolation
is performed across that gap.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "refined_ball_track.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "reconstruction_3d"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "reconstructed_ball_3d.csv"
)


# ============================================================
# VIDEO / CAMERA CONFIGURATION
# ============================================================

# RGB video resolution
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Nominal Intel RealSense D435i RGB FOV
#
# Approximate values from Intel D400-series documentation.
RGB_HORIZONTAL_FOV_DEG = 69.0
RGB_VERTICAL_FOV_DEG = 42.0

# Approximate standard football diameter
BALL_DIAMETER_M = 0.22

# Approximate principal point.
#
# Without factory calibration we assume the optical center
# is approximately the image center.
CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0

# Median filter applied to the measured ball diameter.
#
# This reduces frame-to-frame segmentation noise.
#
# Must be an odd number.
DIAMETER_SMOOTH_WINDOW = 5

# Do not allow absurdly small diameters to create enormous depth.
MIN_DIAMETER_PX = 2.0

# Video FPS.
# The inspected video was 30 FPS.
FPS = 30.0


# ============================================================
# CAMERA MODEL
# ============================================================

def calculate_focal_lengths(
    width,
    height,
    horizontal_fov_deg,
    vertical_fov_deg,
):
    """
    Estimate focal lengths in pixels from image size and FOV.

    Pinhole camera relationship:

        fx = (W / 2) / tan(HFOV / 2)
        fy = (H / 2) / tan(VFOV / 2)
    """

    horizontal_fov_rad = np.deg2rad(
        horizontal_fov_deg
    )

    vertical_fov_rad = np.deg2rad(
        vertical_fov_deg
    )

    fx = (
        width / 2.0
        / np.tan(horizontal_fov_rad / 2.0)
    )

    fy = (
        height / 2.0
        / np.tan(vertical_fov_rad / 2.0)
    )

    return float(fx), float(fy)


# ============================================================
# INPUT VALIDATION
# ============================================================

def check_required_columns(df):
    """
    Verify that the tracker CSV contains the columns required
    for reconstruction.

    The current refined tracker uses the exact names below.
    """

    required_columns = [
        "track_id",
        "frame",
        "time_seconds",
        "measurement_x",
        "measurement_y",
        "diameter",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
            + "\n\n"
            + "Columns actually found:\n"
            + "\n".join(
                f"  - {column}"
                for column in df.columns
            )
        )


# ============================================================
# LOAD 2D TRACK
# ============================================================

def load_track(csv_path):
    """
    Load the refined tracking CSV and normalize the fields
    needed for 3D reconstruction.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nCould not find input CSV:\n{csv_path}\n"
        )

    df = pd.read_csv(csv_path)

    print("=" * 70)
    print("INPUT DATA")
    print("=" * 70)

    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print()

    print("Columns:")
    print(list(df.columns))
    print()

    check_required_columns(df)

    # --------------------------------------------------------
    # Create a clean dataframe.
    # --------------------------------------------------------

    result = pd.DataFrame()

    result["track_id"] = df["track_id"]

    result["frame"] = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    result["time_seconds"] = pd.to_numeric(
        df["time_seconds"],
        errors="coerce",
    )

    result["pixel_x"] = pd.to_numeric(
        df["measurement_x"],
        errors="coerce",
    )

    result["pixel_y"] = pd.to_numeric(
        df["measurement_y"],
        errors="coerce",
    )

    result["diameter_px"] = pd.to_numeric(
        df["diameter"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove rows with invalid essential measurements.
    # --------------------------------------------------------

    before = len(result)

    result = result.dropna(
        subset=[
            "track_id",
            "frame",
            "time_seconds",
            "pixel_x",
            "pixel_y",
            "diameter_px",
        ]
    )

    removed = before - len(result)

    if removed > 0:
        print(
            f"Removed {removed} rows with invalid "
            "essential measurements."
        )

    # --------------------------------------------------------
    # Convert frame to integer.
    # --------------------------------------------------------

    result["frame"] = (
        result["frame"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Remove physically impossible image coordinates.
    # --------------------------------------------------------

    invalid_coordinates = (
        (result["pixel_x"] < 0)
        | (result["pixel_x"] >= IMAGE_WIDTH)
        | (result["pixel_y"] < 0)
        | (result["pixel_y"] >= IMAGE_HEIGHT)
    )

    if invalid_coordinates.any():

        print(
            f"Removing "
            f"{invalid_coordinates.sum()} rows "
            "with coordinates outside the image."
        )

        result = result.loc[
            ~invalid_coordinates
        ].copy()

    # --------------------------------------------------------
    # Remove invalid diameters.
    # --------------------------------------------------------

    invalid_diameter = (
        result["diameter_px"]
        <= MIN_DIAMETER_PX
    )

    if invalid_diameter.any():

        print(
            f"Removing "
            f"{invalid_diameter.sum()} rows "
            "with invalid ball diameters."
        )

        result = result.loc[
            ~invalid_diameter
        ].copy()

    # --------------------------------------------------------
    # Sort.
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "track_id",
            "frame",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print()
    print("Normalized input:")
    print(
        f"  Observations: {len(result)}"
    )

    print(
        f"  Track IDs: "
        f"{sorted(result['track_id'].unique().tolist())}"
    )

    for track_id, group in result.groupby(
        "track_id",
        sort=True,
    ):

        print(
            f"  Track {track_id}: "
            f"frames {group['frame'].min()} -> "
            f"{group['frame'].max()}, "
            f"{len(group)} observations"
        )

    return result


# ============================================================
# DIAMETER SMOOTHING
# ============================================================

def smooth_track_diameter(group):
    """
    Smooth apparent ball diameter within ONE continuous track.

    This function deliberately operates on only one track.

    Therefore Track 0 cannot influence Track 1.
    """

    group = group.copy()

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    # Keep original measurement.
    group["diameter_px_raw"] = (
        group["diameter_px"]
    )

    # Median filtering is useful because segmentation can
    # make the detected ball boundary jump by a few pixels
    # between frames.
    group["diameter_px"] = (
        group["diameter_px"]
        .rolling(
            window=DIAMETER_SMOOTH_WINDOW,
            center=True,
            min_periods=1,
        )
        .median()
    )

    return group


# ============================================================
# RECONSTRUCT 3D POSITION
# ============================================================

def reconstruct_3d(
    df,
    fx,
    fy,
):
    """
    Convert 2D ball observations into camera-relative 3D position.

    Coordinate convention:

        X = right
        Y = down
        Z = forward

    This is the same basic convention used by RealSense.
    """

    # --------------------------------------------------------
    # Process each track independently.
    #
    # We deliberately avoid groupby().apply() here because
    # Pandas versions can handle grouping columns differently.
    # Explicit processing is safer and clearer.
    # --------------------------------------------------------

    processed_groups = []

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        print(
            f"Processing diameter smoothing "
            f"for Track {track_id}..."
        )

        processed = smooth_track_diameter(
            group
        )

        # Explicitly restore track ID.
        processed["track_id"] = track_id

        processed_groups.append(
            processed
        )

    if not processed_groups:
        raise ValueError(
            "No valid tracks remain after input validation."
        )

    df = pd.concat(
        processed_groups,
        ignore_index=True,
    )

    df = df.sort_values(
        [
            "track_id",
            "frame",
        ]
    ).reset_index(drop=True)

    # ========================================================
    # DEPTH
    # ========================================================

    #
    #        f * D
    # Z = ----------
    #          d
    #
    # f = focal length in pixels
    # D = real ball diameter in meters
    # d = apparent ball diameter in pixels
    #

    df["depth_m"] = (
        fx
        * BALL_DIAMETER_M
        / df["diameter_px"].clip(
            lower=MIN_DIAMETER_PX
        )
    )

    # ========================================================
    # X POSITION
    # ========================================================

    df["X_m"] = (
        (
            df["pixel_x"]
            - CX
        )
        * df["depth_m"]
        / fx
    )

    # ========================================================
    # Y POSITION
    # ========================================================

    df["Y_m"] = (
        (
            df["pixel_y"]
            - CY
        )
        * df["depth_m"]
        / fy
    )

    # ========================================================
    # EUCLIDEAN DISTANCE
    # ========================================================

    df["distance_m"] = np.sqrt(
        df["X_m"] ** 2
        + df["Y_m"] ** 2
        + df["depth_m"] ** 2
    )

    return df


# ============================================================
# VELOCITY
# ============================================================

def calculate_track_velocity(group):
    """
    Estimate 3D velocity within one continuous track.

    Velocity components:

        Vx = dX / dt
        Vy = dY / dt
        Vz = dZ / dt

    Speed:

        sqrt(Vx² + Vy² + Vz²)

    Because this function receives only one track, there is
    absolutely no velocity calculation across the visibility gap.
    """

    group = group.copy()

    group = group.sort_values(
        "frame"
    ).reset_index(drop=True)

    n = len(group)

    # Default to NaN.
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
            .to_numpy(dtype=float)
        )

        x = (
            group["X_m"]
            .to_numpy(dtype=float)
        )

        y = (
            group["Y_m"]
            .to_numpy(dtype=float)
        )

        z = (
            group["depth_m"]
            .to_numpy(dtype=float)
        )

        # ----------------------------------------------------
        # np.gradient uses finite differences.
        #
        # We pass actual timestamps rather than assuming
        # perfectly uniform frame timing.
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

    group["Vx_mps"] = vx
    group["Vy_mps"] = vy
    group["Vz_mps"] = vz

    group["speed_mps"] = np.sqrt(
        vx ** 2
        + vy ** 2
        + vz ** 2
    )

    return group


def calculate_velocities(df):
    """
    Calculate velocity separately for every track.
    """

    velocity_groups = []

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        print(
            f"Calculating velocity for "
            f"Track {track_id}..."
        )

        processed = calculate_track_velocity(
            group
        )

        # Explicitly restore track ID.
        processed["track_id"] = track_id

        velocity_groups.append(
            processed
        )

    if not velocity_groups:
        raise ValueError(
            "No tracks available for velocity calculation."
        )

    result = pd.concat(
        velocity_groups,
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
# SAVE CSV
# ============================================================

def save_results(df):
    """
    Save reconstructed 3D trajectory.
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
        f"Saved reconstructed 3D data:\n"
        f"{OUTPUT_CSV}"
    )


# ============================================================
# PLOT 1 — X/Y/Z
# ============================================================

def plot_position_components(df):
    """
    Plot X, Y and Z as functions of time.
    """

    plt.figure(
        figsize=(12, 7)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        plt.plot(
            group["time_seconds"],
            group["X_m"],
            label=f"Track {track_id} - X",
        )

        plt.plot(
            group["time_seconds"],
            group["Y_m"],
            label=f"Track {track_id} - Y",
        )

        plt.plot(
            group["time_seconds"],
            group["depth_m"],
            label=f"Track {track_id} - Z",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Position (m)"
    )

    plt.title(
        "Estimated 3D Ball Position"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "position_3d_components.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# PLOT 2 — DISTANCE
# ============================================================

def plot_distance(df):
    """
    Plot camera-to-ball distance.
    """

    plt.figure(
        figsize=(12, 6)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        plt.plot(
            group["time_seconds"],
            group["distance_m"],
            label=f"Track {track_id}",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Distance from camera (m)"
    )

    plt.title(
        "Estimated Ball Distance from Camera"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "distance_from_camera.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# PLOT 3 — X/Z TRAJECTORY
# ============================================================

def plot_xz_trajectory(df):
    """
    Plot the ball trajectory in the horizontal/depth plane.

    X = horizontal position relative to camera
    Z = depth / forward distance
    """

    plt.figure(
        figsize=(10, 7)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        plt.plot(
            group["X_m"],
            group["depth_m"],
            marker=".",
            markersize=3,
            label=f"Track {track_id}",
        )

    plt.xlabel(
        "X (m)"
    )

    plt.ylabel(
        "Z / Depth (m)"
    )

    plt.title(
        "Estimated Ball Trajectory in X-Z Plane"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "trajectory_xz.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# PLOT 4 — SPEED
# ============================================================

def plot_speed(df):
    """
    Plot estimated 3D ball speed.
    """

    plt.figure(
        figsize=(12, 6)
    )

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        plt.plot(
            group["time_seconds"],
            group["speed_mps"],
            label=f"Track {track_id}",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Speed (m/s)"
    )

    plt.title(
        "Estimated 3D Ball Speed"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "speed_3d.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    return output


# ============================================================
# CREATE ALL PLOTS
# ============================================================

def create_plots(df):
    """
    Generate all diagnostic plots.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    outputs = []

    outputs.append(
        plot_position_components(df)
    )

    outputs.append(
        plot_distance(df)
    )

    outputs.append(
        plot_xz_trajectory(df)
    )

    outputs.append(
        plot_speed(df)
    )

    for output in outputs:
        print(output)


# ============================================================
# SANITY CHECKS
# ============================================================

def print_sanity_checks(
    df,
    fx,
    fy,
):
    """
    Print reconstruction statistics.

    These numbers are intended for physical sanity checking,
    not as final validated measurements.
    """

    print()
    print("=" * 70)
    print("CAMERA MODEL")
    print("=" * 70)

    print(
        f"Image size:              "
        f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
    )

    print(
        f"RGB horizontal FOV:      "
        f"{RGB_HORIZONTAL_FOV_DEG:.1f} deg"
    )

    print(
        f"RGB vertical FOV:        "
        f"{RGB_VERTICAL_FOV_DEG:.1f} deg"
    )

    print(
        f"Estimated fx:            "
        f"{fx:.2f} px"
    )

    print(
        f"Estimated fy:            "
        f"{fy:.2f} px"
    )

    print(
        f"Principal point:         "
        f"({CX:.1f}, {CY:.1f})"
    )

    print(
        f"Football diameter:       "
        f"{BALL_DIAMETER_M:.3f} m"
    )

    print()
    print("=" * 70)
    print("3D RECONSTRUCTION SUMMARY")
    print("=" * 70)

    print(
        f"Total observations:      "
        f"{len(df)}"
    )

    print(
        f"Tracks:                  "
        f"{df['track_id'].nunique()}"
    )

    print(
        f"Depth range:             "
        f"{df['depth_m'].min():.2f} -> "
        f"{df['depth_m'].max():.2f} m"
    )

    print(
        f"Distance range:          "
        f"{df['distance_m'].min():.2f} -> "
        f"{df['distance_m'].max():.2f} m"
    )

    print()

    # --------------------------------------------------------
    # Per-track statistics
    # --------------------------------------------------------

    for track_id, group in df.groupby(
        "track_id",
        sort=True,
    ):

        print(
            f"Track {track_id}"
        )

        print(
            f"  Frames:                "
            f"{group['frame'].min()} -> "
            f"{group['frame'].max()}"
        )

        print(
            f"  Observations:          "
            f"{len(group)}"
        )

        print(
            f"  Pixel X range:         "
            f"{group['pixel_x'].min():.1f} -> "
            f"{group['pixel_x'].max():.1f}"
        )

        print(
            f"  Pixel Y range:         "
            f"{group['pixel_y'].min():.1f} -> "
            f"{group['pixel_y'].max():.1f}"
        )

        print(
            f"  Diameter range:        "
            f"{group['diameter_px'].min():.1f} -> "
            f"{group['diameter_px'].max():.1f} px"
        )

        print(
            f"  Median diameter:       "
            f"{group['diameter_px'].median():.2f} px"
        )

        print(
            f"  Depth range:           "
            f"{group['depth_m'].min():.2f} -> "
            f"{group['depth_m'].max():.2f} m"
        )

        print(
            f"  Median depth:          "
            f"{group['depth_m'].median():.2f} m"
        )

        print(
            f"  Distance range:        "
            f"{group['distance_m'].min():.2f} -> "
            f"{group['distance_m'].max():.2f} m"
        )

        # ----------------------------------------------------
        # Velocity statistics
        # ----------------------------------------------------

        valid_speed = (
            group["speed_mps"]
            .dropna()
        )

        if len(valid_speed) > 0:

            print(
                f"  Median speed:          "
                f"{valid_speed.median():.2f} m/s"
            )

            print(
                f"  Maximum speed:         "
                f"{valid_speed.max():.2f} m/s"
            )

        else:

            print(
                "  Speed:                 "
                "not available"
            )

        print()


# ============================================================
# PHYSICAL SANITY CHECKS
# ============================================================

def print_physical_warnings(df):
    """
    Highlight values that deserve investigation.

    These are NOT automatic errors.

    They simply tell us what to inspect next.
    """

    print()
    print("=" * 70)
    print("PHYSICAL SANITY CHECKS")
    print("=" * 70)

    # --------------------------------------------------------
    # Negative / zero depth
    # --------------------------------------------------------

    invalid_depth = (
        df["depth_m"] <= 0
    )

    if invalid_depth.any():

        print(
            "WARNING: Non-positive depth values detected:"
            f" {invalid_depth.sum()}"
        )

    else:

        print(
            "OK: All reconstructed depths are positive."
        )

    # --------------------------------------------------------
    # Very distant estimates
    # --------------------------------------------------------

    far_points = (
        df["depth_m"] > 20.0
    )

    if far_points.any():

        print(
            "WARNING: "
            f"{far_points.sum()} points have "
            "estimated depth > 20 m."
        )

        print(
            "         This may indicate a small/unstable "
            "detected ball diameter."
        )

    else:

        print(
            "OK: No reconstructed depth exceeds 20 m."
        )

    # --------------------------------------------------------
    # Extremely high speed
    # --------------------------------------------------------

    valid_speed = (
        df["speed_mps"]
        .dropna()
    )

    if len(valid_speed) > 0:

        max_speed = valid_speed.max()

        if max_speed > 100.0:

            print(
                "WARNING: Maximum estimated speed is "
                f"{max_speed:.1f} m/s."
            )

            print(
                "         This is likely dominated by "
                "measurement/depth noise."
            )

        elif max_speed > 50.0:

            print(
                "NOTICE: Maximum estimated speed is "
                f"{max_speed:.1f} m/s."
            )

            print(
                "        We should inspect this before "
                "using speed as a final result."
            )

        else:

            print(
                "OK: Maximum estimated speed is "
                f"{max_speed:.1f} m/s."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FOOTBALL 3D RECONSTRUCTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Calculate approximate camera intrinsics
    # --------------------------------------------------------

    fx, fy = calculate_focal_lengths(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        horizontal_fov_deg=RGB_HORIZONTAL_FOV_DEG,
        vertical_fov_deg=RGB_VERTICAL_FOV_DEG,
    )

    # --------------------------------------------------------
    # 2. Load 2D refined tracks
    # --------------------------------------------------------

    df = load_track(
        INPUT_CSV
    )

    # --------------------------------------------------------
    # 3. Reconstruct 3D position
    # --------------------------------------------------------

    df = reconstruct_3d(
        df=df,
        fx=fx,
        fy=fy,
    )

    # --------------------------------------------------------
    # 4. Calculate 3D velocity
    # --------------------------------------------------------

    df = calculate_velocities(
        df
    )

    # --------------------------------------------------------
    # 5. Save CSV
    # --------------------------------------------------------

    save_results(
        df
    )

    # --------------------------------------------------------
    # 6. Generate plots
    # --------------------------------------------------------

    create_plots(
        df
    )

    # --------------------------------------------------------
    # 7. Print statistics
    # --------------------------------------------------------

    print_sanity_checks(
        df=df,
        fx=fx,
        fy=fy,
    )

    # --------------------------------------------------------
    # 8. Physical sanity checks
    # --------------------------------------------------------

    print_physical_warnings(
        df
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 17 COMPLETE")
    print("=" * 70)

    print()
    print(
        "Next step:"
    )

    print(
        "Validate the reconstructed 3D trajectory "
        "and inspect the generated plots."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()