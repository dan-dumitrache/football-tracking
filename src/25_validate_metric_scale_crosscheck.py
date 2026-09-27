import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Vlad\Documents\football_tracking"

RECONSTRUCTION_CSV = os.path.join(
    BASE_DIR,
    "data",
    "output",
    "reconstruction_3d",
    "reconstructed_ball_3d.csv",
)

SMOOTHED_CSV = os.path.join(
    BASE_DIR,
    "data",
    "output",
    "reconstruction_3d",
    "smoothed_ball_3d.csv",
)

HOMOGRAPHY_JSON = os.path.join(
    BASE_DIR,
    "calibration",
    "ground_homography.json",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "output",
    "validation",
    "metric_crosscheck",
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "football_ground_crosscheck.csv",
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "metric_crosscheck_summary.csv",
)

HEIGHT_PLOT = os.path.join(
    OUTPUT_DIR,
    "ball_height_vs_time.png",
)

SCALE_PLOT = os.path.join(
    OUTPUT_DIR,
    "height_error_vs_tile_size.png",
)

CAMERA_HEIGHT_PLOT = os.path.join(
    OUTPUT_DIR,
    "camera_height_vs_tile_size.png",
)


# ------------------------------------------------------------
# Camera / image parameters
# ------------------------------------------------------------

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Nominal D435/D435i RGB field of view used earlier.
RGB_HFOV_DEG = 69.0
RGB_VFOV_DEG = 42.0

FX = (IMAGE_WIDTH / 2.0) / math.tan(
    math.radians(RGB_HFOV_DEG / 2.0)
)

FY = (IMAGE_HEIGHT / 2.0) / math.tan(
    math.radians(RGB_VFOV_DEG / 2.0)
)

CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0


# ------------------------------------------------------------
# Football dimensions
# ------------------------------------------------------------

# Working value.
# IFAB regulation circumference corresponds to approximately
# 0.216 - 0.223 m diameter.
BALL_DIAMETER_M = 0.220
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0


# ------------------------------------------------------------
# Ground calibration
# ------------------------------------------------------------

REFERENCE_TILE_M = 0.60

TEST_TILE_SIZES = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# Expected center height if football is rolling on the floor.
EXPECTED_CENTER_HEIGHT_M = BALL_RADIUS_M


# Robust statistics
OUTLIER_PERCENTILE = 90.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_homography(path):
    """
    Load the image -> ground-plane homography from JSON.

    The calibration step may have saved the matrix under different
    descriptive key names, so this loader accepts several common formats.
    """

    import json
    import numpy as np

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ------------------------------------------------------------
    # Direct/common key names
    # ------------------------------------------------------------
    possible_keys = [
        "homography",
        "H",
        "H_image_to_ground",
        "H_img_to_ground",
        "image_to_ground",
        "image_to_ground_homography",
        "homography_image_to_ground",
    ]

    for key in possible_keys:
        if key in data:
            H = np.asarray(data[key], dtype=float)

            if H.shape == (3, 3):
                print(f"Found homography under JSON key: '{key}'")
                return H

    # ------------------------------------------------------------
    # Sometimes the matrix may be nested inside another dictionary
    # ------------------------------------------------------------
    for parent_key, parent_value in data.items():

        if isinstance(parent_value, dict):

            for key in possible_keys:

                if key in parent_value:
                    H = np.asarray(parent_value[key], dtype=float)

                    if H.shape == (3, 3):
                        print(
                            f"Found homography under "
                            f"'{parent_key}.{key}'"
                        )
                        return H

    # ------------------------------------------------------------
    # Last resort:
    # recursively search dictionaries for any 3x3 numeric matrix
    # ------------------------------------------------------------
    def find_matrix(obj, path="root"):

        if isinstance(obj, dict):

            for key, value in obj.items():

                result = find_matrix(
                    value,
                    f"{path}.{key}"
                )

                if result is not None:
                    return result

        elif isinstance(obj, list):

            try:
                arr = np.asarray(obj, dtype=float)

                if arr.shape == (3, 3):
                    return arr, path

            except (ValueError, TypeError):
                pass

            for i, value in enumerate(obj):

                result = find_matrix(
                    value,
                    f"{path}[{i}]"
                )

                if result is not None:
                    return result

        return None

    result = find_matrix(data)

    if result is not None:

        H, path = result

        print(
            f"Found 3x3 matrix in JSON at: {path}"
        )

        return H

    # ------------------------------------------------------------
    # Diagnostic output
    # ------------------------------------------------------------
    print("\nJSON structure:")
    print(json.dumps(data, indent=2))

    raise ValueError(
        "\nCould not find a 3x3 homography matrix in:\n"
        f"{path}\n"
        "\nExpected one of these keys:\n"
        + ", ".join(possible_keys)
    )


def load_reconstruction():
    """
    Prefer smoothed reconstruction if available,
    otherwise use raw reconstruction.
    """

    if os.path.exists(SMOOTHED_CSV):
        path = SMOOTHED_CSV
        print(f"Using smoothed reconstruction:")
        print(path)
    elif os.path.exists(RECONSTRUCTION_CSV):
        path = RECONSTRUCTION_CSV
        print(f"Using raw reconstruction:")
        print(path)
    else:
        raise FileNotFoundError(
            "Could not find either reconstructed_ball_3d.csv "
            "or smoothed_ball_3d.csv."
        )

    df = pd.read_csv(path)

    print()
    print(f"Rows: {len(df)}")
    print(f"Columns:")
    print(df.columns.tolist())

    return df, path


def find_column(df, candidates, required=True):
    """
    Find the first matching column name.
    """

    for name in candidates:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            f"Could not identify required column.\n"
            f"Tried: {candidates}\n"
            f"Available: {df.columns.tolist()}"
        )

    return None


def normalize_reconstruction(df):
    """
    Convert possible column-name variants into a standard format.
    """

    track_col = find_column(
        df,
        ["track_id", "track", "trajectory_id"]
    )

    frame_col = find_column(
        df,
        ["frame", "frame_id"]
    )

    time_col = find_column(
        df,
        ["time_seconds", "time", "timestamp"]
    )

    x_col = find_column(
        df,
        [
            "X_m",
            "x_m",
            "camera_x_m",
            "X"
        ]
    )

    y_col = find_column(
        df,
        [
            "Y_m",
            "y_m",
            "camera_y_m",
            "Y"
        ]
    )

    z_col = find_column(
        df,
        [
            "Z_m",
            "z_m",
            "depth_m",
            "camera_z_m",
            "depth"
        ]
    )

    diameter_col = find_column(
        df,
        [
            "diameter_px",
            "diameter_px_raw",
            "ball_diameter_px",
            "diameter"
        ],
        required=False
    )

    out = pd.DataFrame()

    out["track_id"] = df[track_col]
    out["frame"] = df[frame_col]
    out["time_seconds"] = df[time_col]

    out["camera_x_m"] = df[x_col]
    out["camera_y_m"] = df[y_col]
    out["camera_z_m"] = df[z_col]

    if diameter_col is not None:
        out["diameter_px"] = df[diameter_col]
    else:
        out["diameter_px"] = np.nan

    # Preserve original row order.
    out = out.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    return out


# ============================================================
# CAMERA MATRIX
# ============================================================

def create_camera_matrix():
    K = np.array(
        [
            [FX, 0.0, CX],
            [0.0, FY, CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return K


# ============================================================
# SCALE HOMOGRAPHY
# ============================================================

def scale_image_to_world_homography(H_reference, tile_size):
    """
    H_reference maps image -> ground coordinates using
    the reference tile size of 0.60 m.

    If the actual tile dimension changes, the world X/Y
    coordinates scale linearly.

    Therefore:

        H_new = S @ H_reference

    where:

        S = diag(scale, scale, 1)

    """

    scale = tile_size / REFERENCE_TILE_M

    S = np.array(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    H_new = S @ H_reference

    return H_new


# ============================================================
# HOMOGRAPHY -> CAMERA POSE
# ============================================================

def homography_to_pose(H_img_to_world, K):
    """
    Recover camera pose from a planar homography.

    H_img_to_world maps image coordinates to ground-plane
    coordinates.

    We first invert it:

        H_world_to_img = inv(H_img_to_world)

    For Z_world = 0:

        H = K [r1 r2 t]

    up to a common scale factor.

    We recover r1, r2 and t and then enforce an orthonormal
    rotation matrix using SVD.
    """

    H_world_to_img = np.linalg.inv(H_img_to_world)

    K_inv = np.linalg.inv(K)

    B = K_inv @ H_world_to_img

    h1 = B[:, 0]
    h2 = B[:, 1]
    h3 = B[:, 2]

    norm1 = np.linalg.norm(h1)
    norm2 = np.linalg.norm(h2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        raise ValueError(
            "Degenerate homography encountered."
        )

    # Common scale factor.
    lam = 0.5 * (norm1 + norm2)

    r1 = h1 / lam
    r2 = h2 / lam
    r3 = np.cross(r1, r2)

    R_approx = np.column_stack(
        [r1, r2, r3]
    )

    # Project approximately-orthogonal matrix onto SO(3).
    U, _, Vt = np.linalg.svd(R_approx)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    t = h3 / lam

    # Camera center in world coordinates:
    #
    # X_camera = R X_world + t
    #
    # Therefore:
    #
    # C_world = -R^T t

    camera_center_world = -R.T @ t

    return R, t, camera_center_world


# ============================================================
# TRANSFORM CAMERA POINTS -> WORLD
# ============================================================

def camera_to_world(points_camera, R, t):
    """
    Transform:

        X_camera = R X_world + t

    into:

        X_world = R^T (X_camera - t)
    """

    points_camera = np.asarray(
        points_camera,
        dtype=np.float64
    )

    points_world = (
        (R.T @ (points_camera - t.reshape(1, 3)).T)
        .T
    )

    return points_world


# ============================================================
# CROSS-CHECK ONE SCALE
# ============================================================

def evaluate_tile_size(df, H_reference, K, tile_size):

    H_scaled = scale_image_to_world_homography(
        H_reference,
        tile_size
    )

    R, t, camera_center = homography_to_pose(
        H_scaled,
        K
    )

    camera_points = df[
        [
            "camera_x_m",
            "camera_y_m",
            "camera_z_m",
        ]
    ].to_numpy(dtype=np.float64)

    world_points = camera_to_world(
        camera_points,
        R,
        t
    )

    result = df.copy()

    result["world_x_m"] = world_points[:, 0]
    result["world_y_m"] = world_points[:, 1]
    result["world_z_m"] = world_points[:, 2]

    result["center_height_error_m"] = (
        result["world_z_m"]
        - EXPECTED_CENTER_HEIGHT_M
    )

    result["absolute_height_error_m"] = np.abs(
        result["center_height_error_m"]
    )

    valid_height = result[
        "absolute_height_error_m"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    median_abs_error = (
        float(valid_height.median())
        if len(valid_height) > 0
        else np.nan
    )

    mean_abs_error = (
        float(valid_height.mean())
        if len(valid_height) > 0
        else np.nan
    )

    p90_abs_error = (
        float(np.percentile(
            valid_height,
            OUTLIER_PERCENTILE
        ))
        if len(valid_height) > 0
        else np.nan
    )

    median_height = (
        float(result["world_z_m"].median())
        if len(result) > 0
        else np.nan
    )

    mean_height = (
        float(result["world_z_m"].mean())
        if len(result) > 0
        else np.nan
    )

    within_5cm = (
        float(
            np.mean(
                valid_height.to_numpy() <= 0.05
            ) * 100.0
        )
        if len(valid_height) > 0
        else np.nan
    )

    within_10cm = (
        float(
            np.mean(
                valid_height.to_numpy() <= 0.10
            ) * 100.0
        )
        if len(valid_height) > 0
        else np.nan
    )

    camera_height = float(
        camera_center[2]
    )

    summary = {
        "tile_size_m": tile_size,
        "scale_factor": tile_size / REFERENCE_TILE_M,
        "camera_height_m": camera_height,
        "median_ball_center_height_m": median_height,
        "mean_ball_center_height_m": mean_height,
        "expected_ball_center_height_m":
            EXPECTED_CENTER_HEIGHT_M,
        "median_abs_height_error_m":
            median_abs_error,
        "mean_abs_height_error_m":
            mean_abs_error,
        "p90_abs_height_error_m":
            p90_abs_error,
        "within_5cm_percent":
            within_5cm,
        "within_10cm_percent":
            within_10cm,
    }

    return result, summary, R, t, camera_center


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print_header(
        "STEP 25 - FOOTBALL / GROUND-PLANE METRIC CROSS-CHECK"
    )

    print_header("INPUT")

    df_raw, reconstruction_path = load_reconstruction()

    df = normalize_reconstruction(df_raw)

    print()
    print("Normalized columns:")
    print(df.columns.tolist())

    print()
    print(f"Valid observations: {len(df)}")

    for track_id, group in df.groupby("track_id"):

        print(
            f"Track {track_id}: "
            f"{len(group)} observations, "
            f"frames {int(group['frame'].min())} "
            f"-> {int(group['frame'].max())}"
        )

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    print_header("GROUND HOMOGRAPHY")

    H_reference = load_homography(
        HOMOGRAPHY_JSON
    )

    print(H_reference)

    print()
    print(
        f"Reference tile size: "
        f"{REFERENCE_TILE_M:.2f} m"
    )

    # --------------------------------------------------------
    # Camera matrix
    # --------------------------------------------------------

    print_header("CAMERA INTRINSICS")

    K = create_camera_matrix()

    print(
        f"Image size: "
        f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
    )

    print(
        f"fx = {FX:.3f} px"
    )

    print(
        f"fy = {FY:.3f} px"
    )

    print(
        f"cx = {CX:.3f} px"
    )

    print(
        f"cy = {CY:.3f} px"
    )

    print()
    print("Camera matrix:")
    print(K)

    # --------------------------------------------------------
    # Football dimensions
    # --------------------------------------------------------

    print_header("FOOTBALL REFERENCE")

    print(
        f"Ball diameter: "
        f"{BALL_DIAMETER_M:.3f} m"
    )

    print(
        f"Ball radius:   "
        f"{BALL_RADIUS_M:.3f} m"
    )

    print()
    print(
        "Because the ball is assumed to remain in "
        "ground contact, its center should be approximately "
        f"{BALL_RADIUS_M:.3f} m above the floor."
    )

    # --------------------------------------------------------
    # Evaluate scales
    # --------------------------------------------------------

    print_header(
        "TESTING TILE-SIZE ASSUMPTIONS"
    )

    all_results = []
    summaries = []

    for tile_size in TEST_TILE_SIZES:

        print()
        print(
            f"Testing tile size "
            f"{tile_size:.2f} m..."
        )

        result, summary, R, t, camera_center = (
            evaluate_tile_size(
                df,
                H_reference,
                K,
                tile_size
            )
        )

        result["tested_tile_size_m"] = tile_size

        all_results.append(result)
        summaries.append(summary)

        print(
            f"  Camera height: "
            f"{summary['camera_height_m']:.3f} m"
        )

        print(
            f"  Median ball-center height: "
            f"{summary['median_ball_center_height_m']:.3f} m"
        )

        print(
            f"  Median absolute height error: "
            f"{summary['median_abs_height_error_m']:.3f} m"
        )

        print(
            f"  Within 5 cm: "
            f"{summary['within_5cm_percent']:.1f}%"
        )

        print(
            f"  Within 10 cm: "
            f"{summary['within_10cm_percent']:.1f}%"
        )

    # --------------------------------------------------------
    # Summary dataframe
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False
    )

    combined_df = pd.concat(
        all_results,
        ignore_index=True
    )

    combined_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print_header("CROSS-CHECK SUMMARY")

    print(
        summary_df[
            [
                "tile_size_m",
                "camera_height_m",
                "median_ball_center_height_m",
                "median_abs_height_error_m",
                "within_5cm_percent",
                "within_10cm_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Find scale with smallest median error
    # --------------------------------------------------------

    valid_summary = summary_df.dropna(
        subset=["median_abs_height_error_m"]
    )

    if len(valid_summary) > 0:

        best_idx = valid_summary[
            "median_abs_height_error_m"
        ].idxmin()

        best = valid_summary.loc[
            best_idx
        ]

        print()
        print(
            "Smallest median height discrepancy "
            "among tested scales:"
        )

        print(
            f"  Tile size: "
            f"{best['tile_size_m']:.2f} m"
        )

        print(
            f"  Median ball-center height: "
            f"{best['median_ball_center_height_m']:.3f} m"
        )

        print(
            f"  Expected height: "
            f"{EXPECTED_CENTER_HEIGHT_M:.3f} m"
        )

        print(
            f"  Median absolute error: "
            f"{best['median_abs_height_error_m']:.3f} m"
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This is NOT a measurement of the actual tile size."
        )

        print(
            "It is a consistency check between:"
        )

        print(
            "  1. assumed football diameter,"
        )

        print(
            "  2. estimated RGB camera intrinsics,"
        )

        print(
            "  3. ground-plane homography,"
        )

        print(
            "  4. observed ball trajectory."
        )

    # --------------------------------------------------------
    # Plot 1: ball center height over time
    # --------------------------------------------------------

    print_header(
        "CREATING BALL-HEIGHT DIAGNOSTIC PLOT"
    )

    reference_result = all_results[
        TEST_TILE_SIZES.index(
            REFERENCE_TILE_M
        )
    ]

    plt.figure(figsize=(12, 6))

    for track_id, group in reference_result.groupby(
        "track_id"
    ):

        plt.plot(
            group["time_seconds"],
            group["world_z_m"],
            label=f"Track {track_id}"
        )

    plt.axhline(
        EXPECTED_CENTER_HEIGHT_M,
        linestyle="--",
        label=(
            f"Expected center height "
            f"({BALL_RADIUS_M:.2f} m)"
        )
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Ball-center height above floor (m)")
    plt.title(
        "Football Center Height vs Ground Plane "
        "(0.60 m Tile Assumption)"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        HEIGHT_PLOT,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {HEIGHT_PLOT}"
    )

    # --------------------------------------------------------
    # Plot 2: scale sensitivity
    # --------------------------------------------------------

    print_header(
        "CREATING SCALE-CROSSCHECK PLOT"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        summary_df["tile_size_m"],
        summary_df["median_abs_height_error_m"],
        marker="o"
    )

    plt.axvline(
        REFERENCE_TILE_M,
        linestyle="--",
        label="Current working assumption (0.60 m)"
    )

    plt.xlabel("Assumed tile size (m)")
    plt.ylabel(
        "Median absolute ball-height error (m)"
    )

    plt.title(
        "Football / Ground-Plane Metric "
        "Consistency vs Tile Size"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        SCALE_PLOT,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {SCALE_PLOT}"
    )

    # --------------------------------------------------------
    # Plot 3: camera height
    # --------------------------------------------------------

    print_header(
        "CREATING CAMERA-HEIGHT DIAGNOSTIC"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        summary_df["tile_size_m"],
        summary_df["camera_height_m"],
        marker="o"
    )

    plt.axvline(
        REFERENCE_TILE_M,
        linestyle="--",
        label="Current working assumption (0.60 m)"
    )

    plt.xlabel("Assumed tile size (m)")
    plt.ylabel("Estimated camera height (m)")

    plt.title(
        "Estimated Camera Height vs "
        "Assumed Tile Size"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        CAMERA_HEIGHT_PLOT,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {CAMERA_HEIGHT_PLOT}"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_header(
        "STEP 25 COMPLETE"
    )

    print()
    print("Saved:")
    print(OUTPUT_CSV)
    print(SUMMARY_CSV)
    print(HEIGHT_PLOT)
    print(SCALE_PLOT)
    print(CAMERA_HEIGHT_PLOT)

    print()
    print(
        "The football diameter and floor homography "
        "have now been cross-checked geometrically."
    )

    print()
    print(
        "Remember:"
    )

    print(
        "The cross-check does not independently establish "
        "the true tile dimension."
    )

    print(
        "It measures consistency between the independent "
        "geometric assumptions used by the pipeline."
    )


if __name__ == "__main__":
    main()