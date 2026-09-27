"""
STEP 31 - CAMERA-MOTION-COMPENSATED TOP VIEW

Purpose
-------
The original top-view trajectory assumes a fixed camera.

Step 30 showed that the camera moves substantially during the first
football appearance, especially around frames ~160-180.

This step estimates a homography from every video frame to a fixed
reference frame (frame 300), then combines that with the existing
frame-300 -> ground-plane homography.

For each football contact point:

    current frame
        |
        v
    H(frame -> reference)
        |
        v
    reference frame (300)
        |
        v
    H(reference -> ground)
        |
        v
    metric top-view coordinates

IMPORTANT
---------
This step is diagnostic only.

It does NOT modify:
    - original detections
    - 3D reconstruction
    - existing top-view trajectory
    - final integrated video

The purpose is to determine whether camera compensation removes
the wobble observed in the original top-view trajectory.
"""

from pathlib import Path
import json
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "data" / "input" / "rgb.avi"

CONTACT_TRAJECTORY_CSV = (
    ROOT
    / "data"
    / "output"
    / "top_view"
    / "top_view_contact_trajectory.csv"
)

GROUND_HOMOGRAPHY_JSON = (
    ROOT
    / "calibration"
    / "ground_homography.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "output"
    / "validation"
    / "camera_compensated_top_view"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

REFERENCE_FRAME = 300

# ORB
ORB_FEATURES = 3000
ORB_FAST_THRESHOLD = 12

# Lowe ratio test
RATIO_TEST = 0.75

# RANSAC
RANSAC_THRESHOLD_PX = 3.0

# Minimum number of matches required before attempting homography
MIN_GOOD_MATCHES = 20

# Minimum number of RANSAC inliers for accepting a homography
MIN_INLIERS = 15

# Minimum inlier ratio
MIN_INLIER_RATIO = 0.20

# Ground/floor region.
#
# The ball is on the floor and the homography we ultimately care about
# is the floor-plane homography, so floor features are preferred.
#
# y >= this fraction of image height.
GROUND_ROI_START_FRACTION = 0.35

# For visualization
TOP_VIEW_SCALE_PX_PER_M = 120

# Padding around the ground map
TOP_VIEW_PADDING_M = 1.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_homography(H):
    """Normalize a homography so H[2,2] = 1."""
    H = np.asarray(H, dtype=np.float64)

    if H.shape != (3, 3):
        raise ValueError(f"Expected 3x3 homography, got {H.shape}")

    if abs(H[2, 2]) < 1e-12:
        return H

    return H / H[2, 2]


def load_ground_homography(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidate_keys = [
        "homography_image_to_ground",
        "homography",
        "H",
    ]

    for key in candidate_keys:
        if key in data:
            H = np.asarray(data[key], dtype=np.float64)
            print(f"Found ground homography under JSON key: '{key}'")
            print(H)
            return normalize_homography(H)

    raise ValueError(
        "Could not find image-to-ground homography in JSON. "
        f"Expected one of: {candidate_keys}"
    )


def load_contact_trajectory(path):
    df = pd.read_csv(path)

    required = [
        "track_id",
        "frame",
        "time_seconds",
        "pixel_contact_x",
        "pixel_contact_y",
        "ground_contact_x_m",
        "ground_contact_y_m",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns in contact trajectory CSV:\n"
            f"{missing}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df = df.copy()

    numeric_columns = [
        "track_id",
        "frame",
        "time_seconds",
        "pixel_contact_x",
        "pixel_contact_y",
        "ground_contact_x_m",
        "ground_contact_y_m",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()

    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)

    return df


def read_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()

    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read frame {frame_number} from video."
        )

    return frame


def create_ground_mask(image):
    """
    Create a mask favoring the lower/floor part of the image.

    We intentionally avoid the upper part of the image because:
      - people may move there
      - the ball may move there
      - the ground plane is our target geometry
    """

    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    y_start = int(h * GROUND_ROI_START_FRACTION)

    mask[y_start:, :] = 255

    return mask


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def create_orb():
    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        fastThreshold=ORB_FAST_THRESHOLD,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
    )


def detect_features(orb, image, mask=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        mask,
    )

    if descriptors is None:
        return [], None

    return keypoints, descriptors


# ============================================================================
# FEATURE MATCHING
# ============================================================================

def match_features(desc_current, desc_reference):
    if desc_current is None or desc_reference is None:
        return []

    if len(desc_current) < 2 or len(desc_reference) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn_matches = matcher.knnMatch(
        desc_current,
        desc_reference,
        k=2,
    )

    good = []

    for pair in knn_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    return good


# ============================================================================
# HOMOGRAPHY ESTIMATION
# ============================================================================

def estimate_frame_to_reference_homography(
    keypoints_current,
    keypoints_reference,
    good_matches,
):
    if len(good_matches) < MIN_GOOD_MATCHES:
        return None

    src_points = np.float32(
        [
            keypoints_current[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    dst_points = np.float32(
        [
            keypoints_reference[m.trainIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
    )

    if H is None or mask is None:
        return None

    H = normalize_homography(H)

    inlier_mask = mask.ravel().astype(bool)

    inlier_count = int(np.sum(inlier_mask))
    match_count = len(good_matches)

    if match_count == 0:
        return None

    inlier_ratio = inlier_count / match_count

    if inlier_count < MIN_INLIERS:
        return None

    if inlier_ratio < MIN_INLIER_RATIO:
        return None

    # Calculate median reprojection error using inliers.
    projected = cv2.perspectiveTransform(
        src_points,
        H,
    ).reshape(-1, 2)

    destination = dst_points.reshape(-1, 2)

    errors = np.linalg.norm(
        projected - destination,
        axis=1,
    )

    inlier_errors = errors[inlier_mask]

    median_error = float(np.median(inlier_errors))
    mean_error = float(np.mean(inlier_errors))
    p90_error = float(np.percentile(inlier_errors, 90))

    return {
        "H": H,
        "match_count": match_count,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_error_px": median_error,
        "mean_reprojection_error_px": mean_error,
        "p90_reprojection_error_px": p90_error,
    }


# ============================================================================
# HOMOGRAPHY SANITY CHECK
# ============================================================================

def transform_point(H, x, y):
    point = np.array(
        [[[float(x), float(y)]]],
        dtype=np.float64,
    )

    transformed = cv2.perspectiveTransform(
        point,
        H,
    )

    return (
        float(transformed[0, 0, 0]),
        float(transformed[0, 0, 1]),
    )


def homography_reasonable(H, width, height):
    """
    Basic sanity check.

    Transform the four image corners and reject homographies that produce
    absurdly large coordinates.

    This is intentionally conservative. It is not intended to prove
    geometric correctness.
    """

    corners = [
        (0, 0),
        (width - 1, 0),
        (width - 1, height - 1),
        (0, height - 1),
    ]

    transformed = []

    for x, y in corners:
        try:
            tx, ty = transform_point(H, x, y)
        except cv2.error:
            return False

        if not np.isfinite(tx) or not np.isfinite(ty):
            return False

        transformed.append((tx, ty))

    max_dimension = max(width, height)

    for x, y in transformed:
        if abs(x) > 10 * max_dimension:
            return False

        if abs(y) > 10 * max_dimension:
            return False

    return True


# ============================================================================
# MAIN HOMOGRAPHY ESTIMATION
# ============================================================================

def estimate_all_frame_homographies(
    video_path,
    reference_frame,
):
    print_header("ESTIMATING FRAME -> REFERENCE HOMOGRAPHIES")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video frames: {frame_count}")
    print(f"Resolution: {width} x {height}")
    print(f"Reference frame: {reference_frame}")
    print(
        f"Ground ROI: y >= "
        f"{GROUND_ROI_START_FRACTION:.2f} * image height"
    )

    if reference_frame < 0 or reference_frame >= frame_count:
        cap.release()
        raise ValueError(
            f"Reference frame {reference_frame} is outside "
            f"0 -> {frame_count - 1}"
        )

    reference_image = read_frame(
        cap,
        reference_frame,
    )

    ground_mask = create_ground_mask(reference_image)

    orb = create_orb()

    print()
    print("Detecting reference features...")

    kp_ref, desc_ref = detect_features(
        orb,
        reference_image,
        ground_mask,
    )

    print(f"Reference ground features: {len(kp_ref)}")

    if desc_ref is None or len(kp_ref) < 20:
        cap.release()
        raise RuntimeError(
            "Not enough reference features were detected."
        )

    results = {}

    # Reference frame maps to itself exactly.
    results[reference_frame] = {
        "H": np.eye(3, dtype=np.float64),
        "match_count": len(kp_ref),
        "inlier_count": len(kp_ref),
        "inlier_ratio": 1.0,
        "median_reprojection_error_px": 0.0,
        "mean_reprojection_error_px": 0.0,
        "p90_reprojection_error_px": 0.0,
        "status": "REFERENCE",
    }

    for frame_number in range(frame_count):

        if frame_number == reference_frame:
            continue

        current = read_frame(
            cap,
            frame_number,
        )

        mask = create_ground_mask(current)

        kp_cur, desc_cur = detect_features(
            orb,
            current,
            mask,
        )

        if desc_cur is None or len(kp_cur) < 20:
            results[frame_number] = {
                "status": "NO_FEATURES",
                "match_count": 0,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
            }
            continue

        good_matches = match_features(
            desc_cur,
            desc_ref,
        )

        estimate = estimate_frame_to_reference_homography(
            kp_cur,
            kp_ref,
            good_matches,
        )

        if estimate is None:
            results[frame_number] = {
                "status": "FAILED",
                "match_count": len(good_matches),
                "inlier_count": 0,
                "inlier_ratio": 0.0,
            }
            continue

        H = estimate["H"]

        if not homography_reasonable(
            H,
            width,
            height,
        ):
            results[frame_number] = {
                "status": "REJECTED_SANITY",
                "match_count": estimate["match_count"],
                "inlier_count": estimate["inlier_count"],
                "inlier_ratio": estimate["inlier_ratio"],
            }
            continue

        estimate["status"] = "OK"

        results[frame_number] = estimate

        if frame_number % 30 == 0:
            print(
                f"Processed frame {frame_number}/{frame_count - 1} | "
                f"matches={estimate['match_count']} | "
                f"inliers={estimate['inlier_count']} | "
                f"ratio={estimate['inlier_ratio']:.3f} | "
                f"median err={estimate['median_reprojection_error_px']:.2f}px"
            )

    cap.release()

    return results, width, height


# ============================================================================
# SAVE HOMOGRAPHY DIAGNOSTICS
# ============================================================================

def save_homography_results(results):
    rows = []

    for frame_number in sorted(results):

        result = results[frame_number]

        row = {
            "frame": frame_number,
            "status": result.get("status", "UNKNOWN"),
            "match_count": result.get("match_count", 0),
            "inlier_count": result.get("inlier_count", 0),
            "inlier_ratio": result.get("inlier_ratio", 0.0),
            "median_reprojection_error_px": result.get(
                "median_reprojection_error_px",
                np.nan,
            ),
            "mean_reprojection_error_px": result.get(
                "mean_reprojection_error_px",
                np.nan,
            ),
            "p90_reprojection_error_px": result.get(
                "p90_reprojection_error_px",
                np.nan,
            ),
        }

        H = result.get("H")

        if H is not None:

            for r in range(3):
                for c in range(3):
                    row[f"H{r+1}{c+1}"] = H[r, c]

        else:

            for r in range(3):
                for c in range(3):
                    row[f"H{r+1}{c+1}"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / "frame_to_reference_homographies.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")

    return df


# ============================================================================
# APPLY CAMERA COMPENSATION
# ============================================================================

def compensate_contact_points(
    contact_df,
    homography_results,
    H_reference_to_ground,
):
    print_header("APPLYING CAMERA-MOTION COMPENSATION")

    rows = []

    missing = 0
    rejected = 0
    success = 0

    for _, row in contact_df.iterrows():

        frame = int(row["frame"])

        result = homography_results.get(frame)

        if result is None:
            missing += 1
            continue

        if result.get("status") != "OK" and result.get("status") != "REFERENCE":
            rejected += 1
            continue

        H_frame_to_reference = result.get("H")

        if H_frame_to_reference is None:
            missing += 1
            continue

        px = float(row["pixel_contact_x"])
        py = float(row["pixel_contact_y"])

        try:
            reference_x, reference_y = transform_point(
                H_frame_to_reference,
                px,
                py,
            )

            ground_x, ground_y = transform_point(
                H_reference_to_ground,
                reference_x,
                reference_y,
            )

        except cv2.error:
            rejected += 1
            continue

        output = row.to_dict()

        output["reference_pixel_x"] = reference_x
        output["reference_pixel_y"] = reference_y

        output["compensated_ground_x_m"] = ground_x
        output["compensated_ground_y_m"] = ground_y

        output["camera_motion_compensation_status"] = "OK"

        output["original_ground_x_m"] = float(
            row["ground_contact_x_m"]
        )

        output["original_ground_y_m"] = float(
            row["ground_contact_y_m"]
        )

        output["ground_delta_x_m"] = (
            ground_x - float(row["ground_contact_x_m"])
        )

        output["ground_delta_y_m"] = (
            ground_y - float(row["ground_contact_y_m"])
        )

        output["ground_delta_m"] = math.sqrt(
            output["ground_delta_x_m"] ** 2
            + output["ground_delta_y_m"] ** 2
        )

        rows.append(output)

        success += 1

    compensated = pd.DataFrame(rows)

    print()
    print(f"Input observations: {len(contact_df)}")
    print(f"Successfully compensated: {success}")
    print(f"Missing homography: {missing}")
    print(f"Rejected homography: {rejected}")

    output_path = OUTPUT_DIR / "camera_compensated_contact_trajectory.csv"

    compensated.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")

    return compensated


# ============================================================================
# TRAJECTORY STATISTICS
# ============================================================================

def calculate_trajectory_statistics(
    df,
    x_col,
    y_col,
):
    results = []

    for track_id, track in df.groupby("track_id"):

        track = track.sort_values("frame").copy()

        x = track[x_col].to_numpy(dtype=float)
        y = track[y_col].to_numpy(dtype=float)

        if len(track) < 2:
            continue

        dx = np.diff(x)
        dy = np.diff(y)

        step_distances = np.sqrt(
            dx ** 2 + dy ** 2
        )

        path_length = float(
            np.sum(step_distances)
        )

        net_displacement = float(
            math.sqrt(
                (x[-1] - x[0]) ** 2
                + (y[-1] - y[0]) ** 2
            )
        )

        results.append(
            {
                "track_id": int(track_id),
                "observations": len(track),
                "first_frame": int(track["frame"].min()),
                "last_frame": int(track["frame"].max()),
                "path_length_m": path_length,
                "net_displacement_m": net_displacement,
                "median_frame_step_m": float(
                    np.median(step_distances)
                ),
                "max_frame_step_m": float(
                    np.max(step_distances)
                ),
            }
        )

    return pd.DataFrame(results)


# ============================================================================
# PLOTS
# ============================================================================

def plot_before_after(compensated):
    print_header("CREATING BEFORE / AFTER TRAJECTORY PLOT")

    plt.figure(figsize=(12, 8))

    for track_id, track in compensated.groupby("track_id"):

        track = track.sort_values("frame")

        plt.plot(
            track["ground_contact_x_m"],
            track["ground_contact_y_m"],
            marker="o",
            markersize=2,
            linewidth=1,
            alpha=0.45,
            label=f"Track {track_id} original",
        )

        plt.plot(
            track["compensated_ground_x_m"],
            track["compensated_ground_y_m"],
            marker="o",
            markersize=2,
            linewidth=1.5,
            label=f"Track {track_id} compensated",
        )

    plt.xlabel("Ground X (m)")
    plt.ylabel("Ground Y (m)")
    plt.title(
        "Original vs Camera-Motion-Compensated Top-View Trajectory"
    )
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    path = OUTPUT_DIR / "before_after_top_view_trajectory.png"

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Saved: {path}")


def plot_compensated_trajectory(compensated):
    print_header("CREATING COMPENSATED TRAJECTORY PLOT")

    plt.figure(figsize=(12, 8))

    for track_id, track in compensated.groupby("track_id"):

        track = track.sort_values("frame")

        plt.plot(
            track["compensated_ground_x_m"],
            track["compensated_ground_y_m"],
            marker="o",
            markersize=2,
            linewidth=1.5,
            label=f"Track {track_id}",
        )

        # Start
        first = track.iloc[0]

        plt.scatter(
            first["compensated_ground_x_m"],
            first["compensated_ground_y_m"],
            s=80,
            marker="o",
            label=f"Track {track_id} start",
        )

        # End
        last = track.iloc[-1]

        plt.scatter(
            last["compensated_ground_x_m"],
            last["compensated_ground_y_m"],
            s=80,
            marker="x",
        )

    plt.xlabel("Ground X (m)")
    plt.ylabel("Ground Y (m)")
    plt.title(
        "Camera-Motion-Compensated Top-View Football Trajectory"
    )
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    path = OUTPUT_DIR / "camera_compensated_top_view.png"

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Saved: {path}")


def plot_compensation_delta(compensated):
    print_header("CREATING COMPENSATION DELTA PLOT")

    plt.figure(figsize=(12, 7))

    for track_id, track in compensated.groupby("track_id"):

        track = track.sort_values("frame")

        plt.plot(
            track["time_seconds"],
            track["ground_delta_m"],
            linewidth=1.5,
            label=f"Track {track_id}",
        )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
        label="No correction",
    )

    plt.xlabel("Time (s)")
    plt.ylabel(
        "Original vs compensated ground-position difference (m)"
    )

    plt.title(
        "Effect of Camera-Motion Compensation Over Time"
    )

    plt.grid(True)
    plt.legend()

    path = OUTPUT_DIR / "camera_compensation_delta_over_time.png"

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Saved: {path}")


def plot_reference_pixel_positions(compensated):
    print_header("CREATING REFERENCE-IMAGE TRAJECTORY PLOT")

    plt.figure(figsize=(12, 7))

    for track_id, track in compensated.groupby("track_id"):

        track = track.sort_values("frame")

        plt.plot(
            track["reference_pixel_x"],
            track["reference_pixel_y"],
            marker="o",
            markersize=2,
            linewidth=1.5,
            label=f"Track {track_id}",
        )

    plt.gca().invert_yaxis()

    plt.xlabel("Reference-frame X (pixels)")
    plt.ylabel("Reference-frame Y (pixels)")

    plt.title(
        "Football Contact Points Stabilized Into Reference Frame 300"
    )

    plt.grid(True)
    plt.legend()

    path = OUTPUT_DIR / "reference_frame_stabilized_trajectory.png"

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Saved: {path}")


# ============================================================================
# DIAGNOSTIC VIDEO
# ============================================================================

def create_diagnostic_video(
    video_path,
    contact_df,
    homography_results,
    H_reference_to_ground,
):
    print_header("CREATING CAMERA-COMPENSATION DIAGNOSTIC VIDEO")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    output_width = 1920
    output_height = 720

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    output_path = (
        OUTPUT_DIR
        / "camera_motion_compensation_diagnostic.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (output_width, output_height),
    )

    # Prepare trajectory lookup.
    contact_lookup = {}

    for _, row in contact_df.iterrows():

        frame = int(row["frame"])

        if frame not in contact_lookup:
            contact_lookup[frame] = []

        contact_lookup[frame].append(
            (
                int(row["track_id"]),
                float(row["pixel_contact_x"]),
                float(row["pixel_contact_y"]),
            )
        )

    # Ground trajectory history.
    histories = {}

    for track_id in contact_df["track_id"].unique():
        histories[int(track_id)] = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in range(frame_count):

        ok, frame = cap.read()

        if not ok:
            break

        # ------------------------------------------------------------
        # Left side: original RGB frame
        # ------------------------------------------------------------

        left = frame.copy()

        cv2.putText(
            left,
            f"Frame {frame_number}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            left,
            "Original camera view",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw current detected contact points.
        current_points = contact_lookup.get(
            frame_number,
            [],
        )

        for track_id, x, y in current_points:

            cv2.circle(
                left,
                (int(round(x)), int(round(y))),
                8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                left,
                f"T{track_id}",
                (
                    int(round(x)) + 10,
                    int(round(y)) - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------
        # Right side: stabilized top view
        # ------------------------------------------------------------

        top = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        # Reference-frame image coordinates.
        result = homography_results.get(frame_number)

        if result is not None and result.get("H") is not None:

            H_current_to_reference = result["H"]

            for track_id, x, y in current_points:

                try:

                    rx, ry = transform_point(
                        H_current_to_reference,
                        x,
                        y,
                    )

                    gx, gy = transform_point(
                        H_reference_to_ground,
                        rx,
                        ry,
                    )

                    histories.setdefault(
                        track_id,
                        [],
                    )

                    histories[track_id].append(
                        (gx, gy)
                    )

                except cv2.error:
                    pass

        # Draw metric trajectory.
        all_points = []

        for track_id, history in histories.items():

            if len(history) < 1:
                continue

            # Determine plot bounds dynamically.
            for point in history:
                all_points.append(point)

        if len(all_points) > 0:

            xs = np.array(
                [p[0] for p in all_points],
                dtype=float,
            )

            ys = np.array(
                [p[1] for p in all_points],
                dtype=float,
            )

            min_x = min(-1.0, float(np.min(xs)) - TOP_VIEW_PADDING_M)
            max_x = max(1.0, float(np.max(xs)) + TOP_VIEW_PADDING_M)

            min_y = min(0.0, float(np.min(ys)) - TOP_VIEW_PADDING_M)
            max_y = max(1.0, float(np.max(ys)) + TOP_VIEW_PADDING_M)

        else:

            min_x, max_x = -2.0, 2.0
            min_y, max_y = -1.0, 7.0

        # Make a useful fixed-ish coordinate system.
        # Use 120 pixels / meter.
        scale = TOP_VIEW_SCALE_PX_PER_M

        map_width = output_width // 2
        map_height = output_height

        origin_x = int(
            map_width * 0.5
        )

        origin_y = int(
            map_height * 0.85
        )

        def ground_to_pixel(x, y):
            px = int(
                origin_x + x * scale
            )

            py = int(
                origin_y - y * scale
            )

            return px, py

        cv2.putText(
            top,
            "Camera-motion-compensated ground view",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            top,
            f"Reference frame: {REFERENCE_FRAME}",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Camera at world origin.
        camera_px = ground_to_pixel(
            0.0,
            0.0,
        )

        cv2.drawMarker(
            top,
            camera_px,
            (255, 255, 255),
            markerType=cv2.MARKER_TRIANGLE_UP,
            markerSize=25,
            thickness=2,
        )

        cv2.putText(
            top,
            "Camera reference",
            (
                camera_px[0] + 15,
                camera_px[1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw each trajectory.
        for track_id, history in histories.items():

            if len(history) < 2:
                continue

            pixels = [
                ground_to_pixel(x, y)
                for x, y in history
            ]

            for i in range(1, len(pixels)):

                cv2.line(
                    top,
                    pixels[i - 1],
                    pixels[i],
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            if len(pixels) > 0:

                cv2.circle(
                    top,
                    pixels[-1],
                    8,
                    (0, 255, 0),
                    -1,
                )

        # ------------------------------------------------------------
        # Compose side by side
        # ------------------------------------------------------------

        left_resized = cv2.resize(
            left,
            (output_width // 2, output_height),
        )

        right_resized = cv2.resize(
            top,
            (output_width // 2, output_height),
        )

        combined = np.hstack(
            [left_resized, right_resized]
        )

        writer.write(combined)

        if frame_number % 30 == 0:
            print(
                f"Diagnostic video frame "
                f"{frame_number}/{frame_count - 1}"
            )

    cap.release()
    writer.release()

    print(f"Saved: {output_path}")


# ============================================================================
# SUMMARY
# ============================================================================

def create_summary(
    homography_df,
    compensated,
):
    print_header("CAMERA-COMPENSATION SUMMARY")

    total_frames = len(homography_df)

    ok = int(
        np.sum(
            homography_df["status"].isin(
                ["OK", "REFERENCE"]
            )
        )
    )

    failed = total_frames - ok

    print(
        f"Frames with usable homography: "
        f"{ok}/{total_frames} "
        f"({100.0 * ok / total_frames:.1f}%)"
    )

    print(
        f"Frames without usable homography: "
        f"{failed}"
    )

    if len(compensated) > 0:

        print()
        print(
            f"Compensated observations: "
            f"{len(compensated)}"
        )

        print(
            f"Original vs compensated median "
            f"ground-position difference: "
            f"{compensated['ground_delta_m'].median():.4f} m"
        )

        print(
            f"Original vs compensated mean "
            f"ground-position difference: "
            f"{compensated['ground_delta_m'].mean():.4f} m"
        )

        print(
            f"Maximum ground-position correction: "
            f"{compensated['ground_delta_m'].max():.4f} m"
        )

        stats_original = calculate_trajectory_statistics(
            compensated,
            "ground_contact_x_m",
            "ground_contact_y_m",
        )

        stats_compensated = calculate_trajectory_statistics(
            compensated,
            "compensated_ground_x_m",
            "compensated_ground_y_m",
        )

        stats_original["version"] = "original"
        stats_compensated["version"] = "camera_compensated"

        stats = pd.concat(
            [
                stats_original,
                stats_compensated,
            ],
            ignore_index=True,
        )

        stats_path = (
            OUTPUT_DIR
            / "trajectory_statistics_before_after.csv"
        )

        stats.to_csv(
            stats_path,
            index=False,
        )

        print(f"Saved: {stats_path}")

    summary = {
        "reference_frame": REFERENCE_FRAME,
        "total_video_frames": total_frames,
        "usable_homographies": ok,
        "failed_homographies": failed,
        "usable_homography_percent": (
            100.0 * ok / total_frames
        ),
        "input_contact_observations": int(
            len(compensated)
        ),
        "median_ground_correction_m": (
            float(compensated["ground_delta_m"].median())
            if len(compensated)
            else None
        ),
        "mean_ground_correction_m": (
            float(compensated["ground_delta_m"].mean())
            if len(compensated)
            else None
        ),
        "max_ground_correction_m": (
            float(compensated["ground_delta_m"].max())
            if len(compensated)
            else None
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "camera_compensation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print(f"Saved: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("STEP 31 - CAMERA-MOTION-COMPENSATED TOP VIEW")
    print("=" * 70)

    print()
    print("This step is diagnostic only.")
    print("No existing tracking or reconstruction output will be modified.")

    # ------------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------------

    print_header("INPUT")

    print(f"Video:")
    print(VIDEO_PATH)

    print()
    print("Contact trajectory:")
    print(CONTACT_TRAJECTORY_CSV)

    print()
    print("Ground homography:")
    print(GROUND_HOMOGRAPHY_JSON)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    if not CONTACT_TRAJECTORY_CSV.exists():
        raise FileNotFoundError(
            f"Contact trajectory not found:\n"
            f"{CONTACT_TRAJECTORY_CSV}"
        )

    if not GROUND_HOMOGRAPHY_JSON.exists():
        raise FileNotFoundError(
            f"Ground homography not found:\n"
            f"{GROUND_HOMOGRAPHY_JSON}"
        )

    # ------------------------------------------------------------------------
    # LOAD CONTACT TRAJECTORY
    # ------------------------------------------------------------------------

    print_header("LOADING EXISTING CONTACT TRAJECTORY")

    contact_df = load_contact_trajectory(
        CONTACT_TRAJECTORY_CSV
    )

    print(
        f"Rows: {len(contact_df)}"
    )

    print(
        f"Tracks: "
        f"{sorted(contact_df['track_id'].unique())}"
    )

    for track_id, track in contact_df.groupby("track_id"):

        print(
            f"Track {track_id}: "
            f"{len(track)} observations, "
            f"frames {track['frame'].min()} -> "
            f"{track['frame'].max()}"
        )

    # ------------------------------------------------------------------------
    # GROUND HOMOGRAPHY
    # ------------------------------------------------------------------------

    print_header("LOADING REFERENCE -> GROUND HOMOGRAPHY")

    H_reference_to_ground = load_ground_homography(
        GROUND_HOMOGRAPHY_JSON
    )

    # ------------------------------------------------------------------------
    # ESTIMATE CAMERA MOTION
    # ------------------------------------------------------------------------

    homography_results, width, height = (
        estimate_all_frame_homographies(
            VIDEO_PATH,
            REFERENCE_FRAME,
        )
    )

    homography_df = save_homography_results(
        homography_results
    )

    # ------------------------------------------------------------------------
    # APPLY COMPENSATION
    # ------------------------------------------------------------------------

    compensated = compensate_contact_points(
        contact_df,
        homography_results,
        H_reference_to_ground,
    )

    if len(compensated) == 0:
        raise RuntimeError(
            "No contact points could be camera-compensated."
        )

    # ------------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------------

    plot_before_after(
        compensated
    )

    plot_compensated_trajectory(
        compensated
    )

    plot_compensation_delta(
        compensated
    )

    plot_reference_pixel_positions(
        compensated
    )

    # ------------------------------------------------------------------------
    # VIDEO
    # ------------------------------------------------------------------------

    create_diagnostic_video(
        VIDEO_PATH,
        contact_df,
        homography_results,
        H_reference_to_ground,
    )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    create_summary(
        homography_df,
        compensated,
    )

    print()
    print("=" * 70)
    print("STEP 31 COMPLETE")
    print("=" * 70)

    print()
    print("Generated diagnostic outputs:")
    print(
        OUTPUT_DIR
        / "frame_to_reference_homographies.csv"
    )
    print(
        OUTPUT_DIR
        / "camera_compensated_contact_trajectory.csv"
    )
    print(
        OUTPUT_DIR
        / "before_after_top_view_trajectory.png"
    )
    print(
        OUTPUT_DIR
        / "camera_compensated_top_view.png"
    )
    print(
        OUTPUT_DIR
        / "camera_compensation_delta_over_time.png"
    )
    print(
        OUTPUT_DIR
        / "reference_frame_stabilized_trajectory.png"
    )
    print(
        OUTPUT_DIR
        / "trajectory_statistics_before_after.csv"
    )
    print(
        OUTPUT_DIR
        / "camera_compensation_summary.json"
    )
    print(
        OUTPUT_DIR
        / "camera_motion_compensation_diagnostic.mp4"
    )

    print()
    print("IMPORTANT:")
    print(
        "The existing final integrated video has NOT been modified."
    )
    print(
        "We will compare the compensated result against the original "
        "before deciding whether to update the final pipeline."
    )


if __name__ == "__main__":
    main()