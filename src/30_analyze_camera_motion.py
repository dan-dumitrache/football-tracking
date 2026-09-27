"""
STEP 30 - CAMERA MOTION ANALYSIS
================================

Purpose
-------
Estimate global camera motion from the original RGB video.

This is a DIAGNOSTIC step.

It does NOT modify:
    - 2D detections
    - ball tracks
    - 3D reconstruction
    - ground homography
    - top-view trajectory
    - final integrated video

Why this step exists
--------------------
The current ground-plane projection assumes a stationary camera.

The video appears to contain a period where the camera moves.
If the camera moves while a fixed image-to-ground homography is used,
the estimated ground position of the football can move even when the
physical ground trajectory does not change accordingly.

This script estimates frame-to-frame global image motion using static
scene features.

Method
------
1. Convert frames to grayscale.
2. Detect ORB features.
3. Match consecutive frames.
4. Estimate a homography using RANSAC.
5. Keep only geometric inliers.
6. Measure the displacement of those inlier features.
7. Estimate an approximate affine motion:
       translation
       rotation
       scale
8. Save diagnostics and plots.

Important
---------
The camera motion estimate is an image-space diagnostic.
It is NOT yet the corrected ground-plane trajectory.

The next step will use the results to decide how camera-motion
compensation should be implemented.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO = PROJECT_ROOT / "data" / "input" / "rgb.avi"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "camera_motion"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "camera_motion.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "camera_motion_summary.json"

PLOT_MOTION = OUTPUT_DIR / "camera_motion_over_time.png"
PLOT_TRANSLATION = OUTPUT_DIR / "camera_translation_over_time.png"
PLOT_QUALITY = OUTPUT_DIR / "camera_motion_quality.png"

DIAGNOSTIC_VIDEO = OUTPUT_DIR / "camera_motion_diagnostic.mp4"

# Analyze every Nth frame.
# 1 = every frame.
FRAME_STEP = 1

# ORB parameters.
ORB_N_FEATURES = 2000
ORB_FAST_THRESHOLD = 12

# Matching.
RATIO_TEST = 0.75

# Homography RANSAC threshold in pixels.
RANSAC_REPROJECTION_THRESHOLD = 3.0

# Minimum number of matches/inliers required.
MIN_MATCHES = 12
MIN_INLIERS = 8

# Feature visualization.
DRAW_MAX_FEATURE_MATCHES = 40

# Camera-motion threshold used only for diagnostic labeling.
# This is NOT a universal physical threshold.
STRONG_MOTION_PX = 3.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def percentile_or_nan(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    return float(np.percentile(values, q))


def safe_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    return float(np.median(values))


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    return float(np.mean(values))


def create_orb() -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=ORB_N_FEATURES,
        fastThreshold=ORB_FAST_THRESHOLD,
        edgeThreshold=15,
        patchSize=31,
        scaleFactor=1.2,
        nlevels=8,
    )


def detect_features(
    orb: cv2.ORB,
    gray: np.ndarray,
):
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if keypoints is None:
        keypoints = []

    return keypoints, descriptors


def match_descriptors(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
):
    if descriptors_a is None or descriptors_b is None:
        return []

    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn_matches = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2,
    )

    good_matches = []

    for pair in knn_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    return good_matches


def estimate_homography(
    keypoints_a,
    keypoints_b,
    matches,
):
    if len(matches) < MIN_MATCHES:
        return None, None, None, None

    pts_a = np.float32(
        [keypoints_a[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    pts_b = np.float32(
        [keypoints_b[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts_a,
        pts_b,
        cv2.RANSAC,
        RANSAC_REPROJECTION_THRESHOLD,
    )

    if H is None or mask is None:
        return None, pts_a, pts_b, None

    inlier_mask = mask.ravel().astype(bool)

    return H, pts_a, pts_b, inlier_mask


def estimate_affine_motion(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    inlier_mask: np.ndarray,
):
    """
    Estimate a robust partial affine transformation using the homography
    inliers.

    The resulting model is:

        [x']   [a -b][x] + [tx]
        [y'] = [b  a][y]   [ty]

    From this:
        scale = sqrt(a^2 + b^2)
        rotation = atan2(b, a)

    This is an interpretable diagnostic, not the final camera model.
    """

    if pts_a is None or pts_b is None:
        return None

    if inlier_mask is None:
        return None

    if np.sum(inlier_mask) < 4:
        return None

    src = pts_a.reshape(-1, 2)[inlier_mask]
    dst = pts_b.reshape(-1, 2)[inlier_mask]

    if len(src) < 4:
        return None

    affine, affine_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )

    if affine is None:
        return None

    a = float(affine[0, 0])
    b = float(affine[1, 0])

    tx = float(affine[0, 2])
    ty = float(affine[1, 2])

    scale = math.sqrt(a * a + b * b)

    rotation_rad = math.atan2(b, a)
    rotation_deg = math.degrees(rotation_rad)

    affine_inliers = 0

    if affine_mask is not None:
        affine_inliers = int(np.sum(affine_mask))

    return {
        "translation_x_px": tx,
        "translation_y_px": ty,
        "translation_magnitude_px": math.sqrt(tx * tx + ty * ty),
        "rotation_deg": rotation_deg,
        "scale": scale,
        "scale_change_percent": (scale - 1.0) * 100.0,
        "affine_inliers": affine_inliers,
    }


def compute_inlier_motion(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    H: np.ndarray,
    inlier_mask: np.ndarray,
):
    """
    Transform the source inlier points with the estimated homography
    and compare them to their observed destination positions.

    Also computes the observed global image displacement.

    Returns robust statistics.
    """

    if H is None or inlier_mask is None:
        return None

    src = pts_a[inlier_mask]
    dst = pts_b[inlier_mask]

    if len(src) == 0:
        return None

    projected = cv2.perspectiveTransform(
        src,
        H,
    )

    projected_xy = projected.reshape(-1, 2)
    dst_xy = dst.reshape(-1, 2)

    # Observed feature displacement.
    displacement = dst_xy - src.reshape(-1, 2)

    displacement_mag = np.linalg.norm(
        displacement,
        axis=1,
    )

    # Homography reprojection residual.
    residual = dst_xy - projected_xy

    residual_mag = np.linalg.norm(
        residual,
        axis=1,
    )

    return {
        "median_feature_displacement_px": safe_median(
            displacement_mag
        ),
        "mean_feature_displacement_px": safe_mean(
            displacement_mag
        ),
        "p90_feature_displacement_px": percentile_or_nan(
            displacement_mag,
            90,
        ),
        "max_feature_displacement_px": (
            float(np.max(displacement_mag))
            if len(displacement_mag)
            else float("nan")
        ),
        "median_homography_residual_px": safe_median(
            residual_mag
        ),
        "p90_homography_residual_px": percentile_or_nan(
            residual_mag,
            90,
        ),
    }


def classify_motion(
    translation_px: float,
    inlier_count: int,
    inlier_ratio: float,
):
    """
    Diagnostic classification only.

    We require enough geometric support before calling a frame pair
    'strong motion'.

    This is NOT a universal camera-motion classifier.
    """

    if not np.isfinite(translation_px):
        return "INSUFFICIENT_DATA"

    if inlier_count < MIN_INLIERS:
        return "LOW_GEOMETRIC_SUPPORT"

    if not np.isfinite(inlier_ratio) or inlier_ratio < 0.20:
        return "LOW_GEOMETRIC_SUPPORT"

    if translation_px >= STRONG_MOTION_PX:
        return "STRONG_MOTION"

    return "LOW_MOTION"


# ============================================================================
# DIAGNOSTIC VIDEO DRAWING
# ============================================================================

def draw_motion_visualization(
    frame_a,
    frame_b,
    pts_a,
    pts_b,
    inlier_mask,
    H,
    frame_number_a,
    frame_number_b,
    motion_label,
):
    """
    Create a side-by-side diagnostic image.

    Left:
        previous frame

    Right:
        current frame

    Green:
        RANSAC inlier correspondence

    Red:
        rejected correspondence

    The diagnostic video is intended for human inspection.
    """

    left = frame_a.copy()
    right = frame_b.copy()

    if pts_a is not None and pts_b is not None and inlier_mask is not None:

        src = pts_a.reshape(-1, 2)
        dst = pts_b.reshape(-1, 2)

        for i, (p1, p2) in enumerate(zip(src, dst)):

            x1, y1 = np.round(p1).astype(int)
            x2, y2 = np.round(p2).astype(int)

            is_inlier = bool(inlier_mask[i])

            if is_inlier:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.circle(
                left,
                (x1, y1),
                3,
                color,
                -1,
            )

            cv2.circle(
                right,
                (x2, y2),
                3,
                color,
                -1,
            )

    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]

    canvas = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    canvas[
        : left.shape[0],
        : left.shape[1],
    ] = left

    canvas[
        : right.shape[0],
        left.shape[1] :,
    ] = right

    cv2.putText(
        canvas,
        f"Frame {frame_number_a}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Frame {frame_number_b}",
        (left.shape[1] + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Motion: {motion_label}",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():

    print_header("STEP 30 - CAMERA MOTION ANALYSIS")

    print()
    print("This step is diagnostic only.")
    print("No existing tracking or reconstruction output will be modified.")

    # ------------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------------

    print_header("INPUT VIDEO")

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found:\n{INPUT_VIDEO}"
        )

    print(f"Input: {INPUT_VIDEO}")

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{INPUT_VIDEO}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = (
        frame_count / fps
        if fps > 0
        else float("nan")
    )

    print(f"Resolution: {width} x {height}")
    print(f"FPS:        {fps:.2f}")
    print(f"Frames:     {frame_count}")
    print(f"Duration:   {duration:.2f} s")

    # ------------------------------------------------------------------------
    # FEATURE DETECTOR
    # ------------------------------------------------------------------------

    print_header("FEATURE DETECTOR")

    orb = create_orb()

    print(f"ORB features: {ORB_N_FEATURES}")
    print(f"ORB FAST threshold: {ORB_FAST_THRESHOLD}")
    print(f"Ratio test: {RATIO_TEST}")
    print(
        f"RANSAC reprojection threshold: "
        f"{RANSAC_REPROJECTION_THRESHOLD:.1f} px"
    )

    # ------------------------------------------------------------------------
    # DIAGNOSTIC VIDEO
    # ------------------------------------------------------------------------

    diagnostic_writer = None

    # Side-by-side video is 2 * original width.
    diagnostic_width = width * 2
    diagnostic_height = height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    diagnostic_writer = cv2.VideoWriter(
        str(DIAGNOSTIC_VIDEO),
        fourcc,
        fps,
        (diagnostic_width, diagnostic_height),
    )

    if not diagnostic_writer.isOpened():
        print(
            "WARNING: Could not create diagnostic video."
        )
        diagnostic_writer = None

    # ------------------------------------------------------------------------
    # PROCESS VIDEO
    # ------------------------------------------------------------------------

    print_header("PROCESSING")

    rows = []

    previous_gray = None
    previous_frame = None
    previous_keypoints = None
    previous_descriptors = None
    previous_frame_number = None

    frame_number = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_number % FRAME_STEP != 0:
            frame_number += 1
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        keypoints, descriptors = detect_features(
            orb,
            gray,
        )

        if previous_gray is None:

            previous_gray = gray
            previous_frame = frame.copy()
            previous_keypoints = keypoints
            previous_descriptors = descriptors
            previous_frame_number = frame_number

            frame_number += 1
            continue

        matches = match_descriptors(
            previous_descriptors,
            descriptors,
        )

        H, pts_a, pts_b, inlier_mask = estimate_homography(
            previous_keypoints,
            keypoints,
            matches,
        )

        total_matches = len(matches)

        inlier_count = (
            int(np.sum(inlier_mask))
            if inlier_mask is not None
            else 0
        )

        inlier_ratio = (
            inlier_count / total_matches
            if total_matches > 0
            else float("nan")
        )

        motion_stats = {
            "median_feature_displacement_px": float("nan"),
            "mean_feature_displacement_px": float("nan"),
            "p90_feature_displacement_px": float("nan"),
            "max_feature_displacement_px": float("nan"),
            "median_homography_residual_px": float("nan"),
            "p90_homography_residual_px": float("nan"),
        }

        if H is not None and inlier_mask is not None:

            computed_motion = compute_inlier_motion(
                pts_a,
                pts_b,
                H,
                inlier_mask,
            )

            if computed_motion is not None:
                motion_stats.update(computed_motion)

        affine_stats = {
            "translation_x_px": float("nan"),
            "translation_y_px": float("nan"),
            "translation_magnitude_px": float("nan"),
            "rotation_deg": float("nan"),
            "scale": float("nan"),
            "scale_change_percent": float("nan"),
            "affine_inliers": 0,
        }

        if (
            pts_a is not None
            and pts_b is not None
            and inlier_mask is not None
        ):
            estimated_affine = estimate_affine_motion(
                pts_a,
                pts_b,
                inlier_mask,
            )

            if estimated_affine is not None:
                affine_stats.update(estimated_affine)

        motion_label = classify_motion(
            affine_stats["translation_magnitude_px"],
            inlier_count,
            inlier_ratio,
        )

        row = {
            "frame_previous": previous_frame_number,
            "frame_current": frame_number,
            "time_previous_s": (
                previous_frame_number / fps
                if fps > 0
                else float("nan")
            ),
            "time_current_s": (
                frame_number / fps
                if fps > 0
                else float("nan")
            ),
            "total_matches": total_matches,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            **motion_stats,
            **affine_stats,
            "motion_class": motion_label,
        }

        rows.append(row)

        # --------------------------------------------------------------------
        # DIAGNOSTIC VIDEO
        # --------------------------------------------------------------------

        if diagnostic_writer is not None:

            visualization = draw_motion_visualization(
                previous_frame,
                frame,
                pts_a,
                pts_b,
                inlier_mask,
                H,
                previous_frame_number,
                frame_number,
                motion_label,
            )

            diagnostic_writer.write(
                visualization
            )

        # --------------------------------------------------------------------
        # UPDATE
        # --------------------------------------------------------------------

        previous_gray = gray
        previous_frame = frame.copy()
        previous_keypoints = keypoints
        previous_descriptors = descriptors
        previous_frame_number = frame_number

        frame_number += 1

        if frame_number % 30 == 0:
            print(
                f"Processed frame {frame_number}/{frame_count}"
            )

    cap.release()

    if diagnostic_writer is not None:
        diagnostic_writer.release()

    # ------------------------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------------------------

    print_header("RESULTS")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No camera-motion measurements were produced."
        )

    df.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    print(f"Measurements: {len(df)}")
    print(f"Saved: {CSV_OUTPUT}")

    # ------------------------------------------------------------------------
    # BASIC STATISTICS
    # ------------------------------------------------------------------------

    valid_motion = df[
        np.isfinite(
            df["translation_magnitude_px"]
        )
    ].copy()

    valid_displacement = df[
        np.isfinite(
            df["median_feature_displacement_px"]
        )
    ].copy()

    print_header("CAMERA MOTION SUMMARY")

    print(
        f"Valid motion estimates: "
        f"{len(valid_motion)} / {len(df)}"
    )

    if not valid_motion.empty:

        print(
            "Median affine translation: "
            f"{valid_motion['translation_magnitude_px'].median():.3f} px"
        )

        print(
            "P90 affine translation: "
            f"{valid_motion['translation_magnitude_px'].quantile(0.90):.3f} px"
        )

        print(
            "Maximum affine translation: "
            f"{valid_motion['translation_magnitude_px'].max():.3f} px"
        )

        print(
            "Median rotation: "
            f"{valid_motion['rotation_deg'].median():.4f} deg"
        )

        print(
            "Maximum absolute rotation: "
            f"{valid_motion['rotation_deg'].abs().max():.4f} deg"
        )

        print(
            "Median scale change: "
            f"{valid_motion['scale_change_percent'].median():.4f}%"
        )

    if not valid_displacement.empty:

        print(
            "Median observed feature displacement: "
            f"{valid_displacement['median_feature_displacement_px'].median():.3f} px"
        )

        print(
            "P90 observed feature displacement: "
            f"{valid_displacement['median_feature_displacement_px'].quantile(0.90):.3f} px"
        )

    # ------------------------------------------------------------------------
    # MOTION CLASS COUNTS
    # ------------------------------------------------------------------------

    print_header("MOTION CLASSIFICATION")

    class_counts = (
        df["motion_class"]
        .value_counts()
        .to_dict()
    )

    for label, count in class_counts.items():

        percent = (
            100.0 * count / len(df)
        )

        print(
            f"{label}: "
            f"{count} ({percent:.1f}%)"
        )

    # ------------------------------------------------------------------------
    # STRONG MOTION WINDOWS
    # ------------------------------------------------------------------------

    strong = df[
        df["motion_class"] == "STRONG_MOTION"
    ].copy()

    print_header("STRONG CAMERA-MOTION WINDOWS")

    if strong.empty:

        print(
            "No frame pairs exceeded the diagnostic "
            f"threshold of {STRONG_MOTION_PX:.1f} px."
        )

    else:

        print(
            f"Frame pairs classified as strong motion: "
            f"{len(strong)}"
        )

        # Group consecutive frame pairs into windows.
        strong_frames = (
            strong["frame_current"]
            .astype(int)
            .sort_values()
            .tolist()
        )

        windows = []

        start = strong_frames[0]
        previous = strong_frames[0]

        for current in strong_frames[1:]:

            if current <= previous + FRAME_STEP:

                previous = current

            else:

                windows.append(
                    (start, previous)
                )

                start = current
                previous = current

        windows.append(
            (start, previous)
        )

        for i, (start_frame, end_frame) in enumerate(
            windows,
            start=1,
        ):

            start_time = start_frame / fps
            end_time = end_frame / fps

            subset = strong[
                (strong["frame_current"] >= start_frame)
                & (strong["frame_current"] <= end_frame)
            ]

            median_motion = subset[
                "translation_magnitude_px"
            ].median()

            max_motion = subset[
                "translation_magnitude_px"
            ].max()

            print(
                f"Window {i}: "
                f"frames {start_frame}->{end_frame}, "
                f"time {start_time:.2f}->{end_time:.2f} s, "
                f"median={median_motion:.2f} px, "
                f"max={max_motion:.2f} px"
            )

    # ------------------------------------------------------------------------
    # FIND LARGEST MOTION EVENTS
    # ------------------------------------------------------------------------

    print_header("LARGEST CAMERA-MOTION EVENTS")

    if not valid_motion.empty:

        top_events = (
            valid_motion
            .sort_values(
                "translation_magnitude_px",
                ascending=False,
            )
            .head(15)
        )

        for _, row in top_events.iterrows():

            print(
                f"Frame "
                f"{int(row['frame_previous'])}"
                f" -> "
                f"{int(row['frame_current'])} | "
                f"time "
                f"{row['time_current_s']:.2f}s | "
                f"translation "
                f"{row['translation_magnitude_px']:.2f}px | "
                f"rotation "
                f"{row['rotation_deg']:.3f}° | "
                f"inliers "
                f"{int(row['inlier_count'])}/"
                f"{int(row['total_matches'])}"
            )

    # ------------------------------------------------------------------------
    # SAVE SUMMARY JSON
    # ------------------------------------------------------------------------

    summary = {
        "input_video": str(INPUT_VIDEO),
        "resolution": {
            "width": width,
            "height": height,
        },
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "frame_step": FRAME_STEP,
        "orb_features": ORB_N_FEATURES,
        "ratio_test": RATIO_TEST,
        "ransac_reprojection_threshold_px": (
            RANSAC_REPROJECTION_THRESHOLD
        ),
        "strong_motion_threshold_px": (
            STRONG_MOTION_PX
        ),
        "measurement_count": int(len(df)),
        "valid_motion_measurements": int(
            len(valid_motion)
        ),
        "motion_class_counts": {
            str(k): int(v)
            for k, v in class_counts.items()
        },
    }

    if not valid_motion.empty:

        summary["median_translation_px"] = float(
            valid_motion[
                "translation_magnitude_px"
            ].median()
        )

        summary["p90_translation_px"] = float(
            valid_motion[
                "translation_magnitude_px"
            ].quantile(0.90)
        )

        summary["max_translation_px"] = float(
            valid_motion[
                "translation_magnitude_px"
            ].max()
        )

        summary["median_rotation_deg"] = float(
            valid_motion[
                "rotation_deg"
            ].median()
        )

        summary["max_absolute_rotation_deg"] = float(
            valid_motion[
                "rotation_deg"
            ].abs().max()
        )

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print(
        f"Saved: {SUMMARY_OUTPUT}"
    )

    # ------------------------------------------------------------------------
    # PLOT 1 — CAMERA MOTION
    # ------------------------------------------------------------------------

    print_header("CREATING CAMERA-MOTION PLOT")

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        df["time_current_s"],
        df["translation_magnitude_px"],
        label="Affine translation magnitude",
    )

    ax.axhline(
        STRONG_MOTION_PX,
        linestyle="--",
        label=(
            f"Diagnostic strong-motion threshold "
            f"({STRONG_MOTION_PX:.1f} px)"
        ),
    )

    ax.set_title(
        "Estimated Global Camera Motion Over Time"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Frame-to-frame translation magnitude (px)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_MOTION,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Saved: {PLOT_MOTION}"
    )

    # ------------------------------------------------------------------------
    # PLOT 2 — TRANSLATION COMPONENTS
    # ------------------------------------------------------------------------

    print_header("CREATING TRANSLATION PLOT")

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        df["time_current_s"],
        df["translation_x_px"],
        label="X translation",
    )

    ax.plot(
        df["time_current_s"],
        df["translation_y_px"],
        label="Y translation",
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_title(
        "Estimated Camera Translation Components"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Translation (px)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_TRANSLATION,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Saved: {PLOT_TRANSLATION}"
    )

    # ------------------------------------------------------------------------
    # PLOT 3 — GEOMETRIC QUALITY
    # ------------------------------------------------------------------------

    print_header("CREATING QUALITY PLOT")

    fig, ax1 = plt.subplots(
        figsize=(14, 7)
    )

    ax1.plot(
        df["time_current_s"],
        df["inlier_ratio"],
        label="RANSAC inlier ratio",
    )

    ax1.set_xlabel(
        "Time (s)"
    )

    ax1.set_ylabel(
        "Inlier ratio"
    )

    ax1.set_ylim(
        0,
        1,
    )

    ax1.grid(
        True,
        alpha=0.3,
    )

    ax2 = ax1.twinx()

    ax2.plot(
        df["time_current_s"],
        df["inlier_count"],
        label="Inlier count",
    )

    ax2.set_ylabel(
        "Inlier count"
    )

    ax1.set_title(
        "Camera-Motion Geometric Support"
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_QUALITY,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Saved: {PLOT_QUALITY}"
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    print_header("STEP 30 COMPLETE")

    print()
    print("Camera-motion analysis completed.")
    print()
    print("Outputs:")
    print(CSV_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(PLOT_MOTION)
    print(PLOT_TRANSLATION)
    print(PLOT_QUALITY)
    print(DIAGNOSTIC_VIDEO)

    print()
    print("IMPORTANT:")
    print(
        "This step does NOT modify the original trajectory "
        "or reconstruction."
    )
    print(
        "The measured motion is image-space camera/scene motion."
    )
    print(
        "We will use these diagnostics to design the "
        "camera-motion compensation step."
    )


if __name__ == "__main__":
    main()