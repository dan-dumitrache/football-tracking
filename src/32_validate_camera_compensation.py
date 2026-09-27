"""
STEP 32 - STATIC GROUND-PLANE CAMERA-COMPENSATION VALIDATION

Purpose
-------
Independently validate the frame -> reference homographies produced in Step 31.

For stationary scene features on the ground plane:

    before compensation:
        feature position changes when the camera moves

    after compensation:
        H(frame -> reference) should map the feature back near its
        reference-frame position

This step is diagnostic only. It does NOT modify the original trajectory,
3D reconstruction, or final integrated video.

Outputs
-------
data/output/validation/camera_compensated_top_view/
    static_feature_stability.csv
    static_feature_stability_summary.json
    static_feature_stability_over_time.png
    residual_displacement_histogram.png
    camera_compensation_validation.mp4
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "rgb.avi"

HOMOGRAPHY_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "camera_compensated_top_view"
    / "frame_to_reference_homographies.csv"
)

CAMERA_MOTION_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "camera_motion"
    / "camera_motion.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "camera_compensated_top_view"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_CSV = OUTPUT_DIR / "static_feature_stability.csv"
SUMMARY_JSON = OUTPUT_DIR / "static_feature_stability_summary.json"
TIME_PLOT = OUTPUT_DIR / "static_feature_stability_over_time.png"
HIST_PLOT = OUTPUT_DIR / "residual_displacement_histogram.png"
DIAGNOSTIC_VIDEO = OUTPUT_DIR / "camera_compensation_validation.mp4"


# ---------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------

REFERENCE_FRAME = 300

# Same basic ground ROI used by Step 31.
GROUND_Y_FRACTION = 0.35

# ORB / matching parameters.
ORB_FEATURES = 2000
ORB_FAST_THRESHOLD = 12
RATIO_TEST = 0.75

# Keep validation focused on good, repeatable correspondences.
MIN_MATCHES = 8
MAX_VALIDATION_POINTS_PER_FRAME = 80

# Diagnostic window containing the strongest measured camera movement
# from Step 30. This is used only as a named reporting interval.
STRONG_MOTION_START = 150
STRONG_MOTION_END = 180

# Video output size.
VIDEO_CODEC = "mp4v"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

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


def find_column(df: pd.DataFrame, candidates: list[str], description: str) -> str:
    normalized = {normalize_name(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    # Slightly more permissive fallback.
    for col in df.columns:
        n = normalize_name(col)
        for candidate in candidates:
            c = normalize_name(candidate)
            if c in n or n in c:
                return col

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Tried:\n{candidates}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


def load_homographies(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Homography CSV not found:\n{path}")

    df = pd.read_csv(path)

    print("Homography CSV columns:")
    print(list(df.columns))

    frame_col = find_column(
        df,
        ["frame", "frame_number", "frame_id"],
        "frame number",
    )

    # Step 31 stores the homography coefficients as H11 ... H33.
    #
    # Important:
    #   H11 H12 H13
    #   H21 H22 H23
    #   H31 H32 H33
    #
    # Internally we use zero-based NumPy indexing:
    #   H[0,0] ... H[2,2]
    #
    # Support both the actual Step 31 naming convention (H11...H33)
    # and the zero-based h00...h22 convention used by earlier versions
    # of this validation script.
    normalized = {normalize_name(c): c for c in df.columns}

    coefficient_columns = []

    for r in range(3):
        row = []

        for c in range(3):

            one_based = f"H{r + 1}{c + 1}"
            zero_based = f"h{r}{c}"

            possibilities = [
                one_based,
                one_based.lower(),
                f"H_{r + 1}{c + 1}",
                f"H_{r + 1}_{c + 1}",
                zero_based,
                f"h_{r}{c}",
                f"h_{r}_{c}",
                f"homography_{r}{c}",
                f"homography_{r}_{c}",
            ]

            found = None

            for p in possibilities:
                key = normalize_name(p)

                if key in normalized:
                    found = normalized[key]
                    break

            if found is None:
                row = []
                break

            row.append(found)

        if row:
            coefficient_columns.append(row)

    if len(coefficient_columns) != 3:
        raise ValueError(
            "Could not find a 3x3 homography in the Step 31 CSV.\n"
            "Expected columns similar to H11,H12,...,H33.\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    homographies: dict[int, np.ndarray] = {}

    for _, row in df.iterrows():
        try:
            frame = int(row[frame_col])

            H = np.array(
                [
                    [float(row[coefficient_columns[0][0]]),
                     float(row[coefficient_columns[0][1]]),
                     float(row[coefficient_columns[0][2]])],
                    [float(row[coefficient_columns[1][0]]),
                     float(row[coefficient_columns[1][1]]),
                     float(row[coefficient_columns[1][2]])],
                    [float(row[coefficient_columns[2][0]]),
                     float(row[coefficient_columns[2][1]]),
                     float(row[coefficient_columns[2][2]])],
                ],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(H)):
                continue

            # Homographies are defined only up to scale.
            if abs(H[2, 2]) > 1e-12:
                H = H / H[2, 2]

            if abs(np.linalg.det(H)) < 1e-12:
                continue

            homographies[frame] = H

        except (TypeError, ValueError):
            continue

    if not homographies:
        raise ValueError("No usable homographies were loaded.")

    return homographies


def create_orb() -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        fastThreshold=ORB_FAST_THRESHOLD,
    )


def detect_features(
    orb: cv2.ORB,
    frame_bgr: np.ndarray,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    mask = np.zeros_like(gray, dtype=np.uint8)

    y0 = int(GROUND_Y_FRACTION * h)
    mask[y0:, :] = 255

    keypoints, descriptors = orb.detectAndCompute(gray, mask)

    if keypoints is None:
        keypoints = []

    return keypoints, descriptors


def ratio_test_matches(
    desc_current: np.ndarray | None,
    desc_reference: np.ndarray | None,
) -> list[cv2.DMatch]:

    if desc_current is None or desc_reference is None:
        return []

    if len(desc_current) < 2 or len(desc_reference) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    knn = matcher.knnMatch(
        desc_current,
        desc_reference,
        k=2,
    )

    good = []

    for pair in knn:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < RATIO_TEST * n.distance:
            good.append(m)

    return good


def apply_homography(
    H: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:

    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(points, H)

    return transformed.reshape(-1, 2)


def robust_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def read_motion_summary() -> dict:
    """
    Optional use of Step 30 output.

    This does not determine compensation. It simply reports whether the
    strongest validation interval overlaps the previously measured
    strong-camera-motion interval.
    """
    if not CAMERA_MOTION_CSV.exists():
        return {
            "available": False,
            "reason": "Step 30 camera_motion.csv not found.",
        }

    try:
        df = pd.read_csv(CAMERA_MOTION_CSV)

        frame_col = find_column(
            df,
            ["frame", "frame_number", "frame_pair_start"],
            "camera-motion frame",
        )

        translation_col = None

        for candidate in [
            "translation_magnitude_px",
            "affine_translation_magnitude_px",
            "translation_px",
            "translation_magnitude",
        ]:
            try:
                translation_col = find_column(
                    df,
                    [candidate],
                    "camera translation",
                )
                break
            except ValueError:
                pass

        if translation_col is None:
            return {
                "available": True,
                "usable": False,
                "reason": "Translation column not identified.",
            }

        frames = pd.to_numeric(df[frame_col], errors="coerce")
        translation = pd.to_numeric(
            df[translation_col],
            errors="coerce",
        )

        mask = (
            frames.between(
                STRONG_MOTION_START,
                STRONG_MOTION_END,
            )
            & np.isfinite(translation)
        )

        values = translation[mask].to_numpy(dtype=float)

        return {
            "available": True,
            "usable": True,
            "strong_window_start_frame": STRONG_MOTION_START,
            "strong_window_end_frame": STRONG_MOTION_END,
            "translation_median_px": (
                float(np.median(values)) if len(values) else None
            ),
            "translation_max_px": (
                float(np.max(values)) if len(values) else None
            ),
            "sample_count": int(len(values)),
        }

    except Exception as exc:
        return {
            "available": True,
            "usable": False,
            "reason": str(exc),
        }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    print_header(
        "STEP 32 - STATIC GROUND-PLANE CAMERA-COMPENSATION VALIDATION"
    )

    print()
    print("This step is diagnostic only.")
    print("No tracking or reconstruction output will be modified.")

    # ---------------------------------------------------------------
    # INPUT CHECK
    # ---------------------------------------------------------------

    print_header("INPUT")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found:\n{VIDEO_PATH}")

    if not HOMOGRAPHY_CSV.exists():
        raise FileNotFoundError(
            f"Step 31 homography CSV not found:\n{HOMOGRAPHY_CSV}"
        )

    print(f"Video:        {VIDEO_PATH}")
    print(f"Homographies: {HOMOGRAPHY_CSV}")

    # ---------------------------------------------------------------
    # LOAD VIDEO
    # ---------------------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print()
    print(f"Resolution: {width} x {height}")
    print(f"FPS:        {fps:.2f}")
    print(f"Frames:     {frame_count}")

    if REFERENCE_FRAME >= frame_count:
        raise ValueError(
            f"Reference frame {REFERENCE_FRAME} is outside the video."
        )

    # ---------------------------------------------------------------
    # LOAD HOMOGRAPHIES
    # ---------------------------------------------------------------

    print_header("LOADING FRAME -> REFERENCE HOMOGRAPHIES")

    homographies = load_homographies(HOMOGRAPHY_CSV)

    print(f"Usable homographies: {len(homographies)}")
    print(
        f"Frame range: "
        f"{min(homographies.keys())} -> {max(homographies.keys())}"
    )

    if REFERENCE_FRAME in homographies:
        print("Reference-frame homography: available")
    else:
        print(
            "WARNING: reference-frame homography is missing. "
            "The validation can still run, but frame 300 will be skipped."
        )

    # ---------------------------------------------------------------
    # OPTIONAL STEP 30 INFORMATION
    # ---------------------------------------------------------------

    motion_info = read_motion_summary()

    # ---------------------------------------------------------------
    # READ REFERENCE FRAME
    # ---------------------------------------------------------------

    print_header("REFERENCE FEATURE EXTRACTION")

    cap.set(cv2.CAP_PROP_POS_FRAMES, REFERENCE_FRAME)
    ok, reference_frame = cap.read()

    if not ok:
        raise RuntimeError(
            f"Could not read reference frame {REFERENCE_FRAME}."
        )

    orb = create_orb()

    reference_keypoints, reference_descriptors = detect_features(
        orb,
        reference_frame,
    )

    print(
        f"Reference frame: {REFERENCE_FRAME}"
    )
    print(
        f"Reference ground features: "
        f"{len(reference_keypoints)}"
    )

    if reference_descriptors is None:
        raise RuntimeError(
            "Could not compute reference descriptors."
        )

    # ---------------------------------------------------------------
    # OUTPUT VIDEO
    # ---------------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

    writer = cv2.VideoWriter(
        str(DIAGNOSTIC_VIDEO),
        fourcc,
        fps if fps > 0 else 30.0,
        (width * 2, height),
    )

    if not writer.isOpened():
        print(
            "WARNING: Could not create diagnostic video. "
            "CSV and plots will still be generated."
        )
        writer = None

    # ---------------------------------------------------------------
    # FRAME-BY-FRAME VALIDATION
    # ---------------------------------------------------------------

    print_header("VALIDATING STATIC GROUND FEATURES")

    records = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_idx in range(frame_count):

        ok, frame = cap.read()

        if not ok:
            break

        # Reference frame itself should have zero displacement by definition.
        H = homographies.get(frame_idx)

        if H is None:
            continue

        current_keypoints, current_descriptors = detect_features(
            orb,
            frame,
        )

        good_matches = ratio_test_matches(
            current_descriptors,
            reference_descriptors,
        )

        if len(good_matches) < MIN_MATCHES:
            continue

        current_points = np.array(
            [
                current_keypoints[m.queryIdx].pt
                for m in good_matches
            ],
            dtype=np.float64,
        )

        reference_points = np.array(
            [
                reference_keypoints[m.trainIdx].pt
                for m in good_matches
            ],
            dtype=np.float64,
        )

        # -----------------------------------------------------------
        # Geometric sanity filter
        # -----------------------------------------------------------
        #
        # Very bad descriptor matches should not dominate the
        # before/after statistics. We use a robust median-distance
        # filter before applying the homography.

        raw_distances = np.linalg.norm(
            current_points - reference_points,
            axis=1,
        )

        median_raw = float(np.median(raw_distances))

        if len(raw_distances) >= 8:
            mad = float(
                np.median(
                    np.abs(raw_distances - median_raw)
                )
            )

            robust_limit = median_raw + max(
                20.0,
                6.0 * 1.4826 * mad,
            )

            keep = raw_distances <= robust_limit

            current_points = current_points[keep]
            reference_points = reference_points[keep]
            raw_distances = raw_distances[keep]

        if len(current_points) < MIN_MATCHES:
            continue

        # Limit the number of points so every frame has comparable
        # computational cost.
        if len(current_points) > MAX_VALIDATION_POINTS_PER_FRAME:
            order = np.argsort(raw_distances)
            order = order[:MAX_VALIDATION_POINTS_PER_FRAME]

            current_points = current_points[order]
            reference_points = reference_points[order]
            raw_distances = raw_distances[order]

        compensated_points = apply_homography(
            H,
            current_points,
        )

        before_error = np.linalg.norm(
            current_points - reference_points,
            axis=1,
        )

        after_error = np.linalg.norm(
            compensated_points - reference_points,
            axis=1,
        )

        # Remove numerically invalid points.
        valid = (
            np.isfinite(before_error)
            & np.isfinite(after_error)
        )

        before_error = before_error[valid]
        after_error = after_error[valid]

        if len(before_error) < MIN_MATCHES:
            continue

        before_stats = robust_stats(before_error)
        after_stats = robust_stats(after_error)

        record = {
            "frame": frame_idx,
            "time_seconds": (
                frame_idx / fps if fps > 0 else np.nan
            ),
            "feature_count": int(len(before_error)),
            "median_before_px": before_stats["median"],
            "mean_before_px": before_stats["mean"],
            "p90_before_px": before_stats["p90"],
            "max_before_px": before_stats["max"],
            "median_after_px": after_stats["median"],
            "mean_after_px": after_stats["mean"],
            "p90_after_px": after_stats["p90"],
            "max_after_px": after_stats["max"],
            "median_improvement_px": (
                before_stats["median"] - after_stats["median"]
            ),
            "improvement_ratio": (
                before_stats["median"]
                / max(after_stats["median"], 1e-9)
            ),
            "strong_motion_window": (
                STRONG_MOTION_START
                <= frame_idx
                <= STRONG_MOTION_END
            ),
        }

        records.append(record)

        # -----------------------------------------------------------
        # Diagnostic video
        # -----------------------------------------------------------

        if writer is not None:

            left = frame.copy()
            right = frame.copy()

            # Draw a subset of current points and their compensated
            # positions on the right image.
            draw_count = min(
                len(current_points),
                40,
            )

            for i in range(draw_count):
                p_current = tuple(
                    np.round(current_points[i]).astype(int)
                )

                p_reference = tuple(
                    np.round(reference_points[i]).astype(int)
                )

                p_comp = tuple(
                    np.round(compensated_points[i]).astype(int)
                )

                cv2.circle(
                    left,
                    p_current,
                    3,
                    (0, 255, 255),
                    -1,
                )

                cv2.circle(
                    right,
                    p_comp,
                    3,
                    (0, 255, 0),
                    -1,
                )

                # Reference location.
                cv2.circle(
                    right,
                    p_reference,
                    2,
                    (255, 0, 0),
                    -1,
                )

            # Labels.
            cv2.putText(
                left,
                "CURRENT FRAME - BEFORE COMPENSATION",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                right,
                "FRAME -> REFERENCE - AFTER COMPENSATION",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                right,
                (
                    f"median: "
                    f"{after_stats['median']:.2f}px"
                ),
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                right,
                (
                    f"features: "
                    f"{len(before_error)}"
                ),
                (20, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            combined = np.hstack([left, right])

            writer.write(combined)

        if frame_idx % 30 == 0:
            print(
                f"Processed frame {frame_idx}/{frame_count - 1} | "
                f"features={len(before_error)} | "
                f"before median={before_stats['median']:.2f}px | "
                f"after median={after_stats['median']:.2f}px"
            )

    cap.release()

    if writer is not None:
        writer.release()

    if not records:
        raise RuntimeError(
            "No valid static-feature validation records were produced."
        )

    results = pd.DataFrame(records)

    results.to_csv(
        STATIC_CSV,
        index=False,
    )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    print_header("VALIDATION SUMMARY")

    before_values = results["median_before_px"].to_numpy()
    after_values = results["median_after_px"].to_numpy()

    strong = results["strong_motion_window"].to_numpy(dtype=bool)

    overall_before = robust_stats(before_values)
    overall_after = robust_stats(after_values)

    summary = {
        "reference_frame": REFERENCE_FRAME,
        "video": str(VIDEO_PATH),
        "homography_csv": str(HOMOGRAPHY_CSV),
        "ground_roi_y_fraction": GROUND_Y_FRACTION,
        "frames_analyzed": int(len(results)),
        "feature_observations": int(
            results["feature_count"].sum()
        ),
        "overall_before_px": overall_before,
        "overall_after_px": overall_after,
        "overall_median_improvement_px": float(
            np.median(
                before_values - after_values
            )
        ),
        "overall_median_improvement_ratio": float(
            np.median(before_values)
            / max(np.median(after_values), 1e-9)
        ),
        "strong_motion_window": {
            "start_frame": STRONG_MOTION_START,
            "end_frame": STRONG_MOTION_END,
            "start_time_s": STRONG_MOTION_START / fps,
            "end_time_s": STRONG_MOTION_END / fps,
            "frames_with_validation": int(strong.sum()),
        },
        "step30_motion_information": motion_info,
    }

    if np.any(strong):
        strong_before = robust_stats(
            results.loc[
                strong,
                "median_before_px",
            ].to_numpy()
        )

        strong_after = robust_stats(
            results.loc[
                strong,
                "median_after_px",
            ].to_numpy()
        )

        summary["strong_motion_before_px"] = strong_before
        summary["strong_motion_after_px"] = strong_after

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

    print(
        f"Validation frames: {len(results)}"
    )
    print()
    print("ALL VALIDATION FRAMES")
    print(
        f"Before compensation:"
        f" median={overall_before['median']:.3f}px, "
        f"P90={overall_before['p90']:.3f}px"
    )
    print(
        f"After compensation:"
        f"  median={overall_after['median']:.3f}px, "
        f"P90={overall_after['p90']:.3f}px"
    )
    print(
        f"Median improvement ratio:"
        f" {summary['overall_median_improvement_ratio']:.2f}x"
    )

    if np.any(strong):
        print()
        print(
            f"STRONG CAMERA-MOTION WINDOW "
            f"({STRONG_MOTION_START}->{STRONG_MOTION_END})"
        )
        print(
            f"Before:"
            f" median={strong_before['median']:.3f}px, "
            f"P90={strong_before['p90']:.3f}px"
        )
        print(
            f"After:"
            f"  median={strong_after['median']:.3f}px, "
            f"P90={strong_after['p90']:.3f}px"
        )

        ratio = (
            strong_before["median"]
            / max(strong_after["median"], 1e-9)
        )

        print(
            f"Strong-motion improvement ratio:"
            f" {ratio:.2f}x"
        )

    # ---------------------------------------------------------------
    # PLOT 1: TIME
    # ---------------------------------------------------------------

    print_header("CREATING TIME-STABILITY PLOT")

    plt.figure(figsize=(14, 7))

    plt.plot(
        results["time_seconds"],
        results["median_before_px"],
        label="Median displacement before compensation",
    )

    plt.plot(
        results["time_seconds"],
        results["median_after_px"],
        label="Median residual after compensation",
    )

    plt.axvspan(
        STRONG_MOTION_START / fps,
        STRONG_MOTION_END / fps,
        alpha=0.15,
        label="Strong camera-motion validation window",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Median static-feature displacement (px)")
    plt.title(
        "Static Ground-Feature Stability Before vs After "
        "Camera Compensation"
    )
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TIME_PLOT, dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # PLOT 2: HISTOGRAM
    # ---------------------------------------------------------------

    print_header("CREATING RESIDUAL-DISPLACEMENT HISTOGRAM")

    plt.figure(figsize=(12, 7))

    plt.hist(
        before_values,
        bins=40,
        alpha=0.55,
        label="Before compensation",
    )

    plt.hist(
        after_values,
        bins=40,
        alpha=0.55,
        label="After compensation",
    )

    plt.xlabel(
        "Median static-feature displacement per frame (px)"
    )
    plt.ylabel("Frame count")
    plt.title(
        "Static Ground-Feature Residual Displacement"
    )
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HIST_PLOT, dpi=150)
    plt.close()

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print_header("STEP 32 COMPLETE")

    print()
    print("Saved:")
    print(STATIC_CSV)
    print(SUMMARY_JSON)
    print(TIME_PLOT)
    print(HIST_PLOT)

    if writer is not None:
        print(DIAGNOSTIC_VIDEO)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This validation does not modify the original trajectory, "
        "3D reconstruction, or final integrated video."
    )
    print(
        "A successful result is indicated by a substantial reduction "
        "in static-feature displacement after compensation, especially "
        "during the strong camera-motion interval."
    )


if __name__ == "__main__":
    main()
