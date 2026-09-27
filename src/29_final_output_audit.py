"""
STEP 29 - FINAL PROJECT OUTPUT AUDIT

Purpose
-------
Audit the complete football-tracking pipeline and produce a concise
machine-readable summary of:

    - input video
    - 2D detection/tracking
    - ground-plane calibration
    - 3D reconstruction
    - top-view trajectory
    - validation
    - 3D quality classification
    - final integrated visualization

This script does NOT modify any existing pipeline outputs.

It only checks that expected artifacts exist and that important
metadata/assumptions are internally consistent.

Outputs
-------
    data/output/final_audit/
        project_summary.csv
        project_summary.json
        project_audit_report.txt
"""


from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "rgb.avi"
)

CALIBRATION_DIR = (
    PROJECT_ROOT
    / "calibration"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
)

AUDIT_DIR = (
    OUTPUT_DIR
    / "final_audit"
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# EXPECTED ARTIFACTS
# ======================================================================

EXPECTED_FILES = {

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------

    "ground_points":
        CALIBRATION_DIR
        / "ground_points.json",

    "ground_homography":
        CALIBRATION_DIR
        / "ground_homography.json",

    # --------------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------------

    "reconstruction_3d":
        OUTPUT_DIR
        / "reconstruction_3d"
        / "reconstructed_ball_3d.csv",

    "smoothed_reconstruction_3d":
        OUTPUT_DIR
        / "reconstruction_3d"
        / "smoothed_ball_3d.csv",

    # --------------------------------------------------------------
    # Top view
    # --------------------------------------------------------------

    "top_view_contact_trajectory":
        OUTPUT_DIR
        / "top_view"
        / "top_view_contact_trajectory.csv",

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    "metric_crosscheck":
        OUTPUT_DIR
        / "validation"
        / "metric_crosscheck"
        / "football_ground_crosscheck.csv",

    "metric_crosscheck_summary":
        OUTPUT_DIR
        / "validation"
        / "metric_crosscheck"
        / "metric_crosscheck_summary.csv",

    "scale_sensitivity":
        OUTPUT_DIR
        / "validation"
        / "scale_sensitivity"
        / "metric_scale_sensitivity.csv",

    # --------------------------------------------------------------
    # Diameter analysis
    # --------------------------------------------------------------

    "diameter_height_diagnostic":
        OUTPUT_DIR
        / "validation"
        / "diameter_height_diagnostic"
        / "height_error_vs_diameter.csv",

    "diameter_threshold_analysis":
        OUTPUT_DIR
        / "validation"
        / "diameter_height_diagnostic"
        / "3d_reliability_vs_diameter_threshold.csv",

    # --------------------------------------------------------------
    # 3D quality
    # --------------------------------------------------------------

    "3d_quality":
        OUTPUT_DIR
        / "validation"
        / "3d_quality"
        / "reconstructed_ball_3d_quality.csv",

    "3d_quality_summary":
        OUTPUT_DIR
        / "validation"
        / "3d_quality"
        / "3d_quality_summary.csv",

    # --------------------------------------------------------------
    # Final visualization
    # --------------------------------------------------------------

    "final_integrated_video":
        OUTPUT_DIR
        / "final"
        / "final_integrated_video.mp4",
}


# ======================================================================
# HELPERS
# ======================================================================

def status(ok):
    return "PASS" if ok else "FAIL"


def file_check(path):

    exists = path.exists()

    if exists:

        size_bytes = path.stat().st_size

        return {
            "exists": True,
            "size_bytes": int(size_bytes),
        }

    return {
        "exists": False,
        "size_bytes": 0,
    }


def load_json(path):

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return None


def load_csv(path):

    if not path.exists():
        return None

    try:

        return pd.read_csv(path)

    except Exception:

        return None


def find_column(df, candidates):

    if df is None:
        return None

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# VIDEO AUDIT
# ======================================================================

def audit_video():

    print_section("INPUT VIDEO")

    result = {
        "exists": False,
        "readable": False,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_seconds": None,
    }

    if not INPUT_VIDEO.exists():

        print(
            f"FAIL - Video not found:\n{INPUT_VIDEO}"
        )

        return result

    result["exists"] = True

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():

        print(
            "FAIL - Video exists but could not be opened."
        )

        return result

    result["readable"] = True

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        frame_count / fps
        if fps > 0
        else None
    )

    result.update({
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    })

    cap.release()

    print(
        f"Status: PASS"
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {frame_count}"
    )

    print(
        f"Duration: {duration:.2f} s"
        if duration is not None
        else "Duration: unknown"
    )

    return result


# ======================================================================
# CALIBRATION AUDIT
# ======================================================================

def audit_calibration():

    print_section("GROUND-PLANE CALIBRATION")

    result = {
        "ground_points_exists": False,
        "homography_exists": False,
        "tile_width_m": None,
        "tile_height_m": None,
        "homography_key": None,
        "mean_reprojection_error_m": None,
        "max_reprojection_error_m": None,
    }

    points_path = (
        EXPECTED_FILES["ground_points"]
    )

    homography_path = (
        EXPECTED_FILES["ground_homography"]
    )

    points = load_json(
        points_path
    )

    homography = load_json(
        homography_path
    )

    if points is not None:

        result["ground_points_exists"] = True

        print(
            "Ground-point calibration: PASS"
        )

    else:

        print(
            "Ground-point calibration: FAIL"
        )

    if homography is not None:

        result["homography_exists"] = True

        print(
            "Ground homography: PASS"
        )

        # ----------------------------------------------------------
        # Tile scale
        # ----------------------------------------------------------

        scale = homography.get(
            "scale_assumption",
            {}
        )

        result["tile_width_m"] = scale.get(
            "tile_width_m"
        )

        result["tile_height_m"] = scale.get(
            "tile_height_m"
        )

        # ----------------------------------------------------------
        # Homography key
        # ----------------------------------------------------------

        for key in [
            "homography_image_to_ground",
            "homography",
            "H",
        ]:

            if key in homography:

                result["homography_key"] = key
                break

        # ----------------------------------------------------------
        # Reprojection
        # ----------------------------------------------------------

        result[
            "mean_reprojection_error_m"
        ] = homography.get(
            "mean_reprojection_error_m"
        )

        result[
            "max_reprojection_error_m"
        ] = homography.get(
            "max_reprojection_error_m"
        )

        print(
            f"Working tile size: "
            f"{result['tile_width_m']} m"
        )

        print(
            "Important: tile size is a working "
            "assumption, not an independent measurement."
        )

        print(
            f"Mean reprojection error: "
            f"{result['mean_reprojection_error_m']}"
        )

        print(
            f"Max reprojection error: "
            f"{result['max_reprojection_error_m']}"
        )

    else:

        print(
            "Ground homography: FAIL"
        )

    return result


# ======================================================================
# RECONSTRUCTION AUDIT
# ======================================================================

def audit_reconstruction():

    print_section("3D RECONSTRUCTION")

    path = (
        EXPECTED_FILES["smoothed_reconstruction_3d"]
    )

    df = load_csv(path)

    result = {
        "exists": path.exists(),
        "rows": 0,
        "tracks": [],
        "frame_min": None,
        "frame_max": None,
        "required_columns": [],
        "missing_columns": [],
    }

    required_columns = [
        "track_id",
        "frame",
        "time_seconds",
        "diameter_px",
        "depth_m_smoothed",
        "X_m_smoothed",
        "Y_m_smoothed",
        "Vx_mps_smoothed",
        "Vy_mps_smoothed",
        "Vz_mps_smoothed",
    ]

    result["required_columns"] = (
        required_columns
    )

    if df is None:

        print(
            "FAIL - Smoothed 3D reconstruction unavailable."
        )

        return result

    result["rows"] = len(df)

    result["missing_columns"] = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if result["missing_columns"]:

        print(
            "FAIL - Missing columns:"
        )

        print(
            result["missing_columns"]
        )

        return result

    result["tracks"] = sorted(
        df["track_id"]
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )

    result["frame_min"] = int(
        df["frame"].min()
    )

    result["frame_max"] = int(
        df["frame"].max()
    )

    print(
        "Smoothed 3D reconstruction: PASS"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Tracks: {result['tracks']}"
    )

    print(
        f"Frame range: "
        f"{result['frame_min']} -> "
        f"{result['frame_max']}"
    )

    for track_id, track_df in df.groupby(
        "track_id"
    ):

        print(
            f"  Track {int(track_id)}: "
            f"{len(track_df)} observations, "
            f"frames "
            f"{int(track_df['frame'].min())} -> "
            f"{int(track_df['frame'].max())}"
        )

    return result


# ======================================================================
# TOP-VIEW AUDIT
# ======================================================================

def audit_top_view():

    print_section("TOP-VIEW TRAJECTORY")

    path = (
        EXPECTED_FILES[
            "top_view_contact_trajectory"
        ]
    )

    df = load_csv(path)

    result = {
        "exists": path.exists(),
        "rows": 0,
        "tracks": [],
        "required_columns": [],
        "missing_columns": [],
    }

    required_columns = [
        "track_id",
        "frame",
        "time_seconds",
        "ground_contact_x_m",
        "ground_contact_y_m",
    ]

    result["required_columns"] = (
        required_columns
    )

    if df is None:

        print(
            "FAIL - Top-view trajectory unavailable."
        )

        return result

    result["rows"] = len(df)

    result["missing_columns"] = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if result["missing_columns"]:

        print(
            "FAIL - Missing columns:"
        )

        print(
            result["missing_columns"]
        )

        return result

    result["tracks"] = sorted(
        df["track_id"]
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )

    print(
        "Top-view contact trajectory: PASS"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Tracks: {result['tracks']}"
    )

    return result


# ======================================================================
# QUALITY AUDIT
# ======================================================================

def audit_quality():

    print_section("3D QUALITY CLASSIFICATION")

    path = (
        EXPECTED_FILES["3d_quality"]
    )

    df = load_csv(path)

    result = {
        "exists": path.exists(),
        "rows": 0,
        "threshold_px": None,
        "acceptable": 0,
        "low_confidence": 0,
        "acceptable_percent": None,
    }

    if df is None:

        print(
            "FAIL - 3D quality file unavailable."
        )

        return result

    result["rows"] = len(df)

    if (
        "3d_reliability_threshold_px"
        in df.columns
    ):

        values = (
            pd.to_numeric(
                df[
                    "3d_reliability_threshold_px"
                ],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        if len(values) > 0:

            result["threshold_px"] = float(
                values[0]
            )

    if "3d_metric_reliable" in df.columns:

        reliable = (
            df["3d_metric_reliable"]
            .astype(bool)
        )

        result["acceptable"] = int(
            reliable.sum()
        )

        result["low_confidence"] = int(
            (~reliable).sum()
        )

        result["acceptable_percent"] = (
            100.0
            * result["acceptable"]
            / len(df)
        )

    print(
        "3D quality classification: PASS"
    )

    print(
        f"Rows: {result['rows']}"
    )

    print(
        f"Threshold: "
        f"{result['threshold_px']} px"
    )

    print(
        f"Acceptable: "
        f"{result['acceptable']} "
        f"({result['acceptable_percent']:.1f}%)"
    )

    print(
        f"Low confidence: "
        f"{result['low_confidence']}"
    )

    return result


# ======================================================================
# VALIDATION AUDIT
# ======================================================================

def audit_validation():

    print_section("VALIDATION ARTIFACTS")

    result = {}

    for name in [
        "metric_crosscheck",
        "metric_crosscheck_summary",
        "scale_sensitivity",
        "diameter_height_diagnostic",
        "diameter_threshold_analysis",
    ]:

        path = EXPECTED_FILES[name]

        exists = path.exists()

        result[name] = exists

        print(
            f"{status(exists)}  "
            f"{name}: "
            f"{path}"
        )

    return result


# ======================================================================
# FINAL VIDEO AUDIT
# ======================================================================

def audit_final_video():

    print_section("FINAL INTEGRATED VIDEO")

    path = (
        EXPECTED_FILES[
            "final_integrated_video"
        ]
    )

    result = {
        "exists": path.exists(),
        "readable": False,
        "frame_count": None,
        "fps": None,
        "width": None,
        "height": None,
    }

    if not path.exists():

        print(
            "FAIL - Final integrated video not found."
        )

        return result

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        print(
            "FAIL - Final video exists but "
            "could not be opened."
        )

        return result

    result["readable"] = True

    result["frame_count"] = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    result["fps"] = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    result["width"] = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    result["height"] = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    print(
        "Final integrated video: PASS"
    )

    print(
        f"Resolution: "
        f"{result['width']} x "
        f"{result['height']}"
    )

    print(
        f"FPS: {result['fps']:.2f}"
    )

    print(
        f"Frames: {result['frame_count']}"
    )

    return result


# ======================================================================
# TRACK CONSISTENCY AUDIT
# ======================================================================

def audit_tracks(reconstruction):

    print_section("TRACK CONSISTENCY")

    result = {
        "track_0_exists": False,
        "track_1_exists": False,
        "gap_206_388_clear": False,
        "track_0_start": None,
        "track_0_end": None,
        "track_1_start": None,
        "track_1_end": None,
    }

    if reconstruction is None:

        print(
            "FAIL - Reconstruction unavailable."
        )

        return result

    df = load_csv(
        EXPECTED_FILES[
            "smoothed_reconstruction_3d"
        ]
    )

    if df is None:

        return result

    for track_id in [0, 1]:

        track = df[
            df["track_id"] == track_id
        ]

        if len(track) == 0:
            continue

        start = int(
            track["frame"].min()
        )

        end = int(
            track["frame"].max()
        )

        result[
            f"track_{track_id}_exists"
        ] = True

        result[
            f"track_{track_id}_start"
        ] = start

        result[
            f"track_{track_id}_end"
        ] = end

        print(
            f"Track {track_id}: "
            f"frames {start} -> {end}"
        )

    # --------------------------------------------------------------
    # Verify that no reconstruction observations exist in the
    # known visibility gap 206-388.
    # --------------------------------------------------------------

    gap_rows = df[
        (
            df["frame"] >= 206
        )
        &
        (
            df["frame"] <= 388
        )
    ]

    result[
        "gap_206_388_clear"
    ] = len(gap_rows) == 0

    if result["gap_206_388_clear"]:

        print(
            "PASS - No reconstructed ball observations "
            "exist in frames 206-388."
        )

    else:

        print(
            "FAIL - Observations found inside "
            "the known visibility gap 206-388."
        )

        print(
            f"Unexpected rows: {len(gap_rows)}"
        )

    return result


# ======================================================================
# BUILD SUMMARY
# ======================================================================

def build_summary(
    video,
    calibration,
    reconstruction,
    top_view,
    quality,
    validation,
    final_video,
    tracks,
):

    # --------------------------------------------------------------
    # Overall checks
    # --------------------------------------------------------------

    checks = {}

    checks["input_video"] = (
        video["exists"]
        and video["readable"]
    )

    checks["ground_calibration"] = (
        calibration["ground_points_exists"]
        and calibration["homography_exists"]
    )

    checks["3d_reconstruction"] = (
        reconstruction["exists"]
        and not reconstruction["missing_columns"]
    )

    checks["top_view"] = (
        top_view["exists"]
        and not top_view["missing_columns"]
    )

    checks["3d_quality"] = (
        quality["exists"]
        and quality["threshold_px"] is not None
    )

    checks["validation"] = all(
        validation.values()
    )

    checks["final_integrated_video"] = (
        final_video["exists"]
        and final_video["readable"]
    )

    checks["track_consistency"] = (
        tracks["track_0_exists"]
        and tracks["track_1_exists"]
        and tracks["gap_206_388_clear"]
    )

    checks["overall"] = all(
        checks.values()
    )

    # --------------------------------------------------------------
    # Assumptions
    # --------------------------------------------------------------

    assumptions = {

        "football_diameter_m":
            0.220,

        "football_radius_m":
            0.110,

        "working_tile_size_m":
            calibration[
                "tile_width_m"
            ],

        "tile_size_independently_verified":
            False,

        "camera_intrinsics":
            "Estimated nominal RGB intrinsics; "
            "not factory-calibrated D435i intrinsics.",

        "depth_method":
            "Monocular apparent-football-size depth estimate.",

        "3d_quality_threshold_px":
            quality["threshold_px"],

        "3d_quality_threshold_basis":
            "Empirical Step 27 analysis.",

        "ground_contact_model":
            "Football treated as rolling/in ground contact; "
            "contact point used for top-view trajectory.",

        "camera_motion":
            "Camera treated as stationary.",

        "visibility_gap":
            "Frames 206-388 contain no reconstructed football "
            "observations; no interpolation across this gap.",
    }

    return {
        "project": {
            "name": "Football Tracking and 3D Reconstruction",
            "audit_step": 29,
        },

        "overall_status": (
            "PASS"
            if checks["overall"]
            else "INCOMPLETE"
        ),

        "checks": checks,

        "input_video": video,

        "calibration": calibration,

        "reconstruction": reconstruction,

        "top_view": top_view,

        "quality": quality,

        "validation": validation,

        "final_video": final_video,

        "tracks": tracks,

        "assumptions": assumptions,
    }


# ======================================================================
# WRITE REPORT
# ======================================================================

def write_report(summary):

    report_path = (
        AUDIT_DIR
        / "project_audit_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FOOTBALL TRACKING PROJECT - FINAL AUDIT\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Overall status: "
            f"{summary['overall_status']}\n\n"
        )

        f.write(
            "PIPELINE CHECKS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for name, value in summary[
            "checks"
        ].items():

            f.write(
                f"{name}: "
                f"{'PASS' if value else 'FAIL'}\n"
            )

        f.write(
            "\nASSUMPTIONS / LIMITATIONS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for key, value in summary[
            "assumptions"
        ].items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\nTRACKS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        tracks = summary["tracks"]

        f.write(
            f"Track 0: "
            f"{tracks['track_0_start']} -> "
            f"{tracks['track_0_end']}\n"
        )

        f.write(
            f"Track 1: "
            f"{tracks['track_1_start']} -> "
            f"{tracks['track_1_end']}\n"
        )

        f.write(
            "Known visibility gap 206-388 clear: "
            f"{tracks['gap_206_388_clear']}\n"
        )

    return report_path


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("STEP 29 - FINAL PROJECT OUTPUT AUDIT")
    print("=" * 70)

    print()
    print(
        "This step does not modify any existing outputs."
    )

    # ------------------------------------------------------------------
    # Individual audits
    # ------------------------------------------------------------------

    video = audit_video()

    calibration = audit_calibration()

    reconstruction = audit_reconstruction()

    top_view = audit_top_view()

    quality = audit_quality()

    validation = audit_validation()

    final_video = audit_final_video()

    tracks = audit_tracks(
        reconstruction
    )

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------

    summary = build_summary(
        video=video,
        calibration=calibration,
        reconstruction=reconstruction,
        top_view=top_view,
        quality=quality,
        validation=validation,
        final_video=final_video,
        tracks=tracks,
    )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------

    json_path = (
        AUDIT_DIR
        / "project_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # ------------------------------------------------------------------
    # Save compact CSV
    # ------------------------------------------------------------------

    csv_rows = []

    for category, value in summary[
        "checks"
    ].items():

        csv_rows.append({
            "check": category,
            "status": (
                "PASS"
                if value
                else "FAIL"
            ),
        })

    csv_path = (
        AUDIT_DIR
        / "project_summary.csv"
    )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        csv_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Write text report
    # ------------------------------------------------------------------

    report_path = write_report(
        summary
    )

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------

    print_section(
        "FINAL AUDIT SUMMARY"
    )

    for name, value in summary[
        "checks"
    ].items():

        print(
            f"{status(value)}  {name}"
        )

    print()

    if summary["overall_status"] == "PASS":

        print(
            "OVERALL STATUS: PASS"
        )

        print()
        print(
            "All expected major pipeline artifacts "
            "are present and internally consistent."
        )

    else:

        print(
            "OVERALL STATUS: INCOMPLETE"
        )

        print()
        print(
            "One or more expected artifacts/checks failed."
        )

    print()
    print("=" * 70)
    print("AUDIT OUTPUTS")
    print("=" * 70)

    print(
        f"JSON:\n{json_path}"
    )

    print(
        f"CSV:\n{csv_path}"
    )

    print(
        f"Report:\n{report_path}"
    )

    print()
    print("=" * 70)
    print("STEP 29 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()