"""
STEP 28 - CLASSIFY 3D RECONSTRUCTION QUALITY

Purpose
-------
Add an empirical quality label to each reconstructed football
observation based on its apparent pixel diameter.

The original 3D reconstruction is NOT modified.

Based on Step 27:

    diameter < 20 px
        -> LOW_3D_CONFIDENCE

    diameter >= 20 px
        -> ACCEPTABLE_3D_CONFIDENCE

The 20 px threshold was selected because it was the first tested
threshold for which the 90th-percentile absolute height error was
below 20 cm while retaining approximately 87% of observations.

Important
---------
This threshold is empirical and specific to this video/pipeline.
It is NOT a universal computer-vision threshold.

Outputs
-------
    data/output/validation/3d_quality/
        reconstructed_ball_3d_quality.csv
        3d_quality_summary.csv
        3d_quality_over_time.png
        3d_quality_by_diameter.png
"""


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "reconstruction_3d"
    / "smoothed_ball_3d.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "3d_quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Empirically selected in Step 27.
MIN_RELIABLE_DIAMETER_PX = 20.0

EXPECTED_CENTER_HEIGHT_M = 0.110

REFERENCE_TILE_SIZE_M = 0.60


# ======================================================================
# HELPERS
# ======================================================================

def find_column(df, candidates, description):

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Tried: {candidates}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


def load_reconstruction():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Could not find reconstruction:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print("=" * 70)
    print("INPUT")
    print("=" * 70)

    print(f"File: {INPUT_CSV}")
    print(f"Rows: {len(df)}")

    print("\nColumns:")
    print(list(df.columns))

    return df


def normalize_columns(df):

    track_col = find_column(
        df,
        ["track_id", "track"],
        "track ID",
    )

    frame_col = find_column(
        df,
        ["frame", "frame_number"],
        "frame number",
    )

    time_col = find_column(
        df,
        ["time_seconds", "time_s"],
        "time",
    )

    diameter_col = find_column(
        df,
        [
            "diameter_px",
            "football_diameter_px",
            "ball_diameter_px",
        ],
        "football diameter",
    )

    # --------------------------------------------------------------
    # Camera-space coordinates
    # --------------------------------------------------------------

    x_col = find_column(
        df,
        [
            "X_m_smoothed",
            "X_m",
            "camera_x_m",
        ],
        "camera X coordinate",
    )

    y_col = find_column(
        df,
        [
            "Y_m_smoothed",
            "Y_m",
            "camera_y_m",
        ],
        "camera Y coordinate",
    )

    z_col = find_column(
        df,
        [
            "depth_m_smoothed",
            "depth_m",
            "camera_z_m",
        ],
        "camera Z/depth",
    )

    # --------------------------------------------------------------
    # Optional velocity columns
    # --------------------------------------------------------------

    velocity_columns = {}

    optional_columns = {
        "Vx_mps": [
            "Vx_mps_smoothed",
            "Vx_mps",
        ],
        "Vy_mps": [
            "Vy_mps_smoothed",
            "Vy_mps",
        ],
        "Vz_mps": [
            "Vz_mps_smoothed",
            "Vz_mps",
        ],
        "speed_mps": [
            "speed_mps_smoothed",
            "speed_mps",
        ],
    }

    for output_name, candidates in optional_columns.items():

        for candidate in candidates:

            if candidate in df.columns:

                velocity_columns[output_name] = candidate
                break

    result = pd.DataFrame()

    result["track_id"] = pd.to_numeric(
        df[track_col],
        errors="coerce",
    )

    result["frame"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    result["time_seconds"] = pd.to_numeric(
        df[time_col],
        errors="coerce",
    )

    result["diameter_px"] = pd.to_numeric(
        df[diameter_col],
        errors="coerce",
    )

    result["X_m"] = pd.to_numeric(
        df[x_col],
        errors="coerce",
    )

    result["Y_m"] = pd.to_numeric(
        df[y_col],
        errors="coerce",
    )

    result["Z_m"] = pd.to_numeric(
        df[z_col],
        errors="coerce",
    )

    for output_name, source_name in velocity_columns.items():

        result[output_name] = pd.to_numeric(
            df[source_name],
            errors="coerce",
        )

    return result


def classify_quality(diameter_px):

    if not np.isfinite(diameter_px):

        return "INVALID"

    if diameter_px < MIN_RELIABLE_DIAMETER_PX:

        return "LOW_3D_CONFIDENCE"

    return "ACCEPTABLE_3D_CONFIDENCE"


def quality_reason(diameter_px):

    if not np.isfinite(diameter_px):

        return "Invalid or missing football diameter"

    if diameter_px < MIN_RELIABLE_DIAMETER_PX:

        return (
            "Football is below empirical 3D reliability "
            "diameter threshold"
        )

    return (
        "Football diameter meets empirical 3D reliability "
        "threshold"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("STEP 28 - 3D RECONSTRUCTION QUALITY CLASSIFICATION")
    print("=" * 70)

    print()
    print(
        f"Minimum reliable football diameter: "
        f"{MIN_RELIABLE_DIAMETER_PX:.1f} px"
    )

    print(
        "This threshold was derived empirically in Step 27."
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    df_raw = load_reconstruction()

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("NORMALIZING RECONSTRUCTION")
    print("=" * 70)

    df = normalize_columns(df_raw)

    print()
    print("Normalized columns:")
    print(list(df.columns))

    # ------------------------------------------------------------------
    # Remove invalid observations
    # ------------------------------------------------------------------

    required = [
        "track_id",
        "frame",
        "time_seconds",
        "diameter_px",
        "X_m",
        "Y_m",
        "Z_m",
    ]

    before = len(df)

    df = df.dropna(
        subset=required
    ).copy()

    df = df[
        df["diameter_px"] > 0
    ].copy()

    after = len(df)

    print()
    print(
        f"Valid observations: {after}"
    )

    print(
        f"Removed observations: "
        f"{before - after}"
    )

    # ------------------------------------------------------------------
    # Add quality labels
    # ------------------------------------------------------------------

    df["3d_quality"] = (
        df["diameter_px"]
        .apply(classify_quality)
    )

    df["3d_quality_reason"] = (
        df["diameter_px"]
        .apply(quality_reason)
    )

    df["3d_reliability_threshold_px"] = (
        MIN_RELIABLE_DIAMETER_PX
    )

    # Boolean flag useful for downstream processing.
    df["3d_metric_reliable"] = (
        df["diameter_px"]
        >= MIN_RELIABLE_DIAMETER_PX
    )

    # ------------------------------------------------------------------
    # Distance from camera
    # ------------------------------------------------------------------

    df["distance_m"] = np.sqrt(
        df["X_m"] ** 2
        + df["Y_m"] ** 2
        + df["Z_m"] ** 2
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)

    total = len(df)

    quality_counts = (
        df["3d_quality"]
        .value_counts()
    )

    for quality, count in quality_counts.items():

        percentage = (
            100.0 * count / total
        )

        print(
            f"{quality}: "
            f"{count} observations "
            f"({percentage:.1f}%)"
        )

    # ------------------------------------------------------------------
    # Track summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("PER-TRACK QUALITY")
    print("=" * 70)

    track_summary_rows = []

    for track_id, track_df in df.groupby(
        "track_id"
    ):

        total_track = len(track_df)

        reliable_track = int(
            track_df[
                "3d_metric_reliable"
            ].sum()
        )

        low_track = (
            total_track
            - reliable_track
        )

        reliable_percent = (
            100.0
            * reliable_track
            / total_track
        )

        row = {
            "track_id": int(track_id),
            "observations": total_track,
            "acceptable_3d": reliable_track,
            "low_3d_confidence": low_track,
            "acceptable_percent": reliable_percent,
            "median_diameter_px": float(
                track_df[
                    "diameter_px"
                ].median()
            ),
            "min_diameter_px": float(
                track_df[
                    "diameter_px"
                ].min()
            ),
            "max_diameter_px": float(
                track_df[
                    "diameter_px"
                ].max()
            ),
        }

        track_summary_rows.append(row)

        print()
        print(
            f"Track {int(track_id)}:"
        )

        print(
            f"  Observations: "
            f"{total_track}"
        )

        print(
            f"  Acceptable 3D: "
            f"{reliable_track} "
            f"({reliable_percent:.1f}%)"
        )

        print(
            f"  Low confidence: "
            f"{low_track}"
        )

        print(
            f"  Median diameter: "
            f"{row['median_diameter_px']:.1f} px"
        )

    track_summary = pd.DataFrame(
        track_summary_rows
    )

    # ------------------------------------------------------------------
    # Save detailed quality CSV
    # ------------------------------------------------------------------

    output_csv = (
        OUTPUT_DIR
        / "reconstructed_ball_3d_quality.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    print()
    print(
        f"Saved detailed quality CSV:\n"
        f"{output_csv}"
    )

    # ------------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------------

    summary_csv = (
        OUTPUT_DIR
        / "3d_quality_summary.csv"
    )

    track_summary.to_csv(
        summary_csv,
        index=False,
    )

    print()
    print(
        f"Saved track summary:\n"
        f"{summary_csv}"
    )

    # ==================================================================
    # PLOT 1 - QUALITY OVER TIME
    # ==================================================================

    print()
    print("=" * 70)
    print("CREATING QUALITY OVER TIME PLOT")
    print("=" * 70)

    plt.figure(
        figsize=(13, 7)
    )

    for track_id, track_df in df.groupby(
        "track_id"
    ):

        reliable = track_df[
            track_df["3d_metric_reliable"]
        ]

        low = track_df[
            ~track_df["3d_metric_reliable"]
        ]

        if len(reliable) > 0:

            plt.scatter(
                reliable["time_seconds"],
                reliable["diameter_px"],
                s=22,
                alpha=0.75,
                label=(
                    f"Track {int(track_id)} "
                    "acceptable"
                ),
            )

        if len(low) > 0:

            plt.scatter(
                low["time_seconds"],
                low["diameter_px"],
                s=28,
                marker="x",
                alpha=0.9,
                label=(
                    f"Track {int(track_id)} "
                    "low confidence"
                ),
            )

    plt.axhline(
        MIN_RELIABLE_DIAMETER_PX,
        linestyle="--",
        label=(
            f"3D reliability threshold "
            f"({MIN_RELIABLE_DIAMETER_PX:.0f} px)"
        ),
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Football apparent diameter (px)"
    )

    plt.title(
        "3D Reconstruction Quality Over Time"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    time_plot = (
        OUTPUT_DIR
        / "3d_quality_over_time.png"
    )

    plt.savefig(
        time_plot,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved:\n{time_plot}"
    )

    # ==================================================================
    # PLOT 2 - QUALITY BY DIAMETER
    # ==================================================================

    print()
    print("=" * 70)
    print("CREATING QUALITY-BY-DIAMETER PLOT")
    print("=" * 70)

    plt.figure(
        figsize=(12, 7)
    )

    for track_id, track_df in df.groupby(
        "track_id"
    ):

        acceptable = track_df[
            track_df["3d_metric_reliable"]
        ]

        low = track_df[
            ~track_df["3d_metric_reliable"]
        ]

        if len(acceptable) > 0:

            plt.scatter(
                acceptable["diameter_px"],
                acceptable["Z_m"],
                s=22,
                alpha=0.7,
                label=(
                    f"Track {int(track_id)} "
                    "acceptable"
                ),
            )

        if len(low) > 0:

            plt.scatter(
                low["diameter_px"],
                low["Z_m"],
                s=30,
                marker="x",
                alpha=0.9,
                label=(
                    f"Track {int(track_id)} "
                    "low confidence"
                ),
            )

    plt.axvline(
        MIN_RELIABLE_DIAMETER_PX,
        linestyle="--",
        label=(
            f"3D reliability threshold "
            f"({MIN_RELIABLE_DIAMETER_PX:.0f} px)"
        ),
    )

    plt.xlabel(
        "Football apparent diameter (px)"
    )

    plt.ylabel(
        "Estimated camera depth Z (m)"
    )

    plt.title(
        "Estimated 3D Depth vs Football Apparent Size"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    diameter_plot = (
        OUTPUT_DIR
        / "3d_quality_by_diameter.png"
    )

    plt.savefig(
        diameter_plot,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved:\n{diameter_plot}"
    )

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print()
    print("=" * 70)
    print("STEP 28 COMPLETE")
    print("=" * 70)

    print()
    print(
        "The original 3D reconstruction has NOT been modified."
    )

    print(
        "A separate quality classification has been generated."
    )

    print()
    print(
        "Quality rule:"
    )

    print(
        f"  diameter < "
        f"{MIN_RELIABLE_DIAMETER_PX:.0f} px"
        "  -> LOW_3D_CONFIDENCE"
    )

    print(
        f"  diameter >= "
        f"{MIN_RELIABLE_DIAMETER_PX:.0f} px"
        " -> ACCEPTABLE_3D_CONFIDENCE"
    )

    print()
    print(
        "Important:"
    )

    print(
        "The 20 px threshold is empirical for this "
        "video/pipeline and is not a universal threshold."
    )

    print()
    print(
        "Outputs:"
    )

    print(output_csv)
    print(summary_csv)
    print(time_plot)
    print(diameter_plot)


if __name__ == "__main__":
    main()