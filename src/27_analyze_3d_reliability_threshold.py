"""
STEP 27 - ANALYZE 3D RECONSTRUCTION RELIABILITY VS BALL SIZE

Purpose
-------
Determine how the apparent football diameter affects the reliability
of the monocular metric 3D reconstruction.

The analysis uses the 0.60 m working tile-size assumption and tests
different minimum apparent football diameters.

This does NOT modify the reconstruction.

It answers:

    "At what apparent football size does metric 3D reconstruction
     become reasonably stable?"

Outputs
-------
    3d_reliability_vs_diameter_threshold.csv
    3d_reliability_summary.json
    3d_reliability_vs_diameter_threshold.png
"""


from pathlib import Path
import json

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
    / "validation"
    / "diameter_height_diagnostic"
    / "height_error_vs_diameter.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "diameter_height_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REFERENCE_TILE_SIZE_M = 0.60

# Thresholds are minimum apparent football diameters in pixels.
THRESHOLDS_PX = [
    15,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    35,
    40,
    45,
    50,
]


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


def load_input():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Could not find:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print("=" * 70)
    print("INPUT")
    print("=" * 70)

    print(f"File: {INPUT_CSV}")
    print(f"Rows: {len(df)}")

    return df


def normalize(df):

    track_col = find_column(
        df,
        ["track_id", "track"],
        "track ID",
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

    error_col = find_column(
        df,
        [
            "absolute_height_error_m",
            "abs_height_error_m",
        ],
        "absolute height error",
    )

    result = pd.DataFrame()

    result["track_id"] = pd.to_numeric(
        df[track_col],
        errors="coerce",
    )

    result["diameter_px"] = pd.to_numeric(
        df[diameter_col],
        errors="coerce",
    )

    result["absolute_height_error_m"] = pd.to_numeric(
        df[error_col],
        errors="coerce",
    )

    # The Step 26 input has already been filtered to 0.60 m.
    # If the column is still present, verify it.
    if "tested_tile_size_m" in df.columns:

        tile_values = pd.to_numeric(
            df["tested_tile_size_m"],
            errors="coerce",
        )

        valid_tile = np.isclose(
            tile_values,
            REFERENCE_TILE_SIZE_M,
            atol=1e-9,
        )

        result = result[valid_tile].copy()

    result = result.dropna(
        subset=[
            "track_id",
            "diameter_px",
            "absolute_height_error_m",
        ]
    )

    result = result[
        result["diameter_px"] > 0
    ].copy()

    return result


def calculate_stats(df, threshold):

    retained = df[
        df["diameter_px"] >= threshold
    ].copy()

    total = len(df)
    n = len(retained)

    if n == 0:
        return {
            "threshold_px": threshold,
            "observations": 0,
            "retained_percent": 0.0,
            "median_abs_error_m": np.nan,
            "mean_abs_error_m": np.nan,
            "p90_abs_error_m": np.nan,
            "max_abs_error_m": np.nan,
        }

    errors = retained[
        "absolute_height_error_m"
    ].to_numpy()

    result = {
        "threshold_px": threshold,
        "observations": n,
        "retained_percent": 100.0 * n / total,
        "median_abs_error_m": float(
            np.median(errors)
        ),
        "mean_abs_error_m": float(
            np.mean(errors)
        ),
        "p90_abs_error_m": float(
            np.percentile(errors, 90)
        ),
        "max_abs_error_m": float(
            np.max(errors)
        ),
    }

    # --------------------------------------------------------------
    # Track-specific statistics
    # --------------------------------------------------------------

    for track_id in sorted(
        retained["track_id"].unique()
    ):

        track = retained[
            retained["track_id"] == track_id
        ]

        errors_track = track[
            "absolute_height_error_m"
        ].to_numpy()

        prefix = f"track_{int(track_id)}"

        result[
            f"{prefix}_observations"
        ] = len(track)

        result[
            f"{prefix}_median_abs_error_m"
        ] = float(
            np.median(errors_track)
        )

        result[
            f"{prefix}_p90_abs_error_m"
        ] = float(
            np.percentile(
                errors_track,
                90,
            )
        )

    return result


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("STEP 27 - 3D RECONSTRUCTION RELIABILITY VS BALL SIZE")
    print("=" * 70)

    print()
    print(
        f"Reference tile size: "
        f"{REFERENCE_TILE_SIZE_M:.2f} m"
    )

    print(
        "\nTesting minimum football diameters:"
    )

    print(
        ", ".join(
            f"{x} px"
            for x in THRESHOLDS_PX
        )
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    df_raw = load_input()

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    df = normalize(df_raw)

    print()
    print("=" * 70)
    print("NORMALIZED DATA")
    print("=" * 70)

    print(
        f"\nValid observations: {len(df)}"
    )

    print(
        f"Diameter range: "
        f"{df['diameter_px'].min():.1f} -> "
        f"{df['diameter_px'].max():.1f} px"
    )

    # ------------------------------------------------------------------
    # Threshold analysis
    # ------------------------------------------------------------------

    results = []

    for threshold in THRESHOLDS_PX:

        stats = calculate_stats(
            df,
            threshold,
        )

        results.append(stats)

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Print overall table
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    display_columns = [
        "threshold_px",
        "observations",
        "retained_percent",
        "median_abs_error_m",
        "mean_abs_error_m",
        "p90_abs_error_m",
        "max_abs_error_m",
    ]

    print()

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Track-specific table
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRACK-SPECIFIC MEDIAN ERRORS")
    print("=" * 70)

    track_columns = [
        "threshold_px",
        "track_0_observations",
        "track_0_median_abs_error_m",
        "track_1_observations",
        "track_1_median_abs_error_m",
    ]

    available_track_columns = [
        col
        for col in track_columns
        if col in results_df.columns
    ]

    print()

    print(
        results_df[
            available_track_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------

    output_csv = (
        OUTPUT_DIR
        / "3d_reliability_vs_diameter_threshold.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    print()
    print(
        f"Saved:\n{output_csv}"
    )

    # ------------------------------------------------------------------
    # Identify useful thresholds
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("THRESHOLD INTERPRETATION")
    print("=" * 70)

    # Find thresholds where median error <= 0.10 m
    reliable_10cm = results_df[
        results_df["median_abs_error_m"] <= 0.10
    ]

    if len(reliable_10cm) > 0:

        first_10cm = reliable_10cm.iloc[0]

        print()
        print(
            "First tested threshold with median absolute "
            "height error <= 10 cm:"
        )

        print(
            f"  {first_10cm['threshold_px']:.0f} px"
        )

        print(
            f"  Median error: "
            f"{first_10cm['median_abs_error_m']:.4f} m"
        )

        print(
            f"  Observations retained: "
            f"{int(first_10cm['observations'])}"
        )

        print(
            f"  Retained: "
            f"{first_10cm['retained_percent']:.1f}%"
        )

    else:

        print(
            "\nNo tested threshold achieved "
            "median error <= 10 cm."
        )

    # Find thresholds where P90 <= 20 cm
    reliable_p90 = results_df[
        results_df["p90_abs_error_m"] <= 0.20
    ]

    if len(reliable_p90) > 0:

        first_p90 = reliable_p90.iloc[0]

        print()
        print(
            "First tested threshold with 90th-percentile "
            "absolute error <= 20 cm:"
        )

        print(
            f"  {first_p90['threshold_px']:.0f} px"
        )

        print(
            f"  P90 error: "
            f"{first_p90['p90_abs_error_m']:.4f} m"
        )

        print(
            f"  Observations retained: "
            f"{int(first_p90['observations'])}"
        )

        print(
            f"  Retained: "
            f"{first_p90['retained_percent']:.1f}%"
        )

    else:

        print(
            "\nNo tested threshold achieved "
            "P90 error <= 20 cm."
        )

    # ------------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------------

    summary = {
        "reference_tile_size_m": REFERENCE_TILE_SIZE_M,
        "total_observations": int(len(df)),
        "diameter_min_px": float(
            df["diameter_px"].min()
        ),
        "diameter_max_px": float(
            df["diameter_px"].max()
        ),
        "thresholds_px": THRESHOLDS_PX,
        "results": results,
    }

    summary_json = (
        OUTPUT_DIR
        / "3d_reliability_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print(
        f"Saved:\n{summary_json}"
    )

    # ==================================================================
    # PLOT
    # ==================================================================

    print()
    print("=" * 70)
    print("CREATING RELIABILITY PLOT")
    print("=" * 70)

    valid = results_df[
        np.isfinite(
            results_df["median_abs_error_m"]
        )
    ]

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        valid["threshold_px"],
        valid["median_abs_error_m"],
        marker="o",
        linewidth=2,
        label="Median absolute error",
    )

    plt.plot(
        valid["threshold_px"],
        valid["p90_abs_error_m"],
        marker="s",
        linewidth=2,
        label="90th-percentile absolute error",
    )

    plt.axhline(
        0.10,
        linestyle="--",
        label="10 cm median-error reference",
    )

    plt.axhline(
        0.20,
        linestyle=":",
        label="20 cm P90-error reference",
    )

    plt.xlabel(
        "Minimum football apparent diameter (pixels)"
    )

    plt.ylabel(
        "Height reconstruction error (m)"
    )

    plt.title(
        "3D Reconstruction Reliability vs Minimum Football Size"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / "3d_reliability_vs_diameter_threshold.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved:\n{plot_path}"
    )

    # ==================================================================
    # FINAL
    # ==================================================================

    print()
    print("=" * 70)
    print("STEP 27 COMPLETE")
    print("=" * 70)

    print()
    print(
        "This step is diagnostic only."
    )

    print(
        "No original detections or trajectories were modified."
    )

    print()
    print(
        "The purpose is to determine whether metric 3D "
        "reconstruction should have a minimum apparent-ball-size "
        "quality threshold."
    )


if __name__ == "__main__":
    main()