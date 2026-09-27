"""
STEP 26 - DIAGNOSE HEIGHT ERROR VS FOOTBALL DIAMETER

Purpose
-------
Determine whether the remaining 3D ball-height error is related to
the apparent football diameter in pixels.

IMPORTANT
---------
Step 25 evaluates several assumed floor-tile sizes and therefore its
CSV contains multiple rows for the same observation.

For this diagnostic we intentionally use ONLY the reference working
scale:

    0.60 m

This prevents the tile-size sensitivity experiment from contaminating
the diameter/error relationship.

The Step 25 CSV already contains:

    center_height_error_m
    absolute_height_error_m
    tested_tile_size_m

Therefore we do not need to reconstruct ball-center height here.

Hypothesis
----------
The monocular depth estimate approximately follows:

    Z = f * D / d

where:
    f = focal length
    D = real football diameter
    d = apparent football diameter in pixels

Therefore, when d is small, pixel measurement errors can produce
larger depth errors.

This script is diagnostic only.
It does NOT modify the reconstruction.

Outputs
-------
    height_error_vs_diameter.csv
    height_error_vs_diameter_bins.csv
    height_error_vs_diameter_summary.json

    signed_height_error_vs_diameter.png
    absolute_height_error_vs_diameter.png
    binned_height_error_vs_diameter.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "metric_crosscheck"
    / "football_ground_crosscheck.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "diameter_height_diagnostic"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CENTER_HEIGHT_M = 0.110

# IMPORTANT:
# Step 26 analyzes ONLY the working reference scale.
REFERENCE_TILE_SIZE_M = 0.60

TILE_SIZE_TOLERANCE_M = 1e-9


# ======================================================================
# HELPERS
# ======================================================================

def find_column(df, candidates, description):
    """
    Find a column using several possible names.
    Matching is case-insensitive.
    """

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
        f"Tried:\n"
        f"{candidates}\n\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


def load_data(path):
    """Load Step 25 CSV."""

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find Step 25 output:\n{path}"
        )

    df = pd.read_csv(path)

    print("=" * 70)
    print("INPUT")
    print("=" * 70)

    print(f"File: {path}")
    print(f"Rows: {len(df)}")

    print("\nColumns:")
    print(list(df.columns))

    return df


def normalize_columns(df):
    """
    Normalize the exact information needed from Step 25.

    Step 25 provides center_height_error_m directly, so we derive
    ball_center_height_m only for readability.
    """

    track_col = find_column(
        df,
        [
            "track_id",
            "track",
        ],
        "track ID",
    )

    frame_col = find_column(
        df,
        [
            "frame",
            "frame_number",
        ],
        "frame number",
    )

    time_col = find_column(
        df,
        [
            "time_seconds",
            "time_s",
            "timestamp_seconds",
        ],
        "time",
    )

    diameter_col = find_column(
        df,
        [
            "diameter_px",
            "football_diameter_px",
            "ball_diameter_px",
            "apparent_diameter_px",
            "diameter_pixels",
        ],
        "football diameter in pixels",
    )

    signed_error_col = find_column(
        df,
        [
            "center_height_error_m",
            "height_error_m",
            "ball_center_height_error_m",
        ],
        "signed ball-center height error",
    )

    absolute_error_col = find_column(
        df,
        [
            "absolute_height_error_m",
            "abs_height_error_m",
        ],
        "absolute ball-center height error",
    )

    tile_col = find_column(
        df,
        [
            "tested_tile_size_m",
            "tile_size_m",
            "assumed_tile_size_m",
        ],
        "tested tile size",
    )

    normalized = pd.DataFrame()

    normalized["track_id"] = pd.to_numeric(
        df[track_col],
        errors="coerce",
    )

    normalized["frame"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    normalized["time_seconds"] = pd.to_numeric(
        df[time_col],
        errors="coerce",
    )

    normalized["diameter_px"] = pd.to_numeric(
        df[diameter_col],
        errors="coerce",
    )

    normalized["height_error_m"] = pd.to_numeric(
        df[signed_error_col],
        errors="coerce",
    )

    normalized["absolute_height_error_m"] = pd.to_numeric(
        df[absolute_error_col],
        errors="coerce",
    )

    normalized["tested_tile_size_m"] = pd.to_numeric(
        df[tile_col],
        errors="coerce",
    )

    # Reconstruct the actual center height for convenience.
    normalized["ball_center_height_m"] = (
        EXPECTED_CENTER_HEIGHT_M
        + normalized["height_error_m"]
    )

    return normalized


def calculate_correlations(df):
    """
    Calculate Pearson and Spearman correlations between apparent
    football diameter and absolute height error.
    """

    x = df["diameter_px"].to_numpy(dtype=float)
    y = df["absolute_height_error_m"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    results = {}

    if len(x) < 3:
        return results

    if SCIPY_AVAILABLE:

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        results["pearson_r"] = float(pearson_r)
        results["pearson_p"] = float(pearson_p)

        results["spearman_r"] = float(spearman_r)
        results["spearman_p"] = float(spearman_p)

    else:

        results["pearson_r"] = float(
            np.corrcoef(x, y)[0, 1]
        )

        results["pearson_p"] = None
        results["spearman_r"] = None
        results["spearman_p"] = None

    return results


def create_bins(df):
    """
    Divide observations into diameter quantile bins.

    Quantile bins ensure that each bin contains a comparable number
    of observations where possible.
    """

    result = df.copy()

    try:

        result["diameter_bin"] = pd.qcut(
            result["diameter_px"],
            q=6,
            duplicates="drop",
        )

    except ValueError:

        result["diameter_bin"] = pd.cut(
            result["diameter_px"],
            bins=6,
        )

    grouped = (
        result
        .groupby(
            "diameter_bin",
            observed=True,
        )
        .agg(
            diameter_min_px=(
                "diameter_px",
                "min",
            ),
            diameter_max_px=(
                "diameter_px",
                "max",
            ),
            diameter_median_px=(
                "diameter_px",
                "median",
            ),
            sample_count=(
                "diameter_px",
                "count",
            ),
            median_height_error_m=(
                "height_error_m",
                "median",
            ),
            median_absolute_height_error_m=(
                "absolute_height_error_m",
                "median",
            ),
            mean_absolute_height_error_m=(
                "absolute_height_error_m",
                "mean",
            ),
        )
        .reset_index(drop=True)
    )

    return grouped


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("STEP 26 - HEIGHT ERROR VS FOOTBALL DIAMETER")
    print("=" * 70)

    print()
    print(
        f"Reference tile-size assumption: "
        f"{REFERENCE_TILE_SIZE_M:.2f} m"
    )

    # ------------------------------------------------------------------
    # Load Step 25
    # ------------------------------------------------------------------

    df_raw = load_data(INPUT_CSV)

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("NORMALIZING INPUT")
    print("=" * 70)

    df = normalize_columns(df_raw)

    print()
    print("Normalized columns:")
    print(list(df.columns))

    # ------------------------------------------------------------------
    # Show available tile sizes
    # ------------------------------------------------------------------

    available_tile_sizes = sorted(
        df["tested_tile_size_m"]
        .dropna()
        .unique()
    )

    print()
    print("Tile-size assumptions found in Step 25 CSV:")

    for tile_size in available_tile_sizes:

        count = (
            df["tested_tile_size_m"]
            .eq(tile_size)
            .sum()
        )

        print(
            f"  {tile_size:.2f} m: "
            f"{count} observations"
        )

    # ------------------------------------------------------------------
    # Select ONLY 0.60 m
    # ------------------------------------------------------------------

    df = df[
        np.isclose(
            df["tested_tile_size_m"],
            REFERENCE_TILE_SIZE_M,
            atol=TILE_SIZE_TOLERANCE_M,
        )
    ].copy()

    print()
    print("=" * 70)
    print("REFERENCE-SCALE FILTER")
    print("=" * 70)

    print(
        f"\nUsing only tile size: "
        f"{REFERENCE_TILE_SIZE_M:.2f} m"
    )

    print(
        f"Rows after filtering: "
        f"{len(df)}"
    )

    # ------------------------------------------------------------------
    # Remove invalid observations
    # ------------------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "track_id",
            "frame",
            "time_seconds",
            "diameter_px",
            "height_error_m",
            "absolute_height_error_m",
        ]
    )

    df = df[
        df["diameter_px"] > 0
    ]

    after = len(df)

    print(
        f"Removed invalid observations: "
        f"{before - after}"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("REFERENCE-SCALE DIAGNOSTIC")
    print("=" * 70)

    print(
        f"\nExpected football-center height: "
        f"{EXPECTED_CENTER_HEIGHT_M:.3f} m"
    )

    print(
        f"Observations: "
        f"{len(df)}"
    )

    print(
        f"Diameter range: "
        f"{df['diameter_px'].min():.1f} -> "
        f"{df['diameter_px'].max():.1f} px"
    )

    print(
        f"Median diameter: "
        f"{df['diameter_px'].median():.1f} px"
    )

    print(
        f"Median ball-center height: "
        f"{df['ball_center_height_m'].median():.4f} m"
    )

    print(
        f"Median signed height error: "
        f"{df['height_error_m'].median():.4f} m"
    )

    print(
        f"Median absolute height error: "
        f"{df['absolute_height_error_m'].median():.4f} m"
    )

    # ------------------------------------------------------------------
    # Per-track statistics
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("PER-TRACK DIAGNOSTIC")
    print("=" * 70)

    for track_id, track_df in df.groupby(
        "track_id"
    ):

        print()
        print(
            f"Track {int(track_id)}:"
        )

        print(
            f"  Observations: "
            f"{len(track_df)}"
        )

        print(
            f"  Diameter range: "
            f"{track_df['diameter_px'].min():.1f} -> "
            f"{track_df['diameter_px'].max():.1f} px"
        )

        print(
            f"  Median diameter: "
            f"{track_df['diameter_px'].median():.1f} px"
        )

        print(
            f"  Median signed height error: "
            f"{track_df['height_error_m'].median():.4f} m"
        )

        print(
            f"  Median absolute height error: "
            f"{track_df['absolute_height_error_m'].median():.4f} m"
        )

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CORRELATION")
    print("=" * 70)

    correlations = calculate_correlations(df)

    if correlations:

        print(
            f"\nPearson correlation "
            f"(diameter vs absolute error): "
            f"{correlations['pearson_r']:.4f}"
        )

        if correlations["pearson_p"] is not None:

            print(
                f"Pearson p-value: "
                f"{correlations['pearson_p']:.6g}"
            )

        print(
            f"\nSpearman correlation "
            f"(diameter vs absolute error): "
            f"{correlations['spearman_r']:.4f}"
        )

        if correlations["spearman_p"] is not None:

            print(
                f"Spearman p-value: "
                f"{correlations['spearman_p']:.6g}"
            )

    # ------------------------------------------------------------------
    # Diameter bins
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DIAMETER-BIN ANALYSIS")
    print("=" * 70)

    binned = create_bins(df)

    print()

    print(
        binned.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Save detailed CSV
    # ------------------------------------------------------------------

    detailed_csv = (
        OUTPUT_DIR
        / "height_error_vs_diameter.csv"
    )

    df.to_csv(
        detailed_csv,
        index=False,
    )

    print()
    print(
        f"Saved detailed diagnostic:"
    )
    print(detailed_csv)

    # ------------------------------------------------------------------
    # Save binned CSV
    # ------------------------------------------------------------------

    binned_csv = (
        OUTPUT_DIR
        / "height_error_vs_diameter_bins.csv"
    )

    binned.to_csv(
        binned_csv,
        index=False,
    )

    print()
    print(
        f"Saved binned diagnostic:"
    )
    print(binned_csv)

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------

    summary = {
        "reference_tile_size_m": REFERENCE_TILE_SIZE_M,
        "expected_center_height_m": EXPECTED_CENTER_HEIGHT_M,
        "observations": int(len(df)),
        "diameter_min_px": float(
            df["diameter_px"].min()
        ),
        "diameter_max_px": float(
            df["diameter_px"].max()
        ),
        "diameter_median_px": float(
            df["diameter_px"].median()
        ),
        "median_ball_center_height_m": float(
            df["ball_center_height_m"].median()
        ),
        "median_signed_height_error_m": float(
            df["height_error_m"].median()
        ),
        "median_absolute_height_error_m": float(
            df["absolute_height_error_m"].median()
        ),
        "mean_absolute_height_error_m": float(
            df["absolute_height_error_m"].mean()
        ),
        "correlations": correlations,
    }

    summary_json = (
        OUTPUT_DIR
        / "height_error_vs_diameter_summary.json"
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
        f"Saved summary:"
    )
    print(summary_json)

    # ==================================================================
    # PLOT 1 - SIGNED ERROR
    # ==================================================================

    print()
    print("=" * 70)
    print("CREATING SIGNED-ERROR PLOT")
    print("=" * 70)

    plt.figure(figsize=(12, 7))

    for track_id, track_df in df.groupby(
        "track_id"
    ):

        plt.scatter(
            track_df["diameter_px"],
            track_df["height_error_m"],
            s=18,
            alpha=0.65,
            label=f"Track {int(track_id)}",
        )

    plt.axhline(
        0.0,
        linestyle="--",
        label="Zero height error",
    )

    plt.xlabel(
        "Football apparent diameter (pixels)"
    )

    plt.ylabel(
        "Signed ball-center height error (m)"
    )

    plt.title(
        "Signed Football Height Error vs Apparent Diameter "
        "(0.60 m Tile Assumption)"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    signed_plot = (
        OUTPUT_DIR
        / "signed_height_error_vs_diameter.png"
    )

    plt.savefig(
        signed_plot,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {signed_plot}"
    )

    # ==================================================================
    # PLOT 2 - ABSOLUTE ERROR
    # ==================================================================

    print()
    print("=" * 70)
    print("CREATING ABSOLUTE-ERROR PLOT")
    print("=" * 70)

    plt.figure(figsize=(12, 7))

    for track_id, track_df in df.groupby(
        "track_id"
    ):

        plt.scatter(
            track_df["diameter_px"],
            track_df["absolute_height_error_m"],
            s=18,
            alpha=0.65,
            label=f"Track {int(track_id)}",
        )

    plt.plot(
        binned["diameter_median_px"],
        binned["median_absolute_height_error_m"],
        marker="o",
        linewidth=2,
        label="Binned median absolute error",
    )

    plt.xlabel(
        "Football apparent diameter (pixels)"
    )

    plt.ylabel(
        "Absolute ball-center height error (m)"
    )

    plt.title(
        "Absolute Football Height Error vs Apparent Diameter "
        "(0.60 m Tile Assumption)"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    absolute_plot = (
        OUTPUT_DIR
        / "absolute_height_error_vs_diameter.png"
    )

    plt.savefig(
        absolute_plot,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {absolute_plot}"
    )

    # ==================================================================
    # PLOT 3 - BINNED ERROR
    # ==================================================================

    print()
    print("=" * 70)
    print("CREATING BINNED-ERROR PLOT")
    print("=" * 70)

    plt.figure(figsize=(12, 7))

    plt.plot(
        binned["diameter_median_px"],
        binned["median_absolute_height_error_m"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "Median football apparent diameter (pixels)"
    )

    plt.ylabel(
        "Median absolute height error (m)"
    )

    plt.title(
        "Binned Median Height Error vs Football Diameter "
        "(0.60 m Tile Assumption)"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    binned_plot = (
        OUTPUT_DIR
        / "binned_height_error_vs_diameter.png"
    )

    plt.savefig(
        binned_plot,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {binned_plot}"
    )

    # ==================================================================
    # INTERPRETATION
    # ==================================================================

    print()
    print("=" * 70)
    print("STEP 26 COMPLETE")
    print("=" * 70)

    if correlations:

        rho = correlations["spearman_r"]

        print()

        if rho < -0.30:

            print(
                "INTERPRETATION:"
            )

            print(
                "There is a negative monotonic relationship "
                "between apparent football diameter and absolute "
                "height error."
            )

            print(
                "Larger apparent football sizes tend to be "
                "associated with smaller reconstruction errors."
            )

            print(
                "This supports diameter/depth measurement sensitivity "
                "as an important error source."
            )

        elif rho > 0.30:

            print(
                "INTERPRETATION:"
            )

            print(
                "Absolute height error tends to increase with "
                "football diameter."
            )

            print(
                "This does not support the simple hypothesis that "
                "small apparent diameter is the dominant error source."
            )

        else:

            print(
                "INTERPRETATION:"
            )

            print(
                "The correlation between apparent football diameter "
                "and absolute height error is weak."
            )

            print(
                "The remaining error may therefore be more strongly "
                "related to camera calibration, distortion, "
                "homography, or other systematic effects."
            )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Correlation does not prove causation."
    )

    print()

    print(
        "Step 26 analyzed ONLY the 0.60 m reference scale."
    )

    print()

    print("Diagnostic outputs:")

    print(signed_plot)
    print(absolute_plot)
    print(binned_plot)
    print(detailed_csv)
    print(binned_csv)
    print(summary_json)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()