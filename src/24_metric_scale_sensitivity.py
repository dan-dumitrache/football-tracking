from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "top_view"
    / "top_view_contact_trajectory.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
    / "scale_sensitivity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# The trajectory was originally calibrated using this assumption.
REFERENCE_TILE_SIZE_M = 0.60

# Plausible alternative tile sizes to test.
TEST_TILE_SIZES_M = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_track_metrics(track_df):
    """
    Calculate ground-motion metrics for one track.

    Input coordinates must already be in meters.
    """

    track_df = track_df.sort_values("frame").reset_index(drop=True)

    x = track_df["ground_x_m"].to_numpy(dtype=float)
    y = track_df["ground_y_m"].to_numpy(dtype=float)
    t = track_df["time_seconds"].to_numpy(dtype=float)

    if len(x) < 2:
        return {
            "observations": len(x),
            "duration_s": 0.0,
            "path_length_m": 0.0,
            "net_displacement_m": 0.0,
            "mean_speed_mps": 0.0,
            "median_speed_mps": 0.0,
            "max_speed_mps": 0.0,
            "mean_frame_movement_m": 0.0,
            "max_frame_movement_m": 0.0,
        }

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    displacement = np.sqrt(dx**2 + dy**2)

    valid_dt = dt > 0

    speed = np.zeros_like(displacement)

    speed[valid_dt] = (
        displacement[valid_dt]
        / dt[valid_dt]
    )

    path_length = float(np.sum(displacement))

    net_displacement = float(
        np.sqrt(
            (x[-1] - x[0]) ** 2
            + (y[-1] - y[0]) ** 2
        )
    )

    duration = float(t[-1] - t[0])

    return {
        "observations": len(x),
        "duration_s": duration,
        "path_length_m": path_length,
        "net_displacement_m": net_displacement,
        "mean_speed_mps": float(np.mean(speed)),
        "median_speed_mps": float(np.median(speed)),
        "max_speed_mps": float(np.max(speed)),
        "mean_frame_movement_m": float(np.mean(displacement)),
        "max_frame_movement_m": float(np.max(displacement)),
    }


# ============================================================
# LOAD INPUT
# ============================================================

def load_input():
    print_header("STEP 24 - METRIC SCALE SENSITIVITY ANALYSIS")

    print()
    print("INPUT")
    print("=" * 70)

    print(f"CSV:")
    print(INPUT_CSV)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print(f"Rows: {len(df)}")
    print()
    print("Columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Identify required columns
    # --------------------------------------------------------

    required = [
        "track_id",
        "frame",
        "time_seconds",
        "ground_contact_x_m",
        "ground_contact_y_m",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Normalize names
    # --------------------------------------------------------

    df = df[
        [
            "track_id",
            "frame",
            "time_seconds",
            "ground_contact_x_m",
            "ground_contact_y_m",
        ]
    ].copy()

    df = df.rename(
        columns={
            "ground_contact_x_m": "ground_x_m",
            "ground_contact_y_m": "ground_y_m",
        }
    )

    df = df.dropna()

    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)

    df = df.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    print()
    print(f"Valid observations: {len(df)}")

    for track_id in sorted(df["track_id"].unique()):

        track = df[df["track_id"] == track_id]

        print(
            f"Track {track_id}: "
            f"{len(track)} observations, "
            f"frames {track['frame'].min()} -> "
            f"{track['frame'].max()}"
        )

    return df


# ============================================================
# SCALE TRAJECTORY
# ============================================================

def scale_trajectory(df, tile_size_m):
    """
    Convert coordinates generated with the 0.60 m reference
    tile into coordinates corresponding to the requested tile size.
    """

    scale_factor = (
        tile_size_m / REFERENCE_TILE_SIZE_M
    )

    scaled = df.copy()

    scaled["ground_x_m"] = (
        scaled["ground_x_m"]
        * scale_factor
    )

    scaled["ground_y_m"] = (
        scaled["ground_y_m"]
        * scale_factor
    )

    return scaled


# ============================================================
# CALCULATE ALL RESULTS
# ============================================================

def calculate_sensitivity(df):

    print_header("SCALE ASSUMPTIONS")

    print(
        f"Reference calibration tile: "
        f"{REFERENCE_TILE_SIZE_M:.2f} m"
    )

    print()
    print("Testing tile sizes:")

    for size in TEST_TILE_SIZES_M:
        factor = size / REFERENCE_TILE_SIZE_M

        print(
            f"  {size:.2f} m "
            f"(scale factor {factor:.6f})"
        )

    results = []

    for tile_size in TEST_TILE_SIZES_M:

        print()
        print(
            f"Calculating for tile size "
            f"{tile_size:.2f} m..."
        )

        scaled_df = scale_trajectory(
            df,
            tile_size,
        )

        for track_id in sorted(
            scaled_df["track_id"].unique()
        ):

            track = scaled_df[
                scaled_df["track_id"] == track_id
            ]

            metrics = calculate_track_metrics(
                track
            )

            metrics["tile_size_m"] = tile_size
            metrics["track_id"] = track_id

            results.append(metrics)

    results_df = pd.DataFrame(results)

    # Put columns in a sensible order.
    results_df = results_df[
        [
            "tile_size_m",
            "track_id",
            "observations",
            "duration_s",
            "path_length_m",
            "net_displacement_m",
            "mean_speed_mps",
            "median_speed_mps",
            "max_speed_mps",
            "mean_frame_movement_m",
            "max_frame_movement_m",
        ]
    ]

    return results_df


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results_df):

    print_header("SENSITIVITY RESULTS")

    for track_id in sorted(
        results_df["track_id"].unique()
    ):

        print()
        print(f"TRACK {track_id}")
        print("-" * 70)

        track_results = results_df[
            results_df["track_id"] == track_id
        ]

        print(
            f"{'Tile':>8} "
            f"{'Path':>10} "
            f"{'Net disp.':>12} "
            f"{'Median v':>12} "
            f"{'Max v':>10}"
        )

        print(
            f"{'(m)':>8} "
            f"{'(m)':>10} "
            f"{'(m)':>12} "
            f"{'(m/s)':>12} "
            f"{'(m/s)':>10}"
        )

        for _, row in track_results.iterrows():

            print(
                f"{row['tile_size_m']:8.2f} "
                f"{row['path_length_m']:10.3f} "
                f"{row['net_displacement_m']:12.3f} "
                f"{row['median_speed_mps']:12.3f} "
                f"{row['max_speed_mps']:10.3f}"
            )


# ============================================================
# CREATE SUMMARY CSV
# ============================================================

def save_results(results_df):

    output_csv = (
        OUTPUT_DIR
        / "metric_scale_sensitivity.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("Saved:")
    print(output_csv)

    return output_csv


# ============================================================
# PLOT 1 - PATH LENGTH
# ============================================================

def plot_path_length(results_df):

    plt.figure(figsize=(10, 6))

    for track_id in sorted(
        results_df["track_id"].unique()
    ):

        track = results_df[
            results_df["track_id"] == track_id
        ]

        plt.plot(
            track["tile_size_m"],
            track["path_length_m"],
            marker="o",
            label=f"Track {track_id}",
        )

    plt.xlabel("Assumed tile size (m)")
    plt.ylabel("Ground path length (m)")
    plt.title("Metric Scale Sensitivity - Path Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "path_length_vs_tile_size.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# PLOT 2 - NET DISPLACEMENT
# ============================================================

def plot_net_displacement(results_df):

    plt.figure(figsize=(10, 6))

    for track_id in sorted(
        results_df["track_id"].unique()
    ):

        track = results_df[
            results_df["track_id"] == track_id
        ]

        plt.plot(
            track["tile_size_m"],
            track["net_displacement_m"],
            marker="o",
            label=f"Track {track_id}",
        )

    plt.xlabel("Assumed tile size (m)")
    plt.ylabel("Net displacement (m)")
    plt.title(
        "Metric Scale Sensitivity - Net Displacement"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "net_displacement_vs_tile_size.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# PLOT 3 - MEDIAN SPEED
# ============================================================

def plot_median_speed(results_df):

    plt.figure(figsize=(10, 6))

    for track_id in sorted(
        results_df["track_id"].unique()
    ):

        track = results_df[
            results_df["track_id"] == track_id
        ]

        plt.plot(
            track["tile_size_m"],
            track["median_speed_mps"],
            marker="o",
            label=f"Track {track_id}",
        )

    plt.xlabel("Assumed tile size (m)")
    plt.ylabel("Median ground speed (m/s)")
    plt.title(
        "Metric Scale Sensitivity - Median Speed"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "median_speed_vs_tile_size.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# PLOT 4 - MAXIMUM SPEED
# ============================================================

def plot_max_speed(results_df):

    plt.figure(figsize=(10, 6))

    for track_id in sorted(
        results_df["track_id"].unique()
    ):

        track = results_df[
            results_df["track_id"] == track_id
        ]

        plt.plot(
            track["tile_size_m"],
            track["max_speed_mps"],
            marker="o",
            label=f"Track {track_id}",
        )

    plt.xlabel("Assumed tile size (m)")
    plt.ylabel("Maximum ground speed (m/s)")
    plt.title(
        "Metric Scale Sensitivity - Maximum Speed"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / "max_speed_vs_tile_size.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# PLOT 5 - ALL METRICS
# ============================================================

def plot_normalized_sensitivity(results_df):

    """
    Show how each quantity changes relative to the
    0.60 m reference case.

    This makes the effect of the scale assumption
    especially easy to explain.
    """

    metrics = [
        (
            "path_length_m",
            "Path length",
        ),
        (
            "net_displacement_m",
            "Net displacement",
        ),
        (
            "median_speed_mps",
            "Median speed",
        ),
        (
            "max_speed_mps",
            "Maximum speed",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    axes = axes.ravel()

    for ax, (column, title) in zip(
        axes,
        metrics,
    ):

        for track_id in sorted(
            results_df["track_id"].unique()
        ):

            track = results_df[
                results_df["track_id"] == track_id
            ].copy()

            reference = track[
                track["tile_size_m"]
                == REFERENCE_TILE_SIZE_M
            ][column].iloc[0]

            normalized = (
                track[column]
                / reference
            )

            ax.plot(
                track["tile_size_m"],
                normalized,
                marker="o",
                label=f"Track {track_id}",
            )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_title(
            f"{title} relative to 0.60 m case"
        )

        ax.set_xlabel(
            "Assumed tile size (m)"
        )

        ax.set_ylabel(
            "Relative value"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

    fig.suptitle(
        "Metric Scale Sensitivity - Relative Change",
        fontsize=15,
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / "normalized_scale_sensitivity.png"
    )

    fig.savefig(
        output,
        dpi=150,
    )

    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_input()

    results_df = calculate_sensitivity(
        df
    )

    print_results(
        results_df
    )

    save_results(
        results_df
    )

    print()
    print_header("CREATING DIAGNOSTIC PLOTS")

    plot_path_length(
        results_df
    )

    plot_net_displacement(
        results_df
    )

    plot_median_speed(
        results_df
    )

    plot_max_speed(
        results_df
    )

    plot_normalized_sensitivity(
        results_df
    )

    print()
    print("=" * 70)
    print("STEP 24 COMPLETE")
    print("=" * 70)

    print()
    print(
        "The effect of the assumed floor-tile size "
        "has been quantified."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "All metric distances and speeds scale "
        "approximately linearly with the assumed "
        "tile dimension."
    )

    print(
        "The 0.60 m case remains the working reference "
        "unless the actual tile dimension is measured."
    )


if __name__ == "__main__":
    main()