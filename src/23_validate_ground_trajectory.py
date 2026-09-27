"""
STEP 23 - GROUND TRAJECTORY / PHYSICS VALIDATION

Validates the football trajectory after projecting the estimated
ground-contact point onto the calibrated floor plane.

Inputs:
    data/output/top_view/top_view_contact_trajectory.csv

Outputs:
    data/output/validation/
        ground_trajectory_validation.csv
        ground_speed.png
        frame_to_frame_displacement.png
        cumulative_path_length.png
        ground_trajectory_validation_summary.png

The metric scale depends on the assumed 0.60 m x 0.60 m calibration tile.
No interpolation is performed across the real visibility gap (frames 206-388).
"""

from pathlib import Path
import json
import math

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
    / "top_view"
    / "top_view_contact_trajectory.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "validation"
)

FPS = 30.0

# The floor calibration currently assumes a 0.60 m x 0.60 m tile.
ASSUMED_TILE_SIZE_M = 0.60

# Frame-to-frame movement above this value is flagged as suspicious.
# This is deliberately a diagnostic threshold, not an automatic rejection.
SUSPICIOUS_DISPLACEMENT_M = 0.50

# Ground-speed threshold used only for diagnostic flagging.
# It is NOT a claim that the football cannot physically exceed this value.
SUSPICIOUS_SPEED_MPS = 15.0


# ======================================================================
# HELPERS
# ======================================================================

def find_column(df, candidates, required=True):
    """Find the first matching column from a list of possible names."""
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    # Also allow compact matching, e.g. ground_x_m vs ground_x.
    compact = {
        "".join(ch for ch in str(c).lower() if ch.isalnum()): c
        for c in df.columns
    }

    for candidate in candidates:
        key = "".join(ch for ch in candidate.lower() if ch.isalnum())
        if key in compact:
            return compact[key]

    if required:
        raise ValueError(
            f"Could not identify required column. Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_input(path):
    print("=" * 70)
    print("INPUT")
    print("=" * 70)
    print(f"CSV: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist:\n{path}")

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print("\nColumns:")
    print(list(df.columns))

    track_col = find_column(
        df,
        ["track_id", "track", "trajectory_id"]
    )

    frame_col = find_column(
        df,
        ["frame", "frame_id", "frame_number"]
    )

    time_col = find_column(
        df,
        ["time_seconds", "time_sec", "time", "timestamp"]
    )

    x_col = find_column(
        df,
        [
            "ground_contact_x_m",
            "contact_ground_x_m",
            "contact_x_m",
            "ground_contact_x",
            "contact_x",
            "ground_x_m",
            "ground_x",
            "x_ground_m",
        ]
    )

    y_col = find_column(
        df,
        [
            "ground_contact_y_m",
            "contact_ground_y_m",
            "contact_y_m",
            "ground_contact_y",
            "contact_y",
            "ground_y_m",
            "ground_y",
            "y_ground_m",
        ]
    )

    print("\nUsing trajectory columns:")
    print(f"  Track ID:    {track_col}")
    print(f"  Frame:       {frame_col}")
    print(f"  Time:        {time_col}")
    print(f"  Ground X:    {x_col}")
    print(f"  Ground Y:    {y_col}")

    out = pd.DataFrame({
        "track_id": pd.to_numeric(df[track_col], errors="coerce"),
        "frame": pd.to_numeric(df[frame_col], errors="coerce"),
        "time_seconds": pd.to_numeric(df[time_col], errors="coerce"),
        "ground_x_m": pd.to_numeric(df[x_col], errors="coerce"),
        "ground_y_m": pd.to_numeric(df[y_col], errors="coerce"),
    })

    out = out.dropna(
        subset=[
            "track_id",
            "frame",
            "time_seconds",
            "ground_x_m",
            "ground_y_m",
        ]
    ).copy()

    out["track_id"] = out["track_id"].astype(int)
    out["frame"] = out["frame"].astype(int)

    out = out.sort_values(
        ["track_id", "frame"]
    ).reset_index(drop=True)

    print("\nNormalized columns:")
    print(list(out.columns))

    print(f"\nValid observations: {len(out)}")

    for track_id, g in out.groupby("track_id"):
        print(
            f"Track {track_id}: "
            f"{len(g)} observations, "
            f"frames {g['frame'].min()} -> {g['frame'].max()}"
        )

    return out


# ======================================================================
# TRAJECTORY METRICS
# ======================================================================

def calculate_track_metrics(group):
    """
    Calculate frame-to-frame ground motion without crossing gaps.

    A pair of consecutive rows is considered continuous only when:
        frame difference == 1

    Therefore the real visibility gap between Track 0 and Track 1
    is never used in a velocity or distance calculation.
    """

    g = group.sort_values("frame").copy()

    x = g["ground_x_m"].to_numpy(dtype=float)
    y = g["ground_y_m"].to_numpy(dtype=float)
    t = g["time_seconds"].to_numpy(dtype=float)
    frames = g["frame"].to_numpy(dtype=int)

    n = len(g)

    dt = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    displacement = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    continuous = np.zeros(n, dtype=bool)

    for i in range(1, n):
        frame_delta = frames[i] - frames[i - 1]

        # Require consecutive video frames.
        if frame_delta != 1:
            continue

        delta_t = t[i] - t[i - 1]

        if not np.isfinite(delta_t) or delta_t <= 0:
            # Fall back to nominal FPS only for consecutive frames
            delta_t = 1.0 / FPS

        dt[i] = delta_t
        dx[i] = x[i] - x[i - 1]
        dy[i] = y[i] - y[i - 1]

        displacement[i] = math.hypot(dx[i], dy[i])
        speed[i] = displacement[i] / delta_t

        continuous[i] = True

    g["dt_s"] = dt
    g["dx_m"] = dx
    g["dy_m"] = dy
    g["frame_to_frame_displacement_m"] = displacement
    g["ground_speed_mps"] = speed
    g["continuous_pair"] = continuous

    # Diagnostic flags.
    g["flag_large_displacement"] = (
        g["frame_to_frame_displacement_m"]
        > SUSPICIOUS_DISPLACEMENT_M
    )

    g["flag_high_speed"] = (
        g["ground_speed_mps"]
        > SUSPICIOUS_SPEED_MPS
    )

    # Cumulative path length only over valid consecutive frame pairs.
    cumulative = np.zeros(n, dtype=float)

    for i in range(1, n):
        if np.isfinite(displacement[i]):
            cumulative[i] = (
                cumulative[i - 1] + displacement[i]
            )
        else:
            # At a visibility gap, retain the previous cumulative
            # value rather than inventing distance across the gap.
            cumulative[i] = cumulative[i - 1]

    g["cumulative_path_length_m"] = cumulative

    return g


def summarize_track(g):
    valid_speed = g["ground_speed_mps"].dropna()
    valid_disp = g["frame_to_frame_displacement_m"].dropna()

    start_x = float(g["ground_x_m"].iloc[0])
    start_y = float(g["ground_y_m"].iloc[0])

    end_x = float(g["ground_x_m"].iloc[-1])
    end_y = float(g["ground_y_m"].iloc[-1])

    displacement_total = math.hypot(
        end_x - start_x,
        end_y - start_y
    )

    path_length = float(
        g["cumulative_path_length_m"].iloc[-1]
    )

    duration = (
        float(g["time_seconds"].iloc[-1])
        - float(g["time_seconds"].iloc[0])
    )

    summary = {
        "track_id": int(g["track_id"].iloc[0]),
        "start_frame": int(g["frame"].iloc[0]),
        "end_frame": int(g["frame"].iloc[-1]),
        "observations": int(len(g)),
        "duration_s": duration,
        "start_x_m": start_x,
        "start_y_m": start_y,
        "end_x_m": end_x,
        "end_y_m": end_y,
        "net_displacement_m": displacement_total,
        "path_length_m": path_length,
        "mean_ground_speed_mps": (
            float(valid_speed.mean())
            if len(valid_speed)
            else np.nan
        ),
        "median_ground_speed_mps": (
            float(valid_speed.median())
            if len(valid_speed)
            else np.nan
        ),
        "maximum_ground_speed_mps": (
            float(valid_speed.max())
            if len(valid_speed)
            else np.nan
        ),
        "mean_frame_displacement_m": (
            float(valid_disp.mean())
            if len(valid_disp)
            else np.nan
        ),
        "maximum_frame_displacement_m": (
            float(valid_disp.max())
            if len(valid_disp)
            else np.nan
        ),
        "flagged_large_displacements": int(
            g["flag_large_displacement"].sum()
        ),
        "flagged_high_speeds": int(
            g["flag_high_speed"].sum()
        ),
        "continuous_pairs": int(
            g["continuous_pair"].sum()
        ),
    }

    return summary


# ======================================================================
# PLOTS
# ======================================================================

def plot_ground_speed(df, summaries, path):
    fig, ax = plt.subplots(figsize=(13, 7))

    for track_id, g in df.groupby("track_id"):
        ax.plot(
            g["time_seconds"],
            g["ground_speed_mps"],
            label=f"Track {track_id}",
            linewidth=1.5,
        )

        flagged = g[g["flag_high_speed"]]
        if not flagged.empty:
            ax.scatter(
                flagged["time_seconds"],
                flagged["ground_speed_mps"],
                marker="x",
                s=60,
                label=f"Track {track_id} flagged",
            )

    ax.set_title("Estimated Ground Speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ground speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_frame_displacement(df, path):
    fig, ax = plt.subplots(figsize=(13, 7))

    for track_id, g in df.groupby("track_id"):
        ax.plot(
            g["frame"],
            g["frame_to_frame_displacement_m"],
            label=f"Track {track_id}",
            linewidth=1.5,
        )

        flagged = g[g["flag_large_displacement"]]
        if not flagged.empty:
            ax.scatter(
                flagged["frame"],
                flagged["frame_to_frame_displacement_m"],
                marker="x",
                s=60,
                label=f"Track {track_id} flagged",
            )

    ax.axhline(
        SUSPICIOUS_DISPLACEMENT_M,
        linestyle="--",
        linewidth=1.2,
        label=f"Diagnostic threshold ({SUSPICIOUS_DISPLACEMENT_M:.2f} m)",
    )

    ax.set_title("Frame-to-Frame Ground Displacement")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Displacement (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cumulative_path(df, path):
    fig, ax = plt.subplots(figsize=(13, 7))

    for track_id, g in df.groupby("track_id"):
        ax.plot(
            g["time_seconds"],
            g["cumulative_path_length_m"],
            label=f"Track {track_id}",
            linewidth=2,
        )

    ax.set_title("Cumulative Ground Path Length")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative path length (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary(summaries, path):
    track_ids = [str(s["track_id"]) for s in summaries]

    path_lengths = [
        s["path_length_m"] for s in summaries
    ]

    net_displacements = [
        s["net_displacement_m"] for s in summaries
    ]

    median_speeds = [
        s["median_ground_speed_mps"] for s in summaries
    ]

    max_speeds = [
        s["maximum_ground_speed_mps"] for s in summaries
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    axes[0, 0].bar(track_ids, path_lengths)
    axes[0, 0].set_title("Path Length")
    axes[0, 0].set_ylabel("m")

    axes[0, 1].bar(track_ids, net_displacements)
    axes[0, 1].set_title("Net Displacement")
    axes[0, 1].set_ylabel("m")

    axes[1, 0].bar(track_ids, median_speeds)
    axes[1, 0].set_title("Median Ground Speed")
    axes[1, 0].set_ylabel("m/s")

    axes[1, 1].bar(track_ids, max_speeds)
    axes[1, 1].set_title("Maximum Ground Speed")
    axes[1, 1].set_ylabel("m/s")

    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Ground Trajectory Validation Summary",
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("STEP 23 - GROUND TRAJECTORY / PHYSICS VALIDATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input(INPUT_CSV)

    print("\n" + "=" * 70)
    print("CALCULATING FRAME-TO-FRAME GROUND MOTION")
    print("=" * 70)

    processed = []

    for track_id, group in df.groupby("track_id"):
        print(f"\nProcessing Track {track_id}...")
        processed.append(calculate_track_metrics(group))

    df = pd.concat(processed, ignore_index=True)

    summaries = []

    for track_id, group in df.groupby("track_id"):
        summaries.append(summarize_track(group))

    # Save detailed validation data.
    output_csv = OUTPUT_DIR / "ground_trajectory_validation.csv"
    df.to_csv(output_csv, index=False)

    # Save numerical summary.
    summary_csv = OUTPUT_DIR / "ground_trajectory_summary.csv"
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_csv, index=False)

    # Save JSON summary for easy programmatic use.
    summary_json = OUTPUT_DIR / "ground_trajectory_summary.json"

    json_data = {
        "assumed_tile_size_m": ASSUMED_TILE_SIZE_M,
        "fps": FPS,
        "suspicious_displacement_threshold_m": (
            SUSPICIOUS_DISPLACEMENT_M
        ),
        "suspicious_speed_threshold_mps": (
            SUSPICIOUS_SPEED_MPS
        ),
        "tracks": summaries,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    total_flagged_displacement = 0
    total_flagged_speed = 0

    for s in summaries:
        print(f"\nTrack {s['track_id']}")
        print(
            f"  Frames:              "
            f"{s['start_frame']} -> {s['end_frame']}"
        )
        print(f"  Observations:        {s['observations']}")
        print(f"  Duration:            {s['duration_s']:.3f} s")
        print(
            f"  Path length:         "
            f"{s['path_length_m']:.3f} m"
        )
        print(
            f"  Net displacement:    "
            f"{s['net_displacement_m']:.3f} m"
        )
        print(
            f"  Mean ground speed:   "
            f"{s['mean_ground_speed_mps']:.3f} m/s"
        )
        print(
            f"  Median ground speed: "
            f"{s['median_ground_speed_mps']:.3f} m/s"
        )
        print(
            f"  Maximum ground speed:"
            f" {s['maximum_ground_speed_mps']:.3f} m/s"
        )
        print(
            f"  Mean frame movement: "
            f"{s['mean_frame_displacement_m']:.4f} m"
        )
        print(
            f"  Maximum frame movement:"
            f" {s['maximum_frame_displacement_m']:.4f} m"
        )
        print(
            f"  Large displacement flags:"
            f" {s['flagged_large_displacements']}"
        )
        print(
            f"  High speed flags:    "
            f"{s['flagged_high_speeds']}"
        )

        total_flagged_displacement += (
            s["flagged_large_displacements"]
        )
        total_flagged_speed += s["flagged_high_speeds"]

    print("\n" + "=" * 70)
    print("VISIBILITY-GAP CHECK")
    print("=" * 70)

    for track_id, group in df.groupby("track_id"):
        frames = group["frame"].to_numpy()

        if len(frames) > 1:
            gaps = np.diff(frames)
            large_gaps = gaps[gaps > 1]

            if len(large_gaps):
                print(
                    f"Track {track_id}: "
                    f"{len(large_gaps)} internal frame gap(s): "
                    f"{large_gaps.tolist()}"
                )
            else:
                print(
                    f"Track {track_id}: "
                    f"continuous within visible segment."
                )

    print("\n" + "=" * 70)
    print("CREATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    speed_plot = OUTPUT_DIR / "ground_speed.png"
    displacement_plot = OUTPUT_DIR / "frame_to_frame_displacement.png"
    path_plot = OUTPUT_DIR / "cumulative_path_length.png"
    summary_plot = OUTPUT_DIR / "ground_trajectory_validation_summary.png"

    plot_ground_speed(df, summaries, speed_plot)
    plot_frame_displacement(df, displacement_plot)
    plot_cumulative_path(df, path_plot)
    plot_summary(summaries, summary_plot)

    print(f"Saved: {speed_plot}")
    print(f"Saved: {displacement_plot}")
    print(f"Saved: {path_plot}")
    print(f"Saved: {summary_plot}")

    print("\n" + "=" * 70)
    print("PHYSICAL / TRACKING SANITY CHECK")
    print("=" * 70)

    if total_flagged_displacement == 0:
        print(
            "OK: No frame-to-frame ground displacement exceeded "
            f"{SUSPICIOUS_DISPLACEMENT_M:.2f} m."
        )
    else:
        print(
            "ATTENTION: "
            f"{total_flagged_displacement} frame transition(s) "
            "exceeded the displacement diagnostic threshold."
        )

    if total_flagged_speed == 0:
        print(
            "OK: No ground-speed sample exceeded "
            f"{SUSPICIOUS_SPEED_MPS:.1f} m/s."
        )
    else:
        print(
            "ATTENTION: "
            f"{total_flagged_speed} ground-speed sample(s) "
            "exceeded the diagnostic threshold."
        )

    print(
        "\nIMPORTANT: these thresholds are diagnostic flags only. "
        "They do not prove that a measurement is wrong."
    )

    print("\n" + "=" * 70)
    print("STEP 23 COMPLETE")
    print("=" * 70)

    print("\nThe ground-contact trajectory has been validated numerically.")
    print(
        "No motion is calculated across the genuine visibility gap "
        "(frames 206-388)."
    )
    print(
        "Metric values remain dependent on the assumed 0.60 m tile size."
    )


if __name__ == "__main__":
    main()
