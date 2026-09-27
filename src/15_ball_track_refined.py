import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_CSV = (
    PROJECT_DIR
    / "data"
    / "output"
    / "all_ball_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "output"
)

PLOT_DIR = (
    OUTPUT_DIR
    / "tracking_plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "refined_ball_track.csv"
)


# ============================================================
# VIDEO PARAMETERS
# ============================================================

FPS = 30.0


# ============================================================
# IMAGE ROI
# ============================================================
#
# The football has consistently appeared in approximately
# this part of the image.
#
# We deliberately use a BROAD ROI.
#
# This is not meant to "draw the answer".
# It simply rejects obvious red objects far away from the
# football's working area.
#
# Image size:
#     1280 x 720
#
# Coordinates:
#     x -> left/right
#     y -> top/bottom
# ============================================================

ROI_X_MIN = 500
ROI_X_MAX = 1050

ROI_Y_MIN = 80
ROI_Y_MAX = 700


# ============================================================
# FOOTBALL ANCHORS
# ============================================================
#
# These are approximate observations from our earlier
# analysis of the video.
#
# They are NOT ground-truth measurements.
#
# They simply tell the tracker:
#
# "This is approximately where the football is known to be."
#
# We have two visible portions of the trajectory.
#
# First visible portion:
#     around frame 140
#
# Second visible portion:
#     around frame 395
#
# The tracker grows each trajectory from these anchors.
# ============================================================

ANCHORS = [
    {
        "name": "first_ball_segment",
        "frame": 140,
        "x": 690.0,
        "y": 220.0,
    },
    {
        "name": "second_ball_segment",
        "frame": 395,
        "x": 570.0,
        "y": 525.0,
    },
]


# How many frames around the anchor we search.
ANCHOR_FRAME_WINDOW = 8

# Maximum spatial distance from the expected anchor.
ANCHOR_MAX_DISTANCE = 100.0


# ============================================================
# TRACKING PARAMETERS
# ============================================================

# Maximum number of frames that can be skipped.
#
# We don't want to bridge long periods of complete ambiguity.
MAX_GAP = 4


# Approximate maximum ball movement in pixels per frame.
#
# This is deliberately generous.
MAX_SPEED_PX_PER_FRAME = 80.0


# Extra tolerance added to the motion gate.
MOTION_GATE_MARGIN = 35.0


# Maximum allowed candidate-to-prediction distance.
MAX_ASSOCIATION_DISTANCE = 120.0


# Maximum relative size change between nearby observations.
#
# We use logarithmic size difference, so this is not a
# percentage directly.
MAX_SIZE_LOG_CHANGE = 0.75


# Cost weights.
DISTANCE_WEIGHT = 1.0
SIZE_WEIGHT = 25.0
QUALITY_WEIGHT = 8.0
SHAPE_WEIGHT = 4.0


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class Candidate:

    frame: int
    time_seconds: float

    x: float
    y: float

    width: float
    height: float

    area: float
    circularity: float
    aspect_ratio: float

    diameter: float
    quality: float


# ============================================================
# LOAD CANDIDATES
# ============================================================

def load_candidates():

    if not CANDIDATE_CSV.exists():

        raise FileNotFoundError(
            f"Candidate file not found:\n"
            f"{CANDIDATE_CSV}\n\n"
            f"Run 14_global_ball_track.py first."
        )

    df = pd.read_csv(
        CANDIDATE_CSV
    )

    required_columns = [
        "frame",
        "time_seconds",
        "x",
        "y",
        "width",
        "height",
        "area",
        "circularity",
        "aspect_ratio",
        "diameter",
        "quality",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Candidate CSV is missing columns:\n"
            + "\n".join(missing)
        )

    candidates_by_frame = {}

    for row in df.itertuples(
        index=False
    ):

        candidate = Candidate(
            frame=int(row.frame),
            time_seconds=float(
                row.time_seconds
            ),

            x=float(row.x),
            y=float(row.y),

            width=float(row.width),
            height=float(row.height),

            area=float(row.area),
            circularity=float(
                row.circularity
            ),

            aspect_ratio=float(
                row.aspect_ratio
            ),

            diameter=float(
                row.diameter
            ),

            quality=float(
                row.quality
            ),
        )

        # ----------------------------------------------------
        # Broad spatial filtering.
        # ----------------------------------------------------

        if not (
            ROI_X_MIN
            <= candidate.x
            <= ROI_X_MAX
        ):
            continue

        if not (
            ROI_Y_MIN
            <= candidate.y
            <= ROI_Y_MAX
        ):
            continue

        candidates_by_frame.setdefault(
            candidate.frame,
            [],
        ).append(candidate)

    return candidates_by_frame


# ============================================================
# DISTANCE
# ============================================================

def distance(
    x1,
    y1,
    x2,
    y2,
):

    dx = x2 - x1
    dy = y2 - y1

    return float(
        np.sqrt(
            dx * dx
            + dy * dy
        )
    )


# ============================================================
# FIND ANCHOR
# ============================================================

def find_anchor_candidate(
    candidates_by_frame,
    anchor,
):

    expected_frame = anchor["frame"]
    expected_x = anchor["x"]
    expected_y = anchor["y"]

    possible = []

    for frame in range(
        expected_frame
        - ANCHOR_FRAME_WINDOW,

        expected_frame
        + ANCHOR_FRAME_WINDOW
        + 1,
    ):

        if frame not in candidates_by_frame:
            continue

        frame_difference = abs(
            frame - expected_frame
        )

        for candidate in candidates_by_frame[frame]:

            spatial_distance = distance(
                candidate.x,
                candidate.y,
                expected_x,
                expected_y,
            )

            if (
                spatial_distance
                > ANCHOR_MAX_DISTANCE
            ):
                continue

            # Anchor score.
            #
            # Lower = better.
            score = (
                spatial_distance
                + 10.0
                * frame_difference
                - 8.0
                * candidate.quality
            )

            possible.append(
                (
                    score,
                    candidate,
                )
            )

    if not possible:

        return None

    possible.sort(
        key=lambda item: item[0]
    )

    return possible[0][1]


# ============================================================
# PREDICT NEXT POSITION
# ============================================================

def predict_position(
    previous,
    current,
    gap,
):

    # If we only have one observation, assume zero velocity.
    if previous is None:

        return (
            current.x,
            current.y,
        )

    vx = (
        current.x
        - previous.x
    )

    vy = (
        current.y
        - previous.y
    )

    predicted_x = (
        current.x
        + vx * gap
    )

    predicted_y = (
        current.y
        + vy * gap
    )

    return (
        predicted_x,
        predicted_y,
    )


# ============================================================
# CANDIDATE COST
# ============================================================

def candidate_cost(
    current,
    candidate,
    predicted_x,
    predicted_y,
    gap,
):

    spatial_distance = distance(
        predicted_x,
        predicted_y,
        candidate.x,
        candidate.y,
    )

    # --------------------------------------------------------
    # Motion gate.
    # --------------------------------------------------------

    max_distance = (
        MAX_SPEED_PX_PER_FRAME
        * gap
        + MOTION_GATE_MARGIN
    )

    if (
        spatial_distance
        > max(
            MAX_ASSOCIATION_DISTANCE,
            max_distance,
        )
    ):

        return np.inf

    # --------------------------------------------------------
    # Size consistency.
    # --------------------------------------------------------

    current_diameter = max(
        current.diameter,
        1.0,
    )

    candidate_diameter = max(
        candidate.diameter,
        1.0,
    )

    size_log_change = abs(
        np.log(
            candidate_diameter
            / current_diameter
        )
    )

    if (
        size_log_change
        > MAX_SIZE_LOG_CHANGE
    ):

        return np.inf

    # --------------------------------------------------------
    # Shape consistency.
    # --------------------------------------------------------

    shape_change = abs(
        candidate.circularity
        - current.circularity
    )

    # --------------------------------------------------------
    # Candidate quality.
    #
    # Higher quality should mean lower cost.
    # --------------------------------------------------------

    quality_bonus = (
        QUALITY_WEIGHT
        * candidate.quality
    )

    # --------------------------------------------------------
    # Final cost.
    # --------------------------------------------------------

    cost = (

        DISTANCE_WEIGHT
        * spatial_distance

        + SIZE_WEIGHT
        * size_log_change

        + SHAPE_WEIGHT
        * shape_change

        - quality_bonus
    )

    return float(cost)


# ============================================================
# TRACK FORWARD
# ============================================================

def track_forward(
    candidates_by_frame,
    seed,
):

    track = [
        seed
    ]

    previous = None
    current = seed

    last_frame = seed.frame

    gap_count = 0

    # We track until the end of the video.
    max_frame = max(
        candidates_by_frame.keys()
    )

    for frame in range(
        seed.frame + 1,
        max_frame + 1,
    ):

        frame_gap = (
            frame
            - last_frame
        )

        if frame_gap <= 0:
            continue

        # ----------------------------------------------------
        # If the gap becomes too large, stop this segment.
        # ----------------------------------------------------

        if frame_gap > MAX_GAP:

            break

        candidates = (
            candidates_by_frame.get(
                frame,
                []
            )
        )

        if not candidates:

            gap_count += 1

            if gap_count >= MAX_GAP:

                break

            continue

        predicted_x, predicted_y = (
            predict_position(
                previous,
                current,
                frame_gap,
            )
        )

        best_candidate = None
        best_cost = np.inf

        for candidate in candidates:

            cost = candidate_cost(
                current,
                candidate,
                predicted_x,
                predicted_y,
                frame_gap,
            )

            if cost < best_cost:

                best_cost = cost
                best_candidate = candidate

        if best_candidate is None:

            gap_count += 1

            if gap_count >= MAX_GAP:

                break

            continue

        # ----------------------------------------------------
        # Successful association.
        # ----------------------------------------------------

        previous = current

        current = best_candidate

        last_frame = (
            best_candidate.frame
        )

        track.append(
            best_candidate
        )

        gap_count = 0

    return track


# ============================================================
# TRACK BACKWARD
# ============================================================

def track_backward(
    candidates_by_frame,
    seed,
):

    track = [
        seed
    ]

    previous = None
    current = seed

    first_frame = seed.frame

    gap_count = 0

    min_frame = min(
        candidates_by_frame.keys()
    )

    for frame in range(
        seed.frame - 1,
        min_frame - 1,
        -1,
    ):

        frame_gap = (
            current.frame
            - frame
        )

        if frame_gap <= 0:
            continue

        if frame_gap > MAX_GAP:

            break

        candidates = (
            candidates_by_frame.get(
                frame,
                []
            )
        )

        if not candidates:

            gap_count += 1

            if gap_count >= MAX_GAP:

                break

            continue

        # ----------------------------------------------------
        # For backward tracking we estimate the previous
        # position using the current velocity.
        # ----------------------------------------------------

        if previous is None:

            predicted_x = current.x
            predicted_y = current.y

        else:

            vx = (
                current.x
                - previous.x
            )

            vy = (
                current.y
                - previous.y
            )

            predicted_x = (
                current.x
                + vx * frame_gap
            )

            predicted_y = (
                current.y
                + vy * frame_gap
            )

        best_candidate = None
        best_cost = np.inf

        for candidate in candidates:

            cost = candidate_cost(
                current,
                candidate,
                predicted_x,
                predicted_y,
                frame_gap,
            )

            if cost < best_cost:

                best_cost = cost
                best_candidate = candidate

        if best_candidate is None:

            gap_count += 1

            if gap_count >= MAX_GAP:

                break

            continue

        previous = current
        current = best_candidate

        track.append(
            best_candidate
        )

        gap_count = 0

    return track


# ============================================================
# BUILD TRACK FROM ANCHOR
# ============================================================

def build_track(
    candidates_by_frame,
    seed,
):

    forward = track_forward(
        candidates_by_frame,
        seed,
    )

    backward = track_backward(
        candidates_by_frame,
        seed,
    )

    # backward currently goes:
    #
    # seed
    # previous
    # previous
    # ...
    #
    # Reverse it and remove duplicate seed.

    backward = list(
        reversed(
            backward
        )
    )

    if backward:

        backward = backward[:-1]

    combined = (
        backward
        + forward
    )

    # Remove duplicate frame/candidate combinations.
    unique = {}

    for candidate in combined:

        unique[candidate.frame] = (
            candidate
        )

    result = list(
        unique.values()
    )

    result.sort(
        key=lambda c: c.frame
    )

    return result


# ============================================================
# CONVERT TRACK TO DATAFRAME
# ============================================================

def track_to_dataframe(
    track,
    track_id,
):

    rows = []

    for candidate in track:

        rows.append(
            {
                "track_id": track_id,

                "frame": candidate.frame,

                "time_seconds": (
                    candidate.time_seconds
                ),

                "detected": 1,

                "measurement_x": (
                    candidate.x
                ),

                "measurement_y": (
                    candidate.y
                ),

                "diameter": (
                    candidate.diameter
                ),

                "area": (
                    candidate.area
                ),

                "circularity": (
                    candidate.circularity
                ),

                "aspect_ratio": (
                    candidate.aspect_ratio
                ),

                "quality": (
                    candidate.quality
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ADD VELOCITY
# ============================================================

def add_velocity(
    df,
):

    if df.empty:

        return df

    df = df.copy()

    df["vx_pixels_per_second"] = np.nan
    df["vy_pixels_per_second"] = np.nan

    for track_id in df[
        "track_id"
    ].unique():

        mask = (
            df["track_id"]
            == track_id
        )

        group = (
            df.loc[mask]
            .sort_values("frame")
        )

        frames = (
            group["frame"]
            .to_numpy()
            .astype(float)
        )

        x = (
            group["measurement_x"]
            .to_numpy()
        )

        y = (
            group["measurement_y"]
            .to_numpy()
        )

        if len(group) < 2:

            continue

        dt = np.gradient(
            frames / FPS
        )

        vx = np.gradient(
            x
        ) / dt

        vy = np.gradient(
            y
        ) / dt

        df.loc[
            group.index,
            "vx_pixels_per_second"
        ] = vx

        df.loc[
            group.index,
            "vy_pixels_per_second"
        ] = vy

    return df


# ============================================================
# PLOT
# ============================================================

def create_plot(
    df,
    column,
    ylabel,
    title,
    filename,
):

    plt.figure(
        figsize=(15, 7)
    )

    for track_id, group in (
        df.groupby("track_id")
    ):

        group = group.sort_values(
            "frame"
        )

        plt.plot(
            group["frame"],
            group[column],
            linewidth=1.5,
            label=f"Track {track_id}",
        )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOT_DIR
        / filename
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


# ============================================================
# SUMMARY
# ============================================================

def print_track_summary(
    df,
):

    print()
    print("=" * 70)
    print("REFINED BALL TRACKING SUMMARY")
    print("=" * 70)

    if df.empty:

        print(
            "No trajectory was generated."
        )

        return

    for track_id, group in (
        df.groupby("track_id")
    ):

        start = int(
            group["frame"].min()
        )

        end = int(
            group["frame"].max()
        )

        length = len(group)

        span = (
            end
            - start
            + 1
        )

        coverage = (
            length / span
        )

        print()
        print(
            f"Track {track_id}"
        )

        print(
            f"  Frames:    {start} -> {end}"
        )

        print(
            f"  Detections:{length}"
        )

        print(
            f"  Coverage:  "
            f"{coverage * 100:.2f}%"
        )

        print(
            f"  Mean size: "
            f"{group['diameter'].mean():.2f} px"
        )

        print(
            f"  Median size:"
            f" {group['diameter'].median():.2f} px"
        )

        print(
            f"  Mean quality:"
            f" {group['quality'].mean():.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("REFINED BALL TRACKER")
    print("=" * 70)

    print()
    print(
        "Loading candidate detections..."
    )

    candidates_by_frame = (
        load_candidates()
    )

    print(
        f"Frames containing candidates: "
        f"{len(candidates_by_frame)}"
    )

    # --------------------------------------------------------
    # Find anchors.
    # --------------------------------------------------------

    tracks = []

    for track_id, anchor in enumerate(
        ANCHORS
    ):

        print()
        print(
            f"Searching anchor: "
            f"{anchor['name']}"
        )

        seed = find_anchor_candidate(
            candidates_by_frame,
            anchor,
        )

        if seed is None:

            print(
                "  WARNING: anchor could "
                "not be matched."
            )

            continue

        print(
            f"  Seed frame: {seed.frame}"
        )

        print(
            f"  Seed position: "
            f"({seed.x:.1f}, "
            f"{seed.y:.1f})"
        )

        print(
            f"  Seed diameter: "
            f"{seed.diameter:.1f} px"
        )

        print(
            f"  Seed quality: "
            f"{seed.quality:.3f}"
        )

        track = build_track(
            candidates_by_frame,
            seed,
        )

        print(
            f"  Track length: "
            f"{len(track)} detections"
        )

        if len(track) >= 5:

            tracks.append(
                track
            )

    # --------------------------------------------------------
    # Convert tracks to dataframe.
    # --------------------------------------------------------

    if not tracks:

        raise RuntimeError(
            "No valid refined tracks were generated."
        )

    dataframes = []

    for track_id, track in enumerate(
        tracks
    ):

        dataframes.append(
            track_to_dataframe(
                track,
                track_id,
            )
        )

    df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Velocity.
    # --------------------------------------------------------

    df = add_velocity(
        df
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "track_id",
            "frame",
        ]
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(
        f"Saved refined trajectory:"
    )

    print(
        OUTPUT_CSV
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print_track_summary(
        df
    )

    # --------------------------------------------------------
    # Plots.
    # --------------------------------------------------------

    x_plot = create_plot(
        df,
        "measurement_x",
        "X position (pixels)",
        "Refined ball trajectory — horizontal position",
        "refined_tracked_x.png",
    )

    y_plot = create_plot(
        df,
        "measurement_y",
        "Y position (pixels)",
        "Refined ball trajectory — vertical position",
        "refined_tracked_y.png",
    )

    print()
    print(
        "Plots saved:"
    )

    print(
        x_plot
    )

    print(
        y_plot
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()