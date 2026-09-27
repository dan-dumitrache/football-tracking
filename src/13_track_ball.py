"""
13_track_ball.py

Trajectory-aware football tracker.

Pipeline:
    ball_detections.csv
        ↓
    candidate filtering
        ↓
    Kalman prediction
        ↓
    candidate scoring
        ↓
    temporal / size / motion consistency
        ↓
    confirmed track
        ↓
    tracked_ball.csv
        ↓
    tracking plots

Input:
    data/output/ball_detections.csv

Output:
    data/output/tracked_ball.csv
    data/output/tracking_plots/tracked_x.png
    data/output/tracking_plots/tracked_y.png

The detector is responsible for finding red/circular candidates.
This script is responsible for deciding which candidate is actually
the football over time.
"""

from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_DIR
    / "data"
    / "input"
    / "rgb.avi"
)

DETECTIONS_PATH = (
    PROJECT_DIR
    / "data"
    / "output"
    / "ball_detections.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "output"
    / "tracked_ball.csv"
)

PLOTS_DIR = (
    PROJECT_DIR
    / "data"
    / "output"
    / "tracking_plots"
)


# ============================================================
# TRACKER PARAMETERS
# ============================================================

# Maximum spatial distance used when associating a candidate
# with the Kalman prediction.
BASE_ASSOCIATION_DISTANCE = 90.0

# Additional distance allowed when the ball is moving quickly.
VELOCITY_DISTANCE_FACTOR = 0.50

# Maximum number of consecutive frames for which we keep predicting
# after losing the measurement.
MAX_MISSES = 12

# Number of consecutive plausible detections required before
# accepting a new track.
INIT_CONFIRMATIONS = 4

# Maximum distance between candidates during initialization.
INIT_MAX_DISTANCE = 80.0

# Maximum relative change in apparent diameter between frames.
MAX_SIZE_CHANGE_RATIO = 0.65

# Maximum relative change in size allowed when initializing.
INIT_MAX_SIZE_CHANGE = 0.45

# Maximum allowed sudden change in velocity.
MAX_VELOCITY_CHANGE = 450.0

# Score threshold for accepting a candidate.
MAX_ASSOCIATION_SCORE = 1.0

# Minimum candidate quality.
MIN_CIRCULARITY = 0.20
MIN_AREA = 20.0

# Candidates which are extremely elongated are less likely to be
# the football.
MAX_ASPECT_RATIO = 2.5


# ============================================================
# KALMAN FILTER PARAMETERS
# ============================================================

# State:
#
#   [x]
#   [y]
#   [vx]
#   [vy]
#
# where x/y are pixels and vx/vy are pixels per second.

PROCESS_NOISE = 1e-2
MEASUREMENT_NOISE = 5.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def scalar(value):
    """
    Convert an OpenCV/Numpy value into a normal Python float.

    OpenCV sometimes returns a 1-element ndarray rather than a
    scalar. Using float(array) can fail on newer NumPy versions.

    This helper avoids that problem.
    """
    arr = np.asarray(value)

    if arr.size == 0:
        return float("nan")

    return float(arr.reshape(-1)[0])


def safe_float(value, default=np.nan):
    """
    Safely convert a value to float.
    """
    try:
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def apparent_diameter(candidate):
    """
    Estimate the candidate's apparent diameter in pixels.

    We use the average of width and height.
    """
    width = safe_float(candidate.get("width", np.nan))
    height = safe_float(candidate.get("height", np.nan))

    if np.isfinite(width) and np.isfinite(height):
        return 0.5 * (width + height)

    return np.nan


def candidate_quality(candidate):
    """
    Basic geometric quality score.

    Higher is better.
    """
    score = 0.0

    circularity = safe_float(
        candidate.get("circularity", np.nan)
    )

    aspect_ratio = safe_float(
        candidate.get("aspect_ratio", np.nan)
    )

    area = safe_float(
        candidate.get("area", np.nan)
    )

    if np.isfinite(circularity):
        score += min(max(circularity, 0.0), 1.0)

    if np.isfinite(aspect_ratio):
        aspect_penalty = abs(aspect_ratio - 1.0)
        score -= 0.25 * aspect_penalty

    if np.isfinite(area):
        if area >= 100:
            score += 0.20
        elif area >= 40:
            score += 0.10

    return score


def get_candidate_xy(candidate):
    """
    Return candidate center coordinates.
    """
    x = safe_float(candidate["center_x"])
    y = safe_float(candidate["center_y"])

    return x, y


def distance(x1, y1, x2, y2):
    """
    Euclidean distance between two image points.
    """
    return float(
        np.hypot(
            x1 - x2,
            y1 - y2
        )
    )


def size_change_ratio(previous_size, current_size):
    """
    Relative apparent-size change.

    Example:
        30 -> 33 pixels
        ratio = 0.10

    Example:
        30 -> 60 pixels
        ratio = 1.00
    """
    if (
        previous_size is None
        or current_size is None
        or not np.isfinite(previous_size)
        or not np.isfinite(current_size)
        or previous_size <= 0
    ):
        return 0.0

    return abs(current_size - previous_size) / previous_size


# ============================================================
# KALMAN FILTER
# ============================================================

def create_kalman(dt, x, y):
    """
    Create and initialize a constant-velocity Kalman filter.
    """

    kalman = cv2.KalmanFilter(4, 2)

    # State transition matrix.
    #
    # x_new = x + vx * dt
    # y_new = y + vy * dt
    #
    # vx_new = vx
    # vy_new = vy

    kalman.transitionMatrix = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # Measurement:
    #
    # measurement = [x, y]

    kalman.measurementMatrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    kalman.processNoiseCov = (
        np.eye(4, dtype=np.float32)
        * PROCESS_NOISE
    )

    kalman.measurementNoiseCov = (
        np.eye(2, dtype=np.float32)
        * MEASUREMENT_NOISE
    )

    kalman.errorCovPost = (
        np.eye(4, dtype=np.float32)
        * 1.0
    )

    kalman.statePost = np.array(
        [
            [x],
            [y],
            [0.0],
            [0.0],
        ],
        dtype=np.float32,
    )

    return kalman


# ============================================================
# DETECTION LOADING
# ============================================================

def load_detections():
    """
    Load candidate detections.

    Expected columns:

        frame
        center_x
        center_y

    Optional:

        width
        height
        area
        circularity
        aspect_ratio
    """

    if not DETECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Detection file not found:\n{DETECTIONS_PATH}"
        )

    df = pd.read_csv(DETECTIONS_PATH)

    required_columns = [
        "frame",
        "center_x",
        "center_y",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Detection CSV is missing required columns: "
            + ", ".join(missing)
        )

    # Add optional columns if they don't exist.
    optional_columns = [
        "width",
        "height",
        "area",
        "circularity",
        "aspect_ratio",
    ]

    for column in optional_columns:
        if column not in df.columns:
            df[column] = np.nan

    df["frame"] = df["frame"].astype(int)

    df = df.sort_values(
        ["frame"]
    ).reset_index(drop=True)

    return df


def group_detections(df):
    """
    Group candidate detections by frame.
    """

    grouped = {}

    for frame_number, group in df.groupby("frame"):
        candidates = []

        for _, row in group.iterrows():
            candidate = row.to_dict()

            # Basic validity filtering.
            x = safe_float(
                candidate.get("center_x")
            )

            y = safe_float(
                candidate.get("center_y")
            )

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            area = safe_float(
                candidate.get("area")
            )

            if np.isfinite(area) and area < MIN_AREA:
                continue

            circularity = safe_float(
                candidate.get("circularity")
            )

            if (
                np.isfinite(circularity)
                and circularity < MIN_CIRCULARITY
            ):
                continue

            aspect_ratio = safe_float(
                candidate.get("aspect_ratio")
            )

            if (
                np.isfinite(aspect_ratio)
                and aspect_ratio > MAX_ASPECT_RATIO
            ):
                continue

            candidates.append(candidate)

        grouped[int(frame_number)] = candidates

    return grouped


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(
    candidate,
    predicted_x,
    predicted_y,
    predicted_vx,
    predicted_vy,
    previous_size,
):
    """
    Score a candidate against the predicted trajectory.

    Lower score = better candidate.

    Components:

        1. spatial distance
        2. size consistency
        3. geometric quality
        4. velocity plausibility
    """

    x, y = get_candidate_xy(candidate)

    spatial_distance = distance(
        x,
        y,
        predicted_x,
        predicted_y,
    )

    # --------------------------------------------------------
    # Spatial gate
    # --------------------------------------------------------

    speed = float(
        np.hypot(
            predicted_vx,
            predicted_vy,
        )
    )

    allowed_distance = (
        BASE_ASSOCIATION_DISTANCE
        + VELOCITY_DISTANCE_FACTOR * speed
    )

    # Prevent the gate becoming enormous.
    allowed_distance = min(
        allowed_distance,
        220.0,
    )

    if spatial_distance > allowed_distance:
        return None

    # --------------------------------------------------------
    # Size consistency
    # --------------------------------------------------------

    current_size = apparent_diameter(
        candidate
    )

    size_ratio = size_change_ratio(
        previous_size,
        current_size,
    )

    if (
        previous_size is not None
        and current_size is not None
        and size_ratio > MAX_SIZE_CHANGE_RATIO
    ):
        return None

    # --------------------------------------------------------
    # Geometric quality
    # --------------------------------------------------------

    circularity = safe_float(
        candidate.get("circularity")
    )

    if np.isfinite(circularity):
        circularity_penalty = max(
            0.0,
            0.75 - circularity,
        )
    else:
        circularity_penalty = 0.20

    aspect_ratio = safe_float(
        candidate.get("aspect_ratio")
    )

    if np.isfinite(aspect_ratio):
        aspect_penalty = abs(
            aspect_ratio - 1.0
        )
    else:
        aspect_penalty = 0.0

    # --------------------------------------------------------
    # Final normalized score
    # --------------------------------------------------------

    spatial_score = (
        spatial_distance
        / max(allowed_distance, 1.0)
    )

    size_score = min(
        size_ratio,
        1.0,
    )

    score = (
        0.65 * spatial_score
        + 0.20 * size_score
        + 0.10 * circularity_penalty
        + 0.05 * aspect_penalty
    )

    return {
        "score": float(score),
        "distance": float(spatial_distance),
        "size_ratio": float(size_ratio),
        "size": current_size,
    }


def choose_candidate(
    candidates,
    predicted_x,
    predicted_y,
    predicted_vx,
    predicted_vy,
    previous_size,
):
    """
    Select the best candidate for the current frame.
    """

    best_candidate = None
    best_info = None

    for candidate in candidates:

        result = score_candidate(
            candidate,
            predicted_x,
            predicted_y,
            predicted_vx,
            predicted_vy,
            previous_size,
        )

        if result is None:
            continue

        if (
            result["score"]
            > MAX_ASSOCIATION_SCORE
        ):
            continue

        if (
            best_info is None
            or result["score"]
            < best_info["score"]
        ):
            best_candidate = candidate
            best_info = result

    return best_candidate, best_info


# ============================================================
# TRACK INITIALIZATION
# ============================================================

def candidate_can_follow(
    previous_candidate,
    current_candidate,
):
    """
    Determine whether two candidates could belong to the
    same physical object during initialization.
    """

    x1, y1 = get_candidate_xy(
        previous_candidate
    )

    x2, y2 = get_candidate_xy(
        current_candidate
    )

    spatial_distance = distance(
        x1,
        y1,
        x2,
        y2,
    )

    if spatial_distance > INIT_MAX_DISTANCE:
        return False

    size1 = apparent_diameter(
        previous_candidate
    )

    size2 = apparent_diameter(
        current_candidate
    )

    if (
        np.isfinite(size1)
        and np.isfinite(size2)
        and size1 > 0
    ):
        change = (
            abs(size2 - size1)
            / size1
        )

        if change > INIT_MAX_SIZE_CHANGE:
            return False

    return True


def update_initialization_history(
    history,
    candidates,
):
    """
    Find a plausible candidate continuation.

    The history contains candidates selected during the
    confirmation phase.
    """

    if not candidates:
        return history

    if not history:
        # Start with the highest-quality candidate.
        best = max(
            candidates,
            key=candidate_quality,
        )

        return [best]

    previous = history[-1]

    compatible = []

    for candidate in candidates:
        if candidate_can_follow(
            previous,
            candidate,
        ):
            compatible.append(candidate)

    if not compatible:
        # Start over.
        best = max(
            candidates,
            key=candidate_quality,
        )

        return [best]

    best = max(
        compatible,
        key=candidate_quality,
    )

    history.append(best)

    # Keep only recent confirmation history.
    if len(history) > INIT_CONFIRMATIONS:
        history = history[-INIT_CONFIRMATIONS:]

    return history


# ============================================================
# MAIN TRACKING FUNCTION
# ============================================================

def track_ball():

    print(
        "Temporal ball tracker"
    )
    print(
        "====================="
    )

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    video.release()

    if fps <= 0:
        raise RuntimeError(
            "Invalid FPS."
        )

    dt = 1.0 / fps

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS:        {fps:.2f}"
    )

    print(
        f"Frames:     {frame_count}"
    )

    print()

    # --------------------------------------------------------
    # Load detections
    # --------------------------------------------------------

    detections_df = load_detections()

    detections_by_frame = (
        group_detections(
            detections_df
        )
    )

    # --------------------------------------------------------
    # Tracker state
    # --------------------------------------------------------

    kalman = None

    tracking = False
    predicting = False

    misses = 0

    previous_size = None

    previous_measurement = None
    previous_velocity = None

    # Candidate history used before establishing a track.
    initialization_history = []

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    rows = []

    measurement_count = 0
    tracking_count = 0
    predicting_count = 0
    searching_count = 0
    lost_count = 0

    association_distances = []

    initialization_events = []
    lost_events = []

    # --------------------------------------------------------
    # Frame loop
    # --------------------------------------------------------

    for frame_number in range(
        frame_count
    ):

        candidates = detections_by_frame.get(
            frame_number,
            [],
        )

        # ====================================================
        # NO ACTIVE TRACK
        # ====================================================

        if kalman is None:

            searching_count += 1

            initialization_history = (
                update_initialization_history(
                    initialization_history,
                    candidates,
                )
            )

            if (
                len(initialization_history)
                >= INIT_CONFIRMATIONS
            ):

                initial_candidate = (
                    initialization_history[-1]
                )

                x, y = get_candidate_xy(
                    initial_candidate
                )

                kalman = create_kalman(
                    dt,
                    x,
                    y,
                )

                tracking = True
                predicting = False
                misses = 0

                previous_size = (
                    apparent_diameter(
                        initial_candidate
                    )
                )

                previous_measurement = (
                    np.array(
                        [x, y],
                        dtype=float,
                    )
                )

                previous_velocity = (
                    np.array(
                        [0.0, 0.0],
                        dtype=float,
                    )
                )

                initialization_events.append(
                    frame_number
                )

                print(
                    f"Tracker initialized at frame "
                    f"{frame_number}"
                )

                initialization_history = []

                # Record the initialization frame.
                rows.append(
                    {
                        "frame": frame_number,
                        "time_seconds":
                            frame_number / fps,
                        "detected": 1,
                        "measurement_x": x,
                        "measurement_y": y,
                        "estimated_x": x,
                        "estimated_y": y,
                        "estimated_vx": 0.0,
                        "estimated_vy": 0.0,
                        "candidate_count":
                            len(candidates),
                        "association_distance": 0.0,
                        "association_score": 0.0,
                        "track_state": "TRACKING",
                        "misses": 0,
                    }
                )

                measurement_count += 1
                tracking_count += 1

                continue

            # No confirmed track yet.
            rows.append(
                {
                    "frame": frame_number,
                    "time_seconds":
                        frame_number / fps,
                    "detected": 0,
                    "measurement_x": np.nan,
                    "measurement_y": np.nan,
                    "estimated_x": np.nan,
                    "estimated_y": np.nan,
                    "estimated_vx": np.nan,
                    "estimated_vy": np.nan,
                    "candidate_count":
                        len(candidates),
                    "association_distance": np.nan,
                    "association_score": np.nan,
                    "track_state": "SEARCHING",
                    "misses": 0,
                }
            )

            continue

        # ====================================================
        # ACTIVE TRACK
        # ====================================================

        prediction = kalman.predict()

        predicted_x = scalar(
            prediction[0]
        )

        predicted_y = scalar(
            prediction[1]
        )

        predicted_vx = scalar(
            prediction[2]
        )

        predicted_vy = scalar(
            prediction[3]
        )

        # ----------------------------------------------------
        # Candidate association
        # ----------------------------------------------------

        candidate, info = choose_candidate(
            candidates,
            predicted_x,
            predicted_y,
            predicted_vx,
            predicted_vy,
            previous_size,
        )

        # ====================================================
        # MEASUREMENT ACCEPTED
        # ====================================================

        if candidate is not None:

            measurement_x, measurement_y = (
                get_candidate_xy(candidate)
            )

            measurement = np.array(
                [
                    [measurement_x],
                    [measurement_y],
                ],
                dtype=np.float32,
            )

            corrected = kalman.correct(
                measurement
            )

            estimated_x = scalar(
                corrected[0]
            )

            estimated_y = scalar(
                corrected[1]
            )

            estimated_vx = scalar(
                corrected[2]
            )

            estimated_vy = scalar(
                corrected[3]
            )

            # ------------------------------------------------
            # Velocity consistency
            # ------------------------------------------------

            current_velocity = np.array(
                [
                    estimated_vx,
                    estimated_vy,
                ],
                dtype=float,
            )

            velocity_change = 0.0

            if previous_velocity is not None:
                velocity_change = float(
                    np.linalg.norm(
                        current_velocity
                        - previous_velocity
                    )
                )

            # If the velocity changes violently, we still keep
            # the measurement if it is spatially close, but we
            # don't immediately trust it as a confirmed physical
            # trajectory.
            #
            # This protects against sudden false-object jumps.

            if (
                previous_velocity is not None
                and velocity_change
                > MAX_VELOCITY_CHANGE
            ):

                # Revert to prediction rather than accepting a
                # physically implausible jump.

                estimated_x = predicted_x
                estimated_y = predicted_y
                estimated_vx = predicted_vx
                estimated_vy = predicted_vy

                # Reinitialize Kalman state at prediction.
                kalman.statePost = np.array(
                    [
                        [estimated_x],
                        [estimated_y],
                        [estimated_vx],
                        [estimated_vy],
                    ],
                    dtype=np.float32,
                )

                predicting = True
                tracking = False

                predicting_count += 1
                misses += 1

                rows.append(
                    {
                        "frame": frame_number,
                        "time_seconds":
                            frame_number / fps,
                        "detected": 0,
                        "measurement_x": np.nan,
                        "measurement_y": np.nan,
                        "estimated_x":
                            estimated_x,
                        "estimated_y":
                            estimated_y,
                        "estimated_vx":
                            estimated_vx,
                        "estimated_vy":
                            estimated_vy,
                        "candidate_count":
                            len(candidates),
                        "association_distance":
                            info["distance"],
                        "association_score":
                            info["score"],
                        "track_state":
                            "PREDICTING",
                        "misses": misses,
                    }
                )

                if misses > MAX_MISSES:
                    lost_events.append(
                        frame_number
                    )

                    print(
                        f"Track lost at frame "
                        f"{frame_number}"
                    )

                    kalman = None
                    tracking = False
                    predicting = False
                    misses = 0
                    previous_size = None
                    previous_measurement = None
                    previous_velocity = None
                    initialization_history = []

                continue

            # ------------------------------------------------
            # Normal accepted measurement
            # ------------------------------------------------

            tracking = True
            predicting = False
            misses = 0

            tracking_count += 1
            measurement_count += 1

            association_distances.append(
                info["distance"]
            )

            current_size = info["size"]

            if (
                current_size is not None
                and np.isfinite(current_size)
            ):
                previous_size = (
                    current_size
                )

            previous_measurement = np.array(
                [
                    measurement_x,
                    measurement_y,
                ],
                dtype=float,
            )

            previous_velocity = (
                current_velocity
            )

            rows.append(
                {
                    "frame": frame_number,
                    "time_seconds":
                        frame_number / fps,
                    "detected": 1,
                    "measurement_x":
                        measurement_x,
                    "measurement_y":
                        measurement_y,
                    "estimated_x":
                        estimated_x,
                    "estimated_y":
                        estimated_y,
                    "estimated_vx":
                        estimated_vx,
                    "estimated_vy":
                        estimated_vy,
                    "candidate_count":
                        len(candidates),
                    "association_distance":
                        info["distance"],
                    "association_score":
                        info["score"],
                    "track_state":
                        "TRACKING",
                    "misses": 0,
                }
            )

            continue

        # ====================================================
        # NO MEASUREMENT
        # ====================================================

        misses += 1

        estimated_x = predicted_x
        estimated_y = predicted_y
        estimated_vx = predicted_vx
        estimated_vy = predicted_vy

        # ----------------------------------------------------
        # Continue predicting temporarily
        # ----------------------------------------------------

        if misses <= MAX_MISSES:

            predicting = True
            tracking = False

            predicting_count += 1

            rows.append(
                {
                    "frame": frame_number,
                    "time_seconds":
                        frame_number / fps,
                    "detected": 0,
                    "measurement_x": np.nan,
                    "measurement_y": np.nan,
                    "estimated_x":
                        estimated_x,
                    "estimated_y":
                        estimated_y,
                    "estimated_vx":
                        estimated_vx,
                    "estimated_vy":
                        estimated_vy,
                    "candidate_count":
                        len(candidates),
                    "association_distance":
                        np.nan,
                    "association_score":
                        np.nan,
                    "track_state":
                        "PREDICTING",
                    "misses": misses,
                }
            )

            continue

        # ----------------------------------------------------
        # Track lost
        # ----------------------------------------------------

        lost_count += 1

        print(
            f"Track lost at frame "
            f"{frame_number}"
        )

        lost_events.append(
            frame_number
        )

        rows.append(
            {
                "frame": frame_number,
                "time_seconds":
                    frame_number / fps,
                "detected": 0,
                "measurement_x": np.nan,
                "measurement_y": np.nan,
                "estimated_x": np.nan,
                "estimated_y": np.nan,
                "estimated_vx": np.nan,
                "estimated_vy": np.nan,
                "candidate_count":
                    len(candidates),
                "association_distance":
                    np.nan,
                "association_score":
                    np.nan,
                "track_state": "LOST",
                "misses": misses,
            }
        )

        # Reset tracker.
        kalman = None
        tracking = False
        predicting = False
        misses = 0

        previous_size = None
        previous_measurement = None
        previous_velocity = None

        initialization_history = []

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "Tracking complete."
    )
    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Tracking summary"
    )
    print(
        "----------------"
    )

    print(
        f"Measurement detections: "
        f"{measurement_count} / {frame_count}"
    )

    print(
        f"Detection rate: "
        f"{100.0 * measurement_count / frame_count:.2f}%"
    )

    state_counts = (
        result_df["track_state"]
        .value_counts()
    )

    for state in [
        "SEARCHING",
        "TRACKING",
        "PREDICTING",
        "LOST",
    ]:

        count = int(
            state_counts.get(
                state,
                0,
            )
        )

        percentage = (
            100.0 * count / frame_count
        )

        print(
            f"  {state:<11}: "
            f"{count:4d} frames "
            f"({percentage:5.1f}%)"
        )

    # ========================================================
    # ASSOCIATION STATISTICS
    # ========================================================

    print()
    print(
        "Association distance"
    )

    print(
        "--------------------"
    )

    if association_distances:

        distances = np.asarray(
            association_distances,
            dtype=float,
        )

        print(
            f"Mean:   "
            f"{np.mean(distances):.2f} px"
        )

        print(
            f"Median: "
            f"{np.median(distances):.2f} px"
        )

        print(
            f"Maximum:"
            f"{np.max(distances):.2f} px"
        )

    else:

        print(
            "No successful associations."
        )

    # ========================================================
    # TRACK EVENTS
    # ========================================================

    print()
    print(
        "Track events"
    )

    print(
        "------------"
    )

    print(
        "Initialization frames:"
    )

    if initialization_events:
        print(
            initialization_events
        )
    else:
        print(
            "None"
        )

    print(
        "Track-loss frames:"
    )

    if lost_events:
        print(
            lost_events
        )
    else:
        print(
            "None"
        )

    # ========================================================
    # PLOTS
    # ========================================================

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # X position
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 7)
    )

    tracked = (
        result_df["estimated_x"]
        .notna()
    )

    measured = (
        result_df["measurement_x"]
        .notna()
    )

    plt.plot(
        result_df.loc[
            tracked,
            "frame"
        ],
        result_df.loc[
            tracked,
            "estimated_x"
        ],
        label="Kalman estimate",
    )

    plt.scatter(
        result_df.loc[
            measured,
            "frame"
        ],
        result_df.loc[
            measured,
            "measurement_x"
        ],
        s=12,
        label="Associated measurements",
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "X position (pixels)"
    )

    plt.title(
        "Tracked ball — horizontal position"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    x_plot_path = (
        PLOTS_DIR
        / "tracked_x.png"
    )

    plt.savefig(
        x_plot_path,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Y position
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 7)
    )

    tracked = (
        result_df["estimated_y"]
        .notna()
    )

    measured = (
        result_df["measurement_y"]
        .notna()
    )

    plt.plot(
        result_df.loc[
            tracked,
            "frame"
        ],
        result_df.loc[
            tracked,
            "estimated_y"
        ],
        label="Kalman estimate",
    )

    plt.scatter(
        result_df.loc[
            measured,
            "frame"
        ],
        result_df.loc[
            measured,
            "measurement_y"
        ],
        s=12,
        label="Associated measurements",
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Y position (pixels)"
    )

    plt.title(
        "Tracked ball — vertical position"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    y_plot_path = (
        PLOTS_DIR
        / "tracked_y.png"
    )

    plt.savefig(
        y_plot_path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        "Plots saved:"
    )

    print(
        x_plot_path
    )

    print(
        y_plot_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    track_ball()