import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_DIR / "data" / "input" / "rgb.avi"

OUTPUT_DIR = PROJECT_DIR / "data" / "output"
PLOT_DIR = OUTPUT_DIR / "tracking_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "global_ball_track.csv"
CANDIDATE_CSV = OUTPUT_DIR / "all_ball_candidates.csv"


# ============================================================
# VIDEO / DETECTION PARAMETERS
# ============================================================

MIN_AREA = 60
MAX_AREA = 5000

MIN_DIAMETER = 10
MAX_DIAMETER = 120

MIN_CIRCULARITY = 0.20
MAX_ASPECT_RATIO = 2.5

# Red HSV ranges.
#
# OpenCV hue is [0,179].
# Red wraps around the HSV hue boundary, therefore we need
# two ranges.
RED1_LOW = np.array([0, 70, 50])
RED1_HIGH = np.array([10, 255, 255])

RED2_LOW = np.array([170, 70, 50])
RED2_HIGH = np.array([179, 255, 255])


# ============================================================
# GLOBAL TRACKING PARAMETERS
# ============================================================

# Maximum number of consecutive frames for which the ball
# may disappear while still remaining part of the same track.
MAX_GAP = 4

# Maximum physically plausible image displacement per frame.
#
# This is intentionally generous because the ball can move
# quickly after the kick.
MAX_SPEED_PX_PER_FRAME = 80.0

# Additional tolerance when comparing a candidate against
# the predicted position.
ASSOCIATION_GATE = 120.0

# How strongly size continuity contributes to the score.
SIZE_WEIGHT = 8.0

# How strongly shape contributes.
SHAPE_WEIGHT = 4.0

# How strongly spatial continuity contributes.
DISTANCE_WEIGHT = 1.0

# Penalty for skipping one frame.
GAP_PENALTY = 15.0

# Minimum number of detections required for an accepted track.
MIN_TRACK_LENGTH = 5


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
# UTILITY FUNCTIONS
# ============================================================

def circularity(area, perimeter):
    """
    Circularity = 4*pi*A/P^2

    A perfect circle has a value close to 1.
    """
    if perimeter <= 0:
        return 0.0

    return float(4.0 * np.pi * area / (perimeter * perimeter))


def candidate_quality(
    area,
    circularity_value,
    aspect_ratio,
    diameter
):
    """
    Heuristic quality score.

    This is NOT a probability.

    It simply tells the trajectory optimizer that a compact,
    approximately circular object is preferable to an irregular
    red blob.
    """

    # Shape quality.
    shape_score = np.clip(circularity_value, 0.0, 1.0)

    # Aspect-ratio quality.
    aspect_error = abs(aspect_ratio - 1.0)
    aspect_score = np.exp(-aspect_error)

    # Diameter preference.
    #
    # The football can change apparent size significantly, so
    # this term is deliberately weak.
    diameter_score = np.clip(diameter / 40.0, 0.0, 1.0)

    score = (
        2.0 * shape_score
        + 1.0 * aspect_score
        + 0.5 * diameter_score
    )

    return float(score)


# ============================================================
# CANDIDATE DETECTION
# ============================================================

def detect_candidates(frame, frame_number, fps):
    """
    Detect all red candidate objects in a frame.

    IMPORTANT:
    We intentionally keep multiple candidates.

    The previous tracker immediately selected one candidate.
    This script keeps the alternatives so that the trajectory
    optimizer can decide later.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, RED1_LOW, RED1_HIGH)
    mask2 = cv2.inRange(hsv, RED2_LOW, RED2_HIGH)

    mask = cv2.bitwise_or(mask1, mask2)

    # Remove isolated noise.
    kernel_small = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1,
    )

    # Connect nearby pixels belonging to the same object.
    kernel_medium = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circ = circularity(area, perimeter)

        if circ < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        if aspect_ratio < 1.0 / MAX_ASPECT_RATIO:
            continue

        diameter = (w + h) / 2.0

        if diameter < MIN_DIAMETER:
            continue

        if diameter > MAX_DIAMETER:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        quality = candidate_quality(
            area,
            circ,
            aspect_ratio,
            diameter,
        )

        candidates.append(
            Candidate(
                frame=frame_number,
                time_seconds=frame_number / fps,

                x=float(center_x),
                y=float(center_y),

                width=float(w),
                height=float(h),

                area=float(area),
                circularity=float(circ),
                aspect_ratio=float(aspect_ratio),

                diameter=float(diameter),

                quality=float(quality),
            )
        )

    # Stronger candidates first.
    candidates.sort(
        key=lambda c: c.quality,
        reverse=True,
    )

    return candidates


# ============================================================
# DETECT ALL CANDIDATES
# ============================================================

def extract_all_candidates():
    print()
    print("=" * 70)
    print("GLOBAL BALL TRACKER")
    print("=" * 70)

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width} x {height}")
    print(f"FPS:        {fps:.2f}")
    print(f"Frames:     {frame_count}")
    print()

    all_candidates = []

    frame_number = 0

    while True:

        ok, frame = video.read()

        if not ok:
            break

        candidates = detect_candidates(
            frame,
            frame_number,
            fps,
        )

        for candidate in candidates:

            all_candidates.append(
                {
                    "frame": candidate.frame,
                    "time_seconds": candidate.time_seconds,

                    "x": candidate.x,
                    "y": candidate.y,

                    "width": candidate.width,
                    "height": candidate.height,

                    "area": candidate.area,
                    "circularity": candidate.circularity,
                    "aspect_ratio": candidate.aspect_ratio,

                    "diameter": candidate.diameter,
                    "quality": candidate.quality,
                }
            )

        frame_number += 1

    video.release()

    df = pd.DataFrame(all_candidates)

    df.to_csv(
        CANDIDATE_CSV,
        index=False,
    )

    print(
        f"Total candidate detections: {len(df)}"
    )

    print(
        f"Saved candidates:\n{CANDIDATE_CSV}"
    )

    return df, fps, frame_count


# ============================================================
# MOTION MODEL
# ============================================================

def predict_position(previous, current):
    """
    Constant-velocity prediction.

    Given:

        previous position
        current position

    estimate where the ball should appear next.
    """

    vx = current.x - previous.x
    vy = current.y - previous.y

    predicted_x = current.x + vx
    predicted_y = current.y + vy

    return predicted_x, predicted_y


# ============================================================
# TRANSITION COST
# ============================================================

def transition_cost(previous, current, gap):
    """
    Cost of moving from one candidate to another.

    Lower = better.

    Components:

        1. distance from predicted position
        2. size change
        3. shape change
        4. gap penalty
    """

    if gap <= 0:
        gap = 1

    # Predict using previous velocity if possible.
    predicted_x, predicted_y = predict_position(
        previous,
        current,
    )

    dx = current.x - predicted_x
    dy = current.y - predicted_y

    prediction_error = np.sqrt(
        dx * dx + dy * dy
    )

    # Actual displacement.
    actual_dx = current.x - previous.x
    actual_dy = current.y - previous.y

    distance = np.sqrt(
        actual_dx * actual_dx
        + actual_dy * actual_dy
    )

    # Maximum expected displacement increases slightly
    # when frames have been missed.
    allowed_distance = (
        MAX_SPEED_PX_PER_FRAME * gap
        + ASSOCIATION_GATE
    )

    if distance > allowed_distance:
        return np.inf

    # Size continuity.
    previous_size = max(previous.diameter, 1.0)
    current_size = max(current.diameter, 1.0)

    size_ratio = current_size / previous_size

    size_change = abs(
        np.log(size_ratio)
    )

    # Shape continuity.
    shape_change = abs(
        current.circularity
        - previous.circularity
    )

    cost = (
        DISTANCE_WEIGHT * prediction_error
        + SIZE_WEIGHT * size_change
        + SHAPE_WEIGHT * shape_change
        + GAP_PENALTY * max(gap - 1, 0)
    )

    # Candidate quality is rewarded.
    cost -= 5.0 * current.quality

    return float(cost)


# ============================================================
# BUILD TRACK SEGMENTS
# ============================================================

def build_track_segments(df, fps):
    """
    Build temporally consistent candidate tracks.

    This is intentionally conservative.

    We do NOT require one track to cover the whole video.

    A long disappearance is treated as:

        ball visible
             ↓
        ball disappears
             ↓
        no reliable observation
             ↓
        ball reappears
             ↓
        new track segment

    This prevents the tracker from following unrelated red
    objects for several seconds.
    """

    if df.empty:
        return []

    candidates_by_frame = {}

    for row in df.itertuples(index=False):

        candidate = Candidate(
            frame=int(row.frame),
            time_seconds=float(row.time_seconds),

            x=float(row.x),
            y=float(row.y),

            width=float(row.width),
            height=float(row.height),

            area=float(row.area),
            circularity=float(row.circularity),
            aspect_ratio=float(row.aspect_ratio),

            diameter=float(row.diameter),

            quality=float(row.quality),
        )

        candidates_by_frame.setdefault(
            candidate.frame,
            [],
        ).append(candidate)

    frames = sorted(candidates_by_frame.keys())

    # --------------------------------------------------------
    # Greedy segment extraction with global candidate scoring.
    #
    # We repeatedly choose the strongest temporally coherent
    # path and remove its candidates.
    # --------------------------------------------------------

    unused = {
        frame: list(candidates)
        for frame, candidates
        in candidates_by_frame.items()
    }

    segments = []

    while True:

        available_frames = sorted(
            frame
            for frame, candidates
            in unused.items()
            if candidates
        )

        if not available_frames:
            break

        best_segment = None
        best_score = -np.inf

        # ----------------------------------------------------
        # Try every candidate as a possible starting point.
        # ----------------------------------------------------

        for start_frame in available_frames:

            start_candidates = unused[start_frame]

            for start_candidate in start_candidates:

                path = [start_candidate]

                score = start_candidate.quality * 5.0

                current = start_candidate
                gap = 0

                for frame in available_frames:

                    if frame <= start_frame:
                        continue

                    frame_gap = frame - current.frame

                    if frame_gap > MAX_GAP:
                        break

                    best_candidate = None
                    best_cost = np.inf

                    for candidate in unused[frame]:

                        cost = transition_cost(
                            current,
                            candidate,
                            frame_gap,
                        )

                        if cost < best_cost:
                            best_cost = cost
                            best_candidate = candidate

                    if best_candidate is None:
                        gap += frame_gap

                        if gap > MAX_GAP:
                            break

                        continue

                    path.append(best_candidate)

                    score -= best_cost

                    current = best_candidate
                    gap = 0

                if len(path) < MIN_TRACK_LENGTH:
                    continue

                # Prefer longer tracks.
                score += 10.0 * len(path)

                # Prefer tracks with good average candidate quality.
                mean_quality = np.mean(
                    [candidate.quality for candidate in path]
                )

                score += 10.0 * mean_quality

                if score > best_score:

                    best_score = score
                    best_segment = path

        if best_segment is None:
            break

        segments.append(best_segment)

        # Remove selected candidates.
        selected_keys = {
            (candidate.frame, candidate.x, candidate.y)
            for candidate in best_segment
        }

        for frame in list(unused.keys()):

            remaining = []

            for candidate in unused[frame]:

                key = (
                    candidate.frame,
                    candidate.x,
                    candidate.y,
                )

                if key not in selected_keys:
                    remaining.append(candidate)

            unused[frame] = remaining

    # Sort by start frame.
    segments.sort(
        key=lambda segment: segment[0].frame
    )

    return segments


# ============================================================
# KALMAN SMOOTHING
# ============================================================

def smooth_segment(segment, fps):
    """
    Apply a Kalman filter AFTER trajectory association.

    This is deliberately different from 13_track_ball.py.

    Here the trajectory has already been selected.

    Kalman is now only a smoothing / interpolation tool.
    """

    if not segment:
        return []

    dt = 1.0 / fps

    kalman = cv2.KalmanFilter(4, 2)

    kalman.transitionMatrix = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    kalman.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    kalman.processNoiseCov = (
        np.eye(4, dtype=np.float32) * 0.03
    )

    kalman.measurementNoiseCov = (
        np.eye(2, dtype=np.float32) * 4.0
    )

    kalman.errorCovPost = (
        np.eye(4, dtype=np.float32)
    )

    first = segment[0]

    kalman.statePost = np.array(
        [
            [first.x],
            [first.y],
            [0.0],
            [0.0],
        ],
        dtype=np.float32,
    )

    results = []

    previous_frame = first.frame

    for candidate in segment:

        # Advance the filter according to the actual
        # frame difference.
        frame_gap = max(
            candidate.frame - previous_frame,
            1,
        )

        kalman.transitionMatrix[0, 2] = (
            dt * frame_gap
        )

        kalman.transitionMatrix[1, 3] = (
            dt * frame_gap
        )

        prediction = kalman.predict()

        measurement = np.array(
            [
                [candidate.x],
                [candidate.y],
            ],
            dtype=np.float32,
        )

        estimate = kalman.correct(
            measurement
        )

        estimated_x = float(
            np.asarray(
                estimate[0]
            ).reshape(-1)[0]
        )

        estimated_y = float(
            np.asarray(
                estimate[1]
            ).reshape(-1)[0]
        )

        estimated_vx = float(
            np.asarray(
                estimate[2]
            ).reshape(-1)[0]
        )

        estimated_vy = float(
            np.asarray(
                estimate[3]
            ).reshape(-1)[0]
        )

        results.append(
            {
                "frame": candidate.frame,
                "time_seconds": candidate.time_seconds,

                "detected": 1,

                "measurement_x": candidate.x,
                "measurement_y": candidate.y,

                "estimated_x": estimated_x,
                "estimated_y": estimated_y,

                "estimated_vx": estimated_vx,
                "estimated_vy": estimated_vy,

                "diameter": candidate.diameter,
                "area": candidate.area,

                "circularity": candidate.circularity,
                "aspect_ratio": candidate.aspect_ratio,

                "quality": candidate.quality,
            }
        )

        previous_frame = candidate.frame

    return results


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

def create_output(segments, fps, frame_count):

    rows = []

    for track_id, segment in enumerate(segments):

        smoothed = smooth_segment(
            segment,
            fps,
        )

        for row in smoothed:

            row["track_id"] = track_id

            rows.append(row)

    if not rows:

        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["frame", "track_id"]
    ).reset_index(drop=True)

    return df


# ============================================================
# SELECT MAIN BALL TRACK
# ============================================================

def select_main_track(df):
    """
    Select the main trajectory.

    We do NOT necessarily want the longest track.

    A useful ball track should combine:

        - length
        - candidate quality
        - temporal continuity
        - spatial consistency
    """

    if df.empty:
        return df

    scores = []

    for track_id, group in df.groupby("track_id"):

        length = len(group)

        mean_quality = group["quality"].mean()

        # Number of frames covered by the track.
        frame_span = (
            group["frame"].max()
            - group["frame"].min()
            + 1
        )

        coverage = length / max(
            frame_span,
            1,
        )

        score = (
            2.0 * length
            + 50.0 * mean_quality
            + 100.0 * coverage
        )

        scores.append(
            {
                "track_id": track_id,
                "length": length,
                "mean_quality": mean_quality,
                "coverage": coverage,
                "score": score,
            }
        )

    score_df = pd.DataFrame(scores)

    score_df = score_df.sort_values(
        "score",
        ascending=False,
    )

    print()
    print("Candidate trajectory ranking")
    print("----------------------------")

    print(
        score_df.to_string(
            index=False
        )
    )

    main_track_id = int(
        score_df.iloc[0]["track_id"]
    )

    print()
    print(
        f"Selected main track: {main_track_id}"
    )

    main = df[
        df["track_id"] == main_track_id
    ].copy()

    return main


# ============================================================
# PLOTS
# ============================================================

def create_plots(main_track):

    if main_track.empty:
        print("No main track available for plotting.")
        return

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    plt.figure(figsize=(14, 7))

    plt.plot(
        main_track["frame"],
        main_track["estimated_x"],
        label="Kalman estimate",
    )

    plt.scatter(
        main_track["frame"],
        main_track["measurement_x"],
        s=12,
        label="Associated measurements",
    )

    plt.title(
        "Global trajectory — horizontal position"
    )

    plt.xlabel("Frame")
    plt.ylabel("X position (pixels)")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    x_path = (
        PLOT_DIR
        / "global_tracked_x.png"
    )

    plt.savefig(
        x_path,
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    plt.figure(figsize=(14, 7))

    plt.plot(
        main_track["frame"],
        main_track["estimated_y"],
        label="Kalman estimate",
    )

    plt.scatter(
        main_track["frame"],
        main_track["measurement_y"],
        s=12,
        label="Associated measurements",
    )

    plt.title(
        "Global trajectory — vertical position"
    )

    plt.xlabel("Frame")
    plt.ylabel("Y position (pixels)")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    y_path = (
        PLOT_DIR
        / "global_tracked_y.png"
    )

    plt.savefig(
        y_path,
        dpi=150,
    )

    plt.close()

    print()
    print("Plots saved:")
    print(x_path)
    print(y_path)


# ============================================================
# SUMMARY
# ============================================================

def print_summary(main_track):

    print()
    print("=" * 70)
    print("GLOBAL TRACKING SUMMARY")
    print("=" * 70)

    if main_track.empty:

        print("No valid ball trajectory found.")
        return

    start_frame = int(
        main_track["frame"].min()
    )

    end_frame = int(
        main_track["frame"].max()
    )

    detections = len(main_track)

    span = end_frame - start_frame + 1

    coverage = (
        detections / span
        if span > 0
        else 0
    )

    print(
        f"Main track frames: {start_frame} → {end_frame}"
    )

    print(
        f"Track span:        {span} frames"
    )

    print(
        f"Measurements:      {detections}"
    )

    print(
        f"Track coverage:    {coverage * 100:.2f}%"
    )

    print()

    print(
        f"Mean candidate quality: "
        f"{main_track['quality'].mean():.3f}"
    )

    print(
        f"Mean diameter: "
        f"{main_track['diameter'].mean():.2f} px"
    )

    print(
        f"Median diameter: "
        f"{main_track['diameter'].median():.2f} px"
    )

    print()

    print(
        "This track is now ready for the next stage:"
    )

    print(
        "3D reconstruction from image geometry."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df_candidates, fps, frame_count = (
        extract_all_candidates()
    )

    if df_candidates.empty:

        raise RuntimeError(
            "No candidates were detected."
        )

    print()
    print("Building temporally consistent trajectories...")

    segments = build_track_segments(
        df_candidates,
        fps,
    )

    print(
        f"Trajectory segments found: {len(segments)}"
    )

    if not segments:

        raise RuntimeError(
            "No valid trajectory segments were found."
        )

    print()
    print("Smoothing trajectory segments with Kalman filter...")

    all_tracks = create_output(
        segments,
        fps,
        frame_count,
    )

    if all_tracks.empty:

        raise RuntimeError(
            "No tracked output was generated."
        )

    main_track = select_main_track(
        all_tracks
    )

    # Save main track.
    main_track.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(
        f"Saved main trajectory:\n{OUTPUT_CSV}"
    )

    print_summary(
        main_track
    )

    create_plots(
        main_track
    )


if __name__ == "__main__":
    main()