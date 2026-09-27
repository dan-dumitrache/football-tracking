import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_DIR / "data" / "input" / "rgb.avi"
CANDIDATES_PATH = PROJECT_DIR / "data" / "output" / "all_ball_candidates.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "candidate_visualization"
OUTPUT_VIDEO = OUTPUT_DIR / "candidate_visualization.mp4"

# Frames we want to inspect.
INSPECTION_FRAMES = [
    20,
    50,
    100,
    140,
    180,
    200,
    250,
    300,
    350,
    390,
    410,
    430,
    450,
]

# Maximum number of candidates displayed on a frame.
MAX_CANDIDATES = 12

# Candidate selection radius.
#
# The candidate closest to the selected/expected position is
# highlighted as the "best candidate".
#
# This is ONLY for visualization. It does not modify the tracker.
MAX_HIGHLIGHT_DISTANCE = 100.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):
    """Convert a value to float without crashing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_candidate_columns(df):
    """
    Try to identify the important columns in the candidate CSV.

    Different versions of the candidate generator may use slightly
    different names, so we support several common variants.
    """

    possible_x = [
        "center_x",
        "measurement_x",
        "x",
        "cx",
    ]

    possible_y = [
        "center_y",
        "measurement_y",
        "y",
        "cy",
    ]

    possible_w = [
        "width",
        "w",
    ]

    possible_h = [
        "height",
        "h",
    ]

    possible_diameter = [
        "diameter",
        "size",
    ]

    possible_quality = [
        "quality",
        "candidate_quality",
        "score",
    ]

    def find_column(possible_names, required=True):
        for name in possible_names:
            if name in df.columns:
                return name

        if required:
            raise ValueError(
                f"Could not find any of these columns: {possible_names}\n"
                f"Available columns:\n{list(df.columns)}"
            )

        return None

    return {
        "x": find_column(possible_x),
        "y": find_column(possible_y),
        "width": find_column(possible_w, required=False),
        "height": find_column(possible_h, required=False),
        "diameter": find_column(possible_diameter, required=False),
        "quality": find_column(possible_quality, required=False),
    }


def calculate_diameter(row, columns):
    """
    Return candidate diameter.

    Priority:
        1. explicit diameter
        2. mean(width, height)
        3. width
        4. height
    """

    diameter_column = columns["diameter"]

    if diameter_column is not None:
        value = safe_float(row[diameter_column])

        if np.isfinite(value) and value > 0:
            return value

    width_column = columns["width"]
    height_column = columns["height"]

    width = np.nan
    height = np.nan

    if width_column is not None:
        width = safe_float(row[width_column])

    if height_column is not None:
        height = safe_float(row[height_column])

    if np.isfinite(width) and np.isfinite(height):
        return (width + height) / 2.0

    if np.isfinite(width):
        return width

    if np.isfinite(height):
        return height

    return np.nan


def draw_candidate(
    frame,
    x,
    y,
    diameter,
    candidate_index,
    quality=None,
    highlighted=False,
):
    """
    Draw one candidate on the image.
    """

    x_int = int(round(x))
    y_int = int(round(y))

    if np.isfinite(diameter) and diameter > 0:
        radius = max(5, int(round(diameter / 2.0)))
    else:
        radius = 10

    # Different appearance for the highlighted candidate.
    if highlighted:
        thickness = 4
        circle_radius = radius + 5
        label = f"BEST #{candidate_index}"
    else:
        thickness = 2
        circle_radius = radius
        label = f"#{candidate_index}"

    cv2.circle(
        frame,
        (x_int, y_int),
        circle_radius,
        (0, 255, 255),
        thickness,
    )

    # Center point
    cv2.circle(
        frame,
        (x_int, y_int),
        3,
        (0, 255, 255),
        -1,
    )

    # Candidate label
    text_x = x_int + circle_radius + 5
    text_y = y_int - circle_radius - 5

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Additional information for the highlighted candidate.
    if highlighted:
        info = f"x={x:.1f} y={y:.1f}"

        if np.isfinite(diameter):
            info += f" d={diameter:.1f}"

        if quality is not None and np.isfinite(quality):
            info += f" q={quality:.2f}"

        cv2.putText(
            frame,
            info,
            (text_x, text_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_header(frame, frame_number, candidate_count):
    """
    Draw diagnostic information at the top of the frame.
    """

    header = (
        f"FRAME {frame_number}    "
        f"CANDIDATES: {candidate_count}"
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 42),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        header,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_candidate_table(frame, candidates, columns):
    """
    Draw a small table in the upper-right area showing candidate
    positions and sizes.
    """

    if len(candidates) == 0:
        return

    x_start = frame.shape[1] - 330
    y_start = 60

    # Background panel
    panel_height = min(
        frame.shape[0] - 80,
        35 + len(candidates) * 25,
    )

    cv2.rectangle(
        frame,
        (x_start - 10, y_start - 25),
        (frame.shape[1] - 10, y_start + panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        "Candidates",
        (x_start, y_start),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for i, (_, row) in enumerate(candidates.iterrows()):

        x = safe_float(row[columns["x"]])
        y = safe_float(row[columns["y"]])

        diameter = calculate_diameter(row, columns)

        quality = None

        if columns["quality"] is not None:
            quality = safe_float(row[columns["quality"]])

        if np.isfinite(diameter):
            diameter_text = f"{diameter:5.1f}"
        else:
            diameter_text = "  -- "

        if quality is not None and np.isfinite(quality):
            quality_text = f"{quality:4.2f}"
        else:
            quality_text = " -- "

        text = (
            f"#{i:<2} "
            f"x={x:6.1f} "
            f"y={y:6.1f} "
            f"d={diameter_text} "
            f"q={quality_text}"
        )

        y_text = y_start + 25 + i * 25

        cv2.putText(
            frame,
            text,
            (x_start, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CANDIDATE VISUALIZATION")
    print("=" * 70)

    print()
    print(f"Video:      {VIDEO_PATH}")
    print(f"Candidates: {CANDIDATES_PATH}")
    print()

    # --------------------------------------------------------
    # Load candidate CSV
    # --------------------------------------------------------

    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"Candidate CSV not found:\n{CANDIDATES_PATH}"
        )

    candidates_df = pd.read_csv(CANDIDATES_PATH)

    print("Candidate CSV columns:")
    print(list(candidates_df.columns))
    print()

    if "frame" not in candidates_df.columns:
        raise ValueError(
            "The candidate CSV must contain a 'frame' column."
        )

    columns = get_candidate_columns(candidates_df)

    print("Using columns:")
    print(f"  X:        {columns['x']}")
    print(f"  Y:        {columns['y']}")
    print(f"  Width:    {columns['width']}")
    print(f"  Height:   {columns['height']}")
    print(f"  Diameter: {columns['diameter']}")
    print(f"  Quality:  {columns['quality']}")
    print()

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video:")
    print(f"  Resolution: {width} x {height}")
    print(f"  FPS:        {fps:.2f}")
    print(f"  Frames:     {frame_count}")
    print()

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create output video
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video:\n{OUTPUT_VIDEO}"
        )

    # --------------------------------------------------------
    # Process only selected frames
    # --------------------------------------------------------

    selected_frames = set(INSPECTION_FRAMES)

    print("Inspection frames:")
    print(sorted(selected_frames))
    print()

    for frame_number in sorted(selected_frames):

        if frame_number < 0 or frame_number >= frame_count:
            print(
                f"Skipping frame {frame_number}: "
                f"outside video range."
            )
            continue

        # Seek to requested frame.
        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        success, frame = video.read()

        if not success:
            print(
                f"Could not read frame {frame_number}"
            )
            continue

        # ----------------------------------------------------
        # Get candidates belonging to this frame
        # ----------------------------------------------------

        frame_candidates = candidates_df[
            candidates_df["frame"] == frame_number
        ].copy()

        # Sort by quality if available.
        if columns["quality"] is not None:

            frame_candidates["_sort_quality"] = pd.to_numeric(
                frame_candidates[columns["quality"]],
                errors="coerce",
            )

            frame_candidates = frame_candidates.sort_values(
                "_sort_quality",
                ascending=False,
            )

        # Keep only the best N candidates.
        frame_candidates = frame_candidates.head(
            MAX_CANDIDATES
        )

        # ----------------------------------------------------
        # Determine highlighted candidate
        # ----------------------------------------------------

        highlighted_index = None

        if len(frame_candidates) > 0:

            # At this stage we primarily use candidate quality.
            #
            # This is NOT claiming that the best-quality candidate
            # is the football. It simply makes the diagnostic image
            # easier to inspect.

            if columns["quality"] is not None:

                quality_values = pd.to_numeric(
                    frame_candidates[columns["quality"]],
                    errors="coerce",
                )

                if quality_values.notna().any():
                    highlighted_index = int(
                        quality_values.idxmax()
                    )

            if highlighted_index is None:
                highlighted_index = frame_candidates.index[0]

        # ----------------------------------------------------
        # Draw candidates
        # ----------------------------------------------------

        for display_index, (original_index, row) in enumerate(
            frame_candidates.iterrows()
        ):

            x = safe_float(row[columns["x"]])
            y = safe_float(row[columns["y"]])

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            diameter = calculate_diameter(
                row,
                columns,
            )

            quality = None

            if columns["quality"] is not None:
                quality = safe_float(
                    row[columns["quality"]]
                )

            is_highlighted = (
                original_index == highlighted_index
            )

            draw_candidate(
                frame,
                x,
                y,
                diameter,
                display_index,
                quality=quality,
                highlighted=is_highlighted,
            )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        draw_header(
            frame,
            frame_number,
            len(frame_candidates),
        )

        # ----------------------------------------------------
        # Candidate table
        # ----------------------------------------------------

        draw_candidate_table(
            frame,
            frame_candidates,
            columns,
        )

        # ----------------------------------------------------
        # Save individual frame
        # ----------------------------------------------------

        image_path = (
            OUTPUT_DIR
            / f"frame_{frame_number:04d}.jpg"
        )

        cv2.imwrite(
            str(image_path),
            frame,
        )

        # ----------------------------------------------------
        # Add to diagnostic video
        # ----------------------------------------------------

        writer.write(frame)

        print(
            f"Frame {frame_number:3d}: "
            f"{len(frame_candidates):2d} candidates"
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    video.release()
    writer.release()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print("Individual frames saved to:")
    print(OUTPUT_DIR)

    print()
    print("Diagnostic video saved to:")
    print(OUTPUT_VIDEO)


if __name__ == "__main__":
    main()