import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_DIR
    / "data"
    / "output"
    / "ball_detections.csv"
)


# ---------------------------------------------------------
# Manually verified ball positions
#
# These are approximate centers from our earlier analysis.
# ---------------------------------------------------------

ground_truth = {
    0:   (732.0, 212.0),
    50:  (733.0, 217.0),
    100: (750.0, 218.0),
    150: (699.0, 222.0),
    200: (652.0, 591.0),
    400: (615.0, 481.0),
    450: (842.5, 273.0),
}


# ---------------------------------------------------------
# Load automatic detections
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)


print()
print("=" * 70)
print("AUTOMATIC DETECTOR vs MANUAL GROUND TRUTH")
print("=" * 70)


errors = []


for frame, (true_x, true_y) in ground_truth.items():

    row = df[df["frame"] == frame]

    if row.empty:
        print(f"\nFrame {frame}: NO DATA")
        continue

    row = row.iloc[0]


    if row["detected"] == 0:

        print(
            f"\nFrame {frame}: "
            f"DETECTION MISSING"
        )

        continue


    detected_x = row["center_x"]
    detected_y = row["center_y"]


    error = np.sqrt(
        (detected_x - true_x) ** 2
        +
        (detected_y - true_y) ** 2
    )


    errors.append(error)


    print(f"\nFrame {frame}")
    print(
        f"  Ground truth: "
        f"({true_x:.1f}, {true_y:.1f})"
    )

    print(
        f"  Detector:     "
        f"({detected_x:.1f}, {detected_y:.1f})"
    )

    print(
        f"  Error:        "
        f"{error:.1f} pixels"
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)


if errors:

    errors = np.array(errors)

    print(
        f"Mean position error: "
        f"{errors.mean():.1f} pixels"
    )

    print(
        f"Median position error: "
        f"{np.median(errors):.1f} pixels"
    )

    print(
        f"Maximum position error: "
        f"{errors.max():.1f} pixels"
    )

    print(
        f"Within 20 px: "
        f"{np.mean(errors <= 20) * 100:.1f}%"
    )

    print(
        f"Within 50 px: "
        f"{np.mean(errors <= 50) * 100:.1f}%"
    )