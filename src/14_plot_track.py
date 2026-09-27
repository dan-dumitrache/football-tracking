import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_DIR
    / "data"
    / "output"
    / "tracked_ball.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "output"
)


df = pd.read_csv(CSV_PATH)


# ---------------------------------------------------------
# X position
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["frame"],
    df["estimated_x"],
    label="Kalman estimate"
)

measured = df[
    df["detected"] == 1
]

plt.scatter(
    measured["frame"],
    measured["measurement_x"],
    s=10,
    label="Associated measurements"
)

plt.xlabel("Frame")
plt.ylabel("X position (pixels)")
plt.title("Tracked ball — horizontal position")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "tracked_x.png",
    dpi=150
)

plt.show()


# ---------------------------------------------------------
# Y position
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["frame"],
    df["estimated_y"],
    label="Kalman estimate"
)

plt.scatter(
    measured["frame"],
    measured["measurement_y"],
    s=10,
    label="Associated measurements"
)

plt.xlabel("Frame")
plt.ylabel("Y position (pixels)")
plt.title("Tracked ball — vertical position")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "tracked_y.png",
    dpi=150
)

plt.show()