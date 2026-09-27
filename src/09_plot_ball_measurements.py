import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Manually verified ball measurements
# ---------------------------------------------------------
#
# Each entry is:
# frame, x, y, width, height
#
# x and y are the CENTER of the detected ball.
#

measurements = np.array([
    # Initial / kicking phase
    [140, 695.0, 223.0, 30, 30],
    [145, 690.0, 222.0, 28, 28],
    [150, 699.0, 222.0, 30, 28],
    [155, 725.0, 222.0, 30, 28],
    [160, 755.5, 219.0, 29, 26],
    [165, 800.0, 246.0, 32, 32],
    [170, 865.0, 275.0, 38, 34],
    [180, 933.5, 345.0, 49, 42],
    [190, 797.0, 434.0, 58, 56],
    [195, 732.5, 500.0, 65, 64],
    [200, 652.0, 591.0, 76, 82],
    [205, 557.0, 695.0, 86, 50],

    # Missing observations.
    [210, np.nan, np.nan, np.nan, np.nan],
    [215, np.nan, np.nan, np.nan, np.nan],
    [220, np.nan, np.nan, np.nan, np.nan],
    [225, np.nan, np.nan, np.nan, np.nan],
    [230, np.nan, np.nan, np.nan, np.nan],
    [235, np.nan, np.nan, np.nan, np.nan],
    [240, np.nan, np.nan, np.nan, np.nan],
    [245, np.nan, np.nan, np.nan, np.nan],
    [250, np.nan, np.nan, np.nan, np.nan],
    [255, np.nan, np.nan, np.nan, np.nan],
    [260, np.nan, np.nan, np.nan, np.nan],
    [265, np.nan, np.nan, np.nan, np.nan],
    [270, np.nan, np.nan, np.nan, np.nan],
    [275, np.nan, np.nan, np.nan, np.nan],
    [280, np.nan, np.nan, np.nan, np.nan],
    [285, np.nan, np.nan, np.nan, np.nan],
    [290, np.nan, np.nan, np.nan, np.nan],
    [295, np.nan, np.nan, np.nan, np.nan],
    [300, np.nan, np.nan, np.nan, np.nan],
    [305, np.nan, np.nan, np.nan, np.nan],
    [310, np.nan, np.nan, np.nan, np.nan],
    [315, np.nan, np.nan, np.nan, np.nan],
    [320, np.nan, np.nan, np.nan, np.nan],
    [325, np.nan, np.nan, np.nan, np.nan],
    [330, np.nan, np.nan, np.nan, np.nan],
    [335, np.nan, np.nan, np.nan, np.nan],
    [340, np.nan, np.nan, np.nan, np.nan],
    [345, np.nan, np.nan, np.nan, np.nan],
    [350, np.nan, np.nan, np.nan, np.nan],
    [355, np.nan, np.nan, np.nan, np.nan],
    [360, np.nan, np.nan, np.nan, np.nan],
    [365, np.nan, np.nan, np.nan, np.nan],
    [370, np.nan, np.nan, np.nan, np.nan],
    [375, np.nan, np.nan, np.nan, np.nan],
    [380, np.nan, np.nan, np.nan, np.nan],
    [385, np.nan, np.nan, np.nan, np.nan],
    [390, np.nan, np.nan, np.nan, np.nan],

    [395, 570.0, 525.0, 68, 66],
    [400, 615.0, 481.0, 62, 58],
    [405, 655.0, 445.0, 58, 54],
    [410, 688.0, 415.0, 52, 50],
    [415, 716.5, 388.0, 49, 48],
    [420, 742.0, 366.0, 48, 44],
    [425, 763.0, 347.0, 46, 42],
    [430, 784.0, 328.0, 44, 40],
    [435, 800.0, 312.0, 40, 36],
    [440, 815.5, 298.0, 39, 36],
    [445, 831.0, 285.0, 38, 34],
    [450, 842.5, 273.0, 37, 34],
    [455, 854.0, 262.0, 36, 32],
    [460, 865.0, 254.0, 34, 32],
])


frames = measurements[:, 0]
x = measurements[:, 1]
y = measurements[:, 2]

width = measurements[:, 3]
height = measurements[:, 4]

# Average apparent diameter.
diameter = (width + height) / 2


# ---------------------------------------------------------
# Plot 1: X position
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    frames,
    x,
    "o-"
)

plt.xlabel("Frame")
plt.ylabel("Ball center X (pixels)")
plt.title("Ball horizontal position")

plt.grid(True)

plt.tight_layout()

output_path = OUTPUT_DIR / "ball_x_position.png"

plt.savefig(output_path, dpi=150)

plt.show()

print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Plot 2: Y position
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    frames,
    y,
    "o-"
)

plt.xlabel("Frame")
plt.ylabel("Ball center Y (pixels)")
plt.title("Ball vertical position")

plt.grid(True)

plt.tight_layout()

output_path = OUTPUT_DIR / "ball_y_position.png"

plt.savefig(output_path, dpi=150)

plt.show()

print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Plot 3: Apparent ball diameter
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    frames,
    diameter,
    "o-"
)

plt.xlabel("Frame")
plt.ylabel("Approximate diameter (pixels)")
plt.title("Ball apparent size")

plt.grid(True)

plt.tight_layout()

output_path = OUTPUT_DIR / "ball_apparent_size.png"

plt.savefig(output_path, dpi=150)

plt.show()

print(f"Saved: {output_path}")