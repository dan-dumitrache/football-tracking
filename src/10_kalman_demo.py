import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Measured ball positions from the clean second trajectory.
# ---------------------------------------------------------

measurements = np.array([
    [395, 570.0, 525.0],
    [400, 615.0, 481.0],
    [405, 655.0, 445.0],
    [410, 688.0, 415.0],
    [415, 716.5, 388.0],
    [420, 742.0, 366.0],
    [425, 763.0, 347.0],
    [430, 784.0, 328.0],
    [435, 800.0, 312.0],
    [440, 815.5, 298.0],
    [445, 831.0, 285.0],
    [450, 842.5, 273.0],
    [455, 854.0, 262.0],
    [460, 865.0, 254.0],
])


# ---------------------------------------------------------
# Kalman filter
#
# State:
#
# [x]
# [y]
# [vx]
# [vy]
# ---------------------------------------------------------

kalman = cv2.KalmanFilter(4, 2)

# Measurement matrix:
#
# We observe x and y,
# but not velocity directly.
#
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.float32)


# State transition matrix.
#
# This will be updated for the actual time step below.

kalman.transitionMatrix = np.eye(
    4,
    dtype=np.float32
)


# Process noise:
#
# How much do we allow the actual ball motion
# to deviate from our simple constant-velocity model?

kalman.processNoiseCov = np.eye(
    4,
    dtype=np.float32
) * 0.03


# Measurement noise:
#
# How noisy do we believe our visual measurements are?

kalman.measurementNoiseCov = np.eye(
    2,
    dtype=np.float32
) * 2


# Initial uncertainty.

kalman.errorCovPost = np.eye(
    4,
    dtype=np.float32
) * 10


# Initial state.

first_x = measurements[0, 1]
first_y = measurements[0, 2]

second_x = measurements[1, 1]
second_y = measurements[1, 2]

fps = 30.0

dt_initial = (
    measurements[1, 0] - measurements[0, 0]
) / fps

initial_vx = (second_x - first_x) / dt_initial
initial_vy = (second_y - first_y) / dt_initial

kalman.statePost = np.array([
    [first_x],
    [first_y],
    [initial_vx],
    [initial_vy],
], dtype=np.float32)




filtered_positions = []


# ---------------------------------------------------------
# Run the filter
# ---------------------------------------------------------

previous_frame = measurements[0, 0]

for frame, measured_x, measured_y in measurements:

    dt = (frame - previous_frame) / fps

    # Update state transition matrix.
    kalman.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float32)

    # Prediction.
    prediction = kalman.predict()

    # Camera measurement.
    measurement = np.array([
        [measured_x],
        [measured_y],
    ], dtype=np.float32)

    # Measurement update.
    estimate = kalman.correct(measurement)

    filtered_positions.append([
        frame,
        estimate[0, 0],
        estimate[1, 0],
    ])

    previous_frame = frame


filtered_positions = np.array(filtered_positions)


# ---------------------------------------------------------
# Plot X
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    measurements[:, 0],
    measurements[:, 1],
    "o-",
    label="Measured"
)

plt.plot(
    filtered_positions[:, 0],
    filtered_positions[:, 1],
    "o-",
    label="Kalman estimate"
)

plt.xlabel("Frame")
plt.ylabel("X position (pixels)")
plt.title("Measured vs Kalman-filtered X position")

plt.legend()
plt.grid(True)
plt.tight_layout()

output_path = OUTPUT_DIR / "kalman_x_demo.png"

plt.savefig(output_path, dpi=150)

plt.show()


# ---------------------------------------------------------
# Plot Y
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    measurements[:, 0],
    measurements[:, 2],
    "o-",
    label="Measured"
)

plt.plot(
    filtered_positions[:, 0],
    filtered_positions[:, 2],
    "o-",
    label="Kalman estimate"
)

plt.xlabel("Frame")
plt.ylabel("Y position (pixels)")
plt.title("Measured vs Kalman-filtered Y position")

plt.legend()
plt.grid(True)
plt.tight_layout()

output_path = OUTPUT_DIR / "kalman_y_demo.png"

plt.savefig(output_path, dpi=150)

plt.show()