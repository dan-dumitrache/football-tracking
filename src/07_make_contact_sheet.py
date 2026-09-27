import cv2
from pathlib import Path
import math


PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_DIR / "data" / "input" / "rgb.avi"
OUTPUT_PATH = PROJECT_DIR / "data" / "output" / "video_contact_sheet.jpg"


video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


frame_numbers = list(range(0, 462, 10))

thumbnails = []

for frame_number in frame_numbers:

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = video.read()

    if not success:
        continue

    # Resize while preserving aspect ratio.
    thumbnail_width = 256
    scale = thumbnail_width / frame.shape[1]

    thumbnail_height = int(frame.shape[0] * scale)

    thumbnail = cv2.resize(
        frame,
        (thumbnail_width, thumbnail_height)
    )

    # Add frame number at the top.
    cv2.rectangle(
        thumbnail,
        (0, 0),
        (180, 30),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        thumbnail,
        f"Frame {frame_number}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    thumbnails.append(thumbnail)


video.release()


# Grid configuration.
columns = 5
rows = math.ceil(len(thumbnails) / columns)

cell_width = 256
cell_height = thumbnails[0].shape[0]

sheet = cv2.imread(
    str(OUTPUT_PATH)
) if OUTPUT_PATH.exists() else None

# Create blank contact sheet.
sheet = 255 * (
    __import__("numpy").ones(
        (rows * cell_height, columns * cell_width, 3),
        dtype=__import__("numpy").uint8
    )
)

for index, thumbnail in enumerate(thumbnails):

    row = index // columns
    column = index % columns

    x = column * cell_width
    y = row * cell_height

    sheet[
        y:y + cell_height,
        x:x + cell_width
    ] = thumbnail


cv2.imwrite(str(OUTPUT_PATH), sheet)

print("Contact sheet created.")
print(f"Saved to: {OUTPUT_PATH}")