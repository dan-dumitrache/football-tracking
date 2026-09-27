# Football Tracking Challenge

A computer-vision solution for detecting, tracking and reconstructing a football from an RGB video.

The project was deliberately kept understandable and reproducible: OpenCV + NumPy/Pandas, without a deep-learning detector.

## What the project does

The pipeline covers:

1. **2D football detection** – detects the red football in the RGB video and produces an annotated video.
2. **3D football estimation** – estimates the ball position relative to the camera using its apparent size and a pinhole-camera model.
3. **Trajectory tracking** – keeps a temporally consistent ball trajectory and handles periods where the ball leaves the image.
4. **2D top-view map** – projects the ground-contact trajectory onto a metric ground plane and compensates for camera movement.

The final output combines these results into one video.

## Input

- Video: `data/input/rgb.avi`
- Resolution: 1280 × 720
- FPS: 30
- Frames: 462
- Duration: ~15.4 s

## Main pipeline

```text
RGB video
   |
   v
HSV red segmentation
   |
   v
Morphology + contours + candidate filtering
   |
   v
Temporal ball tracking
   |
   +----------------------+
   |                      |
   v                      v
2D trajectory        Ball apparent diameter
                           |
                           v
                    Monocular 3D estimate
                    X / Y / Z / distance
                           |
                           v
                    Ground-contact point
                           |
                           v
                 Camera-motion estimation
                           |
                           v
                 Frame-to-reference homography
                           |
                           v
                 Camera-motion compensation
                           |
                           v
                    Ground-plane mapping
                           |
                           v
                       Top view
```

## Important implementation choices

### 2D detection

The football is red and the video has a controlled visual environment, so HSV thresholding is a good first solution. Candidates are filtered using geometry and temporal consistency rather than treating every red blob as the ball.

Main cues:
- area
- bounding-box aspect ratio
- circularity
- apparent diameter
- temporal continuity

### 3D estimation

The ball diameter is used as a known physical reference.

For a pinhole camera:

```text
Z ≈ f D / d
```

where `D` is the assumed real football diameter, `d` is its measured pixel diameter, `f` is focal length in pixels, and `Z` is estimated depth.

Then the image coordinates are converted to camera-relative X/Y using the camera intrinsics.

This is a **monocular estimate**, not a measurement from the D435i depth sensor.

### Ground plane

Four floor-tile corners were used to define an image-to-ground homography.

The working tile-size assumption is:

```text
0.60 m × 0.60 m
```

This value was not independently measured, so metric ground distances should be treated as estimates.

### Camera motion

The original top view exposed an important problem: the camera was not actually stationary.

Static floor features were used to estimate frame-to-reference homographies with feature matching and RANSAC. The resulting transforms were used to compensate the football trajectory before projecting it onto the ground plane.

## Important observations

The ball is visible in two main segments:

- Track 0: frames 19–205
- Track 1: frames 389–461

The ball is out of view between frames 206–388. The pipeline intentionally does **not** invent positions during that gap.

The final reconstruction contains 259 observations.

## Validation

The project includes several validation steps rather than relying only on visual inspection.

### 3D reliability

A 20-pixel apparent-diameter threshold was used for the main 3D confidence classification:

- 225 / 259 observations: acceptable 3D confidence
- 34 / 259 observations: low 3D confidence

Small apparent ball sizes produce noticeably larger depth uncertainty.

### Camera-motion compensation

Static scene features provide an independent check of the compensation.

The measured residual displacement dropped from roughly 29 px before compensation to roughly 1.2 px after compensation.

This is useful because the validation does not depend on the football itself: stationary parts of the scene should remain stationary.

### Final audit

Step 34 performs a read-only audit of the final project.

Current status:

```text
Required checks: 37 PASS
Required failures: 0
Optional failures: 0
Overall: PASS
```

## Final outputs

Main final video:

```text
data/output/final/final_integrated_video_camera_compensated.mp4
```

Original integrated video, kept unchanged:

```text
data/output/final/final_integrated_video.mp4
```

Final audit:

```text
data/output/final_audit/final_project_audit.json
data/output/final_audit/final_project_audit.txt
```

## Limitations

- The football detector uses scene-specific colour/shape cues, so it is not a general-purpose football detector.
- The 3D reconstruction is monocular and is sensitive to ball-size measurement and camera intrinsics.
- The 0.60 m tile size is an assumption and determines the metric scale of the ground map.
- The project does not use actual RealSense depth frames.
- Camera-motion compensation assumes that the visible floor behaves approximately as a plane.
- Compensation stabilizes the ground-plane view; it does not turn monocular reconstruction into ground-truth 3D.

## Reproducibility

The project was developed with Python 3.13, OpenCV, NumPy, SciPy, Pandas, Matplotlib and OpenPyXL in a local virtual environment.

The video is excluded from Git because of its size.

See `BIBLIOGRAPHY.md` for the technical references.