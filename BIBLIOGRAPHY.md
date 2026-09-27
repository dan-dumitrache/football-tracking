# Bibliography

A short list of references directly relevant to the methods used in this project.

## 1. OpenCV — HSV thresholding

**OpenCV Documentation — Thresholding Operations using `inRange`**

https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html

Used for the red-colour segmentation stage. Covers HSV and `cv.inRange`.

## 2. OpenCV — Morphological operations

**OpenCV Documentation — Eroding and Dilating**

https://docs.opencv.org/4.13.0/db/df6/tutorial_erosion_dilatation.html

Used for cleaning the binary colour mask before extracting candidate regions.

## 3. OpenCV — Camera calibration

**OpenCV Documentation — Camera Calibration**

https://docs.opencv.org/5.0/tutorials/calib3d/camera_calibration/camera_calibration.html

Reference for camera intrinsics, the pinhole model and distortion.

## 4. Hartley & Zisserman — Multiple View Geometry

**Richard Hartley and Andrew Zisserman, _Multiple View Geometry in Computer Vision_, 2nd ed., Cambridge University Press.**

https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6F289C78B2B23F596CAA76D3D43F7A

Especially relevant: projective geometry, camera geometry, estimation and scene planes/homographies.

## 5. OpenCV — Homography

**OpenCV Documentation — Features2D + Homography**

https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html

Used for the practical image-to-image transformations in camera-motion compensation.

## 6. Welch & Bishop — Kalman filter

**Greg Welch and Gary Bishop, _An Introduction to the Kalman Filter_, UNC Chapel Hill, TR 95-041.**

https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

Practical reference for prediction/update and temporal state estimation.

## 7. RealSense — Projection mathematics

**RealSense SDK 2.0 — Projection in RealSense SDK 2.0**

https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/

Relevant to the challenge's RealSense component and camera/3D coordinate conventions.

Note: the final implementation uses monocular RGB estimation rather than actual D435i depth measurements.

## 8. Intel RealSense D435i — Specifications

**Intel RealSense D435i product specifications**

https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html

Used to understand the intended hardware and its published depth capabilities.

## 9. IFAB — Football dimensions

**IFAB, Law 2 — The Ball**

https://www.theifab.com/laws/latest/the-ball/

IFAB specifies a circumference of 68–70 cm for a regulation football. That corresponds to roughly 0.216–0.223 m diameter, so 0.22 m was used as the working value for monocular depth estimation.