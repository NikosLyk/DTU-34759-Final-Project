# DTU-34759-Final-Project

# Final Project Perception

## 2. Stereo Camera Calibration --> Nikos & Elena

**Goal: Obtain intrinsics, extrinsics, and the Q matrix.**

2.1. Load the calibration pattern images (left and right).
2.2. Detect chessboard corners on both images (`findChessboardCorners`).
2.3. Compute:

* Left and right intrinsic matrices
* Distortion coefficients
* Rotation and translation between cameras

2.4. Compute stereo rectification:

* `stereoRectify`
* `initUndistortRectifyMap`
* Apply remapping to raw stereo pairs

2.5. Compare your rectification with the provided rectified images:

* Check horizontal alignment of epipolar lines
* Measure vertical disparity difference
* Visual difference in disparity map quality

---

## 3. Disparity Map and 3D Reconstruction --> Elena & Nikos

**Goal: Derive depth for 3D tracking.**

3.1. Use rectified images to compute disparity (SGBM recommended).

3.2. Generate disparity map for each stereo pair.

3.3. Convert disparity to 3D coordinates using the Q matrix (`reprojectImageTo3D`).

3.4. Optionally store point clouds or per-pixel 3D coordinates.

---

## 4. Object Detection (pedestrians / cyclists / cars) --> Lara

4.1. Choose a detection method:

* Recommended: YOLOv8 / YOLOv9 trained on your own dataset
* Traditional methods (HOG, cascades) are possible but less robust

4.2. Apply detection to:

* Sequence 1
* Sequence 2
* Sequence 3

4.3. For each bounding box, estimate object depth:

* Take median or mean depth value inside the bounding box
* Store `(x, y, z)` for tracking

---

## 5. 3D Tracking --> Simon

**Goal: Maintain tracks even under occlusions.**

5.1. Represent each detection as a 3D state:

```
(x, y, z, vx, vy, vz)
```

5.2. Set up a 3D Kalman Filter per tracked object.
5.3. Perform data association:

* IoU in 2D + 3D distance
  or
* Hungarian algorithm

5.4. Handle occlusions:

* If no detection → use Kalman prediction only
* Keep track alive for N frames before deleting it

5.5. Save trajectories for visualization and evaluation.

---

## 6. Classification System (Machine Learning) --> Francisca

**Goal: Classify images into 3 classes using your own training set.**

6.1. Build your training set:

* Collect images online or capture your own
* Create folders: `pedestrian/`, `cyclist/`, `car/`

6.2. Preprocess:

* Resize
* Normalize
* Data augmentation

6.3. Train your model:

* CNN or pretrained model (ResNet / MobileNet)
* Fine-tune with your dataset

6.4. Validate using sequence 1 and sequence 2.
6.5. Test using sequence 3.
6.6. Generate confusion matrix and accuracy metrics.

---

## 7. Full Pipeline Integration

**Goal: Build a real-time inference pipeline.**

For every frame:

1. (If raw images) Apply rectification
2. Compute disparity
3. Reconstruct 3D
4. Detect objects
5. Classify each detected object
6. Track using Kalman filter
7. Render:

   * Bounding boxes
   * Class label
   * Depth
   * Trajectory
8. Export the full annotated video

---

## 8. Evaluation

**Using ground truth from sequences 1 and 2**

### Detection:

* Precision
* Recall
* Mean IoU

### Depth estimation:

* Absolute depth error vs ground truth

### Tracking:

* MOTA / MOTP or custom metrics

### Classification:

* Confusion matrix
* Overall accuracy

