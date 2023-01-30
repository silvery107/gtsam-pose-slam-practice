# GTSAM Pose SLAM Practice
This repo is a small pratice and example to solve the pose graph SLAM problem using the GTSAM library on INTEL and Garage dataset provied by [Dr. Luca Carlone](https://lucacarlone.mit.edu/datasets/). In cases below, both batch (Gauss-Newton) and incremental (iSAM) solutions are presented.

A detailed report can be found [here](docs/HW_SLAM.pdf).

## Quick Start
Run
```
python hw_slam.py # All optimized trajectory figures are saved under figures/
```


## 2D Pose SLAM
### Intel Lab Dataset
<img src="figures/INTEL_eg2o-300x267.jpg" width="200">

### Optimization Result
<img src="figures/solve_pose_slam_2d_incremental.png" width="400">

## 3D Pose SLAM
### Garage Dataset
<img src="figures/parking-garage-300x155.png" width="250">

### Optimization Result
<img src="figures/solve_pose_slam_3d_batch.png" width="500">

<img src="figures/solve_pose_slam_3d_batch_x_y.png" width="400">

## Dependencies
- Python 3.8 or above
- See [requirements.txt](requirements.txt)