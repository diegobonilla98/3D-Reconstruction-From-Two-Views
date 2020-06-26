# 3D-Reconstruction-From-Two-Views
3D reconstruction from two images. Main class from [here](https://www.packtpub.com/eu/data/opencv-4-with-python-blueprints-second-edition).


Simple 3D reconstruction from two images of the same scene from two different viewpoints. There are many many books and articles which explain it way better ([for example(https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)), but I'll just point the steps I took:

1. Find the camera matrix. (Not in the code)
2. Get the keypoints and descriptors using SIFT, SURF, ...
3. Calculate the optical flow using the Lukas-Kanade algorithm.
4. Find the essential matrix using Singular Value Decomposition.
5. Triangulate.
6. Visualize.


## Result

![](ezgif.com-video-to-gif.gif)

