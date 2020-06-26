#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module that contains an algorithm for 3D scene reconstruction """

import cv2
import numpy as np
import sys


class SceneReconstruction3D:
    """3D scene reconstruction

        This class implements an algorithm for 3D scene reconstruction using
        stereo vision and structure-from-motion techniques.

        A 3D scene is reconstructed from a pair of images that show the same
        real-world scene from two different viewpoints. Feature matching is
        performed either with rich feature descriptors or based on optic flow.
        3D coordinates are obtained via triangulation.

        Note that a complete structure-from-motion pipeline typically includes
        bundle adjustment and geometry fitting, which are out of scope for
        this project.
    """

    def __init__(self, K, dist):
        """Constructor

            This method initializes the scene reconstruction algorithm.

            :param K: 3x3 intrinsic camera matrix
            :param dist: vector of distortion coefficients
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)  # store inverse for fast access
        self.d = dist

    def load_image_pair(
            self,
            img1: np.array,
            img2: np.array) -> None:

        self.img1, self.img2 = [cv2.undistort(path, self.K, self.d) for path in (img1, img2)]

    def plot_optic_flow(self):
        """Plots optic flow field

            This method plots the optic flow between the first and second
            image.
        """
        self._extract_keypoints_flow()

        img = np.copy(self.img1)
        for pt1, pt2 in zip(self.match_pts1, self.match_pts2):
            cv2.arrowedLine(img, tuple(pt1), tuple(pt2),
                            color=(255, 0, 0))

        return img

    def plot_point_cloud(self, feat_mode="sift"):
        """Plots 3D point cloud

            This method generates and plots a 3D point cloud of the recovered
            3D scene.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("sift") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        # triangulate points
        first_inliers = np.array(self.match_inliers1)[:, :2]
        second_inliers = np.array(self.match_inliers2)[:, :2]
        pts4D = cv2.triangulatePoints(self.Rt1, self.Rt2, first_inliers.T,
                                      second_inliers.T).T

        # convert from homogeneous coordinates to 3D
        pts3D = pts4D[:, :3] / pts4D[:, 3, None]

        # plot with matplotlib
        Xs, Zs, Ys = [pts3D[:, i] for i in range(3)]

        return Xs, Zs, Ys

    def _extract_keypoints(self, feat_mode):
        """Extracts keypoints

            This method extracts keypoints for feature matching based on
            a specified mode:
            - "sift": use rich sift descriptor
            - "flow": use optic flow

            :param feat_mode: keypoint extraction mode ("sift" or "flow")
        """
        # extract features
        if feat_mode.lower() == "sift":
            # feature matching via sift and BFMatcher
            self._extract_keypoints_sift()
        elif feat_mode.lower() == "flow":
            # feature matching via optic flow
            self._extract_keypoints_flow()
        else:
            sys.exit(f"Unknown feat_mode {feat_mode}. Use 'sift' or 'FLOW'")

    def _extract_keypoints_sift(self):
        """Extracts keypoints via sift descriptors"""
        # extract keypoints and descriptors from both images
        # detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.11, edgeThreshold=10)
        detector = cv2.xfeatures2d.SIFT_create()
        first_key_points, first_desc = detector.detectAndCompute(self.img1,
                                                                 None)
        second_key_points, second_desc = detector.detectAndCompute(self.img2,
                                                                   None)
        # match descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L1, True)
        matches = matcher.match(first_desc, second_desc)

        # generate lists of point correspondences
        self.match_pts1 = np.array(
            [first_key_points[match.queryIdx].pt for match in matches])
        self.match_pts2 = np.array(
            [second_key_points[match.trainIdx].pt for match in matches])

    def _extract_keypoints_flow(self):
        """Extracts keypoints via optic flow"""
        # find FAST features
        fast = cv2.FastFeatureDetector_create()
        first_key_points = fast.detect(self.img1)

        first_key_list = [i.pt for i in first_key_points]
        first_key_arr = np.array(first_key_list).astype(np.float32)

        second_key_arr, status, err = cv2.calcOpticalFlowPyrLK(
            self.img1, self.img2, first_key_arr, None)

        # filter out the points with high error
        # keep only entries with status=1 and small error
        condition = (status == 1) * (err < 5.)
        concat = np.concatenate((condition, condition), axis=1)
        first_match_points = first_key_arr[concat].reshape(-1, 2)
        second_match_points = second_key_arr[concat].reshape(-1, 2)

        self.match_pts1 = first_match_points
        self.match_pts2 = second_match_points

    def _find_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.1, 0.99)

    def _find_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """
        self.E = self.K.T.dot(self.F).dot(self.K)

    def _find_camera_matrices_rt(self):
        """Finds the [R|t] camera matrix"""
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        for pt1, pt2, mask in zip(
                self.match_pts1, self.match_pts2, self.Fmask):
            if mask:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.K_inv.dot([pt1[0], pt1[1], 1.0]))
                second_inliers.append(self.K_inv.dot([pt2[0], pt2[1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras

        R = T = None
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]
        for r in (U.dot(W).dot(Vt), U.dot(W.T).dot(Vt)):
            for t in (U[:, 2], -U[:, 2]):
                if self._in_front_of_both_cameras(
                        first_inliers, second_inliers, r, t):
                    R, T = r, t

        assert R is not None, "Camera matricies were never found"

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

    def _draw_epipolar_lines_helper(self, img1, img2, lines, pts1, pts2):
        """Helper method to draw epipolar lines and features """
        if img1.shape[2] == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.shape[2] == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        c = img1.shape[1]
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img1, tuple(pt1), 5, color, -1)
            cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    def _in_front_of_both_cameras(self, first_points, second_points, rot,
                                  trans):
        """Determines whether point correspondences are in front of both
           images"""
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :],
                             trans) / np.dot(rot[0, :] - second[0] * rot[2, :],
                                             second)
            first_3d_point = np.array([first[0] * first_z,
                                       second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
                                                                     trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True

    def _linear_ls_triangulation(self, u1, P1, u2, P2):
        """Triangulation via Linear-LS method"""
        # build A matrix for homogeneous equation system Ax=0
        # assume X = (x,y,z,1) for Linear-LS method
        # which turns it into AX=B system, where A is 4x3, X is 3x1 & B is 4x1
        A = np.array([u1[0] * P1[2, 0] - P1[0, 0], u1[0] * P1[2, 1] - P1[0, 1],
                      u1[0] * P1[2, 2] - P1[0, 2], u1[1] * P1[2, 0] - P1[1, 0],
                      u1[1] * P1[2, 1] - P1[1, 1], u1[1] * P1[2, 2] - P1[1, 2],
                      u2[0] * P2[2, 0] - P2[0, 0], u2[0] * P2[2, 1] - P2[0, 1],
                      u2[0] * P2[2, 2] - P2[0, 2], u2[1] * P2[2, 0] - P2[1, 0],
                      u2[1] * P2[2, 1] - P2[1, 1],
                      u2[1] * P2[2, 2] - P2[1, 2]]).reshape(4, 3)

        B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
                      -(u1[1] * P1[2, 3] - P1[1, 3]),
                      -(u2[0] * P2[2, 3] - P2[0, 3]),
                      -(u2[1] * P2[2, 3] - P2[1, 3])]).reshape(4, 1)

        ret, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        return X.reshape(1, 3)
