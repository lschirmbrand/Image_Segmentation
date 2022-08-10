import numpy as np
from sfm import *


class Baseline:
    def __init__(self, img_con1, img_con2, match_object):
        self._img_con1 = img_con1
        self._img_con2 = img_con2
        self._img_con1.set_mat_rot(np.eye(3, 3))
        self._match_object = match_object

    def get_pose(self, K):
        """ Calculates the rotation & translation for the second image and returns it """

        F = remove_outliers_with_f(self._img_con1, self._img_con2, self._match_object)
        E = K.T @ F @ K  # compute essential matrix

        return self.validate_pose(E, K)

    def validate_pose(self, E, K):
        """ Retrieves rotation & translation from E and verifying the solutions """

        rot1, rot2, trl1, trl2 = get_camera_by_E(E)

        if not check_determinant(rot1):
            rot1, rot2, trl1, trl2 = get_camera_by_E(-E)  # change sign of E

        # check the 4 solutions

        # solution 1
        reprojection_error, points_3d = self.triangulate(K, rot1, trl1)
        if reprojection_error > 100.0 or not check_triangulate(points_3d, np.hstack((rot1, trl1))):
            # solution 2
            reprojection_error, points_3d = self.triangulate(K, rot1, trl2)
            if reprojection_error > 100.0 or not check_triangulate(points_3d, np.hstack((rot1, trl2))):
                # solution 3
                reprojection_error, points_3d = self.triangulate(K, rot2, trl1)
                if reprojection_error > 100.0 or not check_triangulate(points_3d, np.hstack((rot2, trl1))):
                    # solution 4
                    return rot2, trl1
                else:
                    return rot2, trl2
            else:
                return rot1, trl2
        else:
            return rot1, trl1

    def triangulate(self, K, R, t):
        """ Triangulates points between baseline views and calculate mean reprojection error of triangulation """

        K_inv = np.linalg.inv(K)
        p1 = np.hstack((self._img_con1.get_mat_rot(), self._img_con1.get_vec_trl()))
        p2 = np.hstack((R, t))

        # only reconstruct the inlier points after filtering by using F
        px_points1, px_points2 = get_keypoints_by_indices(keypoints1=self._img_con1.get_keypoints(),
                                                          indices1=self._match_object.get_inliers1(),
                                                          keypoints2=self._img_con2.get_keypoints(),
                                                          indices2=self._match_object.get_inliers2())
        # convert pixel into voxels in homogeneous coordinates
        vx_points1 = cv.convertPointsToHomogeneous(px_points1)[:, 0, :]
        vx_points2 = cv.convertPointsToHomogeneous(px_points2)[:, 0, :]

        reprojection_errors = []

        points_3d = np.zeros((0, 3))

        for i in range(len(vx_points1)):
            u1 = vx_points1[i, :]
            u2 = vx_points2[i, :]

            # convert hom. coords to normalized device coords
            u1_normalized = K_inv.dot(u1)
            u2_normalized = K_inv.dot(u1)

            point_3d = get_3d_point(u1_normalized, p1, u2_normalized, p2)

            # calc and append reprojection error

            error = calculate_reprojection_error(point_3d, u2[0:2], K, R, t)

            reprojection_errors.append(error)

            # concatenate points
            points_3d = np.concatenate((points_3d, point_3d.T), axis=0)

        return np.mean(reprojection_errors), points_3d


def get_3d_point(u1_normalized, p1, u2_normalized, p2):
    """ Computes 3D point with homo. 2D points & respective camera matrices """

    mat_a = np.array([[u1_normalized[0] * p1[2, 0] - p1[0, 0], u1_normalized[0] * p1[2, 1] - p1[0, 1],
                       u1_normalized[0] * p1[2, 2] - p1[0, 2]],
                      [u1_normalized[1] * p1[2, 0] - p1[1, 0], u1_normalized[1] * p1[2, 1] - p1[1, 1],
                       u1_normalized[1] * p1[2, 2] - p1[1, 2]],
                      [u2_normalized[0] * p2[2, 0] - p2[0, 0], u2_normalized[0] * p2[2, 1] - p2[0, 1],
                       u2_normalized[0] * p2[2, 2] - p2[0, 2]],
                      [u2_normalized[1] * p2[2, 0] - p2[1, 0], u2_normalized[1] * p2[2, 1] - p2[1, 1],
                       u2_normalized[1] * p2[2, 2] - p2[1, 2]]])

    mat_b = np.array([-(u1_normalized[0] * p1[2, 3] - p1[0, 3]),
                      -(u1_normalized[1] * p1[2, 3] - p1[1, 3]),
                      -(u2_normalized[0] * p2[2, 3] - p2[0, 3]),
                      -(u2_normalized[1] * p2[2, 3] - p2[1, 3])])

    x = cv.solve(mat_a, mat_b, flags=cv.DECOMP_SVD)
    return x[1]


def calculate_reprojection_error(point_3d, point_2d, K, R, t):
    """ Calc reprojection error for 3D point by projecting it back into the image plane"""

    reprojected_point = K.dot(R.dot(point_3d) + t)
    reprojected_point = cv.convertPointsFromHomogeneous(reprojected_point.T)[:, 0, :].T
    error = np.linalg.norm(point_2d.reshape((2, 1)) - reprojected_point)
    return error


def check_triangulate(points_3d, P):
    """ Checks position of reconstructed points located in front of camera """

    P = np.vstack((P, np.array([0, 0, 0, 1])))
    reprojected_points = cv.perspectiveTransform(src=points_3d[np.newaxis], m=P)
    z = reprojected_points[0, :, -1]
    if (np.sum(z > 0) / z.shape[0]) < 0.75:
        return False
    else:
        return True
