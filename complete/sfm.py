import ImageContainer
import match
import os
from Baseline import Baseline
from Baseline import *
import numpy as np
import cv2 as cv
import open3d as o3d


class SFM:
    def __init__(self, image_container: list[ImageContainer], matches, K, path):
        self._image_container = image_container
        self._matches = matches
        self._names = list[str]
        self._constructed = []
        self.K = K
        self._points_3d = np.zeros((0, 3))
        self._point_counter = 0
        self._point_map = {}
        self._errors = []
        self._results_path = path

        for con in image_container:
            self._names.append(con.get_name())

    def reconstruct(self):

        baseline1, baseline2 = self._image_container[0], self._image_container[1]
        self.compute_pose_estimation(img_con1=baseline1, img_con2=baseline2, is_baseline=True)
        self.plot_points()

        for i in range(2, len(self.views)):
            self.compute_pose_estimation(img_con1=self._image_container[i])
            self.plot_points()

    def compute_pose_estimation(self, img_con1, img_con2=None, is_baseline=False):
        if is_baseline and img_con2:
            match_object = self._matches[(img_con1.get_name(), img_con2.get_name())]
            baseline_pose = Baseline(img_con1, img_con2, match_object)
            rot, trl = baseline_pose.get_pose(self.K)
            img_con2.set_mat_rot(rot)
            img_con2.set_vec_trl(trl)

            rpe1, rpe2 = self.triangulate(img_con1, img_con2)
            self._errors.append(np.mean(rpe1))
            self._errors.append(np.mean(rpe2))

            self._constructed.append(np.mean(img_con1))
            self._constructed.append(np.mean(img_con2))

        # procedure for estimating the pose of all other views
        else:

            rot, trl = self.compute_pose_pnp(img_con1)
            img_con1.set_mat_rot(rot)
            img_con1.set_vec_trl(trl)
            errors = []

            # reconstruct unreconstructed points from all of the previous views
            for i, old_view in enumerate(self._constructed):
                match_object = self._matches[(old_view.name, img_con1.get_name())]
                _ = remove_outliers_with_f(old_view, img_con1, match_object)
                self.remove_mapped_points(match_object, i)
                _, rpe = self.triangulate(old_view, img_con1)
                errors += rpe

            self._constructed.append(img_con1)
            self._errors.append(np.mean(errors))

    def triangulate(self, img_con1, img_con2):
        """ Triangulates 3D points from two views whose poses have been recovered. Also updates the point_map
        dictionary """

        K_inv = np.linalg.inv(self.K)
        p1 = np.hstack((img_con1.get_mat_rot(), img_con1.get_vec_trl()))
        p2 = np.hstack((img_con2.get_mat_rot(), img_con2.get_vec_trl()))

        match_object = self._matches[(img_con1.get_name(), img_con2.get_name())]
        px_points1, px_points2 = get_keypoints_by_indices(keypoints1=img_con1.get_keypoints(),
                                                          indices1=match_object.get_inliers1(),
                                                          keypoints2=img_con2.get_keypoints(),
                                                          indices2=match_object.get_inliers2())
        px_points1 = cv.convertPointsToHomogeneous(px_points1)[:, 0, :]
        px_points2 = cv.convertPointsToHomogeneous(px_points2)[:, 0, :]
        reprojection_error1 = []
        reprojection_error2 = []

        for i in range(len(px_points1)):
            u1 = px_points1[i, :]
            u2 = px_points2[i, :]

            u1_normalized = K_inv.dot(u1)
            u2_normalized = K_inv.dot(u2)

            point_3d = get_3d_point(u1_normalized, p1, u2_normalized, p2)
            self._points_3d = np.concatenate((self._points_3d, point_3d.T), axis=0)

            error1 = calculate_reprojection_error(point_3d, u1[0:2], self.K, img_con1.get_mat_rot(),
                                                  img_con1.get_vec_trl())
            reprojection_error1.append(error1)

            error2 = calculate_reprojection_error(point_3d, u2[0:2], self.K, img_con2.get_mat_rot(),
                                                  img_con2.get_vec_trl())
            reprojection_error2.append(error2)

            # updates point_map with the key (index of view, index of point in the view) and value point_counter
            # multiple keys can have the same value because a 3D point is reconstructed using 2 points
            self._point_map[(self.get_index_of_img_con(img_con1), match_object.inliers1[i])] = self._point_counter
            self._point_map[(self.get_index_of_img_con(img_con2), match_object.inliers2[i])] = self._point_counter
            self._point_counter += 1

        return reprojection_error1, reprojection_error2

    def get_index_of_img_con(self, img_con):
        return self._names.index(img_con.get_name())

    def compute_pose_pnp(self, img_con1):
        """ Computes pose of new view with perspective n-point """

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crosscheck=False)

        # collects all the descriptors of the reconstructed views

        old_descriptors = []
        for old_img in self._constructed:
            old_descriptors.append(old_img.get_descriptors())

        # match old with new descriptors
        matcher.add(old_descriptors)
        matcher.train()
        matches = matcher.match(queryDescriptors=img_con1.get_descriptors())
        points_3d, points_2d = np.zeros((0, 3)), np.zeros((0, 2))

        # build corresponding descriptors array of 2d & 3d points
        for match in matches:
            old_img_idx, new_img_kp_idx, old_img_kp_idx = match.imgIdx, match.queryIdx, match.trainIdx

            if (old_img_idx, old_img_kp_idx) in self._point_map:
                # obtain 2d point from the match
                point_2d = np.array(img_con1.get_keypoints()[new_img_kp_idx].pt).T.reshape((1, 2))
                points_2d = np.concatenate((points_2d, point_2d), axis=0)

                # same with 3d point from point map
                point_3d = self._points_3d[self._point_map[(old_img_idx, old_img_kp_idx)], :].T.reshape((1, 3))
                points_3d = np.concatenate((points_3d, point_3d), axis=0)

        # compute new pose using solvePnPRansac
        _, R, t, _ = cv.solvePnPRansac(points_3d[:, np.newaxis], points_2d[:, np.newaxis], self.K, None,
                                       confidence=0.99, reprojectionError=8.0, flags=cv.SOLVEPNP_DLS)

        R, _ = cv.Rodrigues(R)
        return R, t

    def remove_mapped_points(self, match_object, image_idx):
        """Removes points which already been reconstructed """

        inliers1 = [], inliers2 = []

        for i in range(len(match_object.get_inliers1())):
            if (image_idx, match_object.get_inliers1()[i]) not in self._point_map:
                inliers1.append(match_object.get_inliers1()[i])
                inliers2.append(match_object.get_inliers2()[i])

        match_object.set_inliers1(inliers1)
        match_object.set_inliers2(inliers2)

    def plot_points(self):
        """Saves the reconstructed 3D points to ply files using Open3D"""

        number = len(self._constructed)
        filename = os.path.join(self._results_path, str(number) + '_images.ply')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._points_3d)
        o3d.io.write_point_cloud(filename, pcd)


def remove_outliers_with_f(img_con1: ImageContainer, img_con2: ImageContainer, match_object):
    """ Uses the Fundamental Matrix to remove outliers """

    px_points1, px_points2 = get_keypoints_by_indices(keypoints1=img_con1.get_keypoints(),
                                                      indices1=match_object.get_indices1(),
                                                      keypoints2=img_con2.get_keypoints(),
                                                      indices2=match_object.get_indices2())

    fund_matrix, mask = cv.findFundamentalMat(px_points1, px_points2, method=cv.RANSAC, ransacReprojThreshold=0.9,
                                              confidence=0.99)
    mask = mask.astype(bool).flatten()
    match_object.set_inliers1(np.array(match_object.get_indices1)[mask])
    match_object.set_inliers2(np.array(match_object.get_indices2)[mask])

    return fund_matrix


def get_keypoints_by_indices(keypoints1, keypoints2, indices1, indices2):
    """ Filters keypoints on basis of the given index list """

    p1 = np.array([kp.pt for kp in keypoints1])[indices1]
    p2 = np.array([kp.pt for kp in keypoints2])[indices2]
    return p1, p2


def get_camera_by_E(E):
    """ Calculates rotation & translation from E """

    W = np.array([0, -1, 0], [1, 0, 0], [0, 0, 1])
    W_t = W.T
    u, w, vt = np.linalg.svd(E)  # Single Value Decomposition

    rot1 = u @ W @ vt
    rot2 = u @ W_t @ vt

    trl1 = u[:, -1].reshape((3, 1))
    trl2 = - trl1

    return rot1, rot2, trl1, trl2


def check_determinant(rot1):
    """" Validates by using det of R """
    return not (np.linalg.det(rot1) + 1) < 1e-9
