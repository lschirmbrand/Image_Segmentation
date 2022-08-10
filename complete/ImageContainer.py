import numpy as np


class ImageContainer:
    def __init__(self, image, filename):
        self._image = None
        self.set_image(image)
        self._resized_image = None
        self._gray_image = None
        self._segmented_image = None
        self._segmented_image_gray = None
        self._keypoints = []
        self._descriptors = []
        self._mat_rot = np.zeros((3, 3), dtype=float)
        self._vec_trl = np.zeros((3, 3), dtype=float)
        self._name = filename

    def set_image(self, image):
        self._image = image

    def get_image(self):
        return self._image

    def set_resized_image(self, image):
        self._resized_image = image

    def get_resized_image(self):
        return self._resized_image

    def set_gray_image(self, image):
        self._gray_image = image

    def get_gray_image(self):
        return self._gray_image

    def set_segmented_image(self, image):
        self._segmented_image = image

    def get_segmented_image(self):
        return self._segmented_image

    def set_segmented_image_gray(self, image):
        self._segmented_image_gray = image

    def get_segmented_image_gray(self):
        return self._segmented_image_gray

    def set_keypoints(self, keypoints: []):
        self._keypoints = keypoints

    def get_keypoints(self):
        return self._keypoints

    def set_descriptors(self, descriptors: []):
        self._descriptors = descriptors

    def get_descriptors(self):
        return self._descriptors

    def get_name(self):
        return self._name

    def set_mat_rot(self, val):
        self._mat_rot = val

    def get_mat_rot(self):
        return self._mat_rot

    def set_vec_trl(self, val):
        self._vec_trl = val

    def get_vec_trl(self):
        return self._vec_trl
