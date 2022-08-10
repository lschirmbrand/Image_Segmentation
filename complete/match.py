import ImageContainer


class Match:

    def __init__(self, img_con1: ImageContainer, img_con2: ImageContainer):
        self._indices1 = []
        self._indices2 = []
        self._distances = []
        self._inliers1 = []
        self._inliers2 = []
        self._image1 = img_con1.get_segmented_image()
        self._image2 = img_con2.get_segmented_image()
        self._image1_name = img_con1.get_name()
        self._image2_name = img_con2.get_name()

    def get_indices1(self):
        return self._indices1

    def get_indices2(self):
        return self._indices2

    def get_inliers1(self):
        return self._inliers1

    def get_inliers2(self):
        return self._inliers2

    def set_inliers1(self, inliers: []):
        self._inliers1 = inliers

    def set_inliers2(self, inliers: []):
        self._inliers2 = inliers


def create_matches(image_container: []):
    matches = {}
    for i in range(0, len(image_container) - 1):
        for j in range(i + 1, len(image_container)):
            matches[(image_container[i].get_name(), image_container[j].get_name())] = Match(image_container[i],
                                                                                            image_container[j])
    return matches
