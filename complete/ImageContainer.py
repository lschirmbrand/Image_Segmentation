class ImageContainer:
    def __init__(self, image):
        self.set_image(image)
        self._resized_image = None
        self._gray_image = None
        self._segmented_image = None

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
