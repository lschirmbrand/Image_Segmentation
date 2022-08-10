import os
import time
import config
import numpy as np
from match import *
from ImageContainer import *
from sfm import *
from color_converter import converter
from filehandler import filehandler
from segmentation import (threshold_segmentation as thr_seg)
from resizer import resizer
from feature_extractor import feature_detection as fd

start = time.time()

images_paths = filehandler.get_all_images_from_path(
    config.assets['image_path'])

image_container: list[ImageContainer] = []
for f in images_paths:
    image_container.append(ImageContainer(converter.convert_color_bgr2rgb(
        filehandler.import_image(config.assets['image_path'], f)), f))

for i in image_container:
    i.set_resized_image(resizer.resize_image(
        i.get_image(), config.image['scale']))
    i.set_gray_image(converter.convert_color_rgb2gray(i.get_resized_image()))
    i.set_segmented_image(thr_seg.segment_image_treshold_color(
        i.get_resized_image(), i.get_gray_image()))
    i.set_segmented_image_gray(thr_seg.segment_image_treshold_gray(i.get_gray_image()))

    keypoints, descriptors = fd.calc_orb_keypoints(i)
    i.set_keypoints(keypoints)
    i.set_descriptors(descriptors)

matches = create_matches(image_container)
K = np.loadtxt(os.path.join(config.assets['image_path'], 'images', 'K.txt'))
sfm = sfm(image_container, )

# fd.harris_corner_detection(image_container[0])
# fd.shi_tomasi_corner_detection(image_container[0])
# fd.sift_detection(image_container[0])
# fd.blob_detection(image_container[0])
# fd.histogram_of_oriented_gradients(image_container[0])
# fd.orb_detection(image_container[0])

# fm.match(image_container, 100)


end = time.time()
print(f"Runtime of the program is {end - start}")

# cv.imshow("Matches", image_container[0].get_segmented_image())
# cv.waitKey(0)
# cv.destroyAllWindows()
