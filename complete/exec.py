from os import listdir
from os.path import isfile, join
import time
import config
import cv2 as cv
from skimage import filters


def get_all_images_from_path(path: str):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def resize_image(image, scale: int):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    return cv.resize(image, (width, height))


def import_image(path: str, name: str):
    return cv.imread(path + "/" + name, cv.IMREAD_COLOR)


def convert_color_bgr2rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def convert_color_rgb2gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def segment_image_treshold_gray(gray_img):
    return apply_mask(gray_img, compute_mask_binary(gray_img, compute_threshold(gray_img)))


def segment_image_treshold_color(img, gray_img):
    return apply_mask(img, compute_mask_color(compute_mask_binary(gray_img, compute_threshold(gray_img))))


def compute_threshold(gray_img):
    return filters.threshold_otsu(gray_img)


def compute_mask_binary(img, threshold: float):
    _, ret = cv.threshold(img, thresh=threshold,
                          maxval=255, type=cv.THRESH_BINARY)
    return ret


def compute_mask_binary_inverted(img, threshold: float):
    _, ret = cv.threshold(img, thresh=threshold,
                          maxval=255, type=cv.THRESH_BINARY_INV)
    return ret


def compute_mask_color(mask):
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)


def apply_mask(img, mask):
    return cv.bitwise_and(img, mask)

# starting time
start = time.time()

images_paths = get_all_images_from_path(config.assets['image_path'])

images = []
for f in images_paths:
    images.append(convert_color_bgr2rgb(
        import_image(config.assets['image_path'], f)))

resized_images = []
for i in images:
    resized_images.append(resize_image(i, config.image['scale']))

gray_images = []
for i in resized_images:
    gray_images.append(convert_color_rgb2gray(i))

segmented_images = []
for i, image in enumerate(gray_images):
    # segmented_images.append(segment_image_treshold_gray(image))
    segmented_images.append(segment_image_treshold_color(resized_images[i], image))

end = time.time()
print(f"Runtime of the program is {end - start}")

cv.imshow("Matches", segmented_images[0])
cv.waitKey(0)
cv.destroyAllWindows()
