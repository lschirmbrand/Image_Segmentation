from skimage import filters
import cv2 as cv


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
