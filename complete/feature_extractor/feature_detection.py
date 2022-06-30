import copy
import cv2 as cv
import numpy as np
from skimage.feature import hog


def harris_corner_detection(image_container):
    image = copy.deepcopy(image_container.get_segmented_image())
    dst = cv.cornerHarris(
        image_container.get_segmented_image_gray(), 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    image[dst > 0.01*dst.max()] = [0, 0, 255]
    # cv.imshow('Original', image_container.get_segmented_image())
    cv.imshow('Harris', image)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


def shi_tomasi_corner_detection(image_container):
    image = copy.deepcopy(image_container.get_segmented_image())
    corners = cv.goodFeaturesToTrack(
        image_container.get_segmented_image_gray(), 20, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(image, (x, y), 5, (0, 0, 255), -1)
    # cv.imshow('Original', image_container.get_segmented_image())
    cv.imshow('Shi-Tomasi', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def sift_detection(image_container):
    image = copy.deepcopy(image_container.get_segmented_image())
    sift = cv.SIFT_create()
    keypoints, _ = sift.detectAndCompute(
        image_container.get_segmented_image(), None)
    image = cv.drawKeypoints(image_container.get_segmented_image(
    ), keypoints, image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow('Original', image_container.get_segmented_image())
    cv.imshow('SIFT', image)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


def blob_detection(image_container):
    image = copy.deepcopy(image_container.get_segmented_image())
    detector = cv.SimpleBlobDetector_create()
    keypoints = detector.detect(image)
    im_with_keypoints = cv.drawKeypoints(image, keypoints, np.array(
        []), (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow('Original', image_container.get_segmented_image())
    cv.imshow('BLOB', im_with_keypoints)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def histogram_of_oriented_gradients(image_container):
    img = copy.deepcopy(image_container.get_segmented_image())
    _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    # cv.imshow('Original', image_container.get_segmented_image())
    cv.imshow('HoG', hog_image)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def orb_detection(image_container):
    image = image_container.get_segmented_image()
    orb = cv.ORB_create(nfeatures=200)
    keypoints = orb.detect(image, None)
    keypoints, _ = orb.compute(image, keypoints)
    im_with_keypoints = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    # cv.imshow('Original', image_container.get_segmented_image())
    cv.imshow('ORB', im_with_keypoints)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
