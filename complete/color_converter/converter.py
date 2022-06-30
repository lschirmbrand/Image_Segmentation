import cv2 as cv

def convert_color_bgr2rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def convert_color_rgb2gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)