import cv2 as cv

def resize_image(image, scale: int):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    return cv.resize(image, (width, height))