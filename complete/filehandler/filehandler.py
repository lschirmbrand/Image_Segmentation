from os import listdir
from os.path import isfile, join
import cv2 as cv

def get_all_images_from_path(path: str):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def import_image(path: str, name: str):
    return cv.imread(path + "/" + name, cv.IMREAD_COLOR)