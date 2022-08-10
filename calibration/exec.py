import cv2
from numpy import diag
import numpy as np
import cv2 as cv
import glob

# cv2.imshow('image',cv2.resize(cv2.imread('calibration/calibrated_seg_iphone.jpg'), (1024, 768)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

class Resolution:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class ObjectResolution:
    def __init__(self, width, height, width_px, height_px):
        self.width = width
        self.height = height
        self.width_px = width_px
        self.height_px = height_px


def calculate_fxfy(object_resolution, distance):
    fx = (object_resolution.width_px / object_resolution.width) * distance
    print(object_resolution.width_px)
    print(object_resolution.width )
    print(distance)
    fy = (object_resolution.height_px / object_resolution.height) * distance
    return fx, fy

def calibrate(row, col, fx, fy):
    fx = fx * col / 1920
    fy = fy * row / 1080
    K = diag([fx, fy, 1])
    K[0, 2] = col / 2
    K[1, 2] = row / 2
    return K.round(1)

# res = Resolution(1920, 1080)
# obj = ObjectResolution(132, 178, 713, 960)

# fx, fy = calculate_fxfy(obj, 206)

# K = calibrate(obj.height_px, obj.width_px, fx, fy)


# obj = ObjectResolution(132, 178, 968, 1328)
# fx, fy = calculate_fxfy(obj, 220)
# K_iphone = calibrate(obj.height_px, obj.width_px, fx, fy)
# print(K)
# print(K_iphone)




# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*5,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('calibration/calibration_cam2/*.jpg')
print(len(images))
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,5), None)
    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # print(corners2)
        imgpoints.append(corners2)
        # Draw and display the corners
        # cv.drawChessboardCorners(img, (8,5), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(0)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(dist)
f = open("calibration/K.txt", "w+")
f.write(np.array2string(mtx))
f.close()
f = open("calibration/dist.txt", "w+")
f.write(np.array2string(dist))
f.close()

print(mtx)
print(dist)


# dist = np.asarray([0.11351048, -0.19936451,  0.00238028,  0.00473524, -0.03718523])
# print(dist)

img = cv.imread(images[8])
h,w = img.shape[:2]
newcamera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print(newcamera_matrix)


# undistort
dst = cv.undistort(img, mtx, dist, None, mtx)
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite('calibration/result/calibresult.png', dst)
cv.imwrite('calibration/result/org_image.png', img)