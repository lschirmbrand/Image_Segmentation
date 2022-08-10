import cv2 as cv
from skimage.feature import canny
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import sobel
from skimage.segmentation import watershed

cmap = plt.cm.get_cmap("nipy_spectral")

img = cv.cvtColor(cv.imread("images/box_threshold_hard.png", cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

elevation_map = sobel(img_gray)

fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
ax[0].imshow(img, interpolation='nearest')
ax[0].axis('off')
ax[0].set_title('Original image')

markers = np.zeros_like(img_gray)
markers[img_gray < 50] = 1
markers[img_gray > 120] = 2

ax[1].imshow(markers, interpolation='nearest')
ax[1].axis('off')
ax[1].set_title('Markers')

segmentation = watershed(elevation_map, markers)

# mask3 = cv.cvtColor(segmentation, cv.COLOR_GRAY2BGR)  # 3 channel mask
# segm = cv.bitwise_and(img, segmentation)

ax[2].imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
ax[2].axis('off')
ax[2].set_title('Segmentation')

plt.show()