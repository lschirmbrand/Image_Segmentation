import cv2 as cv
from skimage.feature import canny
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology

img = cv.cvtColor(cv.imread("images/pallete_simpel.jpg", cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img)
edges = canny(img_gray/255.)

fig, ax = plt.subplots(ncols=3, figsize=(20, 10))

ax[0].set_title("Original image")
ax[0].imshow(img)
ax[0].axis('off')

fill_coins = ndi.binary_fill_holes(edges)
# ax[1].imshow(fill_coins, cmap=plt.cm.gray, interpolation='nearest')
# ax[1].axis('off')
# ax[1].set_title('Filling the holes')

coins_cleaned = morphology.remove_small_objects(fill_coins, 25)
ax[1].title.set_text('Canny detector')
ax[1].imshow(fill_coins, cmap=plt.cm.gray, interpolation='nearest')
ax[1].axis('off')

print(coins_cleaned)
# coins_cleaned = cv.cvtColor(coins_cleaned, cv.COLOR_BGR2GRAY)
# img_seg = cv.bitwise_and(img, coins_cleaned)

ax[2].imshow(coins_cleaned, cmap=plt.cm.gray, interpolation='nearest')
ax[2].axis('off')
ax[2].set_title('Removing small objects')

plt.show()