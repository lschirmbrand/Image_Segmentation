import matplotlib.pyplot as plt
from skimage import (filters, io, morphology, color)
import cv2 as cv
import numpy as np
import skimage.exposure

# change default colormap
plt.rcParams['image.cmap'] = 'gray'

im_color = cv.cvtColor(cv.imread("images/box_threshold_hard.png", cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
im_gray = cv.cvtColor(im_color, cv.COLOR_BGR2GRAY)

threshold = filters.threshold_otsu(im_gray)

_, mask = cv.threshold(im_gray, thresh=threshold,
                       maxval=255, type=cv.THRESH_BINARY_INV)
im_thresh_gray = cv.bitwise_and(im_gray, mask)

mask3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)  # 3 channel mask
#
im_thresh_color = cv.bitwise_and(im_color, mask3)

# lab = cv.cvtColor(im_color, cv.COLOR_BGR2LAB)

# A = lab[:, :, 1]

# thresh = cv.threshold(A, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

# blur = cv.GaussianBlur(thresh, (0, 0), sigmaX=5, sigmaY=5, borderType=cv.BORDER_DEFAULT)

# # stretch so that 255 -> 255 and 127.5 -> 0
# mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255)).astype(np.uint8)

# # add mask to image as alpha channel
# result = im_color.copy()
# result = cv.cvtColor(im_color, cv.COLOR_BGR2BGRA)
# result[:, :, 3] = mask
# # result = cv.cvtColor(result, cv.COLOR_BGR2RGB)

# # save output

# cv.imwrite('greenscreen_antialiased.png', result)

# # Display various images to see the steps
# cv.imshow('result', result)

# cv.waitKey(0)
# cv.destroyAllWindows()

# image = io.imread("images/pallete_simpel.jpg")

# grayscale_image = color.rgb2gray(image)

# ## Median Filter
# median_image = median_filter(grayscale_image, 4)

# # Compute Historgam
# histogram, bin_edges = np.histogram(median_image, bins=256, range=(0, 255))

# # Compute Threshold
# threshold = filters.threshold_otsu(median_image)
# print("Threshold locatetd at grey-value: {}".format(threshold))

# # Compute Binary Mask
# binary_mask = median_image > threshold

# # Compute masked image
# masked_image = np.zeros_like(median_image)
# # masked_image = image[binary_mask]
# res = cv.bitwise_and(image,masked_image)


fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
ax[0].title.set_text('Original image')
ax[0].imshow(im_color)
ax[1].title.set_text('Mask')
ax[1].imshow(mask3)
ax[2].title.set_text('Segmented Image')
ax[2].imshow(im_thresh_color)
# ax[1].title.set_text('Mask')
# ax[1].imshow(mask)
# # ax[1].imshow(mask3d)
# # ax[3].title.set_text('Histogram')
# # ax[3].imshow(histogram)
# ax[2].imshow(masked_image)
#
axis = plt.gca()

for a in ax:
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
#

cv.imwrite('images/calibrated_seg_iphone.jpg', im_thresh_color)
plt.axis('off')
plt.show()
