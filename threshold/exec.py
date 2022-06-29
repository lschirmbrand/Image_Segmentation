import matplotlib.pyplot as plt
from skimage import (filters, io, morphology, color)
import cv2 as cv

# change default colormap
plt.rcParams['image.cmap'] = 'gray'

im_color = cv.cvtColor(
    cv.imread("images/pallete_simpel.jpg", cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
im_gray = cv.cvtColor(im_color, cv.COLOR_BGR2GRAY)

threshold = filters.threshold_otsu(im_gray)

_, mask = cv.threshold(im_gray, thresh=threshold,
                       maxval=255, type=cv.THRESH_BINARY)
im_thresh_gray = cv.bitwise_and(im_gray, mask)

mask3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)  # 3 channel mask

im_thresh_color = cv.bitwise_and(im_color, mask3)


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
# ax[1].title.set_text('Grayscale image')
# ax[1].imshow(im_gray)
ax[1].title.set_text('Mask')
ax[1].imshow(mask3)
# ax[3].title.set_text('Histogram')
# ax[3].imshow(histogram)
ax[2].title.set_text('Segmented Image')
ax[2].imshow(im_thresh_color)

axis = plt.gca()

for a in ax:
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

plt.axis('off')
plt.show()
