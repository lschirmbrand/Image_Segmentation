from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import ndimage
from sklearn.cluster import KMeans
import cv2 as cv

# Scaling the image pixels values within 0-1
img = imread('./images/box_threshold_hard.png') / 255
# plt.imshow(img)
# plt.title('Original')
# plt.show()

# For clustering the image using k-means, we first need to convert it into a 2-dimensional array
image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
# Use KMeans clustering algorithm from sklearn.cluster to cluster pixels in image


# tweak the cluster size and see what happens to the Output
kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
clustered = kmeans.cluster_centers_[kmeans.labels_]
# Reshape back the image from 2D to 3D image
clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
# plt.imshow(clustered_3D)
# plt.title('Clustered Image')
# plt.show()

fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
ax[0].imshow(img)
ax[0].title.set_text('Original image')
ax[0].axis('off')
ax[1].imshow(clustered_3D)
ax[1].title.set_text('Clustered Image')
ax[1].axis('off')

plt.show()