import matplotlib.pyplot as plt
from skimage import filters
import cv2 as cv
# import numpy as np

# change default colormap
plt.rcParams['image.cmap'] = 'gray'

images_color = []
# images_color.append(cv.cvtColor(cv.imread("images/palette_1.jpg", cv.IMREAD_COLOR),
#   cv.COLOR_BGR2RGB))
images_color.append(cv.cvtColor(cv.imread("images/palette_2.jpg", cv.IMREAD_COLOR),
                                cv.COLOR_BGR2RGB))
images_color.append(cv.cvtColor(cv.imread("images/palette_3.jpg", cv.IMREAD_COLOR),
                                cv.COLOR_BGR2RGB))


images_gray = []
i = 0
for image in images_color:
    images_gray.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
    i += 1


thresholds = []
i = 0
for image in images_gray:
    thresholds.append(filters.threshold_otsu(image))
    i += 1

masks = []
i = 0
for image in images_gray:
    _, temp = cv.threshold(
        image, thresh=thresholds[i], maxval=255, type=cv.THRESH_BINARY)
    masks.append(temp)
    i += 1

images_segmented_gray = []
i = 0
for image in images_gray:
    images_segmented_gray.append(cv.bitwise_and(image, masks[i]))
    i += 1

masks3d = []
i = 0
for image in images_gray:
    masks3d.append(cv.cvtColor(masks[i], cv.COLOR_GRAY2BGR))  # 3 channel mask
    i += 1

images_segmented = []
i = 0
for image in images_gray:
    images_segmented.append(cv.bitwise_and(images_color[i], masks3d[i]))
    i += 1


fig, ax = plt.subplots(ncols=len(images_segmented), figsize=(20, 10))
i = 0
for image in images_segmented:
    ax[i].imshow(image)
    i += 1

axis = plt.gca()

for a in ax:
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

plt.axis('off')
plt.show()

query_img = images_segmented[0]
train_img = images_segmented[1]

# Convert it to grayscale
query_img_bw = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
train_img_bw = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

# Initialize the ORB detector algorithm
orb = cv.ORB_create()

# Now detect the keypoints and compute
# the descriptors for the query image
# and train image
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

# Initialize the Matcher for matching
# the keypoints and then match the
# keypoints
matcher = cv.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)

# draw the matches to the final image
# containing both the images the drawMatches()
# function takes both images and keypoints
# and outputs the matched query image with
# its train image
final_img = cv.drawMatches(query_img, queryKeypoints,
                           train_img, trainKeypoints, matches[:20], None)

scale_percent = 50
width = int(final_img.shape[1] * scale_percent / 100)
height = int(final_img.shape[0] * scale_percent / 100)
dim = (width, height)

final_img = cv.cvtColor(cv.resize(final_img, dim), cv.COLOR_BGR2RGB)

# Show the final image
cv.imshow("Matches", final_img)
cv.waitKey(0)
cv.destroyAllWindows()
