import copy
import matplotlib.pyplot as plt
from skimage import filters
import cv2 as cv
# import numpy as np

# change default colormap
plt.rcParams['image.cmap'] = 'gray'

# images_color = [cv.cvtColor(cv.imread("images/9.jpg", cv.IMREAD_COLOR),
#                             cv.COLOR_BGR2RGB),
#                 cv.cvtColor(cv.imread("images/10.jpg", cv.IMREAD_COLOR),
#                             cv.COLOR_BGR2RGB)]
# # images_color.append(cv.cvtColor(cv.imread("images/palette_1.jpg", cv.IMREAD_COLOR),
# #   cv.COLOR_BGR2RGB))
# # images_color.append(cv.cvtColor(cv.imread("images/palette_2.jpg", cv.IMREAD_COLOR),
# #                                 cv.COLOR_BGR2RGB))
# # images_color.append(cv.cvtColor(cv.imread("images/palette_3.jpg", cv.IMREAD_COLOR),
# #                                 cv.COLOR_BGR2RGB))

# images_gray = []
# i = 0
# for image in images_color:
#     images_gray.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
#     i += 1


# thresholds = []
# i = 0
# for image in images_gray:
#     thresholds.append(filters.threshold_otsu(image))
#     i += 1

# masks = []
# i = 0
# for image in images_gray:
#     _, temp = cv.threshold(
#         image, thresh=thresholds[i], maxval=255, type=cv.THRESH_BINARY_INV)
#     masks.append(temp)
#     i += 1

# images_segmented_gray = []
# i = 0
# for image in images_gray:
#     images_segmented_gray.append(cv.bitwise_and(image, masks[i]))
#     i += 1

# masks3d = []
# i = 0
# for image in images_gray:
#     masks3d.append(cv.cvtColor(masks[i], cv.COLOR_GRAY2BGR))  # 3 channel mask
#     i += 1

# images_segmented = []
# i = 0
# for image in images_gray:
#     images_segmented.append(cv.bitwise_and(images_color[i], masks3d[i]))
#     i += 1


# fig, ax = plt.subplots(ncols=len(images_segmented), figsize=(20, 10))
# i = 0
# for image in images_segmented:
#     # ax[i].imshow(image)
#     i += 1

# axis = plt.gca()

# for a in ax:
#     a.get_xaxis().set_visible(False)
#     a.get_yaxis().set_visible(False)

# plt.axis('off')
# plt.show()

# query_img = images_segmented[0]
# train_img = images_segmented[1]
#
# # Convert it to grayscale
# query_img_bw = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
# train_img_bw = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)
#
# # Initialize the ORB detector algorithm
# orb = cv.ORB_create()
#
# # Now detect the keypoints and compute
# # the descriptors for the query image
# # and train image
# queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
# trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)
#
# # Initialize the Matcher for matching
# # the keypoints and then match the
# # keypoints
# matcher = cv.BFMatcher()
# matches = matcher.match(queryDescriptors, trainDescriptors)
#
# # draw the matches to the final image
# # containing both the images the drawMatches()
# # function takes both images and keypoints
# # and outputs the matched query image with
# # its train image
# final_img = cv.drawMatches(query_img, queryKeypoints,
#                            train_img, trainKeypoints, matches[:20], None)
#
# scale_percent = 50


# orb = cv.ORB_create(nfeatures=20)

# kp1, des1 = orb.detectAndCompute(images_segmented[0], None)
# kp2, des2 = orb.detectAndCompute(images_segmented[1], None)

# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
# filtered_matches  = []
# for i in range(len(matches)):
#     if matches[i].distance < 100:
#         filtered_matches.append(matches[i])
# match_img = cv.drawMatches(images_segmented[0], kp1, images_segmented[1], kp2, filtered_matches[:50], None)
# # cv.imshow('original image', img1)
# # cv.imshow('test image', img2)
# cv.imshow('Matches', match_img)
# cv.waitKey()

# image_1 = cv.imread("threshold/images/IMG_3921.jpg", cv.IMREAD_COLOR)
# image_2 = cv.imread("threshold/images/IMG_3922.jpg", cv.IMREAD_COLOR)

image_1 = cv.imread("threshold/images/box_qr_seg_1.jpg", cv.IMREAD_COLOR)
image_2 = cv.imread("threshold/images/box_qr_seg_2.jpg", cv.IMREAD_COLOR)

# image_1 = cv.imread("threshold/images/img(1)_snip.jpg", cv.IMREAD_COLOR)
# image_2 = cv.imread("threshold/images/img(2)_snip.jpg", cv.IMREAD_COLOR)
# 
# image_1 = cv.imread("threshold/images/color-bomb-ken-figurski.jpg", cv.IMREAD_COLOR)
# image_2 = cv.imread("threshold/images/color-bomb-ken-figurski_changed.jpg", cv.IMREAD_COLOR)

detector = cv.SIFT_create()
keypoints1, descriptors1 = detector.detectAndCompute(image_1, None)
keypoints2, descriptors2 = detector.detectAndCompute(image_2, None)

# out_img = copy.deepcopy(image_2)
# out_img = cv.drawKeypoints(image_2,keypoints2, out_img,
#                                    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# scale_percent = 40
# width = int(out_img.shape[1] * scale_percent / 100)
# height = int(out_img.shape[0] * scale_percent / 100)
# dim = (width, height)
# out_img = cv.resize(out_img, dim)
# cv.imshow("SIFT", out_img)

# # final_img = cv.cvtColor(cv.resize(final_img, dim), cv.COLOR_BGR2RGB)

# # Show the final image
# # cv.imshow("Matches", final_img)
# cv.waitKey(0)
# cv.destroyAllWindows()


matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

matches = matcher.knnMatch(descriptors1,descriptors2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(image_1,keypoints1,image_2,keypoints2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


scale_percent = 40
width = int(img3.shape[1] * scale_percent / 100)
height = int(img3.shape[0] * scale_percent / 100)
dim = (width, height)
img3 = cv.resize(img3, dim)

cv.imshow("SIFT", img3)
cv.waitKey(0)
cv.destroyAllWindows()