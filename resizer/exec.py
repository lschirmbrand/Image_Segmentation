import cv2 as cv
import matplotlib.pyplot as plt
from skimage import filters
import matplotlib.gridspec as gridspec

image = cv.cvtColor(cv.imread("images/palette_3.jpg", cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

im_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

threshold = filters.threshold_otsu(im_gray)

_, mask = cv.threshold(im_gray, thresh=threshold, maxval=255, type=cv.THRESH_BINARY)
im_thresh_gray = cv.bitwise_and(im_gray, mask)

mask3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)  # 3 channel mask

im_color = cv.bitwise_and(image, mask3)

print(len(im_color))
print(len(im_color[0]))

def dimension_creator(n):
    width = int(im_color.shape[1] * n / 100)
    height = int(im_color.shape[0] * n / 100)
    return (width, height)   


gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
ax[0][0].title.set_text('Größe: 20 x 15')
ax[0][0].imshow(cv.resize(im_color, dimension_creator(1)))
ax[0][1].title.set_text('Größe: 60 x 45')
ax[0][1].imshow(cv.resize(im_color, dimension_creator(3)))
ax[1][0].title.set_text('Größe: 200 x 150')
ax[1][0].imshow(cv.resize(im_color, dimension_creator(10)))
ax[1][1].title.set_text('Größe: 600 x 450')
ax[1][1].imshow(cv.resize(im_color, dimension_creator(30)))
# ax[4][0].title.set_text('Größe: 600 x 450')
# ax[4][0].imshow(cv.resize(im_color, dimension_creator(30)))

plt.show()