import cv2

def match(image_container, nfeatures=10):
    img1 = image_container[0].get_segmented_image()
    img2 = image_container[0].get_segmented_image()

    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    cv2.imshow('original image', img1)
    cv2.imshow('test image', img2)
    cv2.imshow('Matches', match_img)
    cv2.waitKey()
