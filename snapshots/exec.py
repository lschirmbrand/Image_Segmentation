import cv2
camera = cv2.VideoCapture(1)
captured = 0
images = []
while captured < 5:
    return_value,image = camera.read()
    # gray = cv2.cvtColor(image)
    cv2.imshow('image',image)
    if cv2.waitKey(1)& 0xFF == ord('s'):
        # cv2.imwrite('./snapshots/results/test.jpg',image)
        images.append(image)
        captured += 1
camera.release()
cv2.destroyAllWindows()

for i in range(5):
    cv2.imwrite('./snapshots/moretests/' + str(i) + '.jpg', images[i])