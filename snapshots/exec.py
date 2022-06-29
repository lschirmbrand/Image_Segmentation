import cv2
camera = cv2.VideoCapture(0)
while True:
    return_value,image = camera.read()
    # gray = cv2.cvtColor(image)
    cv2.imshow('image',image)
    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite('./snapshots/results/test.jpg',image)
        break
camera.release()
cv2.destroyAllWindows()