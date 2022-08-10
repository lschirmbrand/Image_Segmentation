import cv2
# camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# camera2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

try:
    # camera.set(3, 1920)
    # camera.set(4, 1080)

    # camera2.set(3, 1920)
    # camera2.set(4, 1080)
    stop = False

    images = []
    print("Capturing...")
    while not stop:
        # gray = cv2.cvtColor(image)
        # if cv2.waitKey(1)& 0xFF == ord('s'):
        camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        camera.set(3, 1920)
        camera.set(4, 1080)
        return_value,image = camera.read()
        # camera.release()
        # camera = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        # return_value,image2 = camera.read()
        # camera.release()
        cv2.imshow('image',image)
        cv2.waitKey(0)
        # cv2.imshow('image',image2)
        # cv2.waitKey(0)
        # cv2.imshow('image2',image2)
        stop = True
finally:
        camera.release()
        cv2.destroyAllWindows()