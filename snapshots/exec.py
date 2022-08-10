import cv2
cameras = []
try:
    # cameras.append(cv2.VideoCapture(1, cv2.CAP_DSHOW))
    # cameras.append(cv2.VideoCapture(2, cv2.CAP_DSHOW))
    # cameras.append(cv2.VideoCapture(3, cv2.CAP_DSHOW))
    # cameras.append(cv2.VideoCapture(4, cv2.CAP_DSHOW))

    # for camera in cameras:

    camera = cv2.VideoCapture(4, cv2.CAP_DSHOW)
    camera.set(3, 1920)
    camera.set(4, 1080)
    captured = 0

    images = []
    print("Capturing...")
    while captured < 25:
        return_value,image = camera.read()
        # gray = cv2.cvtColor(image)
        cv2.imshow('image',image)
        if cv2.waitKey(1)& 0xFF == ord('s'):
            
            return_value,image = camera.read()
            images.append(image)

            captured += 1
    print(len(images))
    for i in range(len(images)):
        cv2.imwrite('./snapshots/calibration_cam1/' + str(i) + '.jpg', images[i])
finally:
    camera.release()