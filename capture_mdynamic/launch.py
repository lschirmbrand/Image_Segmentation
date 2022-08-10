import os
import random
import string
from time import sleep
import cv2
from scipy import rand
from checkports import list_ports
import sys


available_ports = list_ports()

# remove WebCam from the list of available ports
available_ports = []
available_ports.append(0)
available_ports.append(1)
available_ports.append(2)
available_ports.append(4)
print(available_ports)

def capture_image(cam):
    print("Capturing image...")
    _, image = cam.read()
    return image

def setup_camera(port):
    cam = cv2.VideoCapture(port, cv2.CAP_DSHOW)
    cam.set(3, 1920)
    cam.set(4, 1080)
    return cam

images = []

for i in range(4):
    for port in available_ports:
        cam = setup_camera(port)
        images.append(capture_image(cam))
        cam.release()
    print("move")
    sleep(6)

random = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
print('Savd under ' + random)
path = os.path.join("./capture_mdynamic/", random)
os.mkdir(path)
for i, image in enumerate(images):
    cv2.imwrite(os.path.join(path,  str(i) + '.png'), image)