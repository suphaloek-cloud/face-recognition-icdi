import cv2
import numpy as np
import os
import random

# Playing video from file:
cap = cv2.VideoCapture('person.mp4')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    num = random.randint(0, 100000000)
    # Saves image of the current frame in jpg file
    name = './data/' + str(num) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
