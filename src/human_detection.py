import os
import time

import cv2 as cv

SSD = 'ssd_mobilenet_v2_coco_2018_03_29'
FASTER_RCNN = 'faster_rcnn_inception_v2_coco_2018_01_28'

MODEL_DIR = os.path.join(os.sep, 'models', FASTER_RCNN)
WEIGHT_FILE = os.path.join(MODEL_DIR, 'weights.pb')
CONFIG_FILE = os.path.join(MODEL_DIR, 'config.pbtxt')

INPUT_IMG_FILE = os.path.join(os.sep, 'images', 'input', '1.jpg')
OUTPUT_IMG_FILE = os.path.join(os.sep, 'images', 'output', '1.jpg')

cvNet = cv.dnn.readNetFromTensorflow(WEIGHT_FILE, CONFIG_FILE)

img = cv.imread(INPUT_IMG_FILE)
rows = img.shape[0]
cols = img.shape[1]
blob = cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)
cvNet.setInput(blob)
t1 = time.time()
cvOut = cvNet.forward()

for detection in cvOut[0, 0, :, :]:
    score = float(detection[2])
    if score > 0.5:
        print(detection[:3])
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)),
                     (23, 230, 210), thickness=2)
print(time.time() - t1)
cv.imwrite(filename=OUTPUT_IMG_FILE, img=img)
