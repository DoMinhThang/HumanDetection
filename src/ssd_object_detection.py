# USAGE
# python ssd_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --input ../example_videos/guitar.mp4 --output ../output_videos/ssd_guitar.avi --display 0
# python ssd_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --input ../example_videos/guitar.mp4 --output ../output_videos/ssd_guitar.avi --display 0 --use-gpu 1

# import the necessary packages
import os
import time

from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
CONFIDENCE = 0.3

SSD_MOBILENET_CAFFE = 'ssd_mobilenet_caffe'

MODEL_DIR = os.path.join(os.sep, 'models', SSD_MOBILENET_CAFFE)
WEIGHT_FILE = os.path.join(MODEL_DIR, 'MobileNetSSD_deploy.prototxt')
CONFIG_FILE = os.path.join(MODEL_DIR, 'MobileNetSSD_deploy.caffemodel')

INPUT_IMG_FILE = os.path.join(os.sep, 'images', 'input', '1.jpg')
OUTPUT_IMG_FILE = os.path.join(os.sep, 'images', 'output', '1.jpg')

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(WEIGHT_FILE, CONFIG_FILE)

# resize the frame, grab the frame dimensions, and convert it to
# a blob
t1 = time.time()
frame = cv2.imread('/images/input/1.jpg')
frame = imutils.resize(frame, width=400)
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the prediction
	confidence = detections[
		0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > CONFIDENCE:
		# extract the index of the class label from the
		# `detections`, then compute the (x, y)-coordinates of
		# the bounding box for the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the prediction on the frame
		label = "{}: {:.2f}%".format(CLASSES[idx],
			confidence * 100)
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(frame, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
print(time.time() - t1)
cv2.imwrite(filename=OUTPUT_IMG_FILE, img=frame)

