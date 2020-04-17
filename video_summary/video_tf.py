import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import os
import glob
import argparse
import time
import json

import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


arg = argparse.ArgumentParser()
arg.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
arg.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
arg.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(arg.parse_args())

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# step 1: read the video, extract frames from it and save them as jpg files
count = 0
videoFile = "/home/kimchi/graspinglab/computer_vision/video_summary/blazers.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path

try:
    if not os.path.exists('frames'):
        os.makedirs('frames')
except OSError:
    print("Error")

current_frame = 0
while(True):
    breaks
    #captrue frame by frame
    ret, frame = cap.read()

    if not ret:
        break

    #save the image of the current frame as a jpg file
    name = './frames/frame' + str(current_frame) + '.jpg'
    print('Creating ...' + name)
    cv2.imwrite(name, frame)

    current_frame += 1

cap.release()
cv2.destroyAllWindows()

#step 2: load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


image_num = len(glob.glob1("./frames", "*.jpg"))
print(image_num)
json_data = {}
json_data['localisation'] = []
json_data['localisation'].append({
    "thumb": "full/samples-data/examples/kf/amalia01/f_25.jpg",
    "tc": "00:00:01.0000",
    "tclevel": 1
})

with open('json_data.txt', 'w') as outfile:
    json.dump(json_data, outfile)

for i in range(image_num):
    # load our input image and grab its spatial dimensions
    image = cv2.imread("frames/frame%d.jpg" % i)
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[j[0] - 1] for j in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for index in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[index][0], boxes[index][1])
            (w, h) = (boxes[index][2], boxes[index][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[index]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[index]], confidences[index])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    
    # show the output image
    cv2.imwrite("frames/frame%d.jpg" % i, image)
    print("frame%d.jpg is created successfully" % i)


# step 3: create json file and dump img data