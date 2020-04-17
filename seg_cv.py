import cv2
import numpy as np
import argparse

arg = argparse.ArgumentParser()
arg.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
arg.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(arg.parse_args())

if __name__ == "__main__":
	#load YOLO
	net = cv2.dnn.readNetFromDarknet("./trained_data/yolov3-tiny-shape.cfg","./trained_data/yolov3-tiny-shape_best.weights") #shape dataset 

	#load class file (object label)
	classes = []
	#with open("/home/kimchi/graspinglab/darknet/data/coco.names","r") as f:
	with open("/home/kimchi/graspinglab/darknet/data/obj.names","r") as f:
		classes = [line.strip() for line in f.readlines()]
	print(classes)

	# initialize a list of colors to represent each possible class label
	COLORS = np.random.uniform(0,255,size=(len(classes),3))

    # load an image and grab its spatial dimensions
	img = cv2.imread("shape40.jpg")
	img = cv2.resize(img, (800, 600))
	cv2.imshow('input image', img)

	(H, W) = img.shape[:2]
    # determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[j[0] - 1] for j in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
	blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)

	layerOutputs = net.forward(ln)

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
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(classes[classIDs[index]], confidences[index])
			cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			    0.5, color, 2)
	
	cv2.imshow('YOLOed image', img)

    #convert graysale

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	edged = cv2.Canny(gray, 150, 200)
	cv2.imshow('canny edges', edged)
	cv2.waitKey(0)
