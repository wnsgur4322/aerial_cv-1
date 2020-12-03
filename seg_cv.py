import cv2
import numpy as np
import argparse
import imutils

arg = argparse.ArgumentParser()
arg.add_argument('input_image')
arg.add_argument("-w", "--dataset", type=str,
	help="set object detection weight and data file (type original or tiny or shape)")
arg.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
arg.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(arg.parse_args())
args2 = arg.parse_args()

def midpoint(ptA, ptB):
	return ((ptA + ptB) * 0.5, (ptA + ptB) * 0.5)

if __name__ == "__main__":
	if args["dataset"] == None:
		print(args["dataset"])
		print("Error: please set up weight and dataset file argument ex)python3 yolo_webcam.py -w tiny")
		exit()
    
	#load YOLO
	if args["dataset"] == "original":
		net = cv2.dnn.readNetFromDarknet("./trained_data/original_yolov3/yolov3.cfg","./trained_data/original_yolov3/yolov3.weights") # Original yolov3
	if args["dataset"] == "tiny":
		net = cv2.dnn.readNet("trained_data/tiny/yolov3-tiny.weights","trained_data/tiny/yolov3-tiny.cfg") #Tiny Yolo
	if args["dataset"] == "shape":
		net = cv2.dnn.readNet("trained_data/shape/yolov3-tiny-shape_best.weights","trained_data/shape/yolov3-tiny-shape.cfg") #shape dataset


	#load class file (object label)
	classes = []
	if args["dataset"] == "tiny" or "original":
		with open("trained_data/tiny/coco.names","r") as f:
			classes = [line.strip() for line in f.readlines()]
	if args["dataset"] == "shape":
		with open("trained_data/shape/obj.names","r") as f:
		    	classes = [line.strip() for line in f.readlines()]
	print(classes)

	# initialize a list of colors to represent each possible class label
	COLORS = np.random.uniform(0,255,size=(len(classes),3))

    # load an image and grab its spatial dimensions
	img = cv2.imread(args2.input_image)
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

	# initialize the list of threshold methods
	methods = [
		("THRESH_BINARY", cv2.THRESH_BINARY),
		("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
		("THRESH_TRUNC", cv2.THRESH_TRUNC),
		("THRESH_TOZERO", cv2.THRESH_TOZERO),
		("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]

    # ensure at least one detection exists
	if len(idxs) > 0:
        # loop over the indexes we are keeping
		for index in idxs.flatten():
            # extract the bounding box coordinates
			(x, y) = (boxes[index][0], boxes[index][1])
			(w, h) = (boxes[index][2], boxes[index][3])
            # draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[index]]]
			
			
			#rec = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			#text = "{}: {:.4f}".format(classes[classIDs[index]], confidences[index])
			#cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			#    0.5, color, 2)
			
			# unpack the ordered bounding box, then compute the midpoint
			# between the top-left and top-right coordinates, followed by
			# the midpoint between bottom-left and bottom-right coordinates
			print(x)
			(tl, tr) = (x, y)
			(bl, br) = (x+ w , y + h)
			(tltrX, tltrY) = midpoint(tl, tr)
			(blbrX, blbrY) = midpoint(bl, br)
			# compute the midpoint between the top-left and top-right points,
			# followed by the midpoint between the top-righ and bottom-right
			(tlblX, tlblY) = midpoint(tl, bl)
			(trbrX, trbrY) = midpoint(tr, br)
			# draw the midpoints on the image
			cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
			cv2.circle(img, (x + w, y + h), 5, (255, 0, 0), -1)
			cv2.circle(img, (x + w, y), 5, (255, 0, 0), -1)
			cv2.circle(img, (x, y + h), 5, (255, 0, 0), -1)

			roi_color = img[y:y + h, x:x+ w]

			gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (5, 5), 0)
			# threshold the image, then perform a series of erosions +
			# dilations to remove any small regions of noise
			thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
			thresh = cv2.erode(thresh, None, iterations=2)
			thresh = cv2.dilate(thresh, None, iterations=2)

			cv2.imshow('grayed', gray)

			# loop over the threshold methods
			for (threshName, threshMethod) in methods:
			# threshold the image and show it
				(T, thresh) = cv2.threshold(gray, 150, 255, threshMethod)
				cv2.imshow(threshName, thresh)

			#edged = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
			#cv2.imshow('edged', edged)
			
			#retval, thresh = cv2.threshold(gray, 127, 255, 0)
			img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			img_contours = imutils.grab_contours(img_contours)
			c = max(img_contours, key=cv2.contourArea)
			# determine the most extreme points along the contour
			extLeft = tuple(c[c[:, :, 0].argmin()][0])
			extRight = tuple(c[c[:, :, 0].argmax()][0])
			extTop = tuple(c[c[:, :, 1].argmin()][0])
			extBot = tuple(c[c[:, :, 1].argmax()][0])
			cv2.drawContours(roi_color, img_contours, -1, (0, 255, 0))

			cv2.circle(roi_color, extLeft, 4, (0, 0, 255), -1)
			cv2.circle(roi_color, extRight, 4, (0, 255, 0), -1)
			cv2.circle(roi_color, extTop, 4, (255, 0, 0), -1)
			cv2.circle(roi_color, extBot, 4, (255, 255, 0), -1)
			print(extBot)

			cv2.line(roi_color, extTop, extBot, (255,255,255), 1)
			top_to_bottom = extBot[1] - extTop[1]
			cv2.putText(img, "{}".format(top_to_bottom), (extBot[1], extTop[1] + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			cv2.line(roi_color, extRight, extLeft, (255,255,255), 1)
			left_to_right = extRight[0] - extLeft[0] 
			cv2.putText(img, "{}".format(left_to_right), (extLeft[0] + 220, extRight[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			cv2.imshow('contours', roi_color)

			rec = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(classes[classIDs[index]], confidences[index])
			cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			    0.5, color, 2)
			

			

	
	cv2.imshow('YOLOed image', img)

	#combined = cv2.bitwise_and(edged, img)
	#cv2.imshow('combined image', combined )
	cv2.waitKey(0)
