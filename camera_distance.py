#written by Junhyeok Jeong
from imutils import paths
import numpy as np
import imutils
import cv2
import sys
import time

def grab_frame(frame):
	#HSV color
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#cv2.imshow("1", frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(gray, 50, 100)
	cv2.imshow("2", edged)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)


	return cv2.minAreaRect(c)
	

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	cv2.imshow("gray",edged)
 
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
 
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	#compute and return the distacne from the market to the camera
	return (knownWidth * focalLength) / perWidth


if __name__ == "__main__":
	vc = cv2.VideoCapture(0)
	if vc.isOpened(): # try to get the first frame
		rval, frame = vc.read()
	else:
		rval = False
	frame = cv2.resize(frame, (640, 640))

	blue_circle = grab_frame(frame)
	box_2 = cv2.cv.BoxPoints(blue_circle) if imutils.is_cv2() else cv2.boxPoints(blue_circle)
	box_2 = np.int0(box_2)
	cv2.drawContours(frame, [box_2], -1, (255, 0, 0), 3)
	cv2.imshow("blue", frame)


	# initialize the known distance from the camera to the object
	KNOWN_DISTANCE = 12.0
 
	# initialize the known object width, which in this case, the piece of paper
	KNOWN_WIDTH = 5.0
 
	# load the furst image that contains an object that is KNOWN TO BE 2 feet
	# from our camera, then find the paper marker in the image, and initialize the focal length
	image_read = cv2.imread("/home/kimchi/graspinglab/distance1.jpg")
	image = cv2.resize(image_read, (640, 640))
	print(image) #

	marker = find_marker(image)
	focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 
	# draw a bounding box around the image and display it
	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fft" % (inches / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	key = cv2.waitKey(0)
	if key ==27:
		cv2.destroyWindow("image")
