#to do list
# 1. object detection
# 2. recollect data with attached camera angle and train it 
# 3. figure out the size of object https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
#	  - find pixels per metric. ex) 1 pixel = 0.001 cm
#	 - set up reference distance ex) if distance is 1m between object and camera, then 1 pixel is 0.001 cm
#	 - convert pixels of contour box into size scale.
#	  - grasping as much as the size scale.

#Written by Junhyeok Jeong
#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from threading import Thread
import argparse
import imutils
import cv2
import numpy as np
import time
import datetime
import glob
import math
import serial
from sys import platform
if platform == "linux" or platform == "linux2":
	import syslog
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640
ORIGIN_X = VIDEO_WIDTH / 2
ORIGIN_Y = VIDEO_HEIGHT / 2
FOCAL_LENGTH = 3.67			# 3.67 mm, Logitech C615 webcam


def midpoint(ptA, ptB):
	return ((ptA + ptB) * 0.5, (ptA + ptB) * 0.5)


FOCAL_LENGTH = 3.67 # logitech c920 webcam
# doesn't work well
def distance_to_camera(frame, box):
	count = 0
	distance_inch = 0.0
	x,y,w,h = boxes[i]

	if len(box) >= 0: ### Increase the value of count if there are more than one rectangle in a given frame
			count += 1
	distance_inch = (2 * FOCAL_LENGTH * 180) / (w + h * 360) * 1000 #+ 3 ### Distance measuring in Inch
	print(distance_inch)
		# distance = distance_inch * 2.54 #convert to cm
		# print(distance)
	return count, distance_inch

def draw_detections(img, rects, thickness = 1):
	"""
		INPUT :
			img	   : Gets the input frame
			rect : Number from the regression layer (x0,y0,width,height)
		OUTPUT:
			count: Number of objects in a given frame
			distance : Calculates the distance from the rect value
	"""

	count = 0
	distance_inch = 0.0
	for x, y, w, h in rects:
		print(len(rects))

		if len(rects) >= 0: ### Increase the value of count if there are more than one rectangle in a given frame
			count += 1
		distance_inch = (2 * FOCAL_LENGTH * 180) / (w + h * 360) * 1000 + 3 ### Distance measuring in Inch
		# print(distance_inch)
		# distance = distance_inch * 2.54 #convert to cm
		# print(distance)
		# the HOG detector returns slightly larger rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(0.15*w), int(0.05*h)
		cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
	return count, distance_inch


def stack_frames(scale, imgArray):
		rows = len(imgArray)
		cols = len(imgArray[0])
		rowsAvailable = isinstance(imgArray[0], list)
		width = imgArray[0][0].shape[1]
		height = imgArray[0][0].shape[0]
		if rowsAvailable:
				for x in range ( 0, rows):
						for y in range(0, cols):
								if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
										imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
								else:
										imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
								if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
				imageBlank = np.zeros((height, width, 3), np.uint8)
				hor = [imageBlank]*rows
				hor_con = [imageBlank]*rows
				for x in range(0, rows):
						hor[x] = np.hstack(imgArray[x])
				ver = np.vstack(hor)
		else:
				for x in range(0, rows):
						if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
								imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
						else:
								imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
						if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
				hor= np.hstack(imgArray)
				ver = hor
		return ver

def get_contours(img, imgContour, frame, x, y, w, h, roi_color):
		contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for cnt in contours:
				area = cv2.contourArea(cnt)
				areaMin = cv2.getTrackbarPos("Area", "Parameters")
				if area > areaMin:
						try:
							# compute the center of the contour
							M = cv2.moments(cnt)
							cX = int(M["m10"] / M["m00"])
							cY = int(M["m01"] / M["m00"])
							cv2.circle(roi_color, (cX, cY), 3, (255, 255, 255), -1)
						except:
							continue

						cv2.drawContours(roi_color, cnt, -1, (255, 0, 255), 1)
						peri = cv2.arcLength(cnt, True)
						approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
						x2 , y2 , w2, h2 = cv2.boundingRect(approx)
						cv2.rectangle(frame, (x , y), (x + w + pad_w , y + h + pad_h), (0, 255, 0), 1)

						cv2.putText(frame, "Points: " + str(len(approx)), (x + w + 20, y + 20), font, 0.7, (0, 255, 0), 1)
						#cv2.putText(frame, "Area: " + str(int(area)), (x + w + 20, y + 45), font, 0.7, (0, 255, 0), 1)

		return contours, cX, cY

def empty(a):
		pass


if __name__ == "__main__":
	print("start")
	#termination crietria
	#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	#objp=np.zeros((6 * 7, 3), np.float32)
	#objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	ap = argparse.ArgumentParser()
	ap.add_argument("-w", "--dataset", type=str,
	help="set object detection weight and data file (type original, tiny, shape, 10_food, 46_food, 46_food_noised, 4_food)")
	ap.add_argument("-u", "--ultrasonic", type=int, default=0,
	help="set ultrasonic sensor usage flag to measure distance (type -u 1 or 0 ")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
	ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
	ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
	ap.add_argument("-m", "--methods", type=int, default=1,
	help="segmentation threshold methods:\n0.THRESH_BINARY\n1. THRESH_BINARY_INV\n2. THRESH_TRUNC\n3. THRESH_TOZERO\n4.THRESH_TOZERO_INV")
	ap.add_argument("-cam", "--camera_number", type=int, default=0,
	help="0 is built-in webcam, 1 or 2 is usb webcam")
	ap.add_argument("-csv_filename", "--name", type=str, default="data",
	help="put your desired file name for data csv file")
	args = vars(ap.parse_args())
	if args["dataset"] == None:
		print(args["dataset"])
		print("Error: please set up weight and dataset file argument ex)python3 yolo_webcam.py -w tiny")
		exit()
	# csv part
	fields_ultra = ['Object', 'Distance (cm)', 'height','x axis', 'y axis', 'Accuracy(%)']
	fields_non = ['Object', 'Accuracy(%)']

	if os.path.exists("./"+ args["name"] + ".csv"):
		os.remove("./"+ args["name"] + ".csv")
		print("existed data removed")
	else:
		print("can not delete old csv file as it doesn't exists")

	if args["ultrasonic"] == 0:
		with open(args["name"] +".csv", 'w') as f:
			csvwriter = csv.DictWriter(f, fields_non)
			csvwriter.writeheader()
	if args["ultrasonic"] == 1:
		with open(args["name"] +".csv", 'w') as f:
			csvwriter = csv.DictWriter(f, fields_ultra)
			csvwriter.writeheader()


#load YOLO	  
	if args["dataset"] == "original":
		net = cv2.dnn.readNet("trained_data/original_yolov3/yolov3.weights","trained_data/original_yolov3/yolov3.cfg") #original Yolov3
	if args["dataset"] == "tiny":
		net = cv2.dnn.readNet("trained_data/tiny/yolov3-tiny.weights","trained_data/tiny/yolov3-tiny.cfg") #Tiny Yolo
	if args["dataset"] == "shape":
		net = cv2.dnn.readNet("trained_data/shape/yolov3-tiny-shape_best.weights","trained_data/shape/yolov3-tiny-shape.cfg") #shape dataset
	if args["dataset"] == "46_food":
		net = cv2.dnn.readNet("trained_data/46_food/yolov4-tiny-food_best.weights","trained_data/46_food/yolov4-tiny-food.cfg") # 46 classes food
	if args["dataset"] == "46_food_noised":
		net = cv2.dnn.readNet("trained_data/46_noised_food/yolov4-tiny-food-noise_best.weights","trained_data/46_food/yolov4-tiny-food-noise.cfg") # 46 classes(noised) food
	if args["dataset"] == "4_food":
		net = cv2.dnn.readNet("trained_data/4_food/yolov4-tiny-food-4class_best.weights","trained_data/4_food/yolov4-tiny-food-4class.cfg") # 4 classes food

# set GPU run
	#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#load class file
	classes = []
	if args["dataset"] == "tiny" or "original":
		with open("trained_data/tiny/coco.names","r") as f:
			classes = [line.strip() for line in f.readlines()]
	if args["dataset"] == "shape":
		with open("trained_data/shape/obj.names","r") as f:
					classes = [line.strip() for line in f.readlines()]
	if args["dataset"] == "46_food":
		with open("trained_data/46_food/obj.names","r") as f:
					classes = [line.strip() for line in f.readlines()]
	if args["dataset"] == "46_food_noised":
		with open("trained_data/46_noised_food/obj.names","r") as f:
					classes = [line.strip() for line in f.readlines()]
	if args["dataset"] == "4_food":
		with open("trained_data/4_food/obj.names","r") as f:
				classes = [line.strip() for line in f.readlines()]

	print(classes)



	layer_names = net.getLayerNames()
	outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors= np.random.uniform(0,255,size=(len(classes),3))


	#loading image
	#if use open cv 4.2 +, then put cv2.CAP_V4L to prevent gstreaming error
	#cap=cv2.VideoCapture(int(args["camera_number"]), cv2.CAP_V4L) #0 for 1st webcam
	cap=cv2.VideoCapture(int(args["camera_number"])) #0 for 1st webcam
	cv2.namedWindow("Parameters")
	cv2.createTrackbar("Threshold1","Parameters",23,255,empty)
	cv2.createTrackbar("Threshold2","Parameters",20,255,empty)
	cv2.createTrackbar("Area","Parameters",5000,30000,empty)

	font = cv2.FONT_HERSHEY_PLAIN
	starting_time= time.time()
	frame_id = 0

	if cap.isOpened():
		camera_w = cap.get(3)
		camera_h = cap.get(4)
		
		#aspect ratio is the ratio of height to width
		aspect_ratio=camera_h/camera_w
		
		#diagonal fov is the angle between the rays of the corner of the image
		d_fov=78
		#we use the aspect ratio and the diagonal fov to calculate the horizontal and vertical fovs
		xfov=2*(np.arctan(np.tan(d_fov*np.pi/180)*np.cos(np.arctan(aspect_ratio))))
		yfov=2*(np.arctan(np.tan(d_fov*np.pi/180)*np.sin(np.arctan(aspect_ratio))))
		#using these, we can find the maximum values for x and y given a maximum visible depth, zmax
		zmax=1
		xmax=np.sin(xfov)*zmax
		ymax=np.sin(yfov)*zmax
		
		#we can then put these into a scale matrix. Multiplying a point in xyz space by this matrix will get us that point in [-1,1],[-1,1],[0,1]
		scale_mtx=np.array([[1/xmax,0,0],[0,1/ymax,0],[0,0,1/zmax]])
		#we can invert this matrix to get us a matrix that undoes this operation
		unscale_mtx=np.linalg.inv(scale_mtx)
		#we then use this to transform the point 1,1,1 to the world coordinates. as a check, this should be equal to [xmax,ymax,zmax]
		maxs=np.matmul(unscale_mtx,[1,1,1])

		#we can then check to make sure our calculations worked by using trig to find the d_fov of our world system
		print(maxs)
		print('using cos',360/np.pi*np.arccos(maxs[-1]/(np.sqrt(maxs[0]**2+maxs[1]**2+maxs[2]**2))))
		print('using sin',360/np.pi*np.arcsin((np.sqrt(maxs[0]**2+maxs[1]**2))/(np.sqrt(maxs[0]**2+maxs[1]**2+maxs[2]**2))))


	#read data from 
	if args["ultrasonic"] == 1:
		arduino_data = serial.Serial ('/dev/ttyACM0',9600) #change comX, Serial.begin(value)
		time.sleep(3)
		distance_in_cm = 0
		
		arduino_data.flush()
			#arduino_data.write('s'.encode())		  #'s', read range once
		arduino_data.write('c'.encode())		  #'c', read range continuously
			#arduino_data.write('t'.encode())		  #'t', timed measurement
			#arduino_data.write('.'.encode())		  #'.', stop measurement
			#arduino_data.write('d'.encode())		  #'d', dump corrleation record


	assert cap.isOpened(), 'Cannot capture source'
	
	while cap.isOpened():
		_,frame= cap.read()
		frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

		
		frame_id += 1
		height,width,channels = frame.shape

		# get center of blob image
		# https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
		# try:
		# 	gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# 	r,t = cv2.threshold(gray_image,127,255,0)
		# 	M = cv2.moments(t)

		# 	cX = int(M["m10"] / M["m00"])
		# 	cY = int(M["m01"] / M["m00"])
		# 	cv2.circle(frame, (cX, cY), 4, (255, 255, 255), -1)
		
		# except:
		# 	continue

		# center of image with white circle
		cv2.circle(frame, (int(ORIGIN_X), int(ORIGIN_Y)), 2, (255, 255, 255), -1)


		
		#detecting objects
		blob = cv2.dnn.blobFromImage(frame,0.00392,(VIDEO_WIDTH, VIDEO_HEIGHT),(0,0,0),True,crop=False)		   

			
		net.setInput(blob)
		outs = net.forward(outputlayers)

		#Showing info on screen/ get confidence score of algorithm in detecting an object in blob
		class_ids=[]
		confidences=[]
		boxes=[]
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					#object detected
					center_x= int(detection[0]*width)
					center_y= int(detection[1]*height)
					w = int(detection[2]*width)
					h = int(detection[3]*height)
					#cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
					#rectangle co-ordinaters
					x=int(center_x - w/2)
					y=int(center_y - h/2)
					#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

					boxes.append([x,y,w,h]) #put all rectangle areas
					confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
					class_ids.append(class_id) #name of the object tha was detected

		indexes = cv2.dnn.NMSBoxes(boxes,confidences,args["confidence"], args["threshold"])

			# initialize the list of threshold methods
		methods = [
		("THRESH_BINARY", cv2.THRESH_BINARY),
		("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
		("THRESH_TRUNC", cv2.THRESH_TRUNC),
		("THRESH_TOZERO", cv2.THRESH_TOZERO),
		("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]

		label = None

		without_bounding = frame.copy()
		food_counter = 0
		layover_flag = 0
		top_to_bottom = 0
		object_height = 0.0
		object_width = 0.0
		y_from_origin = 0.0
		x_from_origin = 0.0
		c = [None]*100
		for i in range(len(boxes)):
			if i in indexes:
				print(i)
				
				#hog = cv2.HOGDescriptor()	   #
				#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())		 #
				#found,w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.00) #
				print("found ", str(classes[class_ids[i]])) #

				# for food portion : cracker
				if str(classes[class_ids[i]]) == 'cracker':
					food_counter += 1
					

				#distane measurement
				x,y,w,h = boxes[i]
				get_number_of_object, get_distance= distance_to_camera(frame, boxes[i])
				#print(get_number_of_object, get_distance)
				#if get_number_of_object >=1 and get_distance!=0:
				#	 print("{}".format(get_number_of_object)+ " " + classes[class_ids[i]] +" at {}".format(round(get_distance))+" inches")
				label = str(classes[class_ids[i]])
				confidence= confidences[i]
				color = colors[class_ids[i]]
				pad_w, pad_h = int(0.3*w), int(0.05*h)
				cv2.rectangle(frame,(x, y), (x + w + pad_w, y + h + pad_h), color, 1)
				cv2.putText(frame, "x from center (pixel) : " + str(ORIGIN_X - ((x + x + w ) / 2)),(x, y + 95), font, 0.7, (255,255,255), 1)
				cv2.putText(frame, "y from center (pixel) : " + str(ORIGIN_Y - ((y + y + h ) / 2)),(x, y + 105), font, 0.7, (255,255,255), 1)
				
				cv2.putText(frame, label+" "+str(round(confidence,2)),(x,y-10),font,1,(255,255,255),2)

				(tl, tr) = (x, y)
				(bl, br) = (x+ w , y + h)
				(tltrX, tltrY) = midpoint(tl, tr)
				(blbrX, blbrY) = midpoint(bl, br)
				# compute the midpoint between the top-left and top-right points,
				# followed by the midpoint between the top-righ and bottom-right
				(tlblX, tlblY) = midpoint(tl, bl)
				(trbrX, trbrY) = midpoint(tr, br)

				# slice only detected part
				try:
					
					roi_color = without_bounding[y - pad_h : y + h + pad_h, x - pad_w : x + w + pad_w]
					#cv2.circle(frame, (y, x), 4, (0, 0, 255), -1)
					

					# draw contours part
					imgContour = roi_color.copy()
					blur = cv2.GaussianBlur(roi_color, (7, 7), 1)
					gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
					threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
					threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
					canny = cv2.Canny(gray,threshold1,threshold2)
					kernel = np.ones((5, 5))
					imgDil = cv2.dilate(canny, kernel, iterations=1)
					roi_color = frame[y - pad_h : y + h + pad_h, x - pad_w : x + w + pad_w]
					contours, cX, cY = get_contours(imgDil, imgContour, frame, x, y, w, h, roi_color)
					print("len:",len(contours))
					print("area:",cv2.contourArea(contours[i]))
					cv2.putText(frame, "Area: " + str(int(cv2.contourArea(contours[i]))), (x + w + 20, y + 45), font, 0.7, (0, 255, 0), 1)
					# try:
					# 	cv2.putText(frame, "y axis: " + str(math.dist((cX, cY), (160, 160))), (x + w + 20, y + 55), font, 0.7, (0, 255, 0), 1)
					# except:
					# 	continue

					c[i] = max(contours, key=cv2.contourArea)
					# determine the most extreme points along the contour
					extLeft = tuple(c[i][c[i][:, :, 0].argmin()][0])
					extRight = tuple(c[i][c[i][:, :, 0].argmax()][0])
					extTop = tuple(c[i][c[i][:, :, 1].argmin()][0])
					extBot = tuple(c[i][c[i][:, :, 1].argmax()][0])

					cv2.circle(roi_color, extLeft, 4, (0, 0, 255), -1)
					cv2.circle(roi_color, extRight, 4, (0, 255, 0), -1)
					cv2.circle(roi_color, extTop, 4, (255, 0, 0), -1)
					cv2.circle(roi_color, extBot, 4, (255, 255, 0), -1)
					
					cv2.line(roi_color, extTop, extBot, (255,255,255), 1)
					top_to_bottom = extBot[1] - extTop[1]
					cv2.putText(frame, "height: {}".format(top_to_bottom), (x + w + 20, y + 65), font, 0.7, color, 1)

					cv2.line(roi_color, extRight, extLeft, (255,255,255), 1)
					left_to_right = extRight[0] - extLeft[0] 
					cv2.putText(frame, "width: {}".format(left_to_right), (x + w + 20, y + 75), font, 0.7, color, 1)
					
					#cv2.circle(roi_color, (extTop - extBot) / 2 + extBot, 4, (0,255,0), -1)
					#cv2.circle(roi_color, (extRight - extLeft) / 2 + extLeft, 4, (0,255,0), -1)

				except:
					continue
				
				#x,y,radius = cv2.minEnclosingCircle(boxes[i])
				#center = (int(x+w/2),int(y+h/2))
				#radius = int(abs(math.tan(w/2)) * get_distance)
				#cv2.circle(frame,center,int(math.sqrt((w**2) + (h**2))/4),(0,255,0),2)

				#read data from US-100 ultrasonic sensor
				if args["ultrasonic"] == 1:
					arduino_data.flush()
					distance_cm = arduino_data.readline().strip()
					distance_in_cm = distance_cm.decode('utf-8')[:]
					distance_in_cm = float(''.join(filter(str.isdigit, distance_in_cm)))
					cv2.putText(frame, "x axis: "+ distance_in_cm, (x+150,y+30), font, 0.7, (0, 255, 0), 1)

					# x axis (object height) formula with y axis (the distance from camera to object)
					# Real Object Height = (Distance to Object x Object Height on sensor) / Camera Focal Length 
					object_height = (distance_in_cm * top_to_bottom * 0.1) / (FOCAL_LENGTH * 10)	# unit = cm
					object_width = (distance_in_cm * left_to_right * 0.1) / (FOCAL_LENGTH * 10)	# unit = cm
					cv2.putText(frame, "object height: " + str(object_height) + " cm", (x + w + 20, y + 80), font, 0.7, (0, 255, 0), 1)

					# convert x & y axis in cm
					#x_from_origin = (distance_in_cm * (ORIGIN_X - ((x + x + w ) / 2)) * 0.1) / (FOCAL_LENGTH * 10)
					#y_from_origin = (distance_in_cm * (ORIGIN_Y - ((y + y + h ) / 2)) * 0.1) / (FOCAL_LENGTH * 10)

					#this updates those values from normalized (-1,1) to pixel coords
					upix=((ORIGIN_X - ((x + x + w ) / 2)+1)*camera_w/2
					vpix=((ORIGIN_Y - ((y + y + h ) / 2)+1)*camera_h/2
					#this finds the x and y coordinates on a normalized world frame
					x=(ORIGIN_X - ((x + x + w ) / 2)*distance_in_cm
					y=(ORIGIN_Y - ((y + y + h ) / 2)*distance_in_cm
					#this then takes those coordinates and scales them back to real world coordinates
					coord_list=np.matmul(unscale_mtx,[x,y,distance_in_cm])
					x,y,Z=coord_list[0],coord_list[1],coord_list[2]
					#this adds the other end of the bar
					extra_point=np.asarray([x + FOCAL_LENGTH, y, Z])
					#this takes the bar's world coordinates and scales them back to normalized world coordinates
					norm_point=np.matmul(scale_mtx,extra_point)
					#this calculates the normalized image coordinates for the other end of the bar
					point_u=norm_point[0]/norm_point[2]
					point_v=norm_point[1]/norm_point[2]
					#this converts the normalized image coordinates to pixel coordinates
					pix_u,pix_v=(point_u+1)*camera_w/2,(point_v+1)*camera_h/2
					x_from_origin = pix_u
					y_from_origin = pix_v
					
					
					

		elapsed_time = time.time() - starting_time
		fps_cal=frame_id/elapsed_time
		cv2.putText(frame,"FPS:"+str(round(fps_cal,2)),(10,50),font,2,(0,0,0),1)
		cv2.putText(frame, "Press 's' to save data in data.csv", (5, VIDEO_HEIGHT - 20), font, 1, (0,255,0),1)
		
	#	 if ret == True:
	#		 objpoints.append(objp)
	#		 corners2 = cv2.cornersubPix(frame_calibration, (11, 11), (-1, -1), criteria)
	#		 imgpoints.append(corners2)
	#		 frame_calibration = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

		cv2.imshow("Video", frame)
		key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
		
		if key == 27: #esc key stops the process
			break
		if key == ord('s'):
			if label != None and args["ultrasonic"] == 0:
				print("-- save data -- ")
				print("object : {}, accuracy : {} %".format(label, str(round(confidence,2))))
				with open(args["name"] + ".csv", 'a') as csvfile:
					csvwriter = csv.writer(csvfile)
					csvwriter.writerow([label, round(confidence,2)])
			else:
				print("ERROR : any object doesn't be detected !!")

			if label != None and args["ultrasonic"] == 1:
				print("-- save data -- ")
				print("object : {}, distance : {} cm, height : {} cm, x & y from origin : ({},{}) cm, accuracy : {} %".format(label, distance_in_cm, object_height, x_from_origin, y_from_origin, str(round(confidence,2))))
				with open(args["name"] + ".csv", 'a') as csvfile:
					csvwriter = csv.writer(csvfile)
					# [x0, y0, z0, w, h, shape]
					# X0 and y0 are the center of the object, z0 is distance from camera to object. 
					# W and h are width and height, and shape is object class
					csvwriter.writerow([x_from_origin, y_from_origin, distance_in_cm, object_width, object_height, label, round(confidence,2)])
			else:
				print("ERROR : any object doesn't be detected !!")


	# check elasped time
	cap.release()		 
	cv2.destroyAllWindows()
