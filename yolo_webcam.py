#to do list
# 1. object detection
# 2. recollect data with attached camera angle and train it 
# 3. figure out the size of object https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
# 	- find pixels per metric. ex) 1 pixel = 0.001 cm
#	- set up reference distance ex) if distance is 1m between object and camera, then 1 pixel is 0.001 cm
#	- convert pixels of contour box into size scale.
# 	- grasping as much as the size scale.

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
import syslog
import csv
import os

VIDEO_HEIGHT = 320
VIDEO_WIDTH = 320


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
			img  : Gets the input frame
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
            cv2.drawContours(roi_color, cnt, -1, (255, 0, 255), 1)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            x2 , y2 , w2, h2 = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x , y), (x + w + pad_w , y + h + pad_h), (0, 255, 0), 1)

            cv2.putText(frame, "Points: " + str(len(approx)), (x + w + 20, y + 20), font, 0.7, (0, 255, 0), 1)
            cv2.putText(frame, "Area: " + str(int(area)), (x + w + 20, y + 45), font, 0.7, (0, 255, 0), 1)

    return contours

def empty(a):
    pass


if __name__ == "__main__":
	#termination crietria
	#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	#objp=np.zeros((6 * 7, 3), np.float32)
	#objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	ap = argparse.ArgumentParser()
	ap.add_argument("-w", "--dataset", type=str,
	help="set object detection weight and data file (type original or tiny or shape)")
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
	fields = ['Object', 'Distance (cm)', 'Accuracy(%)']

	if os.path.exists("./"+ args["name"] + ".csv"):
		os.remove("./"+ args["name"] + ".csv")
		print("existed data removed")
	else:
		print("can not delete old csv file as it doesn't exists")

	with open(args["name"] +".csv", 'w') as f:
		csvwriter = csv.DictWriter(f, fields)
		csvwriter.writeheader()

#load YOLO	
	if args["dataset"] == "original":
		net = cv2.dnn.readNet("trained_data/original_yolov3/yolov3.weights","trained_data/original_yolov3/yolov3.cfg") #original Yolov3
	if args["dataset"] == "tiny":
		net = cv2.dnn.readNet("trained_data/tiny/yolov3-tiny.weights","trained_data/tiny/yolov3-tiny.cfg") #Tiny Yolo
	if args["dataset"] == "shape":
		net = cv2.dnn.readNet("trained_data/shape/yolov3-tiny-shape_best.weights","trained_data/shape/yolov3-tiny-shape.cfg") #shape dataset

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

	print(classes)



	layer_names = net.getLayerNames()
	outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors= np.random.uniform(0,255,size=(len(classes),3))


#loading image
	#if use open cv 4.2 +, then put cv2.CAP_V4L to prevent gstreaming error
	cap=cv2.VideoCapture(args["camera_number"], cv2.CAP_V4L) #0 for 1st webcam
	cv2.namedWindow("Parameters")
	cv2.createTrackbar("Threshold1","Parameters",23,255,empty)
	cv2.createTrackbar("Threshold2","Parameters",20,255,empty)
	cv2.createTrackbar("Area","Parameters",5000,30000,empty)

	font = cv2.FONT_HERSHEY_PLAIN
	starting_time= time.time()
	frame_id = 0

	#read data from arduino
	#arduino_data = serial.Serial ('/dev/ttyACM0',9600) #change comX, Serial.begin(value)
	#time.sleep(3)
	distance_in_cm = 0
	
	#arduino_data.flush()
    #arduino_data.write('s'.encode())     #'s', read range once
	#arduino_data.write('c'.encode())     #'c', read range continuously
    #arduino_data.write('t'.encode())     #'t', timed measurement
    #arduino_data.write('.'.encode())     #'.', stop measurement
    #arduino_data.write('d'.encode())     #'d', dump corrleation record


	assert cap.isOpened(), 'Cannot capture source'

	while cap.isOpened():
		_,frame= cap.read() # 
		frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

		# imgContour = frame.copy()
		# blur = cv2.GaussianBlur(frame, (7, 7), 1)
		# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
		# threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
		# threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
		# canny = cv2.Canny(gray,threshold1,threshold2)
		# kernel = np.ones((5, 5))
		# imgDil = cv2.dilate(canny, kernel, iterations=1)
		# get_contours(imgDil,imgContour)
		# stacks = stack_frames(0.8,([frame, canny], [imgDil, imgContour]))
		
		frame_id += 1
		height,width,channels = frame.shape
		
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
				if confidence > 0.6:
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


		for i in range(len(boxes)):
			if i in indexes:
				print(i)
				
				#hog = cv2.HOGDescriptor()   #
				#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())    #
				#found,w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.00) #
				print("found ", str(classes[class_ids[i]])) #

				#distane measurement
				x,y,w,h = boxes[i]
				get_number_of_object, get_distance= distance_to_camera(frame, boxes[i])
				#print(get_number_of_object, get_distance)
				#if get_number_of_object >=1 and get_distance!=0:
				#	print("{}".format(get_number_of_object)+ " " + classes[class_ids[i]] +" at {}".format(round(get_distance))+" inches")
				label = str(classes[class_ids[i]])
				confidence= confidences[i]
				color = colors[class_ids[i]]
				pad_w, pad_h = int(0.15*w), int(0.05*h)
				cv2.rectangle(frame,(x, y), (x+ w + pad_w, y + h + pad_h), color,  1)
				#cv2.rectangle(frame,(x + pad_w, y + pad_h), (x+w - pad_w, y+h - pad_h), color, 2)
				
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
					roi_color = frame[y : y + h + pad_h, x : x + w + pad_w]

					# draw contours part
					imgContour = roi_color.copy()
					blur = cv2.GaussianBlur(roi_color, (7, 7), 1)
					gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
					threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
					threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
					canny = cv2.Canny(gray,threshold1,threshold2)
					kernel = np.ones((5, 5))
					imgDil = cv2.dilate(canny, kernel, iterations=1)
					get_contours(imgDil, imgContour, frame, x, y, w, h, roi_color)
				except:
					continue
				# c = max(imgContour, key=cv2.contourArea)

				# # determine the most extreme points along the contour
				# extLeft = tuple(c[c[:, :, 0].argmin()][0])
				# extRight = tuple(c[c[:, :, 0].argmax()][0])
				# extTop = tuple(c[c[:, :, 1].argmin()][0])
				# extBot = tuple(c[c[:, :, 1].argmax()][0])
				# cv2.drawContours(roi_color, cts, -1, (0, 255, 0))

				# cv2.circle(roi_color, extLeft, 4, (0, 0, 255), -1)
				# cv2.circle(roi_color, extRight, 4, (0, 255, 0), -1)
				# cv2.circle(roi_color, extTop, 4, (255, 0, 0), -1)
				# cv2.circle(roi_color, extBot, 4, (255, 255, 0), -1)

				# cv2.line(roi_color, extTop, extBot, (255,255,255), 1)
				# top_to_bottom = extBot[1] - extTop[1]
				# cv2.putText(frame, "{}".format(top_to_bottom), (extBot[1], extTop[1] + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

				# cv2.line(roi_color, extRight, extLeft, (255,255,255), 1)
				# left_to_right = extRight[0] - extLeft[0] 
				# cv2.putText(frame, "{}".format(left_to_right), (extLeft[0] + 220, extRight[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				
				#read data from LIDAR lite v3 HP
				#arduino_data.flush()
				#arduino_data.write('s'.encode())     #'s', read range once

				#read data from US-100 ultrasonic sensor
				#arduino_data.flush()
				#distance_cm = arduino_data.readline().strip()
				#distance_in_cm = distance_cm.decode('utf-8')[1:]
				#cv2.putText(frame, distance_in_cm, (x+150,y+30), font, 1,(255,255,255),2)
				#cv2.putText(frame, str(round(get_distance))+" inches", (x+140,y-10), font, 1,(255,255,255),2)
				
				#x,y,radius = cv2.minEnclosingCircle(boxes[i])
				center = (int(x+w/2),int(y+h/2))
				radius = int(abs(math.tan(w/2)) * get_distance)
				#cv2.circle(frame,center,int(math.sqrt((w**2) + (h**2))/4),(0,255,0),2)
				
		elapsed_time = time.time() - starting_time
		fps_cal=frame_id/elapsed_time
		cv2.putText(frame,"FPS:"+str(round(fps_cal,2)),(10,50),font,2,(0,0,0),1)
		cv2.putText(frame, "Press 's' to save data in data.csv", (5, VIDEO_HEIGHT - 20), font, 1, (0,255,0),1)
		
	#	if ret == True:
	#		objpoints.append(objp)
	#		corners2 = cv2.cornersubPix(frame_calibration, (11, 11), (-1, -1), criteria)
	#		imgpoints.append(corners2)
	#		frame_calibration = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

		cv2.imshow("Video", frame)
		key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
		
		if key == 27: #esc key stops the process
			break
		if key == ord('s'):
			if label != None:
				print("-- save data -- ")
				distance_in_cm = "10"
				print("object : {}, distance : {} cm, accuracy : {} %".format(label, distance_in_cm, str(round(confidence,2))))
				with open(args["name"] + ".csv", 'a') as csvfile:
					csvwriter = csv.writer(csvfile)
					csvwriter.writerow([label, distance_in_cm, round(confidence,2)])
			else:
				print("ERROR : any object doesn't be detected !!")


	# check elasped time
	cap.release()    
	cv2.destroyAllWindows()
