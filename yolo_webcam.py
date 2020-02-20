#to do list
# 1. object detection
# 2. recollect data with attached camera angle and train it
# 3. figure out the size of object https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

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

FOCAL_LENGTH = 3.67 # logitech c920 webcam

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


if __name__ == "__main__":
	#calibration
	#termination crietria
	#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	#objp=np.zeros((6 * 7, 3), np.float32)
	#objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
	ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
	args = vars(ap.parse_args())

#load YOLO
	#net = cv2.dnn.readNet("/home/kimchi/graspinglab/darknet/yolov3.weights","/home/kimchi/graspinglab/darknet/cfg/yolov3.cfg") # Original yolov3
	#net = cv2.dnn.readNet("/home/kimchi/graspinglab/darknet/yolov3-tiny.weights","/home/kimchi/graspinglab/darknet/cfg/yolov3-tiny.cfg") #Tiny Yolo
	net = cv2.dnn.readNet("/home/kimchi/graspinglab/darknet/backup/yolov3-tiny-shape_best.weights","/home/kimchi/graspinglab/darknet/cfg/yolov3-tiny-shape.cfg") #shape dataset 
	#net = cv2.dnn.readNet("/home/kimchi/graspinglab/darknet/backup/face/yolov3-tiny-face_best.weights","/home/kimchi/graspinglab/darknet/cfg/yolov3-tiny-face.cfg") # face dataset (only me)
# set GPU run
	#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#load class file
	classes = []
	#with open("/home/kimchi/graspinglab/darknet/data/coco.names","r") as f:
	with open("/home/kimchi/graspinglab/darknet/data/obj.names","r") as f:
		classes = [line.strip() for line in f.readlines()]

	print(classes)



	layer_names = net.getLayerNames()
	outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors= np.random.uniform(0,255,size=(len(classes),3))


#loading image
	#if use open cv 4.2 +, then put cv2.CAP_V4L to prevent gstreaming error
	cap=cv2.VideoCapture(2, cv2.CAP_V4L) #0 for 1st webcam

	font = cv2.FONT_HERSHEY_PLAIN
	starting_time= time.time()
	frame_id = 0

	while True:
		_,frame= cap.read() # 
		frame = cv2.resize(frame, (640, 480))
		
		#calibration
		#frame_calibration = frame
		#gray = cv2.cvtColor(frame_calibration, cv2.COLOR_BGR2GRAY)

		#ret, corners = cv2.findChessboardCorners(frame_calibration, (7, 6), None)
		frame_id+=1
		
		height,width,channels = frame.shape
		#detecting objects
		blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False)    

			
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

		indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


		for i in range(len(boxes)):
			if i in indexes:
				print(i)
				
				hog = cv2.HOGDescriptor()   #
				hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())    #
				found,w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.00) #
				print("found ") #

				#distane measurement
				x,y,w,h = boxes[i]
				get_number_of_object, get_distance= distance_to_camera(frame, boxes[i])
				print(get_number_of_object, get_distance)
				if get_number_of_object >=1 and get_distance!=0:
					print("{}".format(get_number_of_object)+ " " + classes[class_ids[i]] +" at {}".format(round(get_distance))+"Inches")
				label = str(classes[class_ids[i]])
				confidence= confidences[i]
				color = colors[class_ids[i]]
				#pad_w, pad_h = int(0.15*w), int(0.05*h)
				cv2.rectangle(frame,(x, y), (x+w, y+h), color, 2)
				#cv2.rectangle(frame,(x + pad_w, y + pad_h), (x+w - pad_w, y+h - pad_h), color, 2)
				
				cv2.putText(frame, label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
				cv2.putText(frame, str(round(get_distance))+" Inches", (x+150,y+30), font, 1,(255,0,0),2)
				
		elapsed_time = time.time() - starting_time
		fps_cal=frame_id/elapsed_time
		cv2.putText(frame,"FPS:"+str(round(fps_cal,2)),(10,50),font,2,(0,0,0),1)
		
	#	if ret == True:
	#		objpoints.append(objp)
	#		corners2 = cv2.cornersubPix(frame_calibration, (11, 11), (-1, -1), criteria)
	#		imgpoints.append(corners2)
	#		frame_calibration = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

		cv2.imshow("Video",frame)
		key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
		
		if key == 27: #esc key stops the process
			break


	# check elasped time
	cap.release()    
	cv2.destroyAllWindows()
