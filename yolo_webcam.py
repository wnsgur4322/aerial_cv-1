#Written by Junhyeok Jeong
#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import time

KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.0

def distance_to_camera(known_width, focal_length, perceived_length):
    #computer and return the distance from the object box to camera
    return (known_width * focal_length) / perceived_length

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
    distancei = 0.0
    for x, y, w, h in rects:
        print(len(rects))

        if len(rects) >= 0: ### Increase the value of count if there are more than one rectangle in a given frame
            count += 1
        distancei = (2 * 3.14 * 180) / (w + h * 360) * 1000 + 3 ### Distance measuring in Inch
        # print(distancei)
        # distance = distancei * 2.54
        # print(distance)
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
    return count, distancei


#load YOLO
#net = cv2.dnn.readNet("/home/kimchi/graspinglab/darknet/yolov3.weights","/home/kimchi/graspinglab/darknet/cfg/yolov3.cfg") # Original yolov3
net = cv2.dnn.readNet("/home/kimchi/graspinglab/darknet/yolov3-tiny.weights","/home/kimchi/graspinglab/darknet/cfg/yolov3-tiny.cfg") #Tiny Yolo
#net = cv2.dnn.readNet("/home/kimchi/graspinglab/darknet/backup/yolov3-tiny-shape_best.weights","/home/kimchi/graspinglab/darknet/cfg/yolov3-tiny-shape.cfg") #shape dataset 
# set GPU run
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#load class file
classes = []
with open("/home/kimchi/graspinglab/darknet/data/obj.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)



layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors= np.random.uniform(0,255,size=(len(classes),3))


#loading image
cap=cv2.VideoCapture(-1) #0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0

while True:
    _,frame= cap.read() # 
    frame = cv2.resize(frame, (640, 640))
    frame_id+=1
    
    height,width,channels = frame.shape
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False)    

        
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])


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
                #onject detected
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
            #focal_length = (i * KNOWN_DISTANCE) / KNOWN_WIDTH #
            #inches = distance_to_camera(KNOWN_WIDTH, focal_length, i) #
            hog = cv2.HOGDescriptor()   #
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())    #
            found,w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05) #

            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
            

    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
    
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break
    
cap.release()    
cv2.destroyAllWindows()
