#written by Junhyeok Jeong
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
import torch
import torch.nn as nn
from darknet import Darknet


def arg_parse():

    #Parse arguements to the detect module
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        #default = "/home/kimchi/graspinglab/darknet/cfg/yolov3.cfg", type = str)
                        default = "/home/kimchi/graspinglab/darknet/cfg/yolov3-tiny-shape.cfg", type = str) #basic shape cfg
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        #default = "/home/kimchi/graspinglab/darknet/yolov3.weights", type = str)
                        default = "/home/kimchi/graspinglab/darknet/backup/yolov3-tiny-shape_best152.weights", type = str) #basic shape weights
    parser.add_argument("--resolution", dest = 'resolution', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 4
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.resolution
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32




