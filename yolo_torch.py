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


