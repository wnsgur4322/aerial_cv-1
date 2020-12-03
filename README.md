# Aerial_CV - repo for aerial team computer vision

## Before run the project
make sure to install opencv +4.x
Since original YOLOv3 weight file is too heavy, please download here:
https://pjreddie.com/media/files/yolov3.weights
and put the file in trained_data/original_yolov3

## How to run the project
1. Detect object of the input image file
- python3 yolo_img.py [image file path] -w [choose one: tiny, shape, original]
2. Detect object with webcam (real time)
- python3 yolo_webcam.py -w [choose one: tiny, shape, original]


# Object Detection Notes
## note 1/1/20
Junhyeok Jeong (Derek)
- To use object detection tool, check out YOLOv3 with Pytorch
- Recommend CUDA (good GPU) to boost real-time FPS
- I use NVIDIA Jetson tx2 development kit and built in camera

## note 1/15/20
Junhyeok Jeong (Derek)
- Jetson has an error on OpenCV (4.1.x) + YOLOv3. I guess it is architecture error.

## note 1/20/20
Junhyeok Jeong (Derek)
- jetson tx2 bug fixed, but too much video FPS drop
- Use my desktop (Nvidia RTX 2070) + CUDA 10.2 in Ubuntu
- training own data set (65 images) with 53 layers, the result was low loss. But, limitied only for few objects and RGB.

## note 1/25/20
Junhyeok Jeong (Derek)
- Training basic shape data set with 152 images and 13 CNN layers


![training result](https://github.com/wnsgur4322/aerial_cv-1/blob/master/152%20shape-images%20training%20result.png)
![training chart](https://github.com/wnsgur4322/aerial_cv-1/blob/master/chart_yolov3-tiny-shape.png)
