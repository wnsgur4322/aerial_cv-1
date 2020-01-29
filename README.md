# Aerial_CV

## repo for aerial team computer vision

## note
Junhyeok Jeong (Derek)
- To use object detection tool, check out YOLOv3 with Pytorch
- Recommend CUDA (good GPU) to boost real-time FPS
- I use NVIDIA Jetson tx2 development kit and built in camera

# note 1/15/20
Junhyeok Jeong (Derek)
- Jetson has an error on OpenCV (4.1.x) + YOLOv3. I guess it is architecture error.

## note 1/20/20
Junhyeok Jeong (Derek)
- jetson tx2 bug fixed, but too much video FPS drop
- Use my desktop (Nvidia RTX 2070) + CUDA 10.2 in Ubuntu
- training own data set (65 images), low loss. But, limitied only for few objects and RGB.

## note 1/25/20
Junhyeok Jeong (Derek)
- Training data set with 152 images

![training result](https://github.com/wnsgur4322/aerial_cv-1/152 shape-images training result.png)
