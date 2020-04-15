import cv2
import numpy as np
import glob

img_arr = []

for filename in glob.glob('./frames/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_arr.append(img)

out = cv2.VideoWriter('image_to_video.mp4', 0, 30, size)

for i in range(len(img_arr)):
    out.write(img_arr[i])

cv2.destroyAllWindows()
out.release()