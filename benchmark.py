import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img = cv.imread('./AGAR_representative/higher-resolution/dark/5271.jpg')
img = cv.imread('water_coins.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

plt.imshow(img)
plt.savefig('./img_out/img.png')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

plt.imshow(gray)
plt.savefig('./img_out/gray.png')

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

plt.imshow(opening)
plt.savefig('./img_out/opening.png')

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

plt.imshow(sure_bg)
plt.savefig('./img_out/sure_bg.png')

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)

plt.imshow(dist_transform)
plt.savefig('./img_out/dist_transform.png')

ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

plt.imshow(sure_fg)
plt.savefig('./img_out/sure_fg.png')

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

plt.imshow(markers)
plt.savefig('./img_out/markers_1.png')

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.imshow(markers)
plt.savefig('./img_out/markers_2.png')

plt.imshow(img)
plt.savefig('./img_out/img_final.png')