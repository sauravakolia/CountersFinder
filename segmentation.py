import cv2
import numpy as np 
import argparse
import time
import os
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2

image_path="cropped_images\\car2.JPG"

image=cv2.imread(image_path)
cv2.imshow("img",image)
cv2.waitKey(0)

edges = cv2.Canny(image,100,200)
cv2.imshow("img",image)
cv2.waitKey(0)



gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
cv2.imshow("black",gray)
cv2.waitKey(0)

# cv2.imshow("gray",gray)
blur = cv2.GaussianBlur(gray, (5,5), 1)
gray = cv2.medianBlur(gray, 3)

# perform otsu thresh (using binary inverse since opencv contours work better with white text)
ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# print("thresh")
cv2.imshow("thresh",thresh)
cv2.waitKey(0)
rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

	# apply dilation 
 
# define the named tupled to store the license plate
# LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])

# # extract the Value component from the HSV color space and apply adaptive thresholding
# # to reveal the characters on the license plate
# V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
# T = threshold_local(V, 29, offset=15, method="gaussian")
# thresh = (V > T).astype("uint8") * 255
# thresh = cv2.bitwise_not(thresh)

# # resize the license plate region to a canonical size
# plate = imutils.resize(image, width=400)
# thresh = imutils.resize(thresh, width=400)
# cv2.imshow("Thresh", thresh)
# cv2.waitKey(0)

# 		# perform a connected components analysis and initialize the mask to store the locations
# 		# of the character candidates
# labels = measure.label(thresh, connectivity=2, background=0)
# charCandidates = np.zeros(thresh.shape, dtype="uint8")


# # loop over the unique components
# for label in np.unique(labels):
# # if this is the background label, ignore it
# 	if label == 0:
# 		continue

# 	# otherwise, construct the label mask to display only connected components for the
# 	# current label, then find contours in the label mask
# 	labelMask = np.zeros(thresh.shape, dtype="uint8")
# 	labelMask[labels == label] = 255
# 	cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 	cnts = cnts[0] if imutils.is_cv2() else cnts[1]





# # ensure at least one contour was found in the mask
# if len(cnts) > 0:
# 	# grab the largest contour which corresponds to the component in the mask, then
# 	# grab the bounding box for the contour
# 	c = max(cnts, key=cv2.contourArea)
# 	(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

# 	# compute the aspect ratio, solidity, and height ratio for the component
# 	aspectRatio = boxW / float(boxH)
# 	solidity = cv2.contourArea(c) / float(boxW * boxH)
# 	heightRatio = boxH / float(plate.shape[0])

# 	# determine if the aspect ratio, solidity, and height of the contour pass
# 	# the rules tests
# 	keepAspectRatio = aspectRatio < 1.0
# 	keepSolidity = solidity > 0.15
# 	keepHeight = heightRatio > 0.4 and heightRatio < 0.95

# 	# check to see if the component passes all the tests
# 	if keepAspectRatio and keepSolidity and keepHeight:
# 		# compute the convex hull of the contour and draw it on the character
# 		# candidates mask
# 		hull = cv2.convexHull(c)
# 		cv2.drawContours(charCandidates, [hull], -1, 255, -1)