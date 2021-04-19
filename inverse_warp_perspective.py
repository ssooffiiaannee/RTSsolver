import numpy as np
import cv2

# https://theailearner.com/tag/cv2-warpperspective/

im = cv2.imread ('sudoku_grid3.png')#('sudoCleanedOfRects.jpg')# # read picture

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # BGR to grayscale

ret, thresh = cv2.threshold(imgray, 120, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	
# Specify input and output coordinates that is used
# to calculate the transformation matrix

up_left = [28, 14]
up_right = [321, 42]
bottom_left = [18, 305]
bottom_right = [295, 318]

input_pts = np.float32([up_left, up_right, bottom_left, bottom_right])

height = max(abs(bottom_left[1] - up_left[1]), abs(up_right[1]-bottom_right[1])) 
width = max(abs(bottom_left[0] - bottom_right[0]), abs(up_left[0]-up_right[0]))
output_pts = np.float32([[0, 0], [width, 0],[0, height],[width, height]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts,output_pts)
 
# Apply the perspective transformation to the imag
out = cv2.warpPerspective(im, M, (width, height), flags = cv2.INTER_LINEAR)
cv2.putText(out, "sofiane", (100, 100),cv2.FONT_HERSHEY_DUPLEX, 2, (200, 0, 0), 2)

# /////////////////////// uncomment this //////////////////
# cv2.fillConvexPoly(out, np.array([(0, 0), (width, 0), (width, height), (0, height)]), (255,255,255))

out2 = cv2.warpPerspective(out, np.linalg.inv(M), (width, height), flags = cv2.INTER_LINEAR)
# out2 = cv2.add(out2, im)

# Display the transformed image
cv2.imshow("out", out)
cv2.imshow("title", out2)
cv2.imshow("img", im)
cv2.waitKey()