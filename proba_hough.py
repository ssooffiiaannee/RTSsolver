import numpy as np
import cv2

import cv2
import numpy as np


img = cv2.imread('sudoku_grid1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# gray = cv2.GaussianBlur(gray, (3, 3), 0)

# ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
             cv2.THRESH_BINARY_INV, 35, 12)

#//////////// probabilistic hough //////////////////////

edges = cv2.Canny(thresh,120,120,apertureSize = 3)
minLineLength = 10 # 100 Minimum length of line. Line segments shorter than this are rejected.
maxLineGap = 1	 # Maximum allowed gap between line segments to treat them as single line.
lines = cv2.HoughLinesP(edges,1,np.pi/1800,100,minLineLength,maxLineGap)

for line in lines:
	for x1,y1,x2,y2 in line:
		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("title", cv2.resize(img, (500, 500)))
# cv2.imshow("thresh", thresh)
cv2.waitKey()

#//////////// hough ////////////////////////////////////


# edges = cv2.Canny(thresh, 50, 200, apertureSize = 3) # min_val max_ val are intensity edges
# lines = cv2.HoughLines(edges, 1, np.pi/180, 100) 	 # (edges, ro_accuracy(pixels), theta_accuracy (rads), minimum_vote)
# print(lines.shape)
# for line in lines:
# 	for rho,theta in line:
# 	    a = np.cos(theta)
# 	    b = np.sin(theta)
# 	    x0 = a*rho
# 	    y0 = b*rho
# 	    x1 = int(x0 + 1000*(-b))
# 	    y1 = int(y0 + 1000*(a))
# 	    x2 = int(x0 - 1000*(-b))
# 	    y2 = int(y0 - 1000*(a))

# 	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

# cv2.imshow("title", img)
# # cv2.imshow("title", cv2.resize(thresh, (500, 500)))
# cv2.waitKey()