import numpy as np
import cv2

im = cv2.imread ('sudoku_grid3.png') #('sudoCleanedOfRects.jpg')# # read picture

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # BGR to grayscale

ret, thresh = cv2.threshold(imgray, 120, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

rects = [cv2.boundingRect(ctr) for ctr in contours]


max_area_rect = 0
max_area_ctr = 0
max_area = 0

for ctr in contours:
	rect = cv2.boundingRect(ctr)

	if(rect[2]*rect[3] > max_area):
		max_area = rect[2]*rect[3]
		max_area_rect = rect
		max_area_ctr = ctr

# cv2.rectangle(im, (max_area_rect[0], max_area_rect[1]), (max_area_rect[0] + max_area_rect[2], max_area_rect[1] + max_area_rect[3]), (0, 0, 255), 1)

epsilon = 0.01 * cv2.arcLength(max_area_ctr, True)
approx = cv2.approxPolyDP(max_area_ctr, epsilon, True)
im = cv2.drawContours(im, [approx], -1, (0, 255, 0), 1)

im_rect =  im[max_area_rect[1]:max_area_rect[1]+max_area_rect[3], max_area_rect[0]:max_area_rect[0]+max_area_rect[2]]
print(approx.shape[0])

up_left = approx[0][0]
up_right = approx[3][0]
bottom_left = approx[1][0]
bottom_right = approx[2][0]

input_pts = np.float32([up_left,up_right,bottom_left,bottom_right])

height = max(abs(bottom_left[1] - up_left[1]), abs(up_right[1]-bottom_right[1]))
width = max(abs(bottom_left[0] - bottom_right[0]), abs(up_left[0]-up_right[0]))
output_pts = np.float32([[0, 0], [width, 0],[0, height],[width, height]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)
 
# Apply the perspective transformation to the image
out = cv2.warpPerspective(im,M,(width, height),flags = cv2.INTER_LINEAR)

# cv2.imshow("Contour", cv2.resize(im, (500, 500)))
cv2.imshow("title", out)

cv2.imshow("bla", im_rect)
cv2.waitKey()
cv2.destroyAllWindows()