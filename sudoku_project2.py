import numpy as np
import cv2



cap = cv2.VideoCapture(0)
out = 0
while(1):
	ret, im = cap.read()
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # BGR to grayscale
	# imgray = cv2.GaussianBlur(imgray, (3, 3), 0)
	thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,35,12)
	# ret, thresh = cv2.threshold(imgray,90, 255, cv2.THRESH_BINARY_INV)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_NONE)

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

	epsilon = 0.01 * cv2.arcLength(max_area_ctr, True)
	approx = cv2.approxPolyDP(max_area_ctr, epsilon, True)
	im = cv2.drawContours(im, [approx], -1, (0, 255, 0), 1)

	if(approx.shape[0] == 4):
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
		cv2.imshow("out", out)
		# cv2.imwrite("out.png", out)
	if(str(out) == "None" or str(out) == '0'):
		continue
	img1 = out
	gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (3, 3), 0)
	# ret, im_th = cv2.threshold(gray, 90, 90, cv2.THRESH_BINARY_INV)
	im_th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY_INV,35,12)

	# cv2.imshow("title", im_th)
	# cv2.waitKey()
	edges = cv2.Canny(im_th, 1, 1, apertureSize = 3)
	# cv2.imshow("im_th", im_th)
	# cv2.waitKey()
	shape = gray.shape
	lines = cv2.HoughLines(edges,1,np.pi/900,100)
	if(str(lines) == "None"):
		continue
	for i in range(len(lines)):
		for rho,theta in lines[i]:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a*rho
		    y0 = b*rho
		    x1 = int(x0 + (1000)*(-b)) # shape[0]
		    y1 = int(y0 + (1000)*(a))  # shape[1]
		    x2 = int(x0 - (1000)*(-b))
		    y2 = int(y0 - (1000)*(a))
		    # print(x1, y1, x2, y2)

		    cv2.line(img1, (x1,y1),(x2,y2),(255,255,255),3) # 2 points ,color, thikness

	cv2.imshow("im", im)
	# cv2.imshow("thresh", thresh)
	k = cv2.waitKey(30) & 0xff 
	if k == 27: # 27 is ascii code for ESC
		break


cap.release()
cv2.destroyAllWindows()