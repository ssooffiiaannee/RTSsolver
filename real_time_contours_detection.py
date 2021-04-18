import numpy as np
import cv2
# https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
cap = cv2.VideoCapture(0)

while(1):
	ret, im = cap.read()
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # BGR to grayscale
	# ret, thresh = cv2.threshold(imgray, 90, 255, cv2.THRESH_BINARY_INV)
	# ret, thresh = cv2.threshold(imgray, 90, 255, cv2.THRESH_BINARY)
	# thresh = cv2.GaussianBlur(imgray, (5, 5), 0)

	thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
             cv2.THRESH_BINARY_INV, 35, 40)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	rects = [cv2.boundingRect(ctr) for ctr in contours]
	if(not len(rects)):
		continue

	max_area_rect = rects[0]
	max_area = rects[0][2] * rects[0][3]

	for rect in rects:
		# print(rect[2]*rect[3])
		if(rect[2]*rect[3] > max_area):
			max_area = rect[2]*rect[3]
			max_area_rect = rect

	cv2.rectangle(im, (max_area_rect[0], max_area_rect[1]), (max_area_rect[0] + max_area_rect[2], max_area_rect[1] + max_area_rect[3]), (0, 0, 255), 1)


	# cv2.imshow("title", thresh)
	cv2.imshow("im",im)

	#https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=waitkey
	k = cv2.waitKey(30) & 0xff 
	if k == 27: # 27 is ascii code for ESC
		break


cap.release()
cv2.destroyAllWindows()