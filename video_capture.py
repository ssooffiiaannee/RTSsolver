import numpy as np
import cv2
# https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
cap = cv2.VideoCapture(0)

while(1):
	ret, im = cap.read()
	cv2.imshow("title", im)

	#https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=waitkey
	k = cv2.waitKey(30) & 0xff 
	if k == 27: # 27 is ascii code for ESC
		break


cap.release()
cv2.destroyAllWindows()

# cv2.imshow("title", img)
# cv2.imwrite("sudoku_grid1.png", img)
# cv2.waitKey()