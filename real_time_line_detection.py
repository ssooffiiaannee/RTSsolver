import numpy as np
import cv2
# https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
cap = cv2.VideoCapture(0)

while(1):
	ret, img = cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# gray = cv2.GaussianBlur(gray, (3, 3), 0)

	ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
	
	edges = cv2.Canny(thresh, 50, 200, apertureSize = 3) # min_val max_ val are intensity edges
	lines = cv2.HoughLines(edges, 1, np.pi/180, 100) 	 # (edges, ro_accuracy(pixels), theta_accuracy (rads), minimum_vote)
	# print(lines)
	if(str(lines) != 'None'):
		for line in lines:
			for rho,theta in line:
			    a = np.cos(theta)
			    b = np.sin(theta)
			    x0 = a*rho
			    y0 = b*rho
			    x1 = int(x0 + 1000*(-b))
			    y1 = int(y0 + 1000*(a))
			    x2 = int(x0 - 1000*(-b))
			    y2 = int(y0 - 1000*(a))

			cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
	cv2.imshow("title", img)
	#https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=waitkey
	k = cv2.waitKey(30) & 0xff 
	if k == 27: # 27 is ascii code for ESC
		break


cap.release()
cv2.destroyAllWindows()

# cv2.imshow("title", img)
# cv2.imwrite("sudoku_grid1.png", img)
# cv2.waitKey()