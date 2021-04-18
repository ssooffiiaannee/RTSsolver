import cv2
import numpy as np

img = cv2.imread('sudoku.png')

img1 = img
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(gray, 1, 1, apertureSize = 3)
# ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
shape = gray.shape
lines = cv2.HoughLines(edges,1,np.pi/180,200)

for i in range(len(lines)):
	for rho,theta in lines[i]:
	    a = np.cos(theta)
	    b = np.sin(theta)
	    x0 = a*rho
	    y0 = b*rho
	    x1 = int(x0 + (shape[0])*(-b))
	    y1 = int(y0 + (shape[1])*(a))
	    x2 = int(x0 - (shape[0])*(-b))
	    y2 = int(y0 - (shape[1])*(a))
	    # print(x1, y1, x2, y2)

	    cv2.line(img1,(x1,y1),(x2,y2),(255,0,255),7) # 2 points ,color, thikness

# cv2.imwrite('houghlines3.jpg',img)
# print(type(img))
# print(img.shape)  # (1200, 1200, 3)
# print(gray.shape) #(1200, 1200)
# cv2.imshow("title", im_th)

# cv2.imwrite('sudoCleanedOfRects.jpg', img1)
cv2.imshow("title", img1)
cv2.waitKey() 