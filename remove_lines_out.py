import cv2
import numpy as np

img = cv2.imread("out.png") # ('sudoku.png')

img1 = img
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
# ret, im_th = cv2.threshold(gray, 90, 90, cv2.THRESH_BINARY_INV)
im_th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,35,12)

edges = cv2.Canny(im_th, 1, 1, apertureSize = 3)
cv2.imshow("im_th", im_th)
cv2.waitKey()
shape = gray.shape
lines = cv2.HoughLines(edges,1,np.pi/3600,100)

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

	    # cv2.line(img1, (x1,y1),(x2,y2),(255,255,255),3) # 2 points ,color, thikness
	    cv2.line(im_th, (x1,y1),(x2,y2),(0,0,0),2)

# cv2.imwrite('houghlines3.jpg',img)
# print(type(img))
# print(img.shape)  # (1200, 1200, 3)
# print(gray.shape) #(1200, 1200)
# cv2.imshow("title", im_th)

# cv2.imwrite('sudoCleanedOfRects.jpg', img1)
# cv2.imshow("img1", img1)
# cv2.waitKey()
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, gray = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
print(im_th.shape)
cv2.imshow("gray", im_th)
cv2.imwrite("out_free_of_lines.png", im_th)
cv2.waitKey() 