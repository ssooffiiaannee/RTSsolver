import numpy as np
import cv2

im = cv2.imread ('sudoku_grid1.png')#('sudoCleanedOfRects.jpg')# # read picture
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3, 3), 0)

ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
edges = cv2.Canny(thresh, 155, 155, apertureSize = 3)

# cv2.imshow("title", edges)
cv2.imshow("title1", thresh)
cv2.waitKey()