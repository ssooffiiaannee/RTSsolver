import numpy as np
import cv2

img = cv2.imread("out.png") # ('sudoku.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
# ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

# The blockSize determines the size of the neighbourhood area and C is a 
# constant that is subtracted from the mean or weighted sum of the neighbourhood pixels
im_th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,35,12)


cv2.imshow("im_th", im_th)
cv2.waitKey()