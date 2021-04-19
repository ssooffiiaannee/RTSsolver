import numpy as np
import cv2

# im = cv2.imread("sudoCleanedOfRects.jpg")
# im2 = im.copy()
# cv2.putText(im2, "sofiane", (100, 100),cv2.FONT_HERSHEY_DUPLEX, 6, (200, 0, 0), 5)

width, height = 800, 600
x1, y1 = 0, 0
x2, y2 = 200, 400
im2 = np.ones((height, width)) * 255

line_thickness = 2
cv2.line(im2, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

cv2.imshow("title", im2)
cv2.waitKey()