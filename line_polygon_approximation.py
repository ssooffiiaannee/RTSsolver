import cv2
import numpy as np

# im = cv2.imread ('sudoku_grid3.png')#('sudoCleanedOfRects.jpg')# # read picture

# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # BGR to grayscale

# ret, thresh = cv2.threshold(imgray, 120, 255, cv2.THRESH_BINARY_INV)

# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# # for i in range(len(contours)):
# # 	epsilon = 0.01 * cv2.arcLength(contours[i], True)
# # 	approx = cv2.approxPolyDP(contours[i], epsilon, True)
# # 	im = cv2.drawContours(im, [approx], -1, (0, 255, 0), 3)

# # ////////////////////////////////////////////////////
# # print(len(contours))

# # epsilon = 0.01 * cv2.arcLength(contours[0], True)
# # approx = cv2.approxPolyDP(contours[0], epsilon, True)
# # im = cv2.drawContours(im, [approx], -1, (0, 255, 0), 1)


# # ////////////////////////////////////////////////////
# rects = [cv2.boundingRect(ctr) for ctr in contours]


# max_area_rect = rects[0]
# max_area = rects[0][2] * rects[0][3]

# for rect in rects:
# 	# print(rect[2]*rect[3])
# 	if(rect[2]*rect[3] > max_area):
# 		max_area = rect[2]*rect[3]
# 		max_area_rect = rect

# cv2.rectangle(im, (max_area_rect[0], max_area_rect[1]), (max_area_rect[0] + max_area_rect[2], max_area_rect[1] + max_area_rect[3]), (0, 0, 255), 1)

# # ////////////////////////////////////////////////

# # for rect in rects:
# 	# cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)


# # im = cv2.drawContours(im, [approx], -1, (0, 255, 0), 3)
# cv2.imshow("Contour", cv2.resize(im, (500, 500)))
# # cv2.imshow("bla", thresh)
# cv2.waitKey()
# cv2.destroyAllWindows()




im = cv2.imread("curve.png")
im = cv2.imread("ex.png") #("sudoCleanedOfRects.jpg") #("digit-reco-1-in.jpg")


im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY)
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_SIMPLE) cv2.RETR_EXTERNAL

cnt = ctrs[0]
canvas = im.copy()

epsilon =  0.01*cv2.arcLength(ctrs[0], True)
approx = cv2.approxPolyDP(ctrs[0], epsilon, True)

canvas = cv2.drawContours(canvas, ctrs, -1, (255, 0, 0), 3)
bla = cv2.drawContours(im, [approx], -1, (0, 255, 0), 3)

cv2.fillConvexPoly(im, approx, (0, 0, 255))

cv2.imshow("title", im)
cv2.waitKey()

cv2.imshow("title1", cv2.resize(canvas, (500, 500)))
cv2.waitKey()

print(bla.shape)

cv2.imshow("title2", cv2.resize(bla, (500, 500)))
cv2.waitKey()