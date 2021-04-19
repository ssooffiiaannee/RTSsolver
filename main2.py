import cv2

im = cv2.imread("sudoCleanedOfRects.jpg") #("digit-reco-1-in.jpg")

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV) # cv2.THRESH_BINARY : I made this mistake and it detects the border of the image

# ctrs : fours points of the corners of the image (extremums)
# ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
## findContours: finds all adjacent black pixels as array of numpy arrays
## boundingRect: find the upper-left (min of x', and min of y's), width and height (max x's - min x's, max y's - min y's)
## >>> help(cv2.findContours)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
rect = rects[1]
from_x = max(0, rect[1] - rect[3]//4) ## wtf ? rect[1] for x, rect[0] for ys
to_x = min(im_th.shape[0], rect[1] + rect[3] + rect[3]//4)
from_y = max(0, rect[0] - rect[2]//4)
to_y = min(im_th.shape[1], rect[0] + rect[2] + rect[2]//4)

portion = im_th[from_x: to_x, from_y:to_y] 
# portion = im_th[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
for rect in rects:
	cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1) # image, topleft, bottom right, color, rectangle_width
print(rects[1])
print(portion.shape)
# portion2 = cv2.dilate(portion, (30, 30))
portion2 = cv2.resize(portion, (28, 28))
cv2.imshow("title", portion)
cv2.imshow("title", portion2)
# cv2.imshow("title2", portion2)
# cv2.imshow("title", cv2.resize(portion2, (200, 200))) 
# cv2.imshow("title", im[0:300, 0:300])
# cv2.imshow("title", cv2.resize(im, (500, 500)))
cv2.waitKey()