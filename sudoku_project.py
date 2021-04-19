import cv2
import numpy as np
# import tensorflow as tf 
from tensorflow.keras.models import load_model
from sudoku_solver import solve
from copy import deepcopy

img = cv2.imread('sudoku.png')
img1 = img
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(gray, 1, 1, apertureSize = 3)
# ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
shape = gray.shape
# print(shape)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
left = 2000
right = 0
up = 2000
down = 0
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
		if(abs(x1 - x2) <= 2):
			left = min(min(left, x1), x2)
			right = max(max(right, x1), x2)
		if(abs(y1 - y2) <= 2):	
			up = min(min(up, y1), y2)
			down = max(max(down, y1), y2)
		cv2.line(img1,(x1,y1),(x2,y2),(255,255,255),7) # 2 points ,color, thikness

# cv2.imwrite('houghlines3.jpg',img)
# print(type(img))
# print(img.shape)  # (1200, 1200, 3)
# print(gray.shape) #(1200, 1200)
# cv2.imshow("title", im_th)

# cv2.imwrite('sudoCleanedOfRects.jpg', img1)
# cv2.imshow("title", img1)
# cv2.waitKey() 

# /////////////////////////////////////////////////////

model = load_model("mnist_model.h5")

sudoku = [[0,0,0,0,0,0,0,0,0],
		  [0,0,0,0,0,0,0,0,0],
		  [0,0,0,0,0,0,0,0,0],
		  [0,0,0,0,0,0,0,0,0],
		  [0,0,0,0,0,0,0,0,0],
		  [0,0,0,0,0,0,0,0,0],
		  [0,0,0,0,0,0,0,0,0],
		  [0,0,0,0,0,0,0,0,0],
		  [0,0,0,0,0,0,0,0,0]]

# im = cv2.imread("sudoCleanedOfRects.jpg") #("digit-reco-1-in.jpg")
# im2 = im.copy()

im = img1.copy()
im2 = img1.copy()

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

square_length = (right - left)//9
square_height = (down - up)//9
# print("left: ", left, " right: ", right, " up: ", up, " down :", down)

for rect in rects:
# rect = rects[0]
	cv2.rectangle(im2, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
	from_x = max(0, rect[1] - rect[3]//4)
	to_x = min(im_th.shape[0], rect[1] + rect[3] + rect[3]//4)
	from_y = max(0, rect[0] - rect[2]//4)
	to_y = min(im_th.shape[1], rect[0] + rect[2] + rect[2]//4)
	nbr_img = im_th[from_x: to_x, from_y:to_y]
	nbr_resized = cv2.resize(nbr_img, (28, 28))
	nbr_resized = nbr_resized.reshape(-1, 28, 28, 1)/255

	N = model.predict(nbr_resized).argmax()
	x = (rect[1] - left)//square_length # yep look main2.py
	y = (rect[0] - up)//square_height
	# print("x: ", x, " y: ", y)
	sudoku[x][y] = N
	cv2.putText(im2, str(N), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 0), 2)


sudoku_copy = deepcopy(sudoku)
b = solve(sudoku_copy, 0, 0, sudoku_copy[0][0])
# print(b)

for i in range(9):
	for j in range(9):
		if(not sudoku[i][j]):
			cv2.putText(im2, str(b[i][j]), (int((j+1/3)*square_height), int((i+2/3)*square_length)), cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255), 2)

cv2.imshow("title", cv2.resize(im2, (500, 500)))

if(b):
	for i in range(9):
		print(b[i])
else:
	print(False)

# for i in range(9):
# 	print(sudoku[i])

cv2.waitKey()