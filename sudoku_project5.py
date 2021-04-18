import numpy as np
import cv2
from tensorflow.keras.models import load_model
# from sudoku_solver import solve
from copy import deepcopy
import time

def put_digits(sudoku, N, sudoku_length, sudoku_height, x_coord, y_coord):
	# x = x_coord/(max(sudoku_length, x_coord)/9)
	# y = y_coord/(max(sudoku_height, y_coord)/9)
	# x = min(8, x)
	# y = min(8, y)
	# sudoku[int(x)][int(y)] = N
	# if(N == 5):
	# 	print("======")
	# 	print(sudoku_length, sudoku_height, x_coord, y_coord)
	x = x_coord/(sudoku_length/9)
	y = y_coord/(sudoku_height/9)
	# if(N == 5):
		# print(x, y)
	sudoku[int(x)][int(y)] = N
	# print("put_digits")

def print_sudoku(copy):
	# print("print_sudoku")
	if(copy):
		for i in range(9):
			print(copy[i])
	else:
		print("cannot be solved")

def solve(a, c, d, k, t):
	if(time.time() - t > 2):
		return a
	a[c][d] = k
	for i in range(9):
		for j in range(9):
			if(not a[i][j]):
				for k in range(1, 10):
					ans = True
					b = False
					for l in range(9):
						if(a[i][l] == k or a[l][j] == k or a[(i//3)*3 + l//3][(j//3)*3 + l%3] == k):
							ans = False
					if(ans):
						b = solve(a, i, j, k, t)
						if(b):
							return b
				a[c][d] = 0
				return False
	return a
# //////////////////////////////////////////////
cap = cv2.VideoCapture(0)
out = 0
model = load_model("computer_mnist_model100.h5") #("computer_mnist_model.h5") #("mnist_model.h5")
# //////////////////////////////////////////////

while(1):
	sudoku = [[0,0,0,0,0,0,0,0,0],
			  [0,0,0,0,0,0,0,0,0],
			  [0,0,0,0,0,0,0,0,0],
			  [0,0,0,0,0,0,0,0,0],
			  [0,0,0,0,0,0,0,0,0],
			  [0,0,0,0,0,0,0,0,0],
			  [0,0,0,0,0,0,0,0,0],
			  [0,0,0,0,0,0,0,0,0],
			  [0,0,0,0,0,0,0,0,0]]

	ret, im = cap.read()
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # BGR to grayscale
	thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
             cv2.THRESH_BINARY_INV, 35, 25)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_NONE)

	rects = [cv2.boundingRect(ctr) for ctr in contours]
	if(not len(rects)):
		continue

	max_area_rect = 0
	max_area_ctr = 0
	max_area = 0

	for ctr in contours:
		rect = cv2.boundingRect(ctr)

		if(rect[2]*rect[3] > max_area):
			max_area = rect[2]*rect[3]
			max_area_rect = rect
			max_area_ctr = ctr

	if(str(max_area_ctr) == '0'):
		continue
	epsilon = 0.01 * cv2.arcLength(max_area_ctr, True)
	approx = cv2.approxPolyDP(max_area_ctr, epsilon, True)
	im = cv2.drawContours(im, [approx], -1, (0, 255, 0), 1)
	# print("approx", approx)
	if(approx.shape[0] != 4):
		# print("73")
		continue
	else:
		print("76")
		bottom_left = approx[1][0]
		up_left = approx[0][0]
		bottom_right = approx[2][0]
		up_right = approx[3][0]
		# up_left = approx[1][0] #bottom left
		# up_right = approx[0][0] # up_left
		# bottom_left = approx[2][0] # bottom_right
		# bottom_right = approx[3][0] #up_right
		input_pts = np.float32([up_left, up_right, bottom_left, bottom_right])
		# ==============
		#bottom_left,up_left,bottom_right,up_right
		# print(up_left, up_right, bottom_left, bottom_right)
		height = max(abs(bottom_left[1] - up_left[1]), abs(up_right[1]-bottom_right[1]))
		width = max(abs(bottom_left[0] - bottom_right[0]), abs(up_left[0]-up_right[0]))
		output_pts = np.float32([[0, 0], [width, 0],[0, height],[width, height]])
# =========================================================================================
		# Compute the perspective transform M
		M = cv2.getPerspectiveTransform(input_pts, output_pts)	 
		# Apply the perspective transformation to the image
		out = cv2.warpPerspective(im, M, (width, height), flags = cv2.INTER_LINEAR)
		# print(up_left, up_right, bottom_left, bottom_right)
		# print(width, height)	
		if(out.shape[0] > 180 and out.shape[1] > 180):# or abs(height - width) > 50):
			thresh_out = cv2.warpPerspective(thresh, M, (width, height), flags = cv2.INTER_LINEAR)

			edges = cv2.Canny(thresh_out, 1, 1, apertureSize = 3)
			lines = cv2.HoughLines(edges,1,np.pi/180,100)
			if(str(lines) == 'None'):
				continue

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
				    if(abs(x1 - x2) > 50 and abs(y1 - y2) > 50):
				    	continue
				    # cv2.line(img1, (x1,y1),(x2,y2),(255,255,255),3) # 2 points ,color, thikness
				    cv2.line(thresh_out, (x1,y1),(x2,y2),(0,0,0),4)

			im_th = thresh_out

			ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			rects = [cv2.boundingRect(ctr) for ctr in ctrs]
			sudoku_area = im_th.shape[0]*im_th.shape[1]//81
			digits_array = []
			min_area_div = 12
			for rect in rects:
				if(rect[2]*rect[3] < sudoku_area//min_area_div):
					continue
				cv2.rectangle(out, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
				from_x = max(0, rect[1] - rect[3]//4)
				to_x = min(im_th.shape[0], rect[1] + rect[3] + rect[3]//4)
				from_y = max(0, rect[0] - rect[2]//4)
				to_y = min(im_th.shape[1], rect[0] + rect[2] + rect[2]//4)
				nbr_img = im_th[from_x: to_x, from_y:to_y]/255
				nbr_resized_disp = cv2.resize(nbr_img, (28, 28))
				nbr_resized = nbr_resized_disp.reshape(-1, 28, 28, 1)
				digits_array.append(nbr_resized)

			pred_array = model.predict(np.array(digits_array).reshape(-1, 28, 28, 1)).argmax(axis = 1)
			N = 0
			for rect in rects:
				if(rect[2]*rect[3] < sudoku_area//min_area_div):
					continue
				# if(out.shape[1] <= rect[1] or out.shape[0] <= rect[0]):
				# 	continue
				# cv2.putText(out, str(pred_array[N]), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 0), 1)	
				put_digits(sudoku, pred_array[N], out.shape[0], out.shape[1], rect[1], rect[0]) # shape[0]:length shape[1] height rect[1]: x, rect[0] : y
				N += 1
			if(N < 17):
				print("not enough")
				continue
			sudoku_copy = deepcopy(sudoku)
			t = time.time()
			# print(out.shape, im_th.shape)
			# print_sudoku(sudoku)
			solve(sudoku_copy, 0, 0, sudoku_copy[0][0], t)
			# print_sudoku(sudoku_copy)
			for i in range(9):
				for j in range(9):
					if(not sudoku[i][j]):
						cv2.putText(out, str(sudoku_copy[i][j]), (int((j+1/3)*out.shape[1]/9), int((i+3/4)*out.shape[0]/9)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
				
	if(str(out) == "0"):
		continue
	cv2.imshow("out", out)
	cv2.imshow("title", im)
	cv2.imshow("thresh", thresh)

	k = cv2.waitKey(30) & 0xff 
	if k == 27: # 27 is ascii code for ESC
		break


cap.release()
cv2.destroyAllWindows()