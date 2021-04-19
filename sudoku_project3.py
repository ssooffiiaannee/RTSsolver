import numpy as np
import cv2
from tensorflow.keras.models import load_model


cap = cv2.VideoCapture(0)
out = 0
model = load_model("computer_mnist_model100.h5")# ("computer_mnist_model.h5") #("mnist_model.h5")

while(1):
	ret, im = cap.read()
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # BGR to grayscale
	# imgray = cv2.GaussianBlur(imgray, (3, 3), 0)
	thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
             cv2.THRESH_BINARY_INV, 35, 12)
	# ret, thresh = cv2.threshold(imgray,90, 255, cv2.THRESH_BINARY_INV)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_NONE)

	rects = [cv2.boundingRect(ctr) for ctr in contours]


	max_area_rect = 0
	max_area_ctr = 0
	max_area = 0

	for ctr in contours:
		rect = cv2.boundingRect(ctr)

		if(rect[2]*rect[3] > max_area):
			max_area = rect[2]*rect[3]
			max_area_rect = rect
			max_area_ctr = ctr

	# cv2.rectangle(im, (max_area_rect[0], max_area_rect[1]), (max_area_rect[0] + max_area_rect[2], max_area_rect[1] + max_area_rect[3]), (0, 0, 255), 1)
	if(str(max_area_ctr) == '0'):
		continue
	epsilon = 0.01 * cv2.arcLength(max_area_ctr, True)
	approx = cv2.approxPolyDP(max_area_ctr, epsilon, True)
	im = cv2.drawContours(im, [approx], -1, (0, 255, 0), 1)

	# im_rect =  im[max_area_rect[1]:max_area_rect[1]+max_area_rect[3], max_area_rect[0]:max_area_rect[0]+max_area_rect[2]]
	if(approx.shape[0] == 4):
		up_left = approx[0][0]
		up_right = approx[3][0]
		bottom_left = approx[1][0]
		bottom_right = approx[2][0]

		input_pts = np.float32([up_left,up_right,bottom_left,bottom_right])

		height = max(abs(bottom_left[1] - up_left[1]), abs(up_right[1]-bottom_right[1]))
		width = max(abs(bottom_left[0] - bottom_right[0]), abs(up_left[0]-up_right[0]))
		output_pts = np.float32([[0, 0], [width, 0],[0, height],[width, height]])

		# Compute the perspective transform M
		M = cv2.getPerspectiveTransform(input_pts, output_pts)
		 
		# Apply the perspective transformation to the image
		out = cv2.warpPerspective(im, M, (width, height), flags = cv2.INTER_LINEAR)
		
		# cv2.imshow("out", out)
		# cv2.imwrite("out.png", out)

		thresh_out = cv2.warpPerspective(thresh, M, (width, height), flags = cv2.INTER_LINEAR)

		# imgray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) # BGR to grayscale
		# thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
  		# cv2.THRESH_BINARY_INV,35,12)

		edges = cv2.Canny(thresh_out, 1, 1, apertureSize = 3)
		# shape = gray.shape
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
			    cv2.line(thresh_out, (x1,y1),(x2,y2),(0,0,0),3)

		im_th = thresh_out

		ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		rects = [cv2.boundingRect(ctr) for ctr in ctrs]
		sudoku_area = im_th.shape[0]*im_th.shape[1]//81
		digits_array = []
		for rect in rects:
			if(rect[2]*rect[3] < sudoku_area//15):
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
			
			# N = model.predict(nbr_resized).argmax()
			# cv2.putText(out, str(N), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 0), 1)

		pred_array = model.predict(np.array(digits_array).reshape(-1, 28, 28, 1)).argmax(axis = 1)
		N = 0
		for rect in rects:
			if(rect[2]*rect[3] < sudoku_area//15):
				continue
			cv2.putText(out, str(pred_array[N]), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 0), 1)
			N += 1
		
		# cv2.imshow("out", out)
		# cv2.imshow("thresh_out", thresh_out)
		# cv2.imshow("thresh", thresh)

	if(str(out) == "0"):
		continue
	cv2.imshow("out", out)
	cv2.imshow("title", im)
	# cv2.imshow("thresh", thresh)

	# print(out.shape)

	k = cv2.waitKey(30) & 0xff 
	if k == 27: # 27 is ascii code for ESC
		break


cap.release()
cv2.destroyAllWindows()