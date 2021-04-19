import cv2
import numpy as np
# import tensorflow as tf 
from tensorflow.keras.models import load_model

# /////////////////////////////////////////

model = load_model("FChollet_model_comp_digits2.h5")

im = cv2.imread("digit_number_img_3.jpg") # ("out_free_of_lines.png") #("digit-reco-1-in.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im2 = cv2.imread("digit_number_img_1.jpg")
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

im = im.reshape(28, 28, 1)/255
im2 = im2.reshape(28, 28, 1)/255

im = np.array([im, im2])

# print(im.shape)


print(model.predict(im))#.argmax(axis = 1))


# /////////////////////////////////////////
# im = cv2.imread("out_free_of_lines.png") #("digit-reco-1-in.jpg")

# im2 = im.copy()

# im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# # im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# im_th = im_gray
# # ret, im_th = cv2.threshold(im_gray, 89, 255,cv2.THRESH_BINARY_INV)	
# # cv2.imshow("title33", im_th)
# # cv2.waitKey()
# ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# rects = [cv2.boundingRect(ctr) for ctr in ctrs]
# sudoku_area = im_th.shape[0]*im_th.shape[1]//81

# for rect in rects:
# 	if(rect[2]*rect[3] < sudoku_area//15):
# 		continue
# 	cv2.rectangle(im2, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
# 	from_x = max(0, rect[1] - rect[3]//4)
# 	to_x = min(im_th.shape[0], rect[1] + rect[3] + rect[3]//4)
# 	from_y = max(0, rect[0] - rect[2]//4)
# 	to_y = min(im_th.shape[1], rect[0] + rect[2] + rect[2]//4)
# 	nbr_img = im_th[from_x: to_x, from_y:to_y]/255
# 	nbr_resized_disp = cv2.resize(nbr_img, (28, 28))
# 	nbr_resized = nbr_resized_disp.reshape(-1, 28, 28, 1)

# 	N = model.predict(nbr_resized).argmax()
# 	cv2.putText(im2, str(N), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 0), 1)

# # cv2.imshow("title", cv2.resize(im2, (500, 500)))
# cv2.imshow("title", im2)
# cv2.waitKey()
