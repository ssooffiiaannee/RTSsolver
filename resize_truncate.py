import cv2
import numpy as np
# resizing and extracting pieces of an image
# img = cv2.imread('sudoku.png')
# img1 = cv2.resize(img, (200, 200))
# img2 = cv2.resize(img, (500, 500))
# cv2.imshow("title", cv2.resize(img1, (600, 600)))
# cv2.waitKey()

with open('test.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)
print(a.shape, b.shape)

c = np.reshape(a, (-1, 28, 28, 1))
print(a[0])
print(c.shape)
cv2.imshow("title", np.reshape(c[3532], (28, 28)))
cv2.waitKey()