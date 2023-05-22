import cv2
import numpy as np

img = cv2.imread("Resources/test.png")

hor = np.hstack((img,img))
ver = np.vstack((img,img))

cv2.imshow("Hor",hor)
cv2.imshow("Ver",ver)

cv2.waitKey(0)