import cv2

img = cv2.imread("Resources/test.png")
print(img.shape)


imgResize = cv2.resize(img,(300,200))
imgCropped = img[200:300,300:500]

cv2.imshow("Output",img)
cv2.imshow("Resize",imgResize)
cv2.imshow("Crop",imgCropped)

cv2.waitKey(0)
