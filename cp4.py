import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)

#img[100:200,300:400]=255,0,0
img[:]=100,100,0

cv2.line(img,(0,0),(300,300),(0,250,0),2)
#cv2.rectangle(img,(0,0),(300,300),(0,0,250),2)
cv2.rectangle(img,(0,0),(200,200),(0,0,250),cv2.FILLED)
cv2.circle(img,(200,200),100,(255,0,0),3)
cv2.putText(img," OPENCV ",(300,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

cv2.imshow("Output",img)

cv2.waitKey(0)