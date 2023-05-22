import cv2
import numpy as np

def empty(a):
    pass
img_name = 'lambo'
img_ex = '.png'
path = "Resources/" + img_name + img_ex
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",6,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",23,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",51,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",251,255,empty)
cv2.createTrackbar("Val Min","TrackBars",70,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min = cv2.getTrackbarPos("Val Min","TrackBars")
    v_max = cv2.getTrackbarPos("Val Max","TrackBars")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("Output",img)
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Mask",mask)
    cv2.imshow("Result",imgResult)
    cv2.waitKey(1)