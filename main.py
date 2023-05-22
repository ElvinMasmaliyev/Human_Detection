import cv2
import time
### Camera ######
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread('Resources/a1.jpg')

(rects, _) = hog.detectMultiScale(image, winStride=(4, 4),
	padding=(20, 20), scale=1.01)
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

print(f"Number of people: {len(rects)}")

cv2.imshow("People Detection", image)
cv2.imwrite('Resources/HOGDescriptor_getDefaultPeopleDetector.png', image)
cv2.waitKey(0)