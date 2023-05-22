import cv2
### Camera ######
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

while True:
    success,img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.05, 10)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Video",img)
    if len(faces) >= 2:
        cv2.imwrite('Resources/HearCasCade_Face_T1.png', img)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# cap = cv2.VideoCapture("Resources/test.mp4")
#
# while True:
#     success,img = cap.read()
#
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(imgGray, 1.05, 10)
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#     cv2.imshow("Video",img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break