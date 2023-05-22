import cv2

# Load the image
image = cv2.imread('Resources/a1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascades classifier for person detection
classifier = cv2.CascadeClassifier('Resources/haarcascade_fullbody.xml')

# Detect people in the image
people = classifier.detectMultiScale(gray)

# Draw a rectangle around each person and count them
count = 0
for (x, y, w, h) in people:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    count += 1

# Add text displaying the count of people
cv2.putText(image, f'Number of people: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Show the image
cv2.imshow('Counted People', image)
cv2.imwrite('Resources/HearCasCade.png',image)
cv2.waitKey(0)
cv2.destroyAllWindows()