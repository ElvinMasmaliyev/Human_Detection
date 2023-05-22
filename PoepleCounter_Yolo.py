import cv2
import numpy as np

# Load the image
image = cv2.imread('Resources/crda.jpg')

# Load the YOLOv4 model
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get the output layer names and colors for the bounding boxes
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = open('coco.names').read().strip().split('\n')
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Pre-process the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Pass the image through the network
net.setInput(blob)
outs = net.forward(output_layers)

# Extract the bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            x = center_x - w // 2
            y = center_y - h // 2
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression to remove redundant boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Count the number of detected people
print(f"Number of people: {len(indices)}")

# Draw the bounding boxes around the detected people
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    color = colors[class_ids[i]]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
