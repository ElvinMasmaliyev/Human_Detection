import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# Load the image
image = cv2.imread('Resources/crdt.jpg')

# Load the DETR configuration
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/detr_resnet50.yaml")
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/datasets/coco/\
detection/detr_resnet50/148943148/model_final.pth"

# Create the DETR predictor
predictor = DefaultPredictor(cfg)

# Run DETR on the image
outputs = predictor(image)

# Extract the people detections
classes = outputs["instances"].pred_classes
scores = outputs["instances"].scores
boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
people = boxes[classes == 0]

# Draw a rectangle around each person and count them
count = 0
for person in people:
    x1, y1, x2, y2 = person
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    count += 1

# Add text displaying the count of people
cv2.putText(image, f'Number of people: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Show the image
cv2.imshow('Counted People', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
