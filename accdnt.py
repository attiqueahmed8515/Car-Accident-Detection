# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from tkinter import Tk, filedialog

# Step 1: Automatically Download YOLOv4 Files
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"{filename} downloaded.")
    else:
        print(f"{filename} already exists.")

# URLs to YOLOv4 files
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
weights_url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"
names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Download files
download_file(cfg_url, "yolov4.cfg")
download_file(weights_url, "yolov4.weights")
download_file(names_url, "coco.names")

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Step 2: Upload an Image via File Dialog
print("Please select an image of a car accident scenario.")
root = Tk()
root.withdraw()  # Hide the main tkinter window
image_path = filedialog.askopenfilename()

# Read and preprocess the image
image = cv2.imread(image_path)
height, width = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(608, 608), swapRB=True, crop=False)
net.setInput(blob)

# Step 3: Perform Detection
layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

# Process YOLO output
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.3 and classes[class_id] == "car":
            center_x, center_y, box_w, box_h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
            x = int(center_x - (box_w / 2))
            y = int(center_y - (box_h / 2))
            boxes.append([x, y, int(box_w), int(box_h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression to reduce overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Fix for AttributeError: 'tuple' object has no attribute 'flatten'
if len(indices) > 0:
    indices = indices.flatten()
else:
    indices = []

# Step 4: Detect Accidents and Highlight Areas
output_image = image.copy()
accident_detected = False
damaged_zones = []

# Draw bounding boxes
for i in indices:
    x, y, w, h = boxes[i]
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(output_image, f"Car ({confidences[i]:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Check for collision zones and potential damage
for i in range(len(indices)):
    for j in range(i + 1, len(indices)):
        box1 = boxes[indices[i]]
        box2 = boxes[indices[j]]
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
            accident_detected = True
            overlap_x1 = max(x1, x2)
            overlap_y1 = max(y1, y2)
            overlap_x2 = min(x1 + w1, x2 + w2)
            overlap_y2 = min(y1 + h1, y2 + h2)
            cv2.rectangle(output_image, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2), (0, 0, 255), -1)
            damaged_zones.append((overlap_x1, overlap_y1, overlap_x2, overlap_y2))

# Draw labels for accident detection
if accident_detected:
    for (x1, y1, x2, y2) in damaged_zones:
        cv2.putText(output_image, "Accident Zone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Step 5: Display Results
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.imshow(output_image_rgb)
plt.axis("off")
plt.title("Accident Detected" if accident_detected else "No Accident Detected")
plt.show()
