🚗 Car Accident Detection using YOLOv4

This project implements a computer vision-based car accident detection system using YOLOv4 (You Only Look Once) and OpenCV. The system detects vehicles in an image, analyzes spatial relationships between them, and identifies potential collision (accident) zones based on overlapping bounding boxes.

📌 Project Overview

The goal of this project is to automatically detect possible car accidents from images using deep learning-based object detection.

It performs the following tasks:

Loads YOLOv4 pre-trained model automatically
Detects cars in an input image
Draws bounding boxes around detected vehicles
Checks for overlaps between vehicles
Highlights collision (accident) zones in red
Displays final result using Matplotlib
⚙️ Features
🔄 Automatic download of YOLOv4 configuration, weights, and class labels
🚗 Vehicle detection using YOLOv4 + OpenCV DNN
📦 Non-Maximum Suppression (NMS) for better detection accuracy
⚠️ Collision detection based on bounding box overlap
🔴 Visual highlighting of accident zones
🖼️ GUI-based image selection using Tkinter
🧠 Technology Used
Python 🐍
OpenCV (cv2)
NumPy
Matplotlib
YOLOv4 (Darknet)
Tkinter (for file selection)
Requests (for downloading model files)
🚀 How It Works
YOLOv4 model files are downloaded automatically if not present
An image is selected via file dialog
YOLO detects objects (cars) in the image
Bounding boxes are generated for each detected car
The system checks overlapping regions between cars
If overlap is found → it is marked as an accident zone (red area)
Final output image is displayed
📷 Output
🟩 Green boxes → Detected cars
🟥 Red highlighted area → Accident/Collision zone

🎯 Future Improvements
Real-time video accident detection (CCTV/live stream)
Integration with Deep Learning classifiers for severity detection
Firebase/Cloud storage for accident logging
Alert system for emergency response

👨‍💻 Author
Developed by Attique Ahmed
BS Artificial Intelligence
