import streamlit as st
import torch
import torchvision.models.detection as detection
from torchvision.transforms import ToTensor
from PIL import Image
import cv2
import numpy as np
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the pre-trained model
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO dataset class labels mapping
class_labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Streamlit web app
st.title("Object Detection")
st.write("Upload an image and get the detected objects.")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    transform = ToTensor()
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get predicted labels and bounding boxes
    boxes = output[0]['boxes'].cpu().numpy().tolist()
    labels = output[0]['labels'].cpu().numpy().tolist()

    # Draw bounding boxes on image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    object_counts = {}  # Dictionary to store object labels and counts
    for box, label in zip(boxes, labels):
        x, y, w, h = map(int, box)
        cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 2)
        cv2.putText(image, class_labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # Count the objects
        if class_labels[label] in object_counts:
            object_counts[class_labels[label]] += 1
        else:
            object_counts[class_labels[label]] = 1

    # Display the image with bounding boxes
    st.image(image, caption="Object Detection", use_column_width=True)

    # Display the count of different objects found
    st.write("Object Counts:")
    for label, count in object_counts.items():
        st.write(f"The Count of Object [{label}] is {count}.")

