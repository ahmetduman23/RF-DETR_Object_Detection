import cv2
import numpy as np
import json
from rfdetr import RFDETRBase
from google.colab.patches import cv2_imshow

# Load the object detection model
model = RFDETRBase()

# Load class names from JSON file
with open('class_id.json', 'r', encoding='utf-8') as file:
    class_names = json.load(file)

# Define colors for each class (in BGR format)
class_colors = {
    'car': (0, 0, 255),        # red
    'motorcycle': (0, 255, 0), # green
    'person': (255, 0, 0),     # blue
}

# Load input video
cap = cv2.VideoCapture('vecteezy_traffic.mov') # https://www.vecteezy.com/video/17745973-traffic-on-jalan-merdeka-barat-is-dominated-by-motorcyclists

# Get video resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video codec and create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

frame_count = 0  # Frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run object detection on current frame
    detections = model.predict(frame, threshold=0.5)

    for i, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(detections.class_id[i])
        label = class_names.get(str(class_id))  # Convert class ID to label name
        
        if label is None:
            label = "Unknown"

        confidence = detections.confidence[i]

        # Get box color, default to yellow for unknown classes
        color = class_colors.get(label, (0, 255, 255))

        # Draw bounding box
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        text = f"{label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Compute text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Draw background rectangle for text
        text_bg_x1, text_bg_y1 = x1, y1 - text_height - 10
        text_bg_x2, text_bg_y2 = x1 + text_width + 4, y1
        cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)

        # Draw text label
        cv2.putText(frame, text, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Show every 30th frame in Colab
    frame_count += 1
    if frame_count % 30 == 0:
        resized_frame = cv2.resize(frame, (width, height))
        cv2_imshow(resized_frame)

    # Write processed frame to output video
    out.write(frame)

# Release video resources
cap.release()
out.release()
