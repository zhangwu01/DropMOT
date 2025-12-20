import os
import cv2
import pandas as pd
import trackpy as tp
import numpy as np
from tqdm import tqdm

# ------------------ CONFIG ------------------
video_path = "input/StorageChamber_short.mp4"
base_name = os.path.splitext(os.path.basename(video_path))[0]

CONF_THRESH = 0.5
NMS_THRESH = 0.4
MEMORY = 5

output_name = f"{base_name}_tracked_output_with_IDs_confidence{CONF_THRESH}_memory{MEMORY}.mp4"
output_path = os.path.join("output", output_name)

# YOLOv4 weights and config
weights_path = "yolov4_1ob_best.weights"
config_path = "yolov4_1ob.cfg"
# --------------------------------------------

# Load YOLOv4 network
net = cv2.dnn.readNet(weights_path, config_path)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(640, 512), scale=1/255, swapRB=True)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Collect all detections first (required by trackpy)
all_detections = []

frame_id = 0
print("Detecting droplets in each frame...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    classes, scores, boxes = model.detect(frame, CONF_THRESH, NMS_THRESH)

    for cls_id, score, box in zip(classes, scores, boxes):
        if score < CONF_THRESH:
            continue
        x, y, w, h = box
        cx, cy = x + w/2, y + h/2  # use center coordinates for TrackPy
        all_detections.append({
            'x': cx,
            'y': cy,
            'frame': frame_id,
            'width': w,
            'height': h
        })

    frame_id += 1

cap.release()

# Convert detections to DataFrame for TrackPy
df = pd.DataFrame(all_detections)

# ------------------ Linking ------------------
# Search range should be roughly the max distance a droplet moves between frames
search_range = 10  # pixels, adjust based on droplet speed
linked_df = tp.link_df(df, search_range=search_range, memory=MEMORY)  # for example memory=5 allows temporary disappearance

# ------------------ Write tracked video ------------------
cap = cv2.VideoCapture(video_path)
frame_id = 0

print("Writing tracked video with IDs...")
pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_detections = linked_df[linked_df['frame'] == frame_id]

    for _, row in frame_detections.iterrows():
        x, y, w, h = int(row['x'] - row['width']/2), int(row['y'] - row['height']/2), int(row['width']), int(row['height'])
        pid = int(row['particle'])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        text = f"{pid}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x + w//2 - text_width//2
        text_y = y + h//2 + text_height//2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    out.write(frame)
    frame_id += 1
    pbar.update(1)

pbar.close()
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nTracked video saved to {output_path}")