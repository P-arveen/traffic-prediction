import cv2
import torch
import os
import csv
import time
import numpy as np
from my_utils import preprocess_frame, postprocess_detections, schedule_lights, draw_results
from lane_counter import LaneCounter
from display import display_dashboard

# Load YOLOv5s model (from local or auto-download)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.4  # confidence threshold

# Define video sources from data folder
video_paths = ['data/video1.mp4', 'data/video2.mp4', 'data/video3.mp4', 'data/video5.mp4']
captures = [cv2.VideoCapture(path) for path in video_paths]

# Initialize lane counters
lane_counters = [LaneCounter(i) for i in range(len(video_paths))]

# Initialize CSV Logging
log_file = open('traffic_analytics_log.csv', mode='w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['Timestamp', 'Lane 1 Count', 'Lane 2 Count', 'Lane 3 Count', 'Lane 4 Count', 'Green Lane ID'])

while True:
    frames = []
    for cap in captures:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if len(frames) < len(video_paths):
        break

    # Inference on each frame
    results = [model(preprocess_frame(f)) for f in frames]

    lane_counts = []
    for i, res in enumerate(results):
        boxes = postprocess_detections(res)
        # Pass the frame shape to calculate ROI
        count = lane_counters[i].count_vehicles(boxes, frames[i].shape)
        lane_counts.append(count)

    # Compute signal duration
    light_schedule = schedule_lights(lane_counts)

    # Dynamically assign Green Light to the lane with highest traffic
    current_green_lane = int(np.argmax(lane_counts))

    # Log data every frame (or could throttle it)
    log_writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S')] + lane_counts + [current_green_lane + 1])

    display_frames = []
    # Display results per lane
    for i in range(len(frames)):
        is_green = (i == current_green_lane)
        frame_out = draw_results(frames[i], lane_counters[i], light_schedule[i], is_green)
        display_frames.append(frame_out)

    # Output unified Major Project Dashboard
    display_dashboard(display_frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
