import cv2
import threading
from flask import Flask, render_template, Response, jsonify
import torch
import numpy as np
import time

from my_utils import preprocess_frame, postprocess_detections, schedule_lights, draw_results
from lane_counter import LaneCounter

app = Flask(__name__)

class TrafficEngine:
    def __init__(self):
        self.model = torch.hub.load('./yolov5', 'custom', path='yolov5s.pt', source='local')
        self.model.conf = 0.4
        self.video_paths = ['data/video1.mp4', 'data/video2.mp4', 'data/video3.mp4', 'data/video5.mp4']
        self.captures = [cv2.VideoCapture(path) for path in self.video_paths]
        self.lane_counters = [LaneCounter(i) for i in range(len(self.video_paths))]
        self.current_green_lane = 0
        self.lock = threading.Lock()
        
        self.latest_frame = None
        self.stats = {
            "counts": [0, 0, 0, 0],
            "raw_counts": [0, 0, 0, 0],
            "emergencies": [False, False, False, False],
            "accidents": [False, False, False, False],
            "red_jumps": [0, 0, 0, 0],
            "green_lane": 0,
            "durations": [15, 15, 15, 15],
            "co2": 0.0,
            "eco_score": 100,
            "trend_prediction": "+0%",
            "rl_reward": 0
        }
        
        # For predicting traffic
        self.traffic_history = []
        
    def start(self):
        print("Starting traffic engine background thread...")
        thread = threading.Thread(target=self.process_frames, daemon=True)
        thread.start()

    def process_frames(self):
        while True:
            frames = []
            for cap in self.captures:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                frames.append(frame)

            if len(frames) < len(self.video_paths):
                continue

            results = [self.model(preprocess_frame(f)) for f in frames]

            lane_counts = [] # These are now weighted scores
            raw_counts = []
            emergencies = []
            accidents = []
            red_jumps = []
            
            for i, res in enumerate(results):
                boxes = postprocess_detections(res)
                is_green_val = (i == self.current_green_lane)
                weighted = self.lane_counters[i].count_vehicles(boxes, frames[i], is_green_val)
                
                lane_counts.append(weighted)
                raw_counts.append(self.lane_counters[i].vehicle_count)
                emergencies.append(self.lane_counters[i].has_emergency)
                accidents.append(self.lane_counters[i].has_accident)
                red_jumps.append(self.lane_counters[i].red_light_jumps)

            light_schedule, self.current_green_lane = schedule_lights(lane_counts, emergencies, self.current_green_lane)

            # Analytics Simulation: Prediction & RL
            current_total = sum(raw_counts)
            self.traffic_history.append(current_total)
            if len(self.traffic_history) > 100:
                self.traffic_history.pop(0)
                
            prediction_pct = 0
            if len(self.traffic_history) > 10:
                recent = sum(self.traffic_history[-10:]) / 10
                older = sum(self.traffic_history[:10]) / 10
                if older > 0:
                    prediction_pct = int(((recent - older) / older) * 100)
            
            trend_str = f"+{prediction_pct}%" if prediction_pct >= 0 else f"{prediction_pct}%"
            rl_reward = int(np.random.normal(85, 5)) # Simulate RL reward stabilization metric

            display_frames = []
            total_co2 = 0
            for i in range(len(frames)):
                is_green = (i == self.current_green_lane)
                frame_out = draw_results(frames[i], self.lane_counters[i], light_schedule[i], is_green)
                total_co2 += (self.lane_counters[i].total_idle_frames / 30.0) * 0.05
                display_frames.append(frame_out)

            resized_frames = [cv2.resize(f, (640, 360)) for f in display_frames]
            top_row = np.hstack((resized_frames[0], resized_frames[1]))
            bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
            dashboard = np.vstack((top_row, bottom_row))

            ret, buffer = cv2.imencode('.jpg', dashboard)
            
            with self.lock:
                if ret:
                    self.latest_frame = buffer.tobytes()
                    
                eco_score = max(0, 100 - int(total_co2 * 10))
                
                self.stats = {
                    "counts": raw_counts,
                    "weighted": lane_counts,
                    "emergencies": emergencies,
                    "accidents": accidents,
                    "red_jumps": red_jumps,
                    "green_lane": self.current_green_lane,
                    "durations": light_schedule,
                    "co2": round(total_co2, 2),
                    "eco_score": eco_score,
                    "trend_prediction": trend_str,
                    "rl_reward": rl_reward
                }
            
            time.sleep(0.05) # Yield CPU to prevent GIL locking Flask

engine = TrafficEngine()
engine.start()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        frame = engine.latest_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    with engine.lock:
        return jsonify(engine.stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
